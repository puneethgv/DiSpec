"""CUDA-graph capture for the batch-1 decode step (bucketed).

DiSpec's decode is launch-bound: a single token's GPU work is ~3 ms but the eager
per-layer Python loop costs ~20 ms in kernel-launch + dispatch overhead. A CUDA graph
records the whole forward once and *replays* it as one launch, removing nearly all of
that overhead — the trick vLLM relies on. Measured ~3.6x faster decode here.

CUDA graphs need static shapes, so attention runs over a fixed MAX_CTX window (padding
masked out). Since a fixed window wastes work on short sequences, we capture several
*buckets* (256/512/1024/2048) and replay the smallest one that fits the current context;
sequences longer than the largest bucket fall back to the eager runner.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dispec.engine.model_runner import ModelRunner, _apply_rope
from dispec.kv.paged_cache import PagedKVCache

DEFAULT_BUCKETS = (256, 512, 1024, 2048)


@dataclass
class _Bucket:
    max_ctx: int
    tok: torch.Tensor
    pos: torch.Tensor
    wslot: torch.Tensor
    cslots: torch.Tensor
    mask: torch.Tensor
    graph: torch.cuda.CUDAGraph
    out: torch.Tensor


class CudaGraphDecoder:
    def __init__(self, runner: ModelRunner, cache: PagedKVCache,
                 buckets: tuple[int, ...] = DEFAULT_BUCKETS):
        self.r = runner
        self.cache = cache
        self.buckets = sorted(buckets)
        self.max_ctx = self.buckets[-1]
        self._b: dict[int, _Bucket] = {}

    def _forward(self, tok, pos, wslot, cslots, mask) -> torch.Tensor:
        r = self.r
        cos, sin = r.rope(pos)
        x = r.embed(tok)
        for i, layer in enumerate(r.layers):
            attn = layer.self_attn
            h = layer.input_layernorm(x)
            if r.fused:
                qkv = F.linear(h, r._qkv_w[i], r._qkv_b[i])
                q, k, v = qkv.split([r._qsz, r._kvsz, r._kvsz], dim=-1)
                q = q.view(1, r.num_heads, r.head_dim)
                k = k.view(1, r.num_kv_heads, r.head_dim)
                v = v.view(1, r.num_kv_heads, r.head_dim)
            else:
                q = attn.q_proj(h).view(1, r.num_heads, r.head_dim)
                k = attn.k_proj(h).view(1, r.num_kv_heads, r.head_dim)
                v = attn.v_proj(h).view(1, r.num_kv_heads, r.head_dim)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
            self.cache.key[i].index_copy_(0, wslot, k)
            self.cache.value[i].index_copy_(0, wslot, v)
            ki = self.cache.key[i][cslots].repeat_interleave(r.kv_groups, dim=1)
            vi = self.cache.value[i][cslots].repeat_interleave(r.kv_groups, dim=1)
            oh = F.scaled_dot_product_attention(
                q.transpose(0, 1), ki.transpose(0, 1), vi.transpose(0, 1),
                attn_mask=mask[None, None, :])
            x = x + attn.o_proj(oh.transpose(0, 1).reshape(1, r.num_heads * r.head_dim))
            x = x + r._mlp(layer, layer.post_attention_layernorm(x), i)
        return r.lm_head(r.norm(x))

    def capture(self) -> None:
        """Capture one graph per bucket. Run while the cache holds no live data."""
        dev = self.r.device
        with torch.no_grad():
            for mc in self.buckets:
                tok = torch.zeros(1, dtype=torch.long, device=dev)
                pos = torch.zeros(1, dtype=torch.long, device=dev)
                wslot = torch.zeros(1, dtype=torch.long, device=dev)
                cslots = torch.zeros(mc, dtype=torch.long, device=dev)
                mask = torch.zeros(mc, dtype=torch.bool, device=dev)
                for _ in range(3):  # warmup before capture
                    self._forward(tok, pos, wslot, cslots, mask)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = self._forward(tok, pos, wslot, cslots, mask)
                self._b[mc] = _Bucket(mc, tok, pos, wslot, cslots, mask, graph, out)

    def can_handle(self, ctx_len: int) -> bool:
        return ctx_len <= self.max_ctx

    @torch.no_grad()
    def decode(self, token: int, position: int, write_slot: int,
               ctx_slots: torch.Tensor) -> torch.Tensor:
        """Replay the smallest-fitting bucket for one token. ctx_slots: context slots."""
        ctx_len = ctx_slots.shape[0]
        mc = next(b for b in self.buckets if b >= ctx_len)
        b = self._b[mc]
        b.tok.fill_(token)
        b.pos.fill_(position)
        b.wslot.fill_(write_slot)
        b.cslots.zero_()
        b.cslots[:ctx_len] = ctx_slots
        b.mask.zero_()
        b.mask[:ctx_len] = True
        b.graph.replay()
        return b.out  # (1, vocab), valid until the next replay using this bucket
