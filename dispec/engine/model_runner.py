"""From-scratch forward pass for Qwen2- and Qwen3-family models over a paged KV cache.

We reuse the HuggingFace *weight modules* (embeddings, RMSNorms, q/k/v/o projections,
SwiGLU MLP, lm_head) but implement the rotary embedding, KV cache read/write, and
attention ourselves. This is what makes DiSpec a real inference engine rather than a
wrapper: we control the cache layout (paged) and the attention mask (needed later for
tree-verification in speculative decoding).

A "batch" is a flat concatenation of tokens from one or more sequences (ragged),
described by `SeqMeta` entries. The same code path serves prefill (q_len = prompt len)
and decode (q_len = 1, or = tree size during speculation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

from dispec.kv.paged_cache import PagedKVCache


@dataclass
class SeqMeta:
    """Describes one sequence's slice within a flat batch for a single forward."""

    block_ids: list[int]
    q_len: int  # number of new (query) tokens contributed this step
    ctx_len: int  # total context length including the new tokens


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (T, n_heads, head_dim); cos/sin: (T, head_dim)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return x * cos + _rotate_half(x) * sin


class ModelRunner:
    def __init__(self, model, attn_backend: str = "native", fuse: bool = False):
        # attn_backend: "native" (PyTorch SDPA) or "triton" (fused paged kernel for decode).
        # fuse: precompute fused QKV and gate/up weights (one GEMM each, fewer launches).
        self.model = model
        self.attn_backend = attn_backend
        cfg = model.config
        self.cfg = cfg
        self.device = next(model.parameters()).device
        # Compute dtype = first floating param. Quantized (GPTQ/AWQ) models pack
        # weights as int32, so we must skip those to find the fp16 compute dtype.
        self.dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()),
                          torch.float16)

        # HF submodules (handle the .model nesting of *ForCausalLM).
        core = model.model
        self.embed = core.embed_tokens
        self.layers = core.layers
        self.norm = core.norm
        self.lm_head = model.lm_head

        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5

        # Qwen3 RMS-normalizes every query and key head (over head_dim) after projection
        # and before RoPE; Qwen2 has no such norm. Detected once, not per layer per step.
        attn0 = self.layers[0].self_attn
        self.qk_norm = hasattr(attn0, "q_norm") and hasattr(attn0, "k_norm")

        # The paged attention below is full-attention only. A sliding-window layer would
        # attend to tokens the real model masks out, and nothing would fail loudly.
        if "sliding_attention" in (getattr(cfg, "layer_types", None) or []):
            raise ValueError(
                f"{type(model).__name__} uses sliding-window attention layers, which the "
                "paged attention path does not implement")

        # Reuse HF's rotary module so cos/sin exactly match the reference model
        # (config layout for rope_theta moved around across transformers versions).
        self.rotary_emb = core.rotary_emb
        self._rope_dummy = torch.zeros(1, 1, device=self.device, dtype=self.dtype)

        self.fused = self._build_fused() if fuse else False

    def _build_fused(self) -> bool:
        """Concatenate q/k/v and gate/up weights into one GEMM each, then free the
        originals (so memory doesn't double). Only for plain float Linear layers; a
        quantized model keeps the per-projection path."""
        layers = self.layers
        a0 = layers[0].self_attn
        if isinstance(a0.q_proj, nn.Identity):
            raise ValueError("model already fused by another ModelRunner; fuse=True mutates "
                             "the model in place — use one fused runner per model instance")
        if not (isinstance(a0.q_proj, nn.Linear) and a0.q_proj.weight.is_floating_point()):
            return False  # quantized (GPTQ/AWQ): keep the per-projection module path
        self._qkv_w, self._qkv_b, self._gate_up_w = [], [], []
        for layer in layers:
            a, m = layer.self_attn, layer.mlp
            self._qkv_w.append(torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0))
            if a.q_proj.bias is not None:
                self._qkv_b.append(torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias], 0))
            else:
                self._qkv_b.append(None)
            self._gate_up_w.append(torch.cat([m.gate_proj.weight, m.up_proj.weight], 0))
            # Free the originals; o_proj/down_proj/act_fn are still used as modules.
            a.q_proj = a.k_proj = a.v_proj = nn.Identity()
            m.gate_proj = m.up_proj = nn.Identity()
        self._qsz = self.num_heads * self.head_dim
        self._kvsz = self.num_kv_heads * self.head_dim
        return True

    # -- rotary --------------------------------------------------------------
    def rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos_ids = positions[None, :]  # (1, T)
        cos, sin = self.rotary_emb(self._rope_dummy, pos_ids)  # (1, T, head_dim)
        return cos[0].to(self.dtype), sin[0].to(self.dtype)

    # -- attention for one layer --------------------------------------------
    def _attention(self, layer, x, cos, sin, write_slots, seq_meta, cache, layer_idx,
                   decode_batch=None):
        attn = layer.self_attn
        T = x.shape[0]
        if self.fused:
            qkv = F.linear(x, self._qkv_w[layer_idx], self._qkv_b[layer_idx])
            q, k, v = qkv.split([self._qsz, self._kvsz, self._kvsz], dim=-1)
            q = q.view(T, self.num_heads, self.head_dim)
            k = k.view(T, self.num_kv_heads, self.head_dim)
            v = v.view(T, self.num_kv_heads, self.head_dim)
        else:
            q = attn.q_proj(x).view(T, self.num_heads, self.head_dim)
            k = attn.k_proj(x).view(T, self.num_kv_heads, self.head_dim)
            v = attn.v_proj(x).view(T, self.num_kv_heads, self.head_dim)

        if self.qk_norm:
            q = attn.q_norm(q)
            k = attn.k_norm(k)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        cache.write(layer_idx, write_slots, k, v)

        # Fast path: one batched Triton kernel for an all-decode step (q_len==1 each).
        # The slot table is identical across layers, so it's built once in forward().
        if decode_batch is not None:
            from dispec.engine.triton_attn import batched_paged_decode_attention
            slot_table, ctx_lens = decode_batch
            o = batched_paged_decode_attention(
                q, cache.key[layer_idx], cache.value[layer_idx],
                slot_table, ctx_lens, self.kv_groups, self.scale)
            return attn.o_proj(o.reshape(T, self.num_heads * self.head_dim))

        out = torch.empty_like(q)
        offset = 0
        for meta in seq_meta:
            end = offset + meta.q_len
            if meta.ctx_len == meta.q_len:
                # Fresh prefill: the whole context is the tokens projected and written
                # above, so attend over them directly rather than gathering them back.
                ki, vi = k[offset:end], v[offset:end]  # (ctx_len, KVH, D)
            else:
                ctx_slots = cache.context_slots(meta.block_ids, meta.ctx_len)
                ki, vi = cache.gather(layer_idx, ctx_slots)  # (ctx_len, KVH, D)

            # New tokens are the last q_len of the context, so the causal mask is aligned
            # bottom-right. As a CausalBias rather than a materialized bool mask -- and
            # with GQA handled inside the kernel rather than by copying KV heads -- SDPA
            # can dispatch to FlashAttention / memory-efficient kernels. The bool mask
            # ruled those out: 8192 tokens of Qwen3-4B took 3.5 s here vs 2.0 s in HF.
            oh = F.scaled_dot_product_attention(
                q[offset:end].transpose(0, 1)[None], ki.transpose(0, 1)[None],
                vi.transpose(0, 1)[None],
                attn_mask=causal_lower_right(meta.q_len, meta.ctx_len), enable_gqa=True,
            )  # (1, H, q_len, D)
            out[offset:end] = oh[0].transpose(0, 1)
            offset = end

        return attn.o_proj(out.reshape(T, self.num_heads * self.head_dim))

    # -- full model forward --------------------------------------------------
    @torch.inference_mode()
    def forward(
        self,
        token_ids: torch.Tensor,  # (T,) long
        positions: torch.Tensor,  # (T,) long absolute positions
        write_slots: torch.Tensor,  # (T,) long flat cache slots for new tokens
        seq_meta: list[SeqMeta],
        cache: PagedKVCache,
    ) -> torch.Tensor:
        """Returns normalized hidden states (T, hidden). Use `logits()` to project."""
        cos, sin = self.rope(positions)
        x = self.embed(token_ids)

        # Build the batched-decode slot table once (it's identical across layers).
        decode_batch = None
        if self.attn_backend == "triton" and seq_meta and all(m.q_len == 1 for m in seq_meta):
            B = len(seq_meta)
            max_ctx = max(m.ctx_len for m in seq_meta)
            slot_table = torch.zeros(B, max_ctx, dtype=torch.int32, device=self.device)
            for i, meta in enumerate(seq_meta):
                slot_table[i, :meta.ctx_len] = cache.context_slots(
                    meta.block_ids, meta.ctx_len).to(torch.int32)
            ctx_lens = torch.tensor([m.ctx_len for m in seq_meta],
                                    dtype=torch.int32, device=self.device)
            decode_batch = (slot_table, ctx_lens)

        for i, layer in enumerate(self.layers):
            h = layer.input_layernorm(x)
            x = x + self._attention(layer, h, cos, sin, write_slots, seq_meta, cache, i,
                                    decode_batch)
            h = layer.post_attention_layernorm(x)
            x = x + self._mlp(layer, h, i)
        return self.norm(x)

    def _mlp(self, layer, h, i):
        if not self.fused:
            return layer.mlp(h)
        gate, up = F.linear(h, self._gate_up_w[i]).chunk(2, dim=-1)
        return layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

    def logits(self, hidden: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        if indices is not None:
            hidden = hidden[indices]
        return self.lm_head(hidden)
