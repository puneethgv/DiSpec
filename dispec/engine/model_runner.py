"""From-scratch forward pass for Qwen2-family models over a paged KV cache.

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
    def __init__(self, model):
        self.model = model
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

        # Reuse HF's rotary module so cos/sin exactly match the reference model
        # (config layout for rope_theta moved around across transformers versions).
        self.rotary_emb = core.rotary_emb
        self._rope_dummy = torch.zeros(1, 1, device=self.device, dtype=self.dtype)

    # -- rotary --------------------------------------------------------------
    def rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos_ids = positions[None, :]  # (1, T)
        cos, sin = self.rotary_emb(self._rope_dummy, pos_ids)  # (1, T, head_dim)
        return cos[0].to(self.dtype), sin[0].to(self.dtype)

    # -- attention for one layer --------------------------------------------
    def _attention(self, layer, x, cos, sin, write_slots, seq_meta, cache, layer_idx):
        attn = layer.self_attn
        T = x.shape[0]
        q = attn.q_proj(x).view(T, self.num_heads, self.head_dim)
        k = attn.k_proj(x).view(T, self.num_kv_heads, self.head_dim)
        v = attn.v_proj(x).view(T, self.num_kv_heads, self.head_dim)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        cache.write(layer_idx, write_slots, k, v)

        out = torch.empty_like(q)
        offset = 0
        for meta in seq_meta:
            qi = q[offset:offset + meta.q_len]  # (q_len, H, D)
            ctx_slots = cache.context_slots(meta.block_ids, meta.ctx_len)
            ki, vi = cache.gather(layer_idx, ctx_slots)  # (ctx_len, KVH, D)

            # GQA: expand kv heads to query heads.
            ki = ki.repeat_interleave(self.kv_groups, dim=1)  # (ctx, H, D)
            vi = vi.repeat_interleave(self.kv_groups, dim=1)

            # (H, q_len, D) and (H, ctx, D)
            qh = qi.transpose(0, 1)
            kh = ki.transpose(0, 1)
            vh = vi.transpose(0, 1)

            # Causal mask aligned to the right: new tokens are the last q_len of ctx.
            past = meta.ctx_len - meta.q_len
            qpos = torch.arange(meta.q_len, device=self.device)[:, None] + past
            kpos = torch.arange(meta.ctx_len, device=self.device)[None, :]
            mask = (kpos <= qpos)  # (q_len, ctx_len) bool, True = keep

            oh = torch.nn.functional.scaled_dot_product_attention(
                qh, kh, vh, attn_mask=mask[None, :, :]
            )  # (H, q_len, D)
            out[offset:offset + meta.q_len] = oh.transpose(0, 1)
            offset += meta.q_len

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
        for i, layer in enumerate(self.layers):
            h = layer.input_layernorm(x)
            x = x + self._attention(layer, h, cos, sin, write_slots, seq_meta, cache, i)
            h = layer.post_attention_layernorm(x)
            x = x + layer.mlp(h)
        return self.norm(x)

    def logits(self, hidden: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        if indices is not None:
            hidden = hidden[indices]
        return self.lm_head(hidden)
