"""Hand-written Triton paged-attention kernel for the decode step.

This is the from-scratch equivalent of what libraries like FlashInfer provide: a fused
attention kernel that reads the KV cache straight out of its paged blocks (via a slot
list) and does the whole attention in one launch, instead of gather + repeat_kv + SDPA
as separate PyTorch ops. It uses the standard flash-attention online softmax so it never
materializes the full score vector, and handles grouped-query attention by mapping each
query head to its KV head.

Decode shape: one query token per call, attending over `ctx_len` cached positions.
One Triton program per query head.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    q_ptr, k_ptr, v_ptr, slots_ptr, o_ptr,
    ctx_len, scale,
    kv_groups,
    stride_kslot, stride_khead,   # k/v cache strides (same layout for both)
    HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
):
    head = tl.program_id(0)
    kv_head = head // kv_groups
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(q_ptr + head * HEAD_DIM + offs_d).to(tl.float32) * scale

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for start in range(0, ctx_len, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < ctx_len
        slots = tl.load(slots_ptr + offs_n, mask=mask_n, other=0)
        base = slots[:, None] * stride_kslot + kv_head * stride_khead + offs_d[None, :]
        k = tl.load(k_ptr + base, mask=mask_n[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(q[None, :] * k, axis=1)            # [BLOCK_N]
        scores = tl.where(mask_n, scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)                         # [BLOCK_N]
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        v = tl.load(v_ptr + base, mask=mask_n[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    tl.store(o_ptr + head * HEAD_DIM + offs_d, (acc / l_i).to(o_ptr.dtype.element_ty))


@triton.jit
def _batched_paged_decode_kernel(
    q_ptr, k_ptr, v_ptr, slot_table_ptr, ctx_lens_ptr, o_ptr,
    scale, kv_groups,
    stride_kslot, stride_khead,
    stride_qb, stride_tb,
    HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    head = tl.program_id(1)
    ctx_len = tl.load(ctx_lens_ptr + b)
    kv_head = head // kv_groups
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(q_ptr + b * stride_qb + head * HEAD_DIM + offs_d).to(tl.float32) * scale
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for start in range(0, ctx_len, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < ctx_len
        slots = tl.load(slot_table_ptr + b * stride_tb + offs_n, mask=mask_n, other=0)
        base = slots[:, None] * stride_kslot + kv_head * stride_khead + offs_d[None, :]
        k = tl.load(k_ptr + base, mask=mask_n[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(q[None, :] * k, axis=1)
        scores = tl.where(mask_n, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        v = tl.load(v_ptr + base, mask=mask_n[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    tl.store(o_ptr + b * stride_qb + head * HEAD_DIM + offs_d,
             (acc / l_i).to(o_ptr.dtype.element_ty))


def batched_paged_decode_attention(q, k_cache, v_cache, slot_table, ctx_lens, kv_groups, scale=None):
    """Fused paged attention for a batch of single-token queries (one launch).

    q:          (B, num_heads, head_dim)
    slot_table: (B, max_ctx) int32 padded slot indices per sequence
    ctx_lens:   (B,) int32 actual context length per sequence
    returns:    (B, num_heads, head_dim)
    """
    B, num_heads, head_dim = q.shape
    scale = scale or head_dim ** -0.5
    o = torch.empty_like(q)
    _batched_paged_decode_kernel[(B, num_heads)](
        q, k_cache, v_cache, slot_table, ctx_lens, o,
        scale, kv_groups,
        k_cache.stride(0), k_cache.stride(1),
        q.stride(0), slot_table.stride(0),
        HEAD_DIM=head_dim, BLOCK_N=128, num_warps=2,
    )
    return o


def paged_decode_attention(q, k_cache, v_cache, slots, kv_groups, scale=None):
    """Fused paged attention for a single query token.

    q:        (num_heads, head_dim)
    k_cache:  (num_slots, num_kv_heads, head_dim)  (v_cache same shape/layout)
    slots:    (ctx_len,) int32/long slot indices of the context tokens
    returns:  (num_heads, head_dim)
    """
    num_heads, head_dim = q.shape
    ctx_len = slots.shape[0]
    scale = scale or head_dim ** -0.5
    o = torch.empty_like(q)
    slots = slots.to(torch.int32)
    # BLOCK_N=128 / num_warps=2 won a small sweep (best at longer contexts).
    _paged_decode_kernel[(num_heads,)](
        q, k_cache, v_cache, slots, o,
        ctx_len, scale, kv_groups,
        k_cache.stride(0), k_cache.stride(1),
        HEAD_DIM=head_dim, BLOCK_N=128, num_warps=2,
    )
    return o
