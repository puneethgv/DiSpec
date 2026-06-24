"""Triton paged-attention kernel: correctness vs a reference, and as an engine backend."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_kernel_matches_reference():
    import torch.nn.functional as F

    from dispec.engine.triton_attn import paged_decode_attention

    torch.manual_seed(0)
    H, KVH, D = 14, 2, 64
    groups = H // KVH
    num_slots, ctx = 2048, 130
    q = torch.randn(H, D, device="cuda", dtype=torch.float16)
    kc = torch.randn(num_slots, KVH, D, device="cuda", dtype=torch.float16)
    vc = torch.randn(num_slots, KVH, D, device="cuda", dtype=torch.float16)
    slots = torch.randperm(num_slots, device="cuda")[:ctx]

    scale = D ** -0.5
    ki = kc[slots].repeat_interleave(groups, dim=1).float().permute(1, 0, 2)  # [H,ctx,D]
    vi = vc[slots].repeat_interleave(groups, dim=1).float().permute(1, 0, 2)
    scores = (q.float()[:, None, :] * ki).sum(-1) * scale
    ref = (scores.softmax(-1)[:, :, None] * vi).sum(1)

    out = paged_decode_attention(q, kc, vc, slots, groups).float()
    assert F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item() > 0.999
    assert (out - ref).abs().max().item() < 1e-2


def test_triton_backend_agrees_with_native():
    from dispec.config import DRAFT_MODEL
    from dispec.engine.engine import LLMEngine
    from dispec.models.loader import load_model, load_tokenizer
    from dispec.sampling import SamplingParams

    model, tok = load_model(DRAFT_MODEL), load_tokenizer(DRAFT_MODEL)
    ids = tok("The capital of France is").input_ids
    p = SamplingParams(0.0)
    native = LLMEngine(model, num_blocks=512).generate(ids, 48, p, eos_token_id=tok.eos_token_id)
    triton = LLMEngine(model, num_blocks=512, attn_backend="triton").generate(
        ids, 48, p, eos_token_id=tok.eos_token_id)
    k = min(len(native), len(triton))
    agree = sum(a == b for a, b in zip(native[:k], triton[:k])) / k
    assert agree >= 0.85, f"triton vs native agreement {agree:.2f}\n{triton}\n{native}"
