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


def test_triton_backend_matches_native_logits():
    """Per-step logits cosine between the triton and native decode paths (robust to the
    bf16 near-tie flips that make free-running greedy comparisons flaky)."""
    from dispec.config import DRAFT_MODEL
    from dispec.engine.model_runner import ModelRunner, SeqMeta
    from dispec.kv.paged_cache import PagedKVCache
    from dispec.models.loader import load_model, load_tokenizer

    model, tok = load_model(DRAFT_MODEL), load_tokenizer(DRAFT_MODEL)
    rn = ModelRunner(model, attn_backend="native")
    rt = ModelRunner(model, attn_backend="triton")
    cache = PagedKVCache.for_model(model.config, 512, 16, rn.dtype, "cuda")
    ids = tok("The capital of France is").input_ids
    n = len(ids)
    sid = 1
    table = cache.manager.allocate(sid, n)
    with torch.inference_mode():
        slots = cache.context_slots(table.block_ids, n)
        table.num_tokens = n
        h = rn.forward(torch.tensor(ids, device="cuda"), torch.arange(n, device="cuda"),
                       slots, [SeqMeta(table.block_ids, n, n)], cache)
        nxt = int(rn.logits(h[-1:]).argmax())
        cur = n
        for _ in range(30):  # crosses block boundaries
            blk, off = cache.manager.append_token(sid)
            ws = blk * 16 + off
            args = (torch.tensor([nxt], device="cuda"), torch.tensor([cur], device="cuda"),
                    torch.tensor([ws], device="cuda"),
                    [SeqMeta(table.block_ids, 1, cur + 1)], cache)
            ln = rn.logits(rn.forward(*args)[-1:]).float()[0]   # native writes KV at ws
            lt = rt.logits(rt.forward(*args)[-1:]).float()[0]   # triton rewrites same KV
            cos = torch.nn.functional.cosine_similarity(ln, lt, dim=0).item()
            assert cos >= 0.999, f"ctx={cur}: cosine {cos:.5f}"
            nxt = int(ln.argmax())
            cur += 1
