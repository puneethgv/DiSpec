"""Prefill attention via a causal bias instead of a materialized mask.

`ModelRunner._attention` expresses the right-aligned causal mask as `causal_lower_right`
and handles GQA inside SDPA, so SDPA can dispatch to FlashAttention or memory-efficient
kernels. The explicit bool mask it replaced ruled those out: DiSpec prefilled 8192 tokens
of Qwen3-4B in 3.5 s on an L4 against 2.0 s for HF.

It has to be a pure speed change, so:

  * the bias is checked against the explicit right-aligned bool mask (with KV heads
    repeated, as the old code did) on every (q_len, ctx_len) shape the scheduler makes;
  * every attention shape the runner produces is checked end to end -- fresh prefill,
    chunked prefill with past context, and one forward mixing a decode row with a fresh
    prefill -- against the equivalent single-shape computation;
  * on CUDA the fast kernels are forced, so a silent fallback to the math path fails.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

from dispec.engine.model_runner import ModelRunner, SeqMeta
from dispec.kv.paged_cache import PagedKVCache
from tests.tiny_models import tiny_qwen2, tiny_qwen3

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
MODELS = {"qwen3": tiny_qwen3, "qwen2": tiny_qwen2}


def _ids(n, seed):
    return torch.randint(0, 128, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _forward(runner, cache, rows):
    """One forward over several sequences. rows: (sid, token_ids, start_pos)."""
    dev = runner.device
    tokens, positions, slots, meta = [], [], [], []
    for sid, ids, start in rows:
        mgr = cache.manager
        if not mgr.has_table(sid):
            mgr.allocate(sid, start + len(ids))
        else:
            for _ in ids:
                mgr.append_token(sid)
        table = mgr.block_table(sid)
        pos = list(range(start, start + len(ids)))
        table.num_tokens = start + len(ids)
        tokens += ids
        positions += pos
        slots += cache.slots_for_positions(table.block_ids, torch.tensor(pos)).tolist()
        meta.append(SeqMeta(table.block_ids, len(ids), start + len(ids)))
    hidden = runner.forward(torch.tensor(tokens, device=dev), torch.tensor(positions, device=dev),
                            torch.tensor(slots, device=dev), meta, cache)
    return runner.logits(hidden).float()


@pytest.mark.parametrize("q_len,ctx_len", [(6, 6), (1, 11), (4, 11), (9, 17), (16, 33)])
def test_causal_bias_equals_right_aligned_bool_mask(q_len, ctx_len):
    g = torch.Generator().manual_seed(q_len * 100 + ctx_len)
    q = torch.randn(1, 4, q_len, 8, generator=g)
    k, v = torch.randn(1, 2, ctx_len, 8, generator=g), torch.randn(1, 2, ctx_len, 8, generator=g)
    past = ctx_len - q_len
    mask = torch.arange(ctx_len)[None, :] <= torch.arange(q_len)[:, None] + past
    old = F.scaled_dot_product_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1),
                                         attn_mask=mask[None, None])
    new = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_lower_right(q_len, ctx_len),
                                         enable_gqa=True)
    torch.testing.assert_close(new, old, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("family", sorted(MODELS))
def test_chunked_prefill_matches_one_prefill(family):
    model = MODELS[family]()
    runner = ModelRunner(model)
    ids = _ids(40, seed=1)
    whole = _forward(runner, PagedKVCache.for_model(model.config, 16, 16, torch.float32, "cpu"),
                     [(1, ids, 0)])
    cache = PagedKVCache.for_model(model.config, 16, 16, torch.float32, "cpu")
    first = _forward(runner, cache, [(1, ids[:17], 0)])
    rest = _forward(runner, cache, [(1, ids[17:], 17)])  # q_len 23 over 40 of context
    torch.testing.assert_close(torch.cat([first, rest]), whole, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("family", sorted(MODELS))
def test_mixed_decode_and_prefill_batch_matches_separate(family):
    model = MODELS[family]()
    runner = ModelRunner(model)
    a, b = _ids(21, seed=2), _ids(13, seed=3)

    mixed = PagedKVCache.for_model(model.config, 16, 16, torch.float32, "cpu")
    _forward(runner, mixed, [(1, a[:20], 0)])
    both = _forward(runner, mixed, [(1, a[20:], 20), (2, b, 0)])  # decode row + fresh prefill

    solo = PagedKVCache.for_model(model.config, 16, 16, torch.float32, "cpu")
    _forward(runner, solo, [(1, a[:20], 0)])
    decode_only = _forward(runner, solo, [(1, a[20:], 20)])
    prefill_only = _forward(runner, PagedKVCache.for_model(model.config, 16, 16, torch.float32,
                                                           "cpu"), [(2, b, 0)])
    torch.testing.assert_close(both[:1], decode_only, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(both[1:], prefill_only, atol=1e-4, rtol=1e-4)


# -- CUDA: the fast kernels are actually used -----------------------------------------
@pytest.fixture(scope="module")
def qwen3_small():
    from dispec.models.loader import load_model, load_tokenizer
    return load_model("Qwen/Qwen3-0.6B"), load_tokenizer("Qwen/Qwen3-0.6B")


PROMPT = ("The Antikythera mechanism is an ancient Greek hand-powered device that has been "
          "identified as the oldest known analogue computer, used to predict astronomical "
          "positions and eclipses decades in advance.")


@needs_cuda
def test_fresh_prefill_runs_on_flash_attention(qwen3_small):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    model, tok = qwen3_small
    ids = tok(PROMPT).input_ids
    runner = ModelRunner(model)
    cache = PagedKVCache.for_model(model.config, 64, 16, runner.dtype, "cuda")
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):  # raises if flash cannot run it
        ours = _forward(runner, cache, [(1, ids, 0)])
    with torch.inference_mode():
        ref = model(torch.tensor([ids], device="cuda")).logits[0].float()
    cos = F.cosine_similarity(ours, ref, dim=-1).mean().item()
    assert cos >= 0.999, f"mean cosine {cos:.5f}"


@needs_cuda
def test_chunked_prefill_avoids_math_fallback(qwen3_small):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    model, tok = qwen3_small
    ids = tok(PROMPT).input_ids
    runner = ModelRunner(model)
    whole = _forward(runner, PagedKVCache.for_model(model.config, 64, 16, runner.dtype, "cuda"),
                     [(1, ids, 0)])
    cache = PagedKVCache.for_model(model.config, 64, 16, runner.dtype, "cuda")
    split = len(ids) // 2
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        _forward(runner, cache, [(1, ids[:split], 0)])
        rest = _forward(runner, cache, [(1, ids[split:], split)])
    cos = F.cosine_similarity(rest, whole[split:], dim=-1).mean().item()
    assert cos >= 0.999, f"mean cosine {cos:.5f}"
