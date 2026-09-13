"""Qwen3 in the from-scratch forward, checked against HuggingFace at the logits level.

Qwen3 RMS-normalizes each query and key head after projection and before RoPE; Qwen2
does not. A forward that skipped the norms would still run and still emit fluent-ish
text, so it is checked against HF directly:

  * a tiny random Qwen3 on CPU, in float32 with a tight tolerance -- runs anywhere;
  * real Qwen3-0.6B weights on CUDA, in bf16 with the cosine/argmax criteria the Qwen2
    tests use, covering prefill, incremental decode, and CUDA-graph decode (which
    carries its own copy of the attention and so needs the norms separately).
"""

import pytest
import torch

from dispec.engine.model_runner import ModelRunner, SeqMeta
from dispec.kv.paged_cache import PagedKVCache
from tests.tiny_models import tiny_qwen2, tiny_qwen3

QWEN3_SMALL = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is Paris, a city famous for its art, food, and history."
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _cache(model, device, dtype, num_blocks=64):
    return PagedKVCache.for_model(model.config, num_blocks, 16, dtype, device)


def _prefill(runner, cache, ids, sid=1):
    n = len(ids)
    dev = runner.device
    table = cache.manager.allocate(sid, n)
    table.num_tokens = n
    hidden = runner.forward(torch.tensor(ids, device=dev), torch.arange(n, device=dev),
                            cache.context_slots(table.block_ids, n),
                            [SeqMeta(table.block_ids, n, n)], cache)
    return table, runner.logits(hidden).float()


def _decode_one(runner, cache, sid, table, token, pos):
    blk, off = cache.manager.append_token(sid)
    dev = runner.device
    hidden = runner.forward(torch.tensor([token], device=dev), torch.tensor([pos], device=dev),
                            torch.tensor([blk * cache.block_size + off], device=dev),
                            [SeqMeta(table.block_ids, 1, pos + 1)], cache)
    return runner.logits(hidden[-1:]).float()[0]


def _random_ids(n, vocab=128, seed=1):
    return torch.randint(0, vocab, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


# -- CPU: tiny random models ----------------------------------------------------
def test_runner_detects_qk_norm_per_family():
    assert ModelRunner(tiny_qwen3()).qk_norm
    assert not ModelRunner(tiny_qwen2()).qk_norm


def test_sliding_window_layers_are_rejected():
    model = tiny_qwen3()
    n = model.config.num_hidden_layers
    model.config.layer_types = ["sliding_attention"] + ["full_attention"] * (n - 1)
    with pytest.raises(ValueError, match="sliding-window"):
        ModelRunner(model)


def test_tiny_qwen3_prefill_matches_hf():
    model = tiny_qwen3()
    runner = ModelRunner(model)
    ids = _random_ids(40)
    _, ours = _prefill(runner, _cache(model, "cpu", torch.float32), ids)
    with torch.inference_mode():
        ref = model(torch.tensor([ids])).logits[0].float()
    torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-4)


def test_tiny_qwen3_incremental_decode_matches_hf():
    model = tiny_qwen3()
    runner = ModelRunner(model)
    cache = _cache(model, "cpu", torch.float32)
    ids = _random_ids(40, seed=2)
    split = 17  # crosses a 16-token block boundary during decode
    with torch.inference_mode():
        ref = model(torch.tensor([ids])).logits[0].float()
    table, ours = _prefill(runner, cache, ids[:split], sid=7)
    torch.testing.assert_close(ours[-1], ref[split - 1], atol=1e-4, rtol=1e-4)
    for i in range(split, len(ids) - 1):
        step = _decode_one(runner, cache, 7, table, ids[i], i)
        torch.testing.assert_close(step, ref[i], atol=1e-4, rtol=1e-4)


# -- CUDA: real Qwen3-0.6B weights ---------------------------------------------
@pytest.fixture(scope="module")
def qwen3():
    from dispec.models.loader import load_model, load_tokenizer
    return load_model(QWEN3_SMALL), load_tokenizer(QWEN3_SMALL)


@needs_cuda
def test_qwen3_prefill_matches_hf(qwen3):
    model, tok = qwen3
    ids = tok(PROMPT).input_ids
    with torch.inference_mode():
        ref = model(torch.tensor([ids], device="cuda")).logits[0].float()
    _, ours = _prefill(ModelRunner(model), _cache(model, "cuda", model.dtype), ids)
    cos = torch.nn.functional.cosine_similarity(ours, ref, dim=-1).mean().item()
    agree = (ours.argmax(-1) == ref.argmax(-1)).float().mean().item()
    assert cos >= 0.999, f"mean cosine {cos:.5f}"
    assert agree >= 0.9, f"argmax agreement {agree:.3f}"


@needs_cuda
def test_qwen3_incremental_decode_matches_hf(qwen3):
    model, tok = qwen3
    ids = tok(PROMPT).input_ids
    with torch.inference_mode():
        ref = model(torch.tensor([ids], device="cuda")).logits[0].float()
    runner = ModelRunner(model)
    cache = _cache(model, "cuda", model.dtype)
    split = max(2, len(ids) // 2)
    table, _ = _prefill(runner, cache, ids[:split], sid=3)
    agree = []
    for i in range(split, len(ids) - 1):
        step = _decode_one(runner, cache, 3, table, ids[i], i)
        cos = torch.nn.functional.cosine_similarity(step, ref[i], dim=0).item()
        assert cos >= 0.999, f"position {i}: cosine {cos:.5f}"
        agree.append(int(step.argmax()) == int(ref[i].argmax()))
    assert sum(agree) / len(agree) >= 0.9, agree


@needs_cuda
def test_qwen3_cuda_graph_matches_eager(qwen3):
    from dispec.engine.cuda_graph import CudaGraphDecoder

    model, tok = qwen3
    runner = ModelRunner(model)
    cache = _cache(model, "cuda", model.dtype, num_blocks=512)
    graph = CudaGraphDecoder(runner, cache, buckets=(256,))
    graph.capture()  # while the cache holds no live data

    ids = tok("Once upon a time, in a small village,").input_ids
    with torch.inference_mode():
        table, logits = _prefill(runner, cache, ids, sid=1)
        nxt, cur = int(logits[-1].argmax()), len(ids)
        for _ in range(40):  # crosses 16-token block boundaries repeatedly
            blk, off = cache.manager.append_token(1)
            wslot = blk * cache.block_size + off
            cslots = cache.context_slots(table.block_ids, cur + 1)
            glog = graph.decode(nxt, cur, wslot, cslots).float().clone()[0]
            elog = runner.logits(runner.forward(
                torch.tensor([nxt], device="cuda"), torch.tensor([cur], device="cuda"),
                torch.tensor([wslot], device="cuda"),
                [SeqMeta(table.block_ids, 1, cur + 1)], cache))[-1].float()
            cos = torch.nn.functional.cosine_similarity(glog, elog, dim=0).item()
            assert cos >= 0.999, f"ctx={cur}: cosine {cos:.5f}"
            nxt, cur = int(elog.argmax()), cur + 1
