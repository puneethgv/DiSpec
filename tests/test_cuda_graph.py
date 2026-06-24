"""CUDA-graph decode must match the eager path at the logits level.

The captured graph replays fixed-shape *masked* attention over a padded window; that's
mathematically identical to the eager ragged gather. We verify per-step with teacher
forcing (both paths decode the same token from the same cache state) and compare logits
directly — robust to bf16 near-ties, unlike comparing free-running greedy sequences. The
loop runs well past a 16-token block boundary. Needs CUDA.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_graph_logits_match_eager_per_step():
    from dispec.config import DRAFT_MODEL
    from dispec.engine.cuda_graph import CudaGraphDecoder
    from dispec.engine.model_runner import ModelRunner, SeqMeta
    from dispec.kv.paged_cache import PagedKVCache
    from dispec.models.loader import load_model, load_tokenizer

    model, tok = load_model(DRAFT_MODEL), load_tokenizer(DRAFT_MODEL)
    runner = ModelRunner(model)
    cache = PagedKVCache.for_model(model.config, 512, 16, runner.dtype, "cuda")
    graph = CudaGraphDecoder(runner, cache, buckets=(256,))
    graph.capture()  # capture while cache empty

    ids = tok("Once upon a time, in a small village,").input_ids
    n = len(ids)
    sid = 1
    table = cache.manager.allocate(sid, n)
    with torch.inference_mode():
        slots = cache.context_slots(table.block_ids, n)
        table.num_tokens = n
        hidden = runner.forward(torch.tensor(ids, device="cuda"),
                                torch.arange(n, device="cuda"), slots,
                                [SeqMeta(table.block_ids, n, n)], cache)
        nxt = int(runner.logits(hidden[-1:]).argmax())
        cur = n
        for _ in range(40):  # crosses the 16-token block boundary repeatedly
            blk, off = cache.manager.append_token(sid)
            wslot = blk * 16 + off
            cslots = cache.context_slots(table.block_ids, cur + 1)
            glog = graph.decode(nxt, cur, wslot, cslots).float().clone()
            elog = runner.logits(runner.forward(
                torch.tensor([nxt], device="cuda"), torch.tensor([cur], device="cuda"),
                torch.tensor([wslot], device="cuda"),
                [SeqMeta(table.block_ids, 1, cur + 1)], cache))[-1].float()
            # Per-step cosine is the robust correctness signal (argmax can flip on a
            # bf16 near-tie even when the full distribution matches).
            cos = torch.nn.functional.cosine_similarity(glog[0], elog, dim=0).item()
            assert cos >= 0.999, f"step ctx={cur}: cosine {cos:.5f}"
            nxt = int(elog.argmax())
            cur += 1
