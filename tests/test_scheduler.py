"""Correctness of continuous batching.

We avoid comparing free-running greedy *sequences* across the batched and single
paths: a batched forward uses different GEMM shapes than a single-row forward, so
bf16 rounding flips near-ties and the sequences drift apart even though both are
correct. Instead:

  1. Numerical: a ragged batched prefill must reproduce each sequence's single-path
     last-token logits (cosine ~1, same argmax).
  2. Structural: three *identical* prompts batched together must yield *identical*
     outputs over many steps (crossing KV block boundaries). Any cross-sequence
     attention/slot leakage would make them differ — and this check is immune to
     batched-vs-single numeric drift.
"""

import pytest
import torch

from dispec.config import DRAFT_MODEL, KVConfig
from dispec.engine.engine import LLMEngine
from dispec.engine.model_runner import SeqMeta
from dispec.sampling import SamplingParams
from dispec.sched.scheduler import ContinuousBatchingEngine

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def model_and_tok():
    from dispec.models.loader import load_model, load_tokenizer
    return load_model(DRAFT_MODEL), load_tokenizer(DRAFT_MODEL)


def test_batched_prefill_logits_match_single(model_and_tok):
    model, tok = model_and_tok
    prompts = ["The capital of France is", "Water boils at a temperature of",
               "The three primary colors are"]
    pids = [tok(p, return_tensors="pt").input_ids[0].tolist() for p in prompts]
    dev = model.device

    eng = LLMEngine(model, kv=KVConfig(), num_blocks=512)
    # Single-path last-token logits per prompt.
    single = []
    for k, pid in enumerate(pids):
        n = len(pid)
        t = eng.cache.manager.allocate(1000 + k, n)
        slots = eng.cache.context_slots(t.block_ids, n)
        h = eng.runner.forward(torch.tensor(pid, device=dev), torch.arange(n, device=dev),
                               slots, [SeqMeta(t.block_ids, n, n)], eng.cache)
        single.append(eng.runner.logits(h[-1:]).float()[0])
        eng.cache.manager.free(1000 + k)

    # One ragged batched prefill of all three.
    eng2 = LLMEngine(model, kv=KVConfig(), num_blocks=512)
    token_ids, positions, write_slots, meta, last = [], [], [], [], []
    cur = 0
    for k, pid in enumerate(pids):
        n = len(pid)
        t = eng2.cache.manager.allocate(2000 + k, n)
        token_ids += pid
        positions += list(range(n))
        write_slots += eng2.cache.context_slots(t.block_ids, n).tolist()
        meta.append(SeqMeta(t.block_ids, n, n))
        cur += n
        last.append(cur - 1)
    h = eng2.runner.forward(torch.tensor(token_ids, device=dev),
                            torch.tensor(positions, device=dev),
                            torch.tensor(write_slots, device=dev), meta, eng2.cache)
    batched = eng2.runner.logits(h[torch.tensor(last, device=dev)]).float()

    for k in range(len(pids)):
        cos = torch.nn.functional.cosine_similarity(batched[k], single[k], dim=0).item()
        assert int(batched[k].argmax()) == int(single[k].argmax())
        assert cos >= 0.999, f"prompt {k}: cosine {cos:.5f}"


def test_no_cross_sequence_leakage(model_and_tok):
    model, tok = model_and_tok
    pid = tok("Once upon a time, in a small village,", return_tensors="pt").input_ids[0].tolist()
    n_new = 48  # enough to cross several 16-token block boundaries

    cb = ContinuousBatchingEngine(model, num_blocks=512)
    ids = [cb.add_request(pid, n_new, SamplingParams(0.0), eos_id=tok.eos_token_id)
           for _ in range(3)]
    cb.run_until_done()
    outs = cb.collect_outputs()

    a, b, c = (outs[i] for i in ids)
    assert a == b == c, "identical prompts diverged => cross-sequence leakage"
    assert len(a) == n_new
