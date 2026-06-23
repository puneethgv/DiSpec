"""Correctness: DiSpec's from-scratch forward must match HuggingFace.

We do NOT compare free-running greedy token sequences — for open-ended text many
next-token logits are near-ties, so bf16 rounding flips them and the sequences
diverge even between two correct implementations (HF itself diverges across its
own attention backends). Instead we use *teacher forcing*: feed identical tokens
to both and compare per-position logits. This isolates correctness and, crucially,
exercises the incremental (decode) KV-cache path against HF's parallel forward.

Skipped if CUDA / weights unavailable. Uses the small 0.5B model for speed.
"""

import pytest
import torch

from dispec.config import DRAFT_MODEL
from dispec.engine.engine import LLMEngine
from dispec.engine.model_runner import SeqMeta

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

PROMPTS = [
    "The capital of France is Paris, a city famous for its art, food, and history.",
    "Once upon a time, in a small village nestled between two mountains, there lived",
]


@pytest.fixture(scope="module")
def model_and_tok():
    from dispec.models.loader import load_model, load_tokenizer
    return load_model(DRAFT_MODEL), load_tokenizer(DRAFT_MODEL)


def _hf_all_logits(model, ids):
    with torch.inference_mode():
        return model(ids).logits[0].float()  # (L, vocab)


@pytest.mark.parametrize("prompt", PROMPTS)
def test_prefill_logits_match_hf(model_and_tok, prompt):
    """A single from-scratch prefill must reproduce HF logits at every position."""
    model, tok = model_and_tok
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    n = ids.shape[1]
    hf = _hf_all_logits(model, ids)

    eng = LLMEngine(model)
    table = eng.cache.manager.allocate(1234, n)
    pos = torch.arange(n, device=model.device)
    slots = eng.cache.context_slots(table.block_ids, n)
    hidden = eng.runner.forward(ids[0], pos, slots, [SeqMeta(table.block_ids, n, n)], eng.cache)
    ds = eng.runner.logits(hidden).float()  # (L, vocab)
    eng.cache.manager.free(1234)

    agree = (ds.argmax(-1) == hf.argmax(-1)).float().mean().item()
    cos = torch.nn.functional.cosine_similarity(ds, hf, dim=-1).mean().item()
    # Per-position cosine is the robust signal; a stray argmax flip is a bf16
    # near-tie, so allow a small fraction of disagreements on short sequences.
    assert cos >= 0.999, f"mean cosine {cos:.5f}"
    assert agree >= 0.9, f"argmax agreement {agree:.3f}"


@pytest.mark.parametrize("prompt", PROMPTS)
def test_incremental_decode_matches_hf(model_and_tok, prompt):
    """Step-by-step decode (one token at a time) must match HF's parallel forward.

    This is the real test of the paged KV cache + incremental attention: we prefill
    part of the sequence, then teacher-force the remaining tokens one at a time and
    check each step's next-token logits against HF's full-forward logits.
    """
    model, tok = model_and_tok
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    seq = ids[0].tolist()
    L = len(seq)
    split = max(2, L // 2)
    hf = _hf_all_logits(model, ids)

    eng = LLMEngine(model)
    mgr = eng.cache.manager
    sid = 4321
    table = mgr.allocate(sid, split)
    dev = model.device

    # Prefill first `split` tokens.
    pos = torch.arange(split, device=dev)
    slots = eng.cache.context_slots(table.block_ids, split)
    hidden = eng.runner.forward(torch.tensor(seq[:split], device=dev), pos, slots,
                                [SeqMeta(table.block_ids, split, split)], eng.cache)
    table.num_tokens = split

    agreements = []
    # Predicted next token at position split-1 (end of prefill).
    ds_last = eng.runner.logits(hidden[-1:]).float()[0]
    agreements.append(int(ds_last.argmax()) == int(hf[split - 1].argmax()))

    # Teacher-force the rest one token at a time.
    for i in range(split, L):
        blk, off = mgr.append_token(sid)
        slot = torch.tensor([blk * eng.block_size + off], device=dev)
        hidden = eng.runner.forward(torch.tensor([seq[i]], device=dev),
                                    torch.tensor([i], device=dev), slot,
                                    [SeqMeta(table.block_ids, 1, i + 1)], eng.cache)
        ds = eng.runner.logits(hidden[-1:]).float()[0]
        if i < L - 1:
            agreements.append(int(ds.argmax()) == int(hf[i].argmax()))
    mgr.free(sid)

    rate = sum(agreements) / len(agreements)
    assert rate >= 0.9, f"incremental argmax agreement {rate:.3f} ({agreements})"
