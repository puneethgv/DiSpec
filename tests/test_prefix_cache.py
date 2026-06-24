"""Prefix cache: correct block reuse + lossless output.

CPU unit tests cover the match/insert/evict logic; a GPU test checks that a second
request sharing a prefix prefills fewer tokens and produces identical output.
"""

import pytest
import torch

from dispec.kv.block_manager import BlockManager
from dispec.kv.prefix_cache import PrefixCache


def test_match_returns_shared_full_blocks():
    bm = BlockManager(num_blocks=20, block_size=4)
    pc = PrefixCache(bm, block_size=4)
    t = bm.allocate(1, 8)  # 2 full blocks
    pc.insert(list(range(8)), t.block_ids)
    assert len(pc) == 2
    # A prompt sharing the first 8 tokens (+ a partial 3rd block) reuses both blocks.
    matched = pc.match(list(range(8)) + [99, 99])
    assert matched == t.block_ids
    # A different prefix shares nothing.
    assert pc.match([100, 101, 102, 103]) == []


def test_insert_pins_blocks_and_evict_releases():
    bm = BlockManager(num_blocks=8, block_size=4)
    pc = PrefixCache(bm, block_size=4)
    t = bm.allocate(1, 8)            # uses 2 blocks
    pc.insert(list(range(8)), t.block_ids)
    bm.free(1)                        # sequence gone, but cache still pins the 2 blocks
    assert bm.num_free_blocks == 6    # 8 - 2 still held by the cache
    assert pc.evict(2) == 2
    assert bm.num_free_blocks == 8    # cache released them


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_prefix_reuse_is_lossless_and_skips_prefill():
    from dispec.config import DRAFT_MODEL
    from dispec.engine.engine import LLMEngine
    from dispec.models.loader import load_model, load_tokenizer
    from dispec.sampling import SamplingParams

    model, tok = load_model(DRAFT_MODEL), load_tokenizer(DRAFT_MODEL)
    # Shared prefix must span at least one full 16-token block to be reusable.
    shared = ("You are a helpful, harmless assistant. Always follow the user's "
              "instructions carefully, think step by step, be concise, and never make "
              "up facts you are unsure about. Here is the user's question. ")
    p1 = tok(shared + "What is the capital of France?").input_ids
    p2 = tok(shared + "Name three primary colors.").input_ids
    assert len(p1) > 32 and len(p2) > 32  # ensure multiple shared blocks
    params = SamplingParams(0.0)

    ref = LLMEngine(model, num_blocks=1024).generate(p2, 24, params, eos_token_id=tok.eos_token_id)

    eng = LLMEngine(model, num_blocks=1024, prefix_cache=True)
    eng.generate(p1, 24, params, eos_token_id=tok.eos_token_id)   # warms the shared prefix
    out = eng.generate(p2, 24, params, eos_token_id=tok.eos_token_id)

    assert eng.last_prefill_tokens < len(p2), "prefix cache did not skip any prefill"
    assert eng.prefix_cache.hits > 0
    assert out == ref, f"prefix-cached output differs\n{out}\n{ref}"
