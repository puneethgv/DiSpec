"""Speculative decoding integration: lossless + actually accepts draft tokens.

Losslessness of the acceptance math is proven in test_rejection. Here we check the
end-to-end engine on real models: with greedy decoding, speculative output must
track the target model's own greedy output (allowing bf16 near-tie drift between
the K-parallel verify and sequential decode), and the draft must achieve a
non-trivial acceptance rate (otherwise speculation buys nothing).

Loads the 1.5B target + 0.5B draft (~4 GB); skipped without CUDA.
"""

import pytest
import torch

from dispec.config import DRAFT_MODEL, TARGET_MODEL, SpecConfig
from dispec.engine.engine import LLMEngine
from dispec.sampling import SamplingParams
from dispec.spec.speculative import SpeculativeEngine

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def models():
    from dispec.models.loader import load_model, load_tokenizer
    tgt = load_model(TARGET_MODEL)
    drf = load_model(DRAFT_MODEL)
    tok = load_tokenizer(TARGET_MODEL)
    return tgt, drf, tok


def test_speculative_tracks_target_and_accepts(models):
    tgt, drf, tok = models
    prompt = "Explain in one sentence why the sky is blue:"
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    n_new = 32
    eos = tok.eos_token_id

    # Target-only greedy reference.
    ref = LLMEngine(tgt, num_blocks=2048).generate(
        ids, n_new, SamplingParams(0.0), eos_token_id=eos)

    spec = SpeculativeEngine(tgt, drf, spec=SpecConfig(num_speculative_tokens=5), num_blocks=2048)
    out, stats = spec.generate(ids, n_new, SamplingParams(0.0), eos_token_id=eos)

    k = min(len(out), len(ref))
    agree = sum(a == b for a, b in zip(out[:k], ref[:k])) / k
    assert agree >= 0.7, f"spec vs target greedy agreement {agree:.2f}\n{out}\n{ref}"
    assert stats.acceptance_rate > 0.2, f"acceptance rate too low: {stats.acceptance_rate:.2f}"
    print(f"\nacceptance_rate={stats.acceptance_rate:.2f} tokens/iter={stats.tokens_per_iter:.2f} agree={agree:.2f}")
