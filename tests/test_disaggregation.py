"""P/D disaggregation: output must match the colocated baseline, exactly.

The decode worker generates from KV computed by a *separate* prefill process and
shipped over CUDA IPC. Since it's the same weights, same greedy path, and a zero-copy
(bit-identical) KV transfer, the disaggregated output must equal the colocated run
token-for-token. This proves the KV transfer + import are correct.

Spawns processes that each load the model; uses the small 0.5B model. Needs CUDA.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_disaggregated_matches_colocated():
    from dispec.config import DRAFT_MODEL
    from dispec.models.loader import load_tokenizer
    from dispec.sampling import SamplingParams
    from dispec.workers.disaggregated import (DisaggConfig, DisaggregatedEngine,
                                              _build, decode_from_kv, prefill_export)

    tok = load_tokenizer(DRAFT_MODEL)
    eos = tok.eos_token_id
    prompts = ["The capital of France is", "Water boils at a temperature of"]
    pids = [tok(p).input_ids for p in prompts]
    n_new = 24

    # Disaggregated: separate prefill + decode processes with CUDA-IPC KV transfer.
    cfg = DisaggConfig(DRAFT_MODEL, num_blocks=256, max_new_tokens=n_new, eos_id=eos)
    eng = DisaggregatedEngine(cfg)
    for i, p in enumerate(pids):
        eng.submit(i, p)
    res = {r["rid"]: r for r in eng.collect(len(pids))}
    eng.shutdown()

    assert all(r["kv_bytes"] > 0 for r in res.values()), "no KV was transferred"

    # Colocated baseline in this process (workers have exited, freeing their VRAM).
    runner, cache = _build(DRAFT_MODEL, 256)
    params = SamplingParams(0.0)
    for i, p in enumerate(pids):
        k, v, ft = prefill_export(runner, cache, 1000 + i, p, params)
        ref = decode_from_kv(runner, cache, 2000 + i, p, k, v, ft, n_new, params, eos)
        assert res[i]["out"] == ref, f"disagg != colocated for prompt {i}\n{res[i]['out']}\n{ref}"
