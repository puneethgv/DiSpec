"""Why speculative decoding is below 1x: per-forward costs and tokens per iteration.

Each speculative iteration (dispec/spec/speculative.py) runs, on the target, one K-token
verify forward plus one single-token correction forward, and on the draft, K-1 single-token
proposal forwards plus one resync forward. The forwards are timed through the engine's own
_forward (paged cache, same code path), median of repeats, and tokens per iteration comes
from generating BENCH_PROMPTS.

Run: python -m bench.spec_profile
"""
import json
import statistics
import time

import torch

from dispec.config import BENCH_PROMPTS, DRAFT_MODEL, TARGET_MODEL, SpecConfig
from dispec.engine.cuda_graph import CudaGraphDecoder
from dispec.engine.model_runner import SeqMeta
from dispec.kv.paged_cache import PagedKVCache
from dispec.models.loader import build_prompt, load_model, load_tokenizer
from dispec.sampling import SamplingParams
from dispec.spec.speculative import SpeculativeEngine

K, MAX_NEW, REPS = 5, 96, 20
target, tok = load_model(TARGET_MODEL), load_tokenizer(TARGET_MODEL)
draft = load_model(DRAFT_MODEL)
eng = SpeculativeEngine(target, draft, spec=SpecConfig(num_speculative_tokens=K), num_blocks=512)
params = SamplingParams(0.0)
pids = [tok(build_prompt(tok, p)).input_ids for p in BENCH_PROMPTS]
eos = tok.eos_token_id

# tokens per iteration and acceptance, end to end
eng.generate(pids[0], 8, params, eos_token_id=eos)
iters = proposed = accepted = produced = 0
for pid in pids:
    out, st = eng.generate(pid, MAX_NEW, params, eos_token_id=eos)
    iters += st.iters; proposed += st.proposed; accepted += st.accepted; produced += len(out)


def median_ms(fn):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(REPS):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


sid, pid = 10**6, pids[0]
m = len(pid)
with torch.inference_mode():
    for runner, cache in ((eng.tgt, eng.tgt_cache), (eng.drf, eng.drf_cache)):
        cache.manager.allocate(sid, m)
        eng._forward(runner, cache, sid, pid, 0)

    def fwd(runner, cache, tokens):
        def call():
            cache.manager.truncate(sid, m)
            eng._forward(runner, cache, sid, tokens, m)
        return call

    t_dec = median_ms(fwd(eng.tgt, eng.tgt_cache, [pid[-1]]))
    t_ver = median_ms(fwd(eng.tgt, eng.tgt_cache, pid[-K:]))
    d_dec = median_ms(fwd(eng.drf, eng.drf_cache, [pid[-1]]))

    # the draft's single-token forward replayed as a CUDA graph
    g_cache = PagedKVCache.for_model(draft.config, 64, 16, eng.drf.dtype, "cuda")
    graph = CudaGraphDecoder(eng.drf, g_cache, buckets=(256,))
    graph.capture()
    table = g_cache.manager.allocate(1, m)
    table.num_tokens = m
    eng.drf.forward(torch.tensor(pid, device="cuda"), torch.arange(m, device="cuda"),
                    g_cache.context_slots(table.block_ids, m), [SeqMeta(table.block_ids, m, m)], g_cache)

    def graph_step():
        g_cache.manager.truncate(1, m)
        blk, off = g_cache.manager.append_token(1)
        graph.decode(pid[-1], m, blk * 16 + off, g_cache.context_slots(table.block_ids, m + 1))

    d_graph = median_ms(graph_step)

tpi = produced / iters
iter_ms = t_ver + t_dec + K * d_dec  # verify + correction on target; K-1 proposals + 1 resync on draft
print(f"target {TARGET_MODEL} / draft {DRAFT_MODEL}, K={K}, {torch.cuda.get_device_name()}")
print(f"  acceptance {accepted / proposed:.0%}, {tpi:.2f} tokens per iteration")
print(f"  target forward, 1 token      {t_dec:6.1f} ms")
print(f"  target forward, {K} tokens     {t_ver:6.1f} ms")
print(f"  draft forward, 1 token       {d_dec:6.1f} ms  (eager)")
print(f"  draft forward, 1 token       {d_graph:6.1f} ms  (CUDA graph)")
print(f"  iteration ~{iter_ms:.1f} ms for {tpi:.2f} tokens = {tpi / iter_ms * 1000:.1f} tok/s  vs  target-only eager {1000 / t_dec:.1f} tok/s")
g_iter = t_ver + t_dec + K * d_graph
print(f"  with a graph-captured draft: ~{g_iter:.1f} ms/iter = {tpi / g_iter * 1000:.1f} tok/s (estimate)")
print("RESULT_JSON " + json.dumps({"kind": "spec_profile", "acceptance": accepted / proposed,
      "tokens_per_iter": tpi, "target_decode_ms": t_dec, "target_verify_ms": t_ver,
      "draft_decode_ms": d_dec, "draft_decode_graph_ms": d_graph}))
