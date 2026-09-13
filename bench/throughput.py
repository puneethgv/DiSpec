"""Continuous-batching throughput sweep for the fast path (Triton + fused).

Warms up first (Triton kernels JIT-compile on first launch), then measures aggregate
decode throughput at several concurrency levels. Pair with bench/vllm_ref.py for the
reference numbers. `fuse=True` mutates the model in place, so we load a fresh model
per measurement.

Run: python -m bench.throughput [--max-new 96]
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from dispec.config import BENCH_PROMPTS, TARGET_MODEL
from dispec.models.loader import build_prompt, load_model, load_tokenizer
from dispec.sampling import SamplingParams
from dispec.sched.scheduler import ContinuousBatchingEngine


def measure(tok, prompts, concurrency, max_new):
    model = load_model(TARGET_MODEL)
    eng = ContinuousBatchingEngine(model, num_blocks=2048, max_batch_tokens=8192,
                                   attn_backend="triton", fuse=True)
    reqs = (prompts * ((concurrency // len(prompts)) + 1))[:concurrency]
    for p in reqs:
        eng.add_request(tok(p).input_ids, max_new, SamplingParams(0.0), eos_id=tok.eos_token_id)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    eng.run_until_done()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n = sum(len(v) for v in eng.collect_outputs().values())
    del eng, model
    torch.cuda.empty_cache()
    return n / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[6, 16, 32])
    ap.add_argument("--repeats", type=int, default=3, help="runs per concurrency; median reported")
    args = ap.parse_args()

    tok = load_tokenizer(TARGET_MODEL)
    prompts = [build_prompt(tok, p) for p in BENCH_PROMPTS]

    # Warmup: compile the Triton kernels (disk-cached for later processes).
    print("Warmup...")
    measure(tok, prompts, 4, 8)

    print(f"\nContinuous batching (Triton + fused), Qwen2.5-1.5B:")
    for c in args.concurrency:
        runs = [measure(tok, prompts, c, args.max_new) for _ in range(args.repeats)]
        print(f"  concurrency={c:>3}: {statistics.median(runs):.1f} tok/s  "
              f"(runs {', '.join(f'{r:.1f}' for r in runs)})")


if __name__ == "__main__":
    main()
