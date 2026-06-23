"""Benchmark the DiSpec engine (single-sequence + continuous batching).

Compares against bench.baseline_hf. Reports TTFT/TPOT for single-stream and
aggregate throughput under concurrent load (continuous batching), so we can show
the batching win over HF's static single-stream generate.

Run: python -m bench.dispec_bench [--max-new 128] [--concurrency 8]
"""

from __future__ import annotations

import argparse
import time

import torch

from bench.common import BenchResult, RequestMetrics
from dispec.config import BENCH_PROMPTS, GenConfig
from dispec.engine.engine import LLMEngine
from dispec.models.loader import build_prompt, load_target
from dispec.sampling import SamplingParams
from dispec.sched.scheduler import ContinuousBatchingEngine


def bench_single(model, tok, prompts, gen) -> BenchResult:
    eng = LLMEngine(model, num_blocks=4096)
    params = SamplingParams(temperature=gen.temperature)
    res = BenchResult(name="dispec_single")
    # warmup
    eng.generate(tok(prompts[0]).input_ids, 8, params, eos_token_id=tok.eos_token_id)
    for p in prompts:
        ids = tok(build_prompt(tok, p)).input_ids
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = eng.generate(ids, gen.max_new_tokens, params, eos_token_id=tok.eos_token_id)
        torch.cuda.synchronize(); total = time.perf_counter() - t0
        # Single-stream: approximate TTFT as total/len (no streaming hook here).
        res.add(RequestMetrics(ttft_s=total / max(len(out), 1), total_s=total,
                               num_prompt_tokens=len(ids), num_output_tokens=len(out)))
    return res


def bench_batched(model, tok, prompts, gen, concurrency) -> BenchResult:
    eng = ContinuousBatchingEngine(model, num_blocks=8192, max_batch_tokens=8192)
    params = SamplingParams(temperature=gen.temperature)
    reqs = (prompts * ((concurrency // len(prompts)) + 1))[:concurrency]
    for p in reqs:
        eng.add_request(tok(build_prompt(tok, p)).input_ids, gen.max_new_tokens,
                        params, eos_id=tok.eos_token_id)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    eng.run_until_done()
    torch.cuda.synchronize(); total = time.perf_counter() - t0

    outs = eng.collect_outputs()
    n_out = sum(len(v) for v in outs.values())
    res = BenchResult(name="dispec_continuous_batching")
    res.extra = {
        "concurrency": concurrency,
        "wall_s": round(total, 3),
        "steps": eng.num_steps,
        "prefill_tokens": eng.num_prefill_tokens,
        "decode_tokens": eng.num_decode_tokens,
        "aggregate_throughput_tok_per_s": round(n_out / total, 1),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()
    gen = GenConfig(max_new_tokens=args.max_new, temperature=args.temperature)

    print("Loading target model...")
    model, tok = load_target()
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM after load: {(total - free) / 1e9:.2f} GB used")

    print("\n== Single-sequence ==")
    bench_single(model, tok, BENCH_PROMPTS, gen).print_summary()

    print(f"\n== Continuous batching (concurrency={args.concurrency}) ==")
    import json
    print(json.dumps(bench_batched(model, tok, BENCH_PROMPTS, gen, args.concurrency).summary(), indent=2))


if __name__ == "__main__":
    main()
