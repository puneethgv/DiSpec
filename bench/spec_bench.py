"""Speculative decoding speedup benchmark.

Compares target-only single-sequence decode (DiSpec LLMEngine) against the
SpeculativeEngine on the same prompts, reporting wall-clock tok/s, speedup, and
acceptance rate. Lossless: greedy output matches target-only.

Run: python -m bench.spec_bench [--max-new 128] [--k 5]
"""

from __future__ import annotations

import argparse
import time

import torch

from dispec.config import BENCH_PROMPTS, GenConfig, SpecConfig
from dispec.engine.engine import LLMEngine
from dispec.models.loader import build_prompt, load_draft, load_target
from dispec.sampling import SamplingParams
from dispec.spec.speculative import SpeculativeEngine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    gen = GenConfig(max_new_tokens=args.max_new, temperature=0.0)
    params = SamplingParams(0.0)

    print("Loading target + draft...")
    target, tok = load_target()
    draft, _ = load_draft()
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM used: {(total - free) / 1e9:.2f} GB")

    prompts = [build_prompt(tok, p) for p in BENCH_PROMPTS]
    pids = [tok(p).input_ids for p in prompts]
    eos = tok.eos_token_id

    base = LLMEngine(target, num_blocks=2048)
    spec = SpeculativeEngine(target, draft, spec=SpecConfig(num_speculative_tokens=args.k),
                             num_blocks=2048)

    # Warmup both paths.
    base.generate(pids[0], 8, params, eos_token_id=eos)
    spec.generate(pids[0], 8, params, eos_token_id=eos)

    base_tps, spec_tps, acc, agree = [], [], [], []
    for pid in pids:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        b = base.generate(pid, gen.max_new_tokens, params, eos_token_id=eos)
        torch.cuda.synchronize(); bt = time.perf_counter() - t0
        base_tps.append(len(b) / bt)

        torch.cuda.synchronize(); t0 = time.perf_counter()
        s, st = spec.generate(pid, gen.max_new_tokens, params, eos_token_id=eos)
        torch.cuda.synchronize(); stime = time.perf_counter() - t0
        spec_tps.append(len(s) / stime)
        acc.append(st.acceptance_rate)
        k = min(len(b), len(s))
        agree.append(sum(x == y for x, y in zip(b[:k], s[:k])) / k)

    import statistics as st
    base_m, spec_m = st.mean(base_tps), st.mean(spec_tps)
    print("\n== Speculative decoding (Qwen2.5-1.5B target / 0.5B draft, greedy) ==")
    print(f"  target-only      : {base_m:5.1f} tok/s")
    print(f"  speculative (K={args.k}): {spec_m:5.1f} tok/s")
    print(f"  speedup          : {spec_m / base_m:.2f}x")
    print(f"  acceptance rate  : {st.mean(acc):.2f}")
    print(f"  lossless agree   : {st.mean(agree):.3f}")


if __name__ == "__main__":
    main()
