"""Unified ablation table: every DiSpec technique on the same model + prompts.

Compares, on one GPU with one model pair, the throughput of:
  - HuggingFace baseline (single stream)        -- the reference
  - DiSpec single-sequence                       -- our engine, no batching
  - DiSpec continuous batching                   -- iteration-level scheduling
  - DiSpec speculative decoding                   -- draft + target verify

So the contribution of each technique is visible side by side. Disaggregation is a
latency/placement feature (correctness + transfer cost shown by bench.disagg_bench and
the disagg tests), not a single-GPU throughput lever, so it's reported separately.

Run: python -m bench.ablations [--max-new 96] [--concurrency 8] [--k 5]
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from dispec.config import BENCH_PROMPTS, SpecConfig
from dispec.engine.engine import LLMEngine
from dispec.models.loader import build_prompt, load_draft, load_target
from dispec.sampling import SamplingParams
from dispec.sched.scheduler import ContinuousBatchingEngine
from dispec.spec.speculative import SpeculativeEngine


def hf_single(model, tok, prompts, max_new):
    # Warm up like every DiSpec row does, so HF is not charged for the CUDA init and kernel
    # selection that the first-ever generate call pays.
    ids = tok(build_prompt(tok, prompts[0]), return_tensors="pt").input_ids.to(model.device)
    model.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
    tps = []
    for p in prompts:
        ids = tok(build_prompt(tok, p), return_tensors="pt").input_ids.to(model.device)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        n = out.shape[1] - ids.shape[1]
        tps.append(n / (time.perf_counter() - t0))
    return statistics.mean(tps), None


def dispec_single(model, tok, prompts, max_new, cuda_graph=False):
    eng = LLMEngine(model, num_blocks=2048, cuda_graph=cuda_graph,
                    graph_buckets=(512, 2048) if cuda_graph else (256,))
    params = SamplingParams(0.0)
    eng.generate(tok(prompts[0]).input_ids, 8, params, eos_token_id=tok.eos_token_id)
    tps = []
    for p in prompts:
        ids = tok(build_prompt(tok, p)).input_ids
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = eng.generate(ids, max_new, params, eos_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        tps.append(len(out) / (time.perf_counter() - t0))
    return statistics.mean(tps), None


def dispec_cb(model, tok, prompts, max_new, concurrency):
    eng = ContinuousBatchingEngine(model, num_blocks=2048, max_batch_tokens=4096)
    params = SamplingParams(0.0)
    reqs = (prompts * ((concurrency // len(prompts)) + 1))[:concurrency]
    for p in reqs:
        eng.add_request(tok(build_prompt(tok, p)).input_ids, max_new, params,
                        eos_id=tok.eos_token_id)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    eng.run_until_done()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n = sum(len(v) for v in eng.collect_outputs().values())
    return n / dt, None


def dispec_spec(target, draft, tok, prompts, max_new, k):
    eng = SpeculativeEngine(target, draft, spec=SpecConfig(num_speculative_tokens=k),
                            num_blocks=2048)
    params = SamplingParams(0.0)
    eng.generate(tok(prompts[0]).input_ids, 8, params, eos_token_id=tok.eos_token_id)
    tps, acc = [], []
    for p in prompts:
        ids = tok(build_prompt(tok, p)).input_ids
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out, st = eng.generate(ids, max_new, params, eos_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        tps.append(len(out) / (time.perf_counter() - t0))
        acc.append(st.acceptance_rate)
    return statistics.mean(tps), statistics.mean(acc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    print("Loading target + draft...")
    target, tok = load_target()
    draft, _ = load_draft()
    prompts = BENCH_PROMPTS

    def free():
        torch.cuda.empty_cache()

    rows = []
    base, _ = hf_single(target, tok, prompts, args.max_new)
    rows.append(("HF baseline (single stream)", base, "")); free()
    s, _ = dispec_single(target, tok, prompts, args.max_new)
    rows.append(("DiSpec single-sequence", s, "")); free()
    sg, _ = dispec_single(target, tok, prompts, args.max_new, cuda_graph=True)
    rows.append(("DiSpec single-seq + CUDA graph", sg, "")); free()
    cb, _ = dispec_cb(target, tok, prompts, args.max_new, args.concurrency)
    rows.append((f"DiSpec continuous batching (c={args.concurrency})", cb, "")); free()
    sp, acc = dispec_spec(target, draft, tok, prompts, args.max_new, args.k)
    rows.append((f"DiSpec speculative (K={args.k})", sp, f"accept={acc:.0%}")); free()

    print(f"\n{'config':<40} {'tok/s':>8} {'vs base':>8}  notes")
    print("-" * 72)
    for name, tps, note in rows:
        print(f"{name:<40} {tps:>8.1f} {tps / base:>7.2f}x  {note}")


if __name__ == "__main__":
    main()
