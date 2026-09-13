"""Cross-model KV transfer benchmark: escalating Qwen3-1.7B -> Qwen3-4B.

Escalation replaces the target prefilling the prompt with: exporting the source's KV,
mapping it into the target's layout (dispec/kv/transfer.py), importing it, and
forwarding only the last prompt token through the target. Timed per prompt length:

  target prefill  what escalation avoids
  warm            map (export + RoPE + per-layer map) + inject (import + 1-token forward)
                  -- the source already served the prompt, so its prefill is sunk cost
  cold            warm + the source's own prefill

Reported for plain ridge and for ridge + the trained residual. Also a quality sanity
check on BENCH_PROMPTS: how often the escalated target's greedy tokens agree with the
target run normally, next to the same agreement for the small source model alone --
the baseline that says whether escalating bought anything. Greedy agreement is a
coarse signal (bf16 near-ties flip tokens); kvxfer measures quality properly.

Needs both models on one GPU (~11.5 GB bf16) plus maps from kvxfer.

Run: python -m bench.kvxfer_bench --artifacts <kvxfer pair dir containing maps/ and residual/>
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from dispec.config import BENCH_PROMPTS, KVXFER_SOURCE_MODEL, KVXFER_TARGET_MODEL
from dispec.engine.model_runner import SeqMeta
from dispec.kv.transfer import CrossModelKVMap
from dispec.models.loader import load_model, load_tokenizer
from dispec.sampling import SamplingParams
from dispec.sched.scheduler import ContinuousBatchingEngine
from dispec.workers.escalation import escalate, map_prefix, prefill_and_map


def _median_ms(fn, repeats: int) -> float:
    fn()  # warmup (allocator, kernel selection)
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def _prefill(runner, cache, sid, ids, keep=False):
    n = len(ids)
    dev = runner.device
    table = cache.manager.allocate(sid, n)
    table.num_tokens = n
    hidden = runner.forward(torch.tensor(ids, device=dev), torch.arange(n, device=dev),
                            cache.context_slots(table.block_ids, n),
                            [SeqMeta(table.block_ids, n, n)], cache)
    runner.logits(hidden[-1:])
    if not keep:
        cache.manager.free(sid)
    return table


def _inject(runner, cache, sid, ids, k, v):
    n = len(ids)
    dev = runner.device
    table = cache.manager.allocate(sid, n)
    cache.import_contiguous(table.block_ids, k, v)
    table.num_tokens = n
    slot = cache.context_slots(table.block_ids, n)[-1:]
    hidden = runner.forward(torch.tensor(ids[-1:], device=dev), torch.tensor([n - 1], device=dev),
                            slot, [SeqMeta(table.block_ids, 1, n)], cache)
    runner.logits(hidden[-1:])
    cache.manager.free(sid)


@torch.inference_mode()
def bench_latency(src_eng, tgt_eng, kv_map, n_tokens: int, repeats: int) -> dict:
    ids = torch.randint(0, 100_000, (n_tokens,),
                        generator=torch.Generator().manual_seed(n_tokens)).tolist()
    sr, sc, tr, tc = src_eng.runner, src_eng.cache, tgt_eng.runner, tgt_eng.cache
    point = {"n_tokens": n_tokens,
             "target_prefill_ms": _median_ms(lambda: _prefill(tr, tc, 10_001, ids), repeats),
             "source_prefill_ms": _median_ms(lambda: _prefill(sr, sc, 10_002, ids), repeats)}

    table = _prefill(sr, sc, 10_003, ids, keep=True)
    for label, use_residual in (("ridge", False), ("residual", True)):
        mapped = {}

        def do_map():
            mapped["kv"] = map_prefix(sr, sc, table.block_ids, n_tokens, kv_map, tr, use_residual)

        map_ms = _median_ms(do_map, repeats)
        k, v = mapped["kv"]
        inject_ms = _median_ms(lambda: _inject(tr, tc, 10_004, ids, k, v), repeats)
        warm = map_ms + inject_ms
        cold = warm + point["source_prefill_ms"]
        point[label] = {"map_ms": map_ms, "inject_ms": inject_ms, "warm_ms": warm, "cold_ms": cold,
                        "warm_speedup": point["target_prefill_ms"] / warm,
                        "cold_speedup": point["target_prefill_ms"] / cold}
        del mapped, k, v
    sc.manager.free(10_003)
    return point


def _generate(engine, ids, n_new, prefix_kv=None):
    params = SamplingParams(0.0)
    if prefix_kv is None:
        rid = engine.add_request(ids, n_new, params)
    else:
        rid = escalate(engine, ids, prefix_kv, n_new, params)
    engine.run_until_done()
    return engine.collect_outputs()[rid]


def _agreement(out, reference) -> dict:
    prefix = 0
    for a, b in zip(out, reference):
        if a != b:
            break
        prefix += 1
    return {"first_token": out[:1] == reference[:1], "matched_prefix": prefix,
            "positional": sum(a == b for a, b in zip(out, reference)) / len(reference)}


@torch.inference_mode()
def bench_quality(src_eng, tgt_eng, kv_map, tok, n_new: int) -> dict:
    rows = []
    for i, prompt in enumerate(BENCH_PROMPTS):
        ids = tok(prompt).input_ids
        target = _generate(tgt_eng, ids, n_new)
        row = {"prompt": prompt, "source": _agreement(_generate(src_eng, ids, n_new), target)}
        for label, use_residual in (("ridge", False), ("residual", True)):
            prefix = prefill_and_map(src_eng.runner, src_eng.cache, 20_000 + i, ids, kv_map,
                                     tgt_eng.runner, use_residual)
            row[label] = _agreement(_generate(tgt_eng, ids, n_new, prefix), target)
        rows.append(row)

    summary = {}
    for label in ("source", "ridge", "residual"):
        summary[label] = {
            "first_token": sum(r[label]["first_token"] for r in rows) / len(rows),
            "matched_prefix": statistics.mean(r[label]["matched_prefix"] for r in rows),
            "positional": statistics.mean(r[label]["positional"] for r in rows),
        }
    return {"new_tokens": n_new, "summary": summary, "prompts": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifacts", required=True,
                    help="kvxfer pair directory holding maps/ and residual/")
    ap.add_argument("--lengths", default="512,1024,2048,4096,8192")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--out", default="kvxfer_bench.json")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    # Each engine holds at most one long sequence at a time: timed prefills free before
    # the kept source sequence is allocated, and the target never holds a timed prefill
    # and an injection together. The extra blocks are for the quality prompts.
    blocks = (max(lengths) + 15) // 16 + 64
    src = load_model(KVXFER_SOURCE_MODEL)
    tgt = load_model(KVXFER_TARGET_MODEL)
    tok = load_tokenizer(KVXFER_TARGET_MODEL)
    src_eng = ContinuousBatchingEngine(src, num_blocks=blocks, max_batch_tokens=max(lengths))
    tgt_eng = ContinuousBatchingEngine(tgt, num_blocks=blocks, max_batch_tokens=max(lengths))
    art = Path(args.artifacts)
    kv_map = CrossModelKVMap.load(art / "maps", src.config, tgt.config,
                                  residual_dir=art / "residual", device="cuda",
                                  dtype=tgt_eng.runner.dtype)

    result = {"source": KVXFER_SOURCE_MODEL, "target": KVXFER_TARGET_MODEL,
              "gpu": torch.cuda.get_device_name(), "dtype": str(tgt_eng.runner.dtype),
              "repeats": args.repeats, "points": []}

    # The first-ever calls pay for allocator growth and kernel selection. Without a
    # discarded pass, the first length's ridge timing came out slower than the next
    # length's (138 ms at 512 tokens vs 104 ms at 1024).
    bench_latency(src_eng, tgt_eng, kv_map, min(lengths), repeats=1)

    print(f"{KVXFER_SOURCE_MODEL} -> {KVXFER_TARGET_MODEL} on {result['gpu']}")
    print(f"{'tokens':>7} {'tgt prefill':>12} {'src prefill':>12} "
          f"{'ridge warm':>11} {'x':>6} {'resid warm':>11} {'x':>6} {'resid cold':>11} {'x':>6}")
    for n in lengths:
        try:
            p = bench_latency(src_eng, tgt_eng, kv_map, n, args.repeats)
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            result["points"].append({"n_tokens": n, "error": f"out of memory: {exc}"[:200]})
            print(f"{n:>7} out of memory")
            continue
        result["points"].append(p)
        r, s = p["ridge"], p["residual"]
        print(f"{n:>7} {p['target_prefill_ms']:>10.1f}ms {p['source_prefill_ms']:>10.1f}ms "
              f"{r['warm_ms']:>9.1f}ms {r['warm_speedup']:>5.2f}x {s['warm_ms']:>9.1f}ms "
              f"{s['warm_speedup']:>5.2f}x {s['cold_ms']:>9.1f}ms {s['cold_speedup']:>5.2f}x",
              flush=True)

    result["quality"] = bench_quality(src_eng, tgt_eng, kv_map, tok, args.new_tokens)
    print(f"\ngreedy agreement with the target over {args.new_tokens} tokens "
          f"({len(BENCH_PROMPTS)} prompts):")
    for label, s in result["quality"]["summary"].items():
        print(f"  {label:9s} first token {s['first_token']:.0%}  matched prefix "
              f"{s['matched_prefix']:.1f}  positional {s['positional']:.0%}")

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
