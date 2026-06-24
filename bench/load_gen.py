"""Closed/open-loop load generator for the DiSpec server.

Fires requests at a target rate with Poisson (exponential inter-arrival) timing — the
standard way to load-test a serving system — and reports latency percentiles and
goodput. Point it at a running server (python -m dispec.router.app).

Run: python -m bench.load_gen --rate 8 --n 64 --max-new 64
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time

import httpx

from dispec.config import BENCH_PROMPTS


async def _one(client: httpx.AsyncClient, url: str, prompt: str, max_new: int) -> tuple[float, int]:
    t0 = time.perf_counter()
    r = await client.post(url, json={"prompt": prompt, "max_new_tokens": max_new})
    r.raise_for_status()
    return time.perf_counter() - t0, r.json()["num_tokens"]


async def run(url: str, rate: float, n: int, max_new: int) -> None:
    lats: list[float] = []
    toks = 0
    tasks: list[asyncio.Task] = []
    async with httpx.AsyncClient(timeout=120) as client:
        wall0 = time.perf_counter()
        for i in range(n):
            prompt = BENCH_PROMPTS[i % len(BENCH_PROMPTS)]
            tasks.append(asyncio.create_task(_one(client, url, prompt, max_new)))
            await asyncio.sleep(random.expovariate(rate))  # Poisson arrivals
        for t in tasks:
            dt, nt = await t
            lats.append(dt)
            toks += nt
        wall = time.perf_counter() - wall0

    lats.sort()
    pct = lambda p: lats[min(int(len(lats) * p), len(lats) - 1)]
    print(f"\nrequests={n} target_rate={rate}/s wall={wall:.1f}s")
    print(f"  latency  p50={pct(0.5)*1e3:.0f}ms  p90={pct(0.9)*1e3:.0f}ms  p99={pct(0.99)*1e3:.0f}ms")
    print(f"  mean latency={statistics.mean(lats)*1e3:.0f}ms")
    print(f"  goodput  {toks/wall:.1f} tok/s  ({n/wall:.1f} req/s completed)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/generate")
    ap.add_argument("--rate", type=float, default=8.0, help="target arrivals/sec (Poisson)")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    asyncio.run(run(args.url, args.rate, args.n, args.max_new))


if __name__ == "__main__":
    main()
