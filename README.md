# DiSpec — Disaggregated Speculative Decoding (from scratch)

A from-scratch LLM inference engine that unifies the two biggest levers in modern
serving: **speculative decoding** (to fight the memory-bandwidth wall in decode) and
**prefill/decode disaggregation** (to stop prefill and decode from fighting over the
same GPU and wrecking latency SLOs).

The whole serving stack is implemented from scratch — paged KV cache, continuous-batching
scheduler, the model forward pass, speculative decoding, and (later) cross-process KV
transfer — using PyTorch + HuggingFace *weight modules* only for the raw matmuls. It is
**not** built on vLLM/TGI; those are used only as reference baselines.

## Why this matters

- **Decode is memory-bandwidth-bound.** Generating one token streams the entire model from
  HBM, so the big target GPU is idle. Speculative decoding lets a small draft model propose
  many tokens that the target verifies in a single forward pass — provably preserving the
  target's output distribution.
- **Prefill and decode interfere.** Prefill is compute-bound and bursty; decode is
  bandwidth-bound and steady. Co-locating them hurts tail latency. Disaggregation runs them
  on separate workers and transfers the KV cache between them.

## What works today (Phases 0–1)

- **Paged KV cache** with a block allocator and copy-on-write forking (`dispec/kv/`).
- **From-scratch Qwen2 forward** over the paged cache: own rotary embeddings, GQA, and a
  right-aligned causal paged attention (`dispec/engine/model_runner.py`). Validated to match
  HuggingFace logits (per-position cosine ≈ 0.99995, identical argmax).
- **Continuous batching** scheduler: iteration-level scheduling that mixes prefill and decode
  tokens in one forward pass, token-budgeted, with no cross-sequence leakage
  (`dispec/sched/scheduler.py`).
- **Benchmarks** vs a HuggingFace baseline (`bench/`).

### Phase-1 results (Qwen2.5-1.5B, RTX 3070 Laptop 8 GB, bf16)

| Config | Throughput |
|---|---|
| HuggingFace baseline (single-stream) | 54.9 tok/s |
| DiSpec single-sequence | 41.0 tok/s |
| **DiSpec continuous batching** | **~92 tok/s (≈1.7×)** |

Throughput plateaus because the per-sequence Python attention loop is CPU-bound — motivating
a fused Triton paged-attention kernel (planned). Correctness, not single-stream speed, is the
Phase-1 goal.

### Phase-2 results — speculative decoding (lossless)

Sequential speculative decoding with a 0.5B draft + rejection sampling
(`dispec/spec/`). The acceptance math is unit-tested to reproduce the target
distribution; on real models it is lossless in practice (greedy output matches
target-only) at **~50% acceptance, ~3.6 tokens per target iteration**.

Wall-clock speedup, however, is currently **<1×** — and profiling shows exactly why,
which is the interesting part:

| Forward (7B-int4 target / 0.5B draft) | Time |
|---|---|
| Target decode, 1 token | 31.9 ms |
| **Target verify, 5 tokens** | **32.0 ms** (≈ same as 1) |
| Draft decode, 1 token | 20.2 ms |

The target forward is **launch-bound**: verifying 5 tokens costs the same as decoding
1, so speculation's core mechanism — amortizing the target — works perfectly. But the
*draft* forward also carries ~20 ms of Python/launch overhead (a 0.5B model is only
~3 ms of real compute), so the K≈5 draft steps cost more than the target call they
save. **Speculation here is bottlenecked by per-forward overhead, not target size** —
the fix is CUDA graphs / `torch.compile`d forwards to make the draft cheap, not a
bigger model. (int4 7B was tried specifically to test the "bigger target" hypothesis;
the profile above is why it didn't help.)

### Phase-3 results — true P/D disaggregation

Prefill and decode run as **separate processes** wired by a pluggable KV-transfer
engine (`dispec/transport/`, `dispec/workers/`). The decode worker generates from KV
computed by the prefill worker and shipped over the transport; output is
**token-for-token identical** to the colocated baseline (verified in tests).

KV-transfer cost (Qwen2.5-7B geometry, RTX 3070):

| prompt len | KV size | TCP transfer | TCP throughput |
|---|---|---|---|
| 512 | 29 MB | 99 ms | 0.30 GB/s |
| 2048 | 117 MB | 350 ms | 0.34 GB/s |

`TcpTransport` (the multi-node fallback) is serialize+copy bound and scales linearly;
`CudaIpcTransport` hands off the same KV by sharing the GPU buffer's IPC handle —
zero-copy, O(1) in payload size. This is the concrete argument for on-node IPC/NVLink
and for an RDMA/NIXL backend over the wire.

## Roadmap

- **Phase 2 (done, analyzed)** — lossless speculative decoding; wall-clock speedup
  pending overhead reduction (CUDA graphs / compiled forwards), see above.
- **Phase 3 (done)** — true P/D disaggregation across processes with a pluggable
  KV-transfer engine (CUDA IPC zero-copy on-node, TCP multi-node); output matches
  colocated. Remaining: SLO-aware router + draft-pool autoscaling.
- **Phase 4 (in progress)** — FastAPI serving layer + Prometheus/Grafana observability
  + Poisson load testing (done); ablations + vLLM comparison (todo).
- **Phase 5** — Triton kernels (paged + tree attention), EAGLE draft head, int4.

### Serving layer + observability (Phase 4)

A FastAPI server runs the continuous-batching engine on a background scheduler thread
(`dispec/router/`): async `/generate` handlers submit requests and await futures the
scheduler resolves on completion. `/metrics` exposes TTFT, TPOT, end-to-end latency,
throughput, queue depth, and batch size (Prometheus); a Grafana dashboard is in
`dashboards/`. `bench/load_gen.py` drives it with Poisson arrivals and reports latency
percentiles + goodput (e.g. 32 reqs @ 8/s → 76 tok/s goodput, with queueing latency
under overload as expected).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m dispec.env_check          # validate GPU / bf16 / triton
.venv/bin/python -m pytest -q                  # run tests

# benchmarks
.venv/bin/python -m bench.baseline_hf          # HF baseline
.venv/bin/python -m bench.dispec_bench         # engine: single-seq vs continuous batching
.venv/bin/python -m bench.spec_bench           # speculative decoding (add --gptq for int4 7B)
.venv/bin/python -m bench.disagg_bench         # KV-transfer latency

# serve + load test
.venv/bin/python -m dispec.router.app          # FastAPI server on :8000 (/generate /metrics)
.venv/bin/python -m bench.load_gen --rate 8 --n 64
```

int4 GPTQ target needs the `quant` extra (`pip install -e ".[quant]"`) and `ninja` on PATH.

## Layout

```
dispec/
  config.py            model choices + generation/KV/spec configs
  sampling.py          greedy / top-k / top-p, built for spec-decode rejection math
  env_check.py         GPU / bf16 / triton validation
  kv/                  block_manager.py (paged allocator + COW), paged_cache.py (GPU pool)
  engine/              model_runner.py (from-scratch Qwen2 fwd), engine.py (single-seq)
  sched/               scheduler.py (continuous batching)
  spec/                rejection.py (lossless verify), speculative.py (draft+target)
  transport/           base/tcp/cuda_ipc — pluggable KV-transfer engine
  workers/             disaggregated.py — prefill/decode worker processes
  router/              app.py (FastAPI), server.py (scheduler thread), metrics.py
bench/                 baseline_hf, dispec_bench, spec_bench, disagg_bench, load_gen
dashboards/            dispec.json (Grafana)
tests/                 17+ tests: cache, runner, scheduler, rejection, spec, transport, disagg, server
```
