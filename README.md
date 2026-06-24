# DiSpec — a from-scratch LLM inference engine

DiSpec is an LLM serving engine I wrote from scratch to understand how systems like vLLM
actually work — by building the pieces myself rather than calling them. It runs real models
(Qwen2.5) on a single 8 GB laptop GPU and implements the whole path: a paged KV cache, a
continuous-batching scheduler, the model forward pass, CUDA-graph decode, speculative
decoding, prefill/decode disaggregation with KV transfer, and an HTTP server with metrics.

The only thing I lean on PyTorch/HuggingFace for is the raw matmuls (and the pretrained
weights). Everything around them — cache layout, attention masking, scheduling, sampling,
the speculative-decoding math, cross-process KV movement — is hand-written. vLLM and
HuggingFace `generate` show up only as baselines to measure against.

The name comes from two of the features (**Di**saggregation + **Spec**ulative decoding), but
honestly the engine is broader than that, and the biggest speedups come from the less
glamorous parts: continuous batching and CUDA graphs.

## Why these techniques exist

Two facts drive almost everything in LLM serving:

- **Decode is memory-bandwidth-bound.** Generating one token reads the entire model out of
  HBM, so a big GPU sits mostly idle during decode. *Speculative decoding* hides this: a
  small draft model guesses several tokens and the big model verifies them in a single
  forward pass — provably without changing the output distribution.
- **Prefill and decode want different things.** Prefill is compute-heavy and bursty; decode
  is bandwidth-heavy and steady. Run them on the same GPU and they hurt each other's
  latency. *P/D disaggregation* splits them onto separate workers and ships the KV cache
  between them.

DiSpec implements both, plus the machinery they need to be useful (paging, batching, a
scheduler, a server).

## What's in it

- **Paged KV cache** (`dispec/kv/`) — a block allocator with copy-on-write forking and a
  GPU block pool, so variable-length sequences share memory without padding waste.
- **From-scratch Qwen2 forward** (`dispec/engine/model_runner.py`) — my own rotary
  embeddings, grouped-query attention, and right-aligned causal masking over the paged
  cache. Verified against HuggingFace at the logits level (per-position cosine ≈ 0.99995,
  same argmax).
- **Continuous batching** (`dispec/sched/scheduler.py`) — iteration-level scheduling that
  mixes prefill and decode tokens in one forward pass, bounded by a token budget, with
  priority-aware admission. No cross-sequence leakage (tested).
- **CUDA-graph decode** (`dispec/engine/cuda_graph.py`) — captures the decode step as a
  replayable graph to kill per-layer launch overhead. This is the single biggest win.
- **Speculative decoding** (`dispec/spec/`) — independent 0.5B draft + rejection sampling,
  lossless (the acceptance math is unit-tested to reproduce the target distribution).
- **P/D disaggregation** (`dispec/transport/`, `dispec/workers/`) — prefill and decode as
  separate processes with a pluggable KV transport: zero-copy CUDA IPC on one node, TCP for
  multi-node.
- **Serving + observability** (`dispec/router/`) — a FastAPI server, Prometheus `/metrics`,
  a built-in live `/dashboard`, an optional Grafana stack, SLO priority routing, and a
  draft-pool autoscaler.

25 tests cover all of it (cache, forward correctness, batching, rejection math, spec
decoding, transports, disaggregation, CUDA graphs, the HTTP server, the autoscaler).

## Numbers

All on Qwen2.5-1.5B, RTX 3070 Laptop (8 GB), bf16, greedy.

**Throughput** — each row adds one technique:

| Config | tok/s | vs HF |
|---|---|---|
| HuggingFace `generate` (single stream) | 51 | 1.0× |
| DiSpec single-sequence, eager | 40 | 0.78× |
| DiSpec single-sequence + CUDA graph | 68 | 1.32× |
| DiSpec continuous batching | 86 | 1.66× |
| *vLLM (reference)* | *528* | *10×* |

The eager engine is slower than HF single-stream — no surprise, it's a Python loop over 28
layers and the launch overhead dominates. CUDA graphs fix that: capturing the decode step
makes it 3.6× faster on the 0.5B and 1.7× on the 1.5B, enough to pass HF. Continuous
batching is the throughput lever.

vLLM is ~6× faster than my continuous batching, and that gap is the honest one: it's
optimized CUDA/Triton kernels, CUDA graphs everywhere, and years of scheduler tuning. DiSpec
has the same *architecture* and is correct — closing the constant factor is the remaining
work, not a redesign. (I also tried Liger kernels; they were *slower* for single-token
decode because they're tuned for training-size shapes.)

**Speculative decoding** is the interesting disappointment. It's lossless and accepts ~50%
of drafted tokens (~3.6 tokens per target step), but the wall-clock speedup is currently
**below 1×**. Profiling says exactly why:

| Forward (7B-int4 target / 0.5B draft) | time |
|---|---|
| target decode, 1 token | 31.9 ms |
| target verify, 5 tokens | 32.0 ms |
| draft decode, 1 token | 20.2 ms |

Verifying 5 tokens costs the same as decoding 1 — so the *idea* works perfectly, the target
forward is pure launch overhead and amortizes for free. The problem is the draft: it's also
~20 ms of launch overhead (a 0.5B model is ~3 ms of actual compute), so the handful of draft
steps cost more than the target call they save. The fix isn't a bigger target (I confirmed
that with an int4 7B — same result); it's making the draft cheap with CUDA graphs. The graph
machinery now exists for plain decode; wiring it into the draft loop is what flips this
positive, and it's the next thing I'd do.

**KV transfer** for disaggregation (Qwen2.5-7B KV geometry):

| prompt length | KV size | TCP transfer | throughput |
|---|---|---|---|
| 512 | 29 MB | 99 ms | 0.30 GB/s |
| 2048 | 117 MB | 350 ms | 0.34 GB/s |

TCP is serialize-and-copy bound and scales with payload size — it's the multi-node fallback.
CUDA IPC moves the same KV by passing the GPU buffer's handle: zero-copy, constant time. The
disaggregated output is token-for-token identical to running it all on one process (tested),
which is the thing that actually has to be true.

## How it fits together

```
            client ──► FastAPI router ──► scheduler (continuous batching, priority)
                          /metrics            │
                          /dashboard          ▼
                                        model runner ──► paged KV cache
                                          (CUDA graph)        │
                                                              │ export/import
       disaggregated mode:  prefill worker ──KV transport──► decode worker
                                              (CUDA IPC / TCP)

       speculative mode:    draft model ──proposes──► target model verifies (rejection sampling)
```

## Running it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

.venv/bin/python -m dispec.env_check     # check GPU / bf16 / triton
.venv/bin/python -m pytest -q            # 25 tests

# benchmarks
.venv/bin/python -m bench.ablations      # the throughput table above
.venv/bin/python -m bench.spec_bench     # speculative decoding (--gptq for the int4 7B target)
.venv/bin/python -m bench.disagg_bench   # KV-transfer cost

# serve it
.venv/bin/python -m dispec.router.app    # http://localhost:8000  (/generate, /metrics, /dashboard)
.venv/bin/python -m bench.load_gen --rate 8 --n 64   # Poisson load
```

For metrics you have two options: open `http://localhost:8000/dashboard` for a built-in live
page (no extra setup), or run `docker compose up -d` to get Prometheus + Grafana with the
dashboard in `dashboards/` auto-loaded.

The int4 GPTQ target (`bench.spec_bench --gptq`) needs the `quant` extra
(`uv pip install -e ".[quant]"`) and `ninja` on `PATH` for the Marlin kernels. vLLM, if you
want to reproduce the reference number, goes in a separate venv (`bench/vllm_ref.py`) since it
ships its own torch.

## Layout

```
dispec/
  config.py          models + generation/KV/spec config
  sampling.py        greedy / top-k / top-p, written for the rejection-sampling math
  env_check.py       GPU / bf16 / triton check
  kv/                block_manager.py (paged allocator + COW), paged_cache.py (GPU pool)
  engine/            model_runner.py (Qwen2 forward), engine.py (single-seq), cuda_graph.py
  sched/             scheduler.py (continuous batching + priority admission)
  spec/              rejection.py (lossless verify), speculative.py (draft + target)
  transport/         base / cuda_ipc / tcp — the KV-transfer engine
  workers/           disaggregated.py — prefill & decode worker processes
  router/            app.py (FastAPI), server.py (scheduler thread), metrics.py,
                     dashboard.py (built-in UI), autoscale.py (draft-pool controller)
bench/               ablations, baseline_hf, dispec_bench, spec_bench, disagg_bench,
                     load_gen, vllm_ref
dashboards/          dispec.json (Grafana)        monitoring/  Prometheus + Grafana config
tests/               25 tests
```

## What's left

Roughly in the order I'd do it:

1. **Graph-accelerate the speculative draft loop** — the one change that flips speculative
   decoding to a real wall-clock win.
2. **Batched CUDA graphs** for continuous batching — pushes server throughput toward vLLM.
3. **Triton kernels** — a real paged-attention kernel (and tree-attention for tree
   speculation), which is where most of the remaining gap to vLLM lives.
4. **EAGLE draft head** — to push acceptance from ~50% toward 80%+.
5. **int4 everywhere** — fit a larger target and free up KV memory.

This is a learning/portfolio project, not a production server — the goal was to build the
real thing end to end and be able to explain every number above, including the ones that
didn't go my way.
