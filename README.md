# DiSpec — a from-scratch LLM inference engine

DiSpec is a from-scratch LLM serving engine that implements the internals of systems like
vLLM directly rather than calling them. It runs real models (Qwen2.5, Qwen3) on a single 8 GB
laptop GPU and covers the whole path: a paged KV cache with prefix sharing, a continuous-batching
scheduler, the model forward pass, CUDA-graph decode, speculative decoding, prefill/decode
disaggregation with KV transfer, cross-model KV transfer for escalating a request from a small
model to a large one, and an OpenAI-compatible HTTP server with metrics.

PyTorch/HuggingFace is used only for the raw matmuls (and the pretrained weights). Everything
around them — cache layout, attention masking, scheduling, sampling, the speculative-decoding
math, cross-process KV movement — is implemented here. vLLM and HuggingFace `generate` serve
only as measurement baselines.

The name comes from two of the features (**Di**saggregation + **Spec**ulative decoding), but
the engine is broader than that, and the largest speedups come from continuous batching, CUDA
graphs, and the Triton attention kernel.

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
- **Prefix caching** (`dispec/kv/prefix_cache.py`) — requests that share a leading prefix
  (system prompt, few-shot preamble, chat history) reuse cached KV blocks instead of
  re-prefilling them, with LRU eviction under memory pressure.
- **From-scratch Qwen2 / Qwen3 forward** (`dispec/engine/model_runner.py`) — custom rotary
  embeddings, grouped-query attention, Qwen3's per-head q/k norm, and right-aligned causal
  masking over the paged cache. Verified against HuggingFace at the logits level (per-position cosine ≈ 0.99995,
  same argmax).
- **Continuous batching** (`dispec/sched/scheduler.py`) — iteration-level scheduling that
  mixes prefill and decode tokens in one forward pass, bounded by a token budget, with
  priority-aware admission, and chunked prefill (long prompts slice into the batch
  instead of stalling decodes). No cross-sequence leakage (tested).
- **CUDA-graph decode** (`dispec/engine/cuda_graph.py`) — captures the decode step as a
  replayable graph to kill per-layer launch overhead.
- **Triton paged-attention kernel** (`dispec/engine/triton_attn.py`) — a hand-written
  fused flash-decoding kernel (online softmax, GQA-aware) that reads the paged cache
  directly, replacing gather + repeat_kv + SDPA. Optional `attn_backend="triton"`.
- **Speculative decoding** (`dispec/spec/`) — independent 0.5B draft + rejection sampling,
  lossless (the acceptance math is unit-tested to reproduce the target distribution).
- **P/D disaggregation** (`dispec/transport/`, `dispec/workers/`) — prefill and decode as
  separate processes with a pluggable KV transport: zero-copy CUDA IPC on one node, TCP for
  multi-node.
- **Cross-model KV transfer** (`dispec/kv/transfer.py`, `dispec/workers/escalation.py`) —
  escalate a request from Qwen3-1.7B to Qwen3-4B without the 4B model prefilling the prompt.
  The small model's KV cache is mapped into the large model's layout by per-layer linear maps
  (plus a small trained residual) fitted offline by
  [kvxfer](https://github.com/puneethgv/kvxfer), then admitted to the scheduler as prefix KV,
  so the 4B model forwards a single token before decoding.
- **Serving + observability** (`dispec/router/`) — a FastAPI server with an
  OpenAI-compatible `/v1/chat/completions` endpoint (streaming + non-streaming), Prometheus
  `/metrics`, a built-in live `/dashboard`, an optional Grafana stack, SLO priority routing,
  and a draft-pool autoscaler.

59 tests cover all of it (cache, prefix cache, forward correctness, batching, rejection
math, spec decoding, transports, disaggregation, CUDA graphs, the HTTP server, the
autoscaler, the Qwen3 forward, cross-model KV transfer).

## Numbers

All on Qwen2.5-1.5B, RTX 3070 Laptop (8 GB), bf16, greedy.

**Single-sequence** — each row adds one technique (`bench/ablations.py`):

| Config | tok/s | vs HF |
|---|---|---|
| HuggingFace `generate` | 51 | 1.0× |
| DiSpec single-sequence, eager | 40 | 0.78× |
| DiSpec single-sequence + CUDA graph | 68 | 1.32× |

Eager single-stream is slower than HF — it's a Python loop over 28 layers and launch
overhead dominates. CUDA-graph capture of the decode step fixes that (3.6× on 0.5B, 1.7× on
1.5B) and passes HF.

**Continuous-batching throughput vs vLLM**, matched concurrency, warmed
(`bench/throughput.py` and `bench/vllm_ref.py`):

| concurrency | DiSpec (Triton + fused) | vLLM | gap |
|---|---|---|---|
| 6 | 373 tok/s | 528 | 1.4× |
| 16 | 858 tok/s | 1337 | 1.6× |
| 32 | 1426 tok/s | 2498 | 1.75× |

The gap to vLLM narrows from ~6× to **~1.5×**. The changes that account for it, in order of
impact: building the per-step attention slot table **once** instead of per-layer; a
**batched** Triton decode kernel (one launch for the whole batch instead of a Python loop over
sequences); fused QKV / gate-up GEMMs; and CUDA-graph decode. The remaining ~1.5× is full
CUDA-graph capture of the *batched* step and FlashAttention-grade kernels — diminishing
returns. (Liger kernels were tried and were *slower* for single-token decode; they target
training-size shapes. flash-attn / FlashInfer have no torch-2.12/CUDA-13 wheel, so the Triton
kernel here is implemented directly.)

**Speculative decoding** is lossless and accepts ~50% of drafted tokens (~3.6 tokens per
target step), but the wall-clock speedup is currently **below 1×**. Profiling shows why:

| Forward (7B-int4 target / 0.5B draft) | time |
|---|---|
| target decode, 1 token | 31.9 ms |
| target verify, 5 tokens | 32.0 ms |
| draft decode, 1 token | 20.2 ms |

Verifying 5 tokens costs the same as decoding 1 — so the *idea* works perfectly, the target
forward is pure launch overhead and amortizes for free. The problem is the draft: it's also
~20 ms of launch overhead (a 0.5B model is ~3 ms of actual compute), so the handful of draft
steps cost more than the target call they save. The fix is not a bigger target (an int4 7B
target gives the same result); it is making the draft cheap with CUDA graphs. The graph
machinery already exists for plain decode; wiring it into the draft loop is what would flip
this positive.

**KV transfer** for disaggregation (Qwen2.5-7B KV geometry):

| prompt length | KV size | TCP transfer | throughput |
|---|---|---|---|
| 512 | 29 MB | 99 ms | 0.30 GB/s |
| 2048 | 117 MB | 350 ms | 0.34 GB/s |

TCP is serialize-and-copy bound and scales with payload size — it's the multi-node fallback.
CUDA IPC moves the same KV by passing the GPU buffer's handle: zero-copy, constant time. The
disaggregated output is token-for-token identical to running it all on one process (tested),
which is the thing that actually has to be true.

**Cross-model KV transfer** — escalating Qwen3-1.7B → Qwen3-4B on one L4, bf16
(`bench/kvxfer_bench.py`). *Target prefill* is what escalation avoids. *warm* is what it
costs instead when the 1.7B model has already served the prompt: map its KV (export + RoPE
strip/re-apply + per-layer map) and inject it (import + one 4B forward for the last prompt
token). *cold* adds the 1.7B prefill.

| tokens | Qwen3-4B prefill | ridge warm | ridge + residual warm | ridge + residual cold |
|---|---|---|---|---|
| 512 | 116.7 ms | 92.0 ms (1.27×) | 97.1 ms (1.20×) | 158.1 ms (0.74×) |
| 1024 | 246.6 ms | 100.5 ms (2.45×) | 107.2 ms (2.30×) | 209.4 ms (1.18×) |
| 2048 | 488.8 ms | 133.1 ms (3.67×) | 148.4 ms (3.29×) | 352.5 ms (1.39×) |
| 4096 | 1240.9 ms | 209.4 ms (5.93×) | 235.8 ms (5.26×) | 749.9 ms (1.65×) |
| 8192 | 3501.8 ms | 396.9 ms (8.82×) | 454.4 ms (7.71×) | 1911.0 ms (1.83×) |

Ratios are against DiSpec's own prefill, which is ~2× slower than vLLM's at 8k tokens. Against
vLLM's prefill on the same GPU type (1653 ms at 8192 tokens, measured in kvxfer), warm is 4.2×
(ridge) / 3.6× (residual) at 8192 and ~1.1× at 512: the gain grows with prompt length and is
negligible for short prompts. The trained residual adds 5–58 ms over ridge. Cold never exceeds
1.83×, because the 1.7B prefill it pays for is already ~40% of the 4B one.

The two models together are ~11.5 GB in bf16, so this does **not** fit the 8 GB GPU the rest
of the numbers are from; the maps add ~0.8 GB. Only models with identical KV heads, head dim
and tokenizer can be paired — which is why DiSpec's own draft/target pair can't use it
(Qwen2.5-0.5B has `head_dim=64`, 1.5B has 128).

It is a **latency** feature, not a quality one. kvxfer's evaluation of the same maps
(prefix-conditioned perplexity over 64 documents; 300-item ARC):

| | perplexity | ARC-Easy | ARC-Challenge |
|---|---|---|---|
| Qwen3-4B, own prefill | 13.56 | 0.850 | 0.497 |
| 4B with mapped cache, ridge + residual | 14.11 | 0.733 | 0.353 |
| 4B with mapped cache, ridge | 14.71 | 0.740 | 0.323 |
| Qwen3-1.7B alone | 16.48 | 0.747 | 0.380 |

The mapped cache makes the 4B model a better language model than the 1.7B model, but not a
more accurate one on ARC — escalating through it buys less than a real 4B prefill would.

Greedy agreement with the 4B model's own output over 32 tokens on the six `BENCH_PROMPTS` — a
coarse signal, since bf16 near-ties flip tokens — points the same way: the escalated 4B model
matches the real 4B model's first token on 67% of prompts (ridge and residual) against 50% for
the 1.7B model alone, with a matched prefix of 3.5–3.7 tokens against 2.5.

Correctness is gated rather than assumed: a model's own KV admitted through the transfer path
(raw, and through an identity map) must reproduce a plain prefill; DiSpec's map must agree with
kvxfer's reference implementation on identical inputs; and the trained map must predict the
4B model's next token with lower KL divergence than a zeroed cache.

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

       escalation mode:     small model ──KV map (kvxfer)──► large model decodes, no prefill
```

## Running it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

.venv/bin/python -m dispec.env_check     # check GPU / bf16 / triton
.venv/bin/python -m pytest -q            # 59 tests (CUDA ones skip without a GPU)

# benchmarks
.venv/bin/python -m bench.ablations      # the throughput table above
.venv/bin/python -m bench.spec_bench     # speculative decoding (--gptq for the int4 7B target)
.venv/bin/python -m bench.disagg_bench   # KV-transfer cost
.venv/bin/python -m bench.kvxfer_bench --artifacts <kvxfer pair dir>   # escalation (~12 GB GPU)

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

Cross-model KV transfer needs maps from [kvxfer](https://github.com/puneethgv/kvxfer)
(`scripts/calibrate.py` → `scripts/fit_maps.py` → `scripts/train_residual.py`). `--artifacts`
points at a directory holding its `maps/` and `residual/` outputs; they are ~1.6 GB and not
committed here. Maps fitted for other models are rejected on load unless the KV geometry and
tokenizer match.

## Layout

```
dispec/
  config.py          models + generation/KV/spec config
  sampling.py        greedy / top-k / top-p, written for the rejection-sampling math
  env_check.py       GPU / bf16 / triton check
  kv/                block_manager.py (paged allocator + COW), paged_cache.py (GPU pool),
                     prefix_cache.py (shared-prefix KV reuse), transfer.py (cross-model KV map)
  engine/            model_runner.py (Qwen2/Qwen3 forward), engine.py (single-seq),
                     cuda_graph.py, triton_attn.py (fused paged-attention kernel)
  sched/             scheduler.py (continuous batching + priority admission)
  spec/              rejection.py (lossless verify), speculative.py (draft + target)
  transport/         base / cuda_ipc / tcp — the KV-transfer engine
  workers/           disaggregated.py — prefill & decode worker processes,
                     escalation.py — small → large model hand-off with mapped KV
  router/            app.py (FastAPI), server.py (scheduler thread), metrics.py,
                     dashboard.py (built-in UI), autoscale.py (draft-pool controller)
bench/               ablations, throughput, baseline_hf, dispec_bench, spec_bench,
                     disagg_bench, kvxfer_bench, load_gen, vllm_ref
dashboards/          dispec.json (Grafana)        monitoring/  Prometheus + Grafana config
tests/               59 tests
```

This is a learning/portfolio project, not a production server — the aim is an end-to-end
implementation where every number above is measured and accounted for.
