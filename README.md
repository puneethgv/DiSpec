# DiSpec — a from-scratch LLM inference engine

DiSpec is a from-scratch LLM serving engine that implements the internals of systems like
vLLM directly rather than calling them. It runs real models (Qwen2.5, Qwen3) on a single GPU
and covers the whole path: a paged KV cache with prefix sharing, a continuous-batching
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
  masking over the paged cache, expressed as a causal bias so prefill runs on FlashAttention. Checked against HuggingFace at
  the logits level (tests require per-position cosine ≥ 0.999 and matching argmax).
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

70 tests cover all of it (cache, prefix cache, forward correctness, batching, rejection
math, spec decoding, transports, disaggregation, CUDA graphs, the HTTP server, the
autoscaler, the Qwen3 forward, prefill attention, cross-model KV transfer).

## Numbers

Measured on one NVIDIA L4 on Modal (8 vCPUs), bf16, greedy decoding, torch 2.14, transformers
5.16.1, vLLM 0.11. Every table up to the cross-model KV transfer section comes from a single run
in one container, using the scripts named. HuggingFace `generate` is warmed up before it is
timed, like every DiSpec configuration.

**Single sequence** — Qwen2.5-1.5B, the six `BENCH_PROMPTS`, 96 new tokens, mean of two runs
(`bench/ablations.py`):

| config | tok/s | vs HF |
|---|---|---|
| HuggingFace `generate` | 27.4 | 1.00× |
| DiSpec single-sequence, eager | 24.5 | 0.90× |
| DiSpec single-sequence + CUDA graph | 62.6 | 2.29× |

Eager single-stream is slower than HF — it is a Python loop over 28 layers, and per-step kernel
launch overhead dominates. Capturing the decode step as a CUDA graph replays it as a single
launch: 2.6× over eager, and past HF. On Qwen3-1.7B the same path runs at 55.0 tok/s against
HF's 21.1 (2.60×).

**Prefill** — time to first token on long prompts, Qwen2.5-1.5B (`bench/prefill_bench.py`):

| tokens | HuggingFace | DiSpec | DiSpec / HF |
|---|---|---|---|
| 512 | 39.3 ms | 39.0 ms | 0.99× |
| 1024 | 75.6 ms | 78.0 ms | 1.03× |
| 2048 | 152.5 ms | 150.8 ms | 0.99× |
| 4096 | 303.1 ms | 305.9 ms | 1.01× |
| 8192 | 726.2 ms | 711.1 ms | 0.98× |

Prefill attention passes its right-aligned causal mask to SDPA as a `causal_lower_right` bias,
with GQA handled inside the kernel, so it dispatches to FlashAttention. With the materialized
boolean mask it used before, the same benchmark measured DiSpec at 1.14–1.59× HF's prefill time
(1136 ms at 8192 tokens). Qwen3-1.7B is also at parity (0.98–1.01×).

**Continuous batching** — Qwen2.5-1.5B on the serving fast path (Triton decode kernel + fused
GEMMs) against vLLM, on the same chat-templated prompts at the same concurrency, 96 new tokens,
median of three runs (`bench/throughput.py`, `bench/vllm_ref.py`). vLLM's prefix caching is off,
since DiSpec's batching engine has none and the replicated prompts would let vLLM skip prefill.

| concurrency | DiSpec | vLLM | vLLM / DiSpec |
|---|---|---|---|
| 6 | 194 tok/s | 395 tok/s | 2.0× |
| 16 | 489 tok/s | 994 tok/s | 2.0× |
| 32 | 864 tok/s | 1758 tok/s | 2.0× |

vLLM ran before and after DiSpec in the same container and moved by at most 0.2%. DiSpec's
batched decode is more sensitive to the host: the same code measured 279 / 681 / 1185 tok/s in a
different L4 container, a 1.4× gap against vLLM's numbers there, while vLLM stayed within ~2% of
the figures above. Expect a 1.4–2.0× gap depending on the machine.

What gets DiSpec this far: building the per-step attention slot table once instead of per layer,
a batched Triton decode kernel (one launch for the whole batch rather than a Python loop over
sequences), and fused QKV and gate/up GEMMs. vLLM additionally captures its batched decode as
CUDA graphs; DiSpec's batched step runs eagerly.

**Speculative decoding** is lossless, but here it is slower than decoding without it:
Qwen2.5-1.5B with the 0.5B draft and K=5 runs at 13.2 tok/s (0.48× HF), accepting 49% of drafted
tokens; Qwen3-1.7B with a 0.6B draft gives 10.1 tok/s (0.48×). `bench/spec_profile.py` shows why:

| forward (Qwen2.5-1.5B target / 0.5B draft) | time |
|---|---|
| target, 1 token | 41.3 ms |
| target, 5 tokens (verify) | 42.4 ms |
| draft, 1 token, eager | 35.0 ms |
| draft, 1 token, CUDA graph | 6.5 ms |

Verifying five tokens costs about the same as decoding one, so the target side works as
intended. The draft is the problem: an eager 0.5B forward costs almost as much as the 1.5B
target's, because both are dominated by per-layer launch overhead rather than compute. Each
iteration runs a verify and a correction forward on the target plus five draft forwards — about
259 ms for 3.35 tokens, 12.9 tok/s, in line with the measured end-to-end rate. Replaying the draft
as a CUDA graph (6.5 ms) would bring an iteration to ~116 ms, ~29 tok/s: faster than eager target
decode (24 tok/s) but under half of what CUDA-graph decode gives the target alone (62.6 tok/s).
Speculation needs the target's verify and correction forwards captured as graphs too.

**KV transfer** for disaggregation, Qwen2.5-7B KV geometry, TCP over localhost
(`bench/disagg_bench.py`, one transfer per size):

| prompt length | KV size | TCP transfer | throughput |
|---|---|---|---|
| 128 | 7.3 MB | 45.8 ms | 0.16 GB/s |
| 512 | 29.4 MB | 112.9 ms | 0.26 GB/s |
| 1024 | 58.7 MB | 168.7 ms | 0.35 GB/s |
| 2048 | 117.4 MB | 381.3 ms | 0.31 GB/s |

TCP is serialize-and-copy bound and its cost grows with the payload — it is the multi-node
fallback. CUDA IPC moves the same KV by passing the GPU buffer's handle instead of the bytes, so
nothing is copied between processes. The disaggregated output is token-for-token identical to
running it all in one process (tested), which is the thing that actually has to be true.

**Cross-model KV transfer** — escalating Qwen3-1.7B → Qwen3-4B on one L4, bf16, on the serving
fast path (Triton decode + fused GEMMs; `bench/kvxfer_bench.py`). Escalation avoids the 4B model
prefilling the prompt. *warm* is what it costs instead when the 1.7B model has already served the
prompt: map its KV (export + RoPE strip/re-apply + per-layer map) and inject it (import + one 4B
forward for the last prompt token). *cold* adds the 1.7B prefill. Speedups are against vLLM's
prefill on the same GPU type (vLLM 0.11, prefix caching off, measured in kvxfer).

| tokens | vLLM 4B prefill | DiSpec 4B prefill | warm | cold | cold, 1.7B prefill on vLLM |
|---|---|---|---|---|---|
| 512 | 106.1 ms | 105.8 ms | 71.0 ms (1.49×) | 119.0 ms (0.89×) | 113.6 ms (0.93×) |
| 1024 | 175.1 ms | 187.5 ms | 76.4 ms (2.29×) | 157.1 ms (1.11×) | 149.7 ms (1.17×) |
| 2048 | 332.9 ms | 374.0 ms | 106.4 ms (3.13×) | 265.5 ms (1.25×) | 254.7 ms (1.31×) |
| 4096 | 723.9 ms | 874.4 ms | 173.9 ms (4.16×) | 532.2 ms (1.36×) | 490.9 ms (1.47×) |
| 8192 | 1653.0 ms | 2000.5 ms | 347.2 ms (4.76×) | 1199.2 ms (1.38×) | 1013.1 ms (1.63×) |

These are the ridge maps; the trained residual adds 5–58 ms (warm 1.35–4.08×, cold 0.84–1.31×).
Below ~1k tokens the ~50 ms one-token inject dominates and the gain is small. Cold beats vLLM from
1k tokens up even with the 1.7B prefill on DiSpec, and reaches 1.63× when that prefill runs on
vLLM. It cannot go much higher for this pair: the 1.7B prefill alone is 40–45% of the 4B one, so
even a free transfer would cap cold at ~2.4×.

Cold only overtook vLLM after one engine change. Prefill attention used an explicit causal mask,
which kept SDPA off its FlashAttention kernel; expressed as a `causal_lower_right` bias instead,
DiSpec's 4B prefill at 8192 tokens dropped from 3502 ms to 2000 ms, and cold from 0.89× to 1.38×
of vLLM. Decode throughput is unchanged (the all-decode step uses the Triton kernel).

The two models together are ~11.5 GB in bf16, and the maps add ~0.8 GB. Only models with
identical KV heads, head dim
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
.venv/bin/python -m pytest -q            # 70 tests (CUDA ones skip without a GPU)

# benchmarks
.venv/bin/python -m bench.ablations      # single-sequence table
.venv/bin/python -m bench.prefill_bench  # prefill vs HuggingFace
.venv/bin/python -m bench.throughput     # continuous batching (pair with bench/vllm_ref.py)
.venv/bin/python -m bench.spec_profile   # why speculative decoding is below 1x
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
bench/               ablations, prefill_bench, throughput, baseline_hf, dispec_bench,
                     spec_bench, spec_profile, disagg_bench, kvxfer_bench, load_gen, vllm_ref
dashboards/          dispec.json (Grafana)        monitoring/  Prometheus + Grafana config
tests/               70 tests
```

This is a learning/portfolio project, not a production server — the aim is an end-to-end
implementation where every number above is measured and accounted for.
