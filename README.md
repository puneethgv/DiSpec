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

## Roadmap

- **Phase 2** — speculative decoding: sequential rejection sampling → tree-based speculation,
  with provable losslessness.
- **Phase 3** — true P/D disaggregation: separate prefill/decode/draft processes + a KV
  transfer engine (CUDA IPC on-node, TCP for multi-node), SLO-aware routing.
- **Phase 4** — observability (Prometheus/Grafana), load testing, ablations, vLLM comparison.
- **Phase 5** — Triton kernels (paged + tree attention), EAGLE draft head, int4.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m dispec.env_check          # validate GPU / bf16 / triton
.venv/bin/python -m pytest -q                  # run tests
.venv/bin/python -m bench.baseline_hf          # HF baseline
.venv/bin/python -m bench.dispec_bench         # DiSpec engine
```

## Layout

```
dispec/
  config.py            model choices + generation/KV/spec configs
  sampling.py          greedy / top-k / top-p, built for spec-decode rejection math
  env_check.py         GPU / bf16 / triton validation
  kv/                  block_manager.py (paged allocator + COW), paged_cache.py (GPU pool)
  engine/              model_runner.py (from-scratch Qwen2 fwd), engine.py (single-seq)
  sched/               scheduler.py (continuous batching)
bench/                 baseline_hf.py, dispec_bench.py, common.py
tests/                 paged cache, runner correctness, scheduler
```
