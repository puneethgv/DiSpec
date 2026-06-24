"""vLLM reference throughput — the production SOTA baseline to position DiSpec against.

Standalone (no DiSpec imports) so it runs in an isolated venv where vLLM brings its own
torch: `uv venv .venv-vllm && uv pip install --python .venv-vllm/bin/python vllm`, then
`.venv-vllm/bin/python bench/vllm_ref.py`. Measures continuous-batching throughput on
the same model/prompts as bench.ablations.
"""

from __future__ import annotations

import time

PROMPTS = [
    "Explain how a transformer attention mechanism works, step by step.",
    "Write a short story about a lighthouse keeper who discovers a hidden door.",
    "What are the trade-offs between prefill/decode disaggregation and colocation?",
    "Summarize the causes of the French Revolution in three paragraphs.",
    "Implement binary search in Python and explain its time complexity.",
    "Describe how speculative decoding preserves the target output distribution.",
]


def main() -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=2048, enforce_eager=False)
    sp = SamplingParams(temperature=0.0, max_tokens=96)

    llm.generate(PROMPTS[:1], sp)  # warmup
    t0 = time.perf_counter()
    outs = llm.generate(PROMPTS, sp)
    dt = time.perf_counter() - t0
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"\nvLLM (Qwen2.5-1.5B, bf16, continuous batching): "
          f"{n / dt:.1f} tok/s  ({n} tokens, {dt:.2f}s, {len(PROMPTS)} prompts)")


if __name__ == "__main__":
    main()
