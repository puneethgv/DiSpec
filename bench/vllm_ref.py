"""vLLM reference throughput, matched to DiSpec's bench/throughput.py.

Standalone (no DiSpec imports) so it runs in an isolated venv where vLLM brings its own
torch: `uv venv .venv-vllm && uv pip install --python .venv-vllm/bin/python vllm`, then
`.venv-vllm/bin/python bench/vllm_ref.py`.

Same model, same chat-templated BENCH_PROMPTS replicated to each concurrency, greedy,
96 new tokens, all requests submitted at once. Prefix caching is off: DiSpec's
continuous-batching engine has none, and the replicated prompts would otherwise let vLLM
skip their prefill. Three runs per concurrency, median reported.
"""
import json
import statistics
import time

from vllm import LLM, SamplingParams

PROMPTS = [
    "Explain how a transformer attention mechanism works, step by step.",
    "Write a short story about a lighthouse keeper who discovers a hidden door.",
    "What are the trade-offs between prefill/decode disaggregation and colocation?",
    "Summarize the causes of the French Revolution in three paragraphs.",
    "Implement binary search in Python and explain its time complexity.",
    "Describe how speculative decoding preserves the target output distribution.",
]
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", dtype="bfloat16", gpu_memory_utilization=0.85,
          max_model_len=2048, enable_prefix_caching=False)
tok = llm.get_tokenizer()
ids = [tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                   add_generation_prompt=True)).input_ids for p in PROMPTS]


def measure(c, max_new):
    reqs = [{"prompt_token_ids": r} for r in (ids * (c // len(ids) + 1))[:c]]
    t0 = time.perf_counter()
    outs = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=max_new), use_tqdm=False)
    dt = time.perf_counter() - t0
    return sum(len(o.outputs[0].token_ids) for o in outs) / dt


measure(4, 8); measure(6, 96)  # warmup
res = {}
for c in (6, 16, 32):
    runs = [measure(c, 96) for _ in range(3)]
    res[c] = {"runs": runs, "median": statistics.median(runs)}
    print(f"  vLLM concurrency={c:>3}: {res[c]['median']:.1f} tok/s  (runs {', '.join(f'{r:.1f}' for r in runs)})", flush=True)
print("RESULT_JSON " + json.dumps({"kind": "vllm_throughput", "results": res}))
