"""Phase-0 baseline: plain HuggingFace `generate` on the target model.

This is the number every later DiSpec phase must beat. Measures TTFT, TPOT, and
single-stream decode throughput with a streamer so TTFT is real.

Run: python -m bench.baseline_hf  [--max-new 128] [--n 6]
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import TextIteratorStreamer

from bench.common import BenchResult, RequestMetrics
from dispec.config import BENCH_PROMPTS, GenConfig
from dispec.models.loader import build_prompt, load_target


def run_one(model, tokenizer, prompt: str, gen: GenConfig) -> RequestMetrics:
    import threading

    text = build_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    n_prompt = inputs.input_ids.shape[1]

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(
        **inputs,
        max_new_tokens=gen.max_new_tokens,
        do_sample=not gen.greedy,
        temperature=gen.temperature if not gen.greedy else None,
        top_p=gen.top_p if not gen.greedy else None,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
    )

    t0 = time.perf_counter()
    thread = threading.Thread(target=model.generate, kwargs=kwargs)
    thread.start()

    ttft = None
    n_out = 0
    for _ in streamer:
        if ttft is None:
            ttft = time.perf_counter() - t0
        n_out += 1
    thread.join()
    total = time.perf_counter() - t0

    return RequestMetrics(
        ttft_s=ttft or total,
        total_s=total,
        num_prompt_tokens=n_prompt,
        num_output_tokens=max(n_out, 1),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--n", type=int, default=len(BENCH_PROMPTS))
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    gen = GenConfig(max_new_tokens=args.max_new, temperature=args.temperature)
    torch.manual_seed(gen.seed)

    print("Loading target model...")
    model, tokenizer = load_target()
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM after load: {(total - free) / 1e9:.2f} GB used, {free / 1e9:.2f} GB free")

    result = BenchResult(name="baseline_hf_generate")
    prompts = (BENCH_PROMPTS * ((args.n // len(BENCH_PROMPTS)) + 1))[: args.n]

    # Warmup (kernel autotune / cudnn) so timings are steady-state.
    print("Warmup...")
    run_one(model, tokenizer, prompts[0], GenConfig(max_new_tokens=16, temperature=args.temperature))

    print("Benchmarking...")
    for i, p in enumerate(prompts):
        m = run_one(model, tokenizer, p, gen)
        result.add(m)
        print(f"  [{i + 1}/{len(prompts)}] ttft={m.ttft_s * 1000:.0f}ms "
              f"tpot={m.tpot_s * 1000:.1f}ms out={m.num_output_tokens}")

    print()
    result.print_summary()


if __name__ == "__main__":
    main()
