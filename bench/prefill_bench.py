"""Time-to-first-token cost: DiSpec prefill vs HuggingFace prefill on long prompts.

Both compute the full prompt forward, fill a KV cache, and project only the last
position through the LM head. HF runs the backbone then lm_head on the last token,
so it is not charged for logits it would discard. Median of repeats, synchronized.

Run: python -m bench.prefill_bench [--model Qwen/Qwen3-1.7B] [--lengths 512,1024,2048,4096,8192]
"""
import argparse
import json
import statistics
import time

import torch

from dispec.config import TARGET_MODEL
from dispec.engine.model_runner import ModelRunner, SeqMeta
from dispec.kv.paged_cache import PagedKVCache
from dispec.models.loader import load_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=TARGET_MODEL)
ap.add_argument("--lengths", default="512,1024,2048,4096,8192")
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--label", default="")
args = ap.parse_args()

lengths = [int(x) for x in args.lengths.split(",")]
model = load_model(args.model)
runner = ModelRunner(model)
cache = PagedKVCache.for_model(model.config, (max(lengths) + 15) // 16 + 8, 16, runner.dtype, "cuda")
vocab = min(model.config.vocab_size, 100_000)


def median_ms(fn):
    fn(); fn()
    ts = []
    for _ in range(args.repeats):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


@torch.inference_mode()
def dispec_prefill(ids):
    n = ids.shape[0]
    table = cache.manager.allocate(1, n)
    table.num_tokens = n
    h = runner.forward(ids, torch.arange(n, device="cuda"), cache.context_slots(table.block_ids, n),
                       [SeqMeta(table.block_ids, n, n)], cache)
    runner.logits(h[-1:])
    cache.manager.free(1)


@torch.inference_mode()
def hf_prefill(ids):
    out = model.model(input_ids=ids[None], use_cache=True)
    model.lm_head(out.last_hidden_state[:, -1:])


points = []
print(f"[{args.label}] {args.model} on {torch.cuda.get_device_name()}")
print(f"{'tokens':>7} {'HF':>9} {'DiSpec':>9} {'DiSpec/HF':>9}")
for n in lengths:
    ids = torch.randint(0, vocab, (n,), device="cuda", generator=torch.Generator(device="cuda").manual_seed(n))
    hf, ds = median_ms(lambda: hf_prefill(ids)), median_ms(lambda: dispec_prefill(ids))
    points.append({"n_tokens": n, "hf_ms": hf, "dispec_ms": ds})
    print(f"{n:>7} {hf:>7.1f}ms {ds:>7.1f}ms {ds / hf:>8.2f}x", flush=True)
print("RESULT_JSON " + json.dumps({"kind": "prefill", "label": args.label, "model": args.model,
                                   "gpu": torch.cuda.get_device_name(), "points": points}))
