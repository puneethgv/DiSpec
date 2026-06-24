"""Single-sequence generation engine (Phase 1 correctness harness).

Drives prefill + step-by-step decode through ModelRunner over the paged cache, with
our own sampling. Continuous batching of many sequences lives in the scheduler
(dispec/sched); this module keeps the happy path simple so we can prove the
from-scratch forward matches HuggingFace token-for-token under greedy decoding.
"""

from __future__ import annotations

import itertools

import torch

from dispec.config import KVConfig
from dispec.engine.cuda_graph import DEFAULT_BUCKETS, CudaGraphDecoder
from dispec.engine.model_runner import ModelRunner, SeqMeta
from dispec.kv.paged_cache import PagedKVCache
from dispec.kv.prefix_cache import PrefixCache
from dispec.sampling import SamplingParams, sample


class LLMEngine:
    _seq_counter = itertools.count()

    def __init__(self, model, kv: KVConfig | None = None, num_blocks: int | None = None,
                 cuda_graph: bool = False, graph_buckets: tuple[int, ...] = DEFAULT_BUCKETS,
                 prefix_cache: bool = False, attn_backend: str = "native"):
        self.runner = ModelRunner(model, attn_backend=attn_backend)
        kv = kv or KVConfig()
        if num_blocks is None:
            num_blocks = kv.max_blocks
        self.cache = PagedKVCache.for_model(
            model.config, num_blocks=num_blocks, block_size=kv.block_size,
            dtype=self.runner.dtype, device=str(self.runner.device),
        )
        self.block_size = kv.block_size
        self.prefix_cache = PrefixCache(self.cache.manager, kv.block_size) if prefix_cache else None
        self.last_prefill_tokens = 0  # tokens actually prefilled on the last generate()
        # Capture decode graphs while the cache is empty (replayed per decode step).
        self.graph = None
        if cuda_graph:
            self.graph = CudaGraphDecoder(self.runner, self.cache, graph_buckets)
            self.graph.capture()

    @torch.inference_mode()
    def generate(self, prompt_ids: list[int], max_new_tokens: int, params: SamplingParams,
                 eos_token_id: int | None = None, generator: torch.Generator | None = None) -> list[int]:
        dev = self.runner.device
        mgr = self.cache.manager
        seq_id = next(self._seq_counter)

        n = len(prompt_ids)
        # Prefix cache: adopt the longest run of cached blocks and prefill only the rest.
        reuse: list[int] = []
        if self.prefix_cache is not None:
            reuse = self.prefix_cache.match(prompt_ids)
            # Keep at least the final token to prefill (we need its next-token logits).
            while reuse and len(reuse) * self.block_size >= n:
                reuse.pop()
        start = len(reuse) * self.block_size  # tokens already cached
        table = mgr.allocate(seq_id, n, reuse_blocks=reuse)
        try:
            q = n - start
            self.last_prefill_tokens = q
            ids = torch.tensor(prompt_ids[start:], device=dev, dtype=torch.long)
            positions = torch.arange(start, n, device=dev)
            write_slots = self.cache.slots_for_positions(table.block_ids, positions)
            hidden = self.runner.forward(
                ids, positions, write_slots,
                [SeqMeta(table.block_ids, q_len=q, ctx_len=n)], self.cache,
            )
            table.num_tokens = n
            if self.prefix_cache is not None:
                self.prefix_cache.insert(prompt_ids, table.block_ids)
            logits = self.runner.logits(hidden[-1:])  # (1, vocab)
            next_id = int(sample(logits, params, generator)[0])

            out = [next_id]
            cur = n
            while len(out) < max_new_tokens and next_id != eos_token_id:
                blk, off = mgr.append_token(seq_id)
                wslot = blk * self.block_size + off
                if self.graph is not None and self.graph.can_handle(cur + 1):
                    # CUDA-graph fast path: replay the captured decode (~3x faster).
                    cslots = self.cache.context_slots(table.block_ids, cur + 1)
                    logits = self.graph.decode(next_id, cur, wslot, cslots)
                else:
                    hidden = self.runner.forward(
                        torch.tensor([next_id], device=dev, dtype=torch.long),
                        torch.tensor([cur], device=dev),
                        torch.tensor([wslot], device=dev),
                        [SeqMeta(table.block_ids, q_len=1, ctx_len=cur + 1)], self.cache,
                    )
                    logits = self.runner.logits(hidden[-1:])
                next_id = int(sample(logits, params, generator)[0])
                out.append(next_id)
                cur += 1
            return out
        finally:
            mgr.free(seq_id)
