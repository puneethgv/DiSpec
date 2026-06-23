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
from dispec.engine.model_runner import ModelRunner, SeqMeta
from dispec.kv.paged_cache import PagedKVCache
from dispec.sampling import SamplingParams, sample


class LLMEngine:
    _seq_counter = itertools.count()

    def __init__(self, model, kv: KVConfig | None = None, num_blocks: int | None = None):
        self.runner = ModelRunner(model)
        kv = kv or KVConfig()
        if num_blocks is None:
            num_blocks = kv.max_blocks
        self.cache = PagedKVCache.for_model(
            model.config, num_blocks=num_blocks, block_size=kv.block_size,
            dtype=self.runner.dtype, device=str(self.runner.device),
        )
        self.block_size = kv.block_size

    @torch.inference_mode()
    def generate(self, prompt_ids: list[int], max_new_tokens: int, params: SamplingParams,
                 eos_token_id: int | None = None, generator: torch.Generator | None = None) -> list[int]:
        dev = self.runner.device
        mgr = self.cache.manager
        seq_id = next(self._seq_counter)

        n = len(prompt_ids)
        table = mgr.allocate(seq_id, n)
        try:
            ids = torch.tensor(prompt_ids, device=dev, dtype=torch.long)
            positions = torch.arange(n, device=dev)
            write_slots = self.cache.context_slots(table.block_ids, n)
            hidden = self.runner.forward(
                ids, positions, write_slots,
                [SeqMeta(table.block_ids, q_len=n, ctx_len=n)], self.cache,
            )
            table.num_tokens = n
            logits = self.runner.logits(hidden[-1:])  # (1, vocab)
            next_id = int(sample(logits, params, generator)[0])

            out = [next_id]
            cur = n
            while len(out) < max_new_tokens and next_id != eos_token_id:
                blk, off = mgr.append_token(seq_id)
                slot = torch.tensor([blk * self.block_size + off], device=dev)
                ids = torch.tensor([next_id], device=dev, dtype=torch.long)
                positions = torch.tensor([cur], device=dev)
                hidden = self.runner.forward(
                    ids, positions, slot,
                    [SeqMeta(table.block_ids, q_len=1, ctx_len=cur + 1)], self.cache,
                )
                logits = self.runner.logits(hidden[-1:])
                next_id = int(sample(logits, params, generator)[0])
                out.append(next_id)
                cur += 1
            return out
        finally:
            mgr.free(seq_id)
