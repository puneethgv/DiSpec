"""Continuous-batching engine (iteration-level scheduling).

Unlike static batching (which waits for the whole batch to finish), this admits and
retires requests every iteration, mixing prefill and decode tokens into a single
forward pass. This keeps the GPU busy and is the foundation real serving systems
(vLLM, TGI) are built on. The paged KV cache is what makes ragged batching possible
without padding waste.

Token budget (`max_batch_tokens`) bounds the tokens per forward so a burst of long
prompts can't OOM; excess prompts wait. Admission also requires enough free KV blocks.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto

import torch

from dispec.config import KVConfig
from dispec.engine.model_runner import ModelRunner, SeqMeta
from dispec.kv.block_manager import OutOfBlocksError
from dispec.kv.paged_cache import PagedKVCache
from dispec.sampling import SamplingParams, sample


class State(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Request:
    id: int
    prompt_ids: list[int]
    params: SamplingParams
    max_new_tokens: int
    eos_id: int | None = None
    output_ids: list[int] = field(default_factory=list)
    state: State = State.WAITING
    cur_len: int = 0  # context length stored in cache
    prefilled: bool = False
    priority: int = 0  # higher = admitted sooner (SLO-aware routing)

    @property
    def done(self) -> bool:
        if len(self.output_ids) >= self.max_new_tokens:
            return True
        return bool(self.output_ids) and self.output_ids[-1] == self.eos_id


class ContinuousBatchingEngine:
    def __init__(self, model, kv: KVConfig | None = None, num_blocks: int = 2048,
                 max_batch_tokens: int = 2048, attn_backend: str = "native"):
        self.runner = ModelRunner(model, attn_backend=attn_backend)
        kv = kv or KVConfig()
        self.block_size = kv.block_size
        self.max_batch_tokens = max_batch_tokens
        self.cache = PagedKVCache.for_model(
            model.config, num_blocks=num_blocks, block_size=kv.block_size,
            dtype=self.runner.dtype, device=str(self.runner.device),
        )
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self._ids = itertools.count()
        # Metrics
        self.num_steps = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0

    def add_request(self, prompt_ids: list[int], max_new_tokens: int,
                    params: SamplingParams | None = None, eos_id: int | None = None,
                    priority: int = 0) -> int:
        req = Request(next(self._ids), list(prompt_ids), params or SamplingParams(),
                      max_new_tokens, eos_id, priority=priority)
        self.waiting.append(req)
        return req.id

    # -- one scheduling iteration -------------------------------------------
    @torch.inference_mode()
    def step(self) -> None:
        dev = self.runner.device
        mgr = self.cache.manager

        batch: list[Request] = []
        token_ids: list[int] = []
        positions: list[int] = []
        write_slots: list[int] = []
        seq_meta: list[SeqMeta] = []
        last_idx: list[int] = []  # flat index of each seq's last token
        budget = self.max_batch_tokens
        cursor = 0

        # 1) Continue all running (decode) sequences first.
        # The block table's num_tokens is the single source of truth for how much
        # KV is stored; the new token's position is the index it will occupy.
        for req in self.running:
            if budget < 1:
                break
            table = mgr.block_table(req.id)
            pos = table.num_tokens  # index of the token we're about to store
            blk, off = mgr.append_token(req.id)  # num_tokens -> pos + 1
            slot = blk * self.block_size + off
            token_ids.append(req.output_ids[-1])
            positions.append(pos)
            write_slots.append(slot)
            seq_meta.append(SeqMeta(table.block_ids, 1, table.num_tokens))
            cursor += 1
            last_idx.append(cursor - 1)
            batch.append(req)
            budget -= 1
            self.num_decode_tokens += 1

        # 2) Admit waiting requests (prefill), highest priority (then oldest) first.
        self.waiting.sort(key=lambda r: (-r.priority, r.id))
        while self.waiting and budget >= len(self.waiting[0].prompt_ids):
            req = self.waiting[0]
            pl = len(req.prompt_ids)
            try:
                table = mgr.allocate(req.id, pl)
            except OutOfBlocksError:
                break
            self.waiting.pop(0)
            slots = self.cache.context_slots(table.block_ids, pl).tolist()
            token_ids.extend(req.prompt_ids)
            positions.extend(range(pl))
            write_slots.extend(slots)
            seq_meta.append(SeqMeta(table.block_ids, pl, pl))
            cursor += pl
            last_idx.append(cursor - 1)
            table.num_tokens = pl
            req.cur_len = pl
            req.prefilled = True
            req.state = State.RUNNING
            self.running.append(req)
            batch.append(req)
            budget -= pl
            self.num_prefill_tokens += pl

        if not batch:
            return

        # 3) Single forward over the mixed prefill+decode batch.
        hidden = self.runner.forward(
            torch.tensor(token_ids, device=dev, dtype=torch.long),
            torch.tensor(positions, device=dev, dtype=torch.long),
            torch.tensor(write_slots, device=dev, dtype=torch.long),
            seq_meta, self.cache,
        )
        idx = torch.tensor(last_idx, device=dev)
        logits = self.runner.logits(hidden[idx])  # (batch, vocab)

        # 4) Sample one token per sequence (greedy path batches cleanly).
        params0 = batch[0].params
        next_ids = sample(logits, params0).tolist()

        # 5) Append outputs and retire finished. KV length is tracked by the block
        # table (updated on the next decode step), so we don't advance it here.
        finished = []
        for req, tid in zip(batch, next_ids):
            req.output_ids.append(int(tid))
            req.cur_len = mgr.block_table(req.id).num_tokens
            if req.done:
                req.state = State.FINISHED
                finished.append(req)
        for req in finished:
            self.running.remove(req)
            self.finished.append(req)
            mgr.free(req.id)
        self.num_steps += 1

    def collect_outputs(self) -> dict[int, list[int]]:
        return {r.id: r.output_ids for r in self.finished + self.running}

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def run_until_done(self, max_steps: int = 100_000) -> None:
        steps = 0
        while self.has_work() and steps < max_steps:
            self.step()
            steps += 1
