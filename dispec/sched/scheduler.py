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
    PREFILLING = auto()  # prompt partially prefilled (chunked prefill)
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
    prefill_pos: int = 0  # prompt tokens prefilled so far (chunked prefill)
    # Contiguous (layers, n, kv_heads, head_dim) KV for the first n prompt tokens,
    # computed elsewhere -- e.g. mapped from a smaller model (dispec/kv/transfer.py).
    # Imported at admission; prefill then resumes at token n. Cleared once imported.
    prefix_kv: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def done(self) -> bool:
        if len(self.output_ids) >= self.max_new_tokens:
            return True
        return bool(self.output_ids) and self.output_ids[-1] == self.eos_id


class ContinuousBatchingEngine:
    def __init__(self, model, kv: KVConfig | None = None, num_blocks: int = 2048,
                 max_batch_tokens: int = 2048, attn_backend: str = "native", fuse: bool = False,
                 chunk_size: int | None = None):
        self.runner = ModelRunner(model, attn_backend=attn_backend, fuse=fuse)
        kv = kv or KVConfig()
        self.block_size = kv.block_size
        self.max_batch_tokens = max_batch_tokens
        # Chunked prefill: cap prefill tokens processed per step so a long prompt
        # co-batches with decodes instead of stalling them. Defaults to no chunking.
        self.chunk_size = chunk_size or max_batch_tokens
        self.cache = PagedKVCache.for_model(
            model.config, num_blocks=num_blocks, block_size=kv.block_size,
            dtype=self.runner.dtype, device=str(self.runner.device),
        )
        self.waiting: list[Request] = []
        self.prefilling: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self._ids = itertools.count()
        # Metrics
        self.num_steps = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0

    def add_request(self, prompt_ids: list[int], max_new_tokens: int,
                    params: SamplingParams | None = None, eos_id: int | None = None,
                    priority: int = 0,
                    prefix_kv: tuple[torch.Tensor, torch.Tensor] | None = None) -> int:
        """Queue a request.

        `prefix_kv` supplies KV for the first n prompt tokens so they are never
        prefilled. It must cover fewer tokens than the prompt: the logits that pick the
        first output token come from forwarding the last prompt token, so at least that
        one has to go through the model.
        """
        if prefix_kv is not None:
            self._check_prefix_kv(prefix_kv, len(prompt_ids))
        req = Request(next(self._ids), list(prompt_ids), params or SamplingParams(),
                      max_new_tokens, eos_id, priority=priority, prefix_kv=prefix_kv)
        self.waiting.append(req)
        return req.id

    def _check_prefix_kv(self, prefix_kv, prompt_len: int) -> None:
        k, v = prefix_kv
        c = self.cache
        expected = (c.num_layers, k.shape[1], c.num_kv_heads, c.head_dim)
        if tuple(k.shape) != expected or tuple(v.shape) != expected:
            raise ValueError(f"prefix_kv shapes {tuple(k.shape)}/{tuple(v.shape)} do not "
                             f"match this model's cache layout {expected}")
        if not 0 < k.shape[1] < prompt_len:
            raise ValueError(f"prefix_kv covers {k.shape[1]} tokens of a {prompt_len}-token "
                             "prompt; it must cover at least one and fewer than all of them")

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

        # 2) Prefill — continue in-progress chunked prefills, then admit new (highest
        #    priority, then oldest). Each request advances at most chunk_size tokens this
        #    step; only the chunk that *finishes* the prompt yields a token to sample.
        sampled = [True] * len(batch)  # decode requests above all produce a token
        self.waiting.sort(key=lambda r: (-r.priority, r.id))
        prefill_queue = self.prefilling + self.waiting
        for req in prefill_queue:
            if budget < 1:
                break
            pl = len(req.prompt_ids)
            if req.state == State.WAITING:
                try:
                    table = mgr.allocate(req.id, pl)
                except OutOfBlocksError:
                    break
                table.num_tokens = 0
                if req.prefix_kv is not None:
                    k, v = req.prefix_kv
                    dev_kv = self.cache.key.device
                    self.cache.import_contiguous(table.block_ids, k.to(dev_kv), v.to(dev_kv))
                    req.prefill_pos = table.num_tokens = k.shape[1]
                    req.prefix_kv = None  # the paged cache owns it now
                req.state = State.PREFILLING
                self.waiting.remove(req)
                self.prefilling.append(req)
            table = mgr.block_table(req.id)
            chunk = min(pl - req.prefill_pos, self.chunk_size, budget)
            pos = list(range(req.prefill_pos, req.prefill_pos + chunk))
            slots = self.cache.slots_for_positions(
                table.block_ids, torch.tensor(pos, device=dev)).tolist()
            token_ids.extend(req.prompt_ids[req.prefill_pos:req.prefill_pos + chunk])
            positions.extend(pos)
            write_slots.extend(slots)
            seq_meta.append(SeqMeta(table.block_ids, chunk, req.prefill_pos + chunk))
            cursor += chunk
            req.prefill_pos += chunk
            table.num_tokens = req.prefill_pos
            budget -= chunk
            self.num_prefill_tokens += chunk
            batch.append(req)
            if req.prefill_pos >= pl:  # prompt fully prefilled -> sample first token
                last_idx.append(cursor - 1)
                sampled.append(True)
                req.prefilled = True
                req.state = State.RUNNING
                self.prefilling.remove(req)
                self.running.append(req)
            else:
                sampled.append(False)  # still prefilling, no token this step

        if not batch:
            return

        # 3) Single forward over the mixed prefill+decode batch.
        hidden = self.runner.forward(
            torch.tensor(token_ids, device=dev, dtype=torch.long),
            torch.tensor(positions, device=dev, dtype=torch.long),
            torch.tensor(write_slots, device=dev, dtype=torch.long),
            seq_meta, self.cache,
        )
        self.num_steps += 1
        if not last_idx:  # all entries were mid-prefill chunks; nothing to sample
            return

        idx = torch.tensor(last_idx, device=dev)
        logits = self.runner.logits(hidden[idx])  # (n_sampling, vocab)

        # 4) Sample one token per request that produced one (decode + finished prefill).
        sample_batch = [r for r, s in zip(batch, sampled) if s]
        next_ids = sample(logits, sample_batch[0].params).tolist()

        # 5) Append outputs and retire finished. KV length is tracked by the block
        # table (updated on the next decode step), so we don't advance it here.
        finished = []
        for req, tid in zip(sample_batch, next_ids):
            req.output_ids.append(int(tid))
            req.cur_len = mgr.block_table(req.id).num_tokens
            if req.done:
                req.state = State.FINISHED
                finished.append(req)
        for req in finished:
            self.running.remove(req)
            self.finished.append(req)
            mgr.free(req.id)

    def collect_outputs(self) -> dict[int, list[int]]:
        return {r.id: r.output_ids for r in self.finished + self.running}

    def has_work(self) -> bool:
        return bool(self.waiting or self.prefilling or self.running)

    def run_until_done(self, max_steps: int = 100_000) -> None:
        steps = 0
        while self.has_work() and steps < max_steps:
            self.step()
            steps += 1
