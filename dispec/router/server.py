"""Async inference server core: a background scheduler thread driving the engine.

A FastAPI handler is async, but the continuous-batching `step()` is a blocking GPU
call. So the engine is owned by a single background thread that loops `step()`; async
handlers submit requests and await an `asyncio.Future` that the scheduler thread
resolves (thread-safely) when the request finishes. A lock serializes engine mutation
(submit vs step). This mirrors how real servers separate the API from the run loop.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field

from dispec.config import KVConfig
from dispec.router import metrics
from dispec.router.autoscale import DraftPoolAutoscaler
from dispec.sampling import SamplingParams
from dispec.sched.scheduler import ContinuousBatchingEngine


@dataclass
class _Pending:
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future
    t_submit: float
    t_first: float = 0.0  # timestamp of first output token (0 = not yet)
    queue: "asyncio.Queue | None" = None  # set for streaming requests
    decoded: str = ""  # text streamed so far (for incremental delta decoding)


class InferenceServer:
    def __init__(self, model, tokenizer, num_blocks: int = 2048,
                 max_batch_tokens: int = 2048, kv: KVConfig | None = None,
                 attn_backend: str = "native", fuse: bool = False,
                 chunk_size: int | None = None):
        self.engine = ContinuousBatchingEngine(
            model, kv=kv, num_blocks=num_blocks, max_batch_tokens=max_batch_tokens,
            attn_backend=attn_backend, fuse=fuse, chunk_size=chunk_size)
        self.tok = tokenizer
        self.autoscaler = DraftPoolAutoscaler()
        self._lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        self._thread.join(timeout=5)

    async def generate(self, prompt: str, max_new_tokens: int = 128,
                       temperature: float = 0.0, priority: int = 0) -> dict:
        ids = self.tok(prompt).input_ids
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        with self._lock:
            rid = self.engine.add_request(ids, max_new_tokens,
                                          SamplingParams(temperature),
                                          eos_id=self.tok.eos_token_id, priority=priority)
            self._pending[rid] = _Pending(loop, fut, time.perf_counter())
        out_ids = await fut
        return {"text": self.tok.decode(out_ids, skip_special_tokens=True),
                "num_tokens": len(out_ids)}

    async def generate_stream(self, prompt: str, max_new_tokens: int = 128,
                              temperature: float = 0.0, priority: int = 0):
        """Yield text deltas as they're produced (for SSE streaming)."""
        ids = self.tok(prompt).input_ids
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            rid = self.engine.add_request(ids, max_new_tokens, SamplingParams(temperature),
                                          eos_id=self.tok.eos_token_id, priority=priority)
            self._pending[rid] = _Pending(loop, loop.create_future(),
                                          time.perf_counter(), queue=q)
        while True:
            delta = await q.get()
            if delta is None:  # end-of-stream sentinel
                break
            yield delta

    # -- background scheduler loop ------------------------------------------
    def _run(self) -> None:
        while not self._stop:
            with self._lock:
                if not self.engine.has_work():
                    idle = True
                else:
                    idle = False
                    before = self.engine.num_prefill_tokens + self.engine.num_decode_tokens
                    t0 = time.perf_counter()
                    self.engine.step()
                    metrics.STEP.observe(time.perf_counter() - t0)
                    after = self.engine.num_prefill_tokens + self.engine.num_decode_tokens
                    metrics.BATCH.observe(max(after - before, 1))
                    metrics.RUNNING.set(len(self.engine.running))
                    metrics.WAITING.set(len(self.engine.waiting))
                    metrics.DRAFT_REPLICAS.set(
                        self.autoscaler.observe(len(self.engine.running)))
                    self._reap()
            if idle:
                time.sleep(0.003)

    def _reap(self) -> None:
        """Record TTFT and resolve futures for finished requests (holds the lock)."""
        now = time.perf_counter()
        by_id = {r.id: r for r in self.engine.running}
        for rid, p in self._pending.items():
            r = by_id.get(rid)
            if r is None or not r.output_ids:
                continue
            if not p.t_first:  # TTFT: first output token while still running
                p.t_first = now
                metrics.TTFT.observe(now - p.t_submit)
            if p.queue is not None:  # streaming: push the new text delta
                self._push_delta(p, r.output_ids)
        # Completion: drain engine.finished.
        if not self.engine.finished:
            return
        for r in self.engine.finished:
            p = self._pending.pop(r.id, None)
            if p is None:
                continue
            t_first = p.t_first or now  # finished same step it first emitted
            metrics.E2E.observe(now - p.t_submit)
            metrics.REQUESTS.inc()
            metrics.TOKENS.inc(len(r.output_ids))
            if len(r.output_ids) > 1:
                metrics.TPOT.observe((now - t_first) / (len(r.output_ids) - 1))
            if p.queue is not None:
                self._push_delta(p, r.output_ids)
                p.loop.call_soon_threadsafe(p.queue.put_nowait, None)  # end sentinel
            else:
                p.loop.call_soon_threadsafe(p.future.set_result, list(r.output_ids))
        self.engine.finished = []

    def _push_delta(self, p: _Pending, output_ids: list[int]) -> None:
        """Decode the full output and push the newly-appeared text suffix to the queue."""
        full = self.tok.decode(output_ids, skip_special_tokens=True)
        if len(full) > len(p.decoded):
            delta = full[len(p.decoded):]
            p.decoded = full
            p.loop.call_soon_threadsafe(p.queue.put_nowait, delta)
