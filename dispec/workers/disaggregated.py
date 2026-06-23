"""Prefill/decode (P/D) disaggregation across separate processes.

Two worker processes share one GPU:

  prefill worker  -- computes the prompt KV (and the first token) --\\
                                                                     >-- KV transfer --> decode worker -- generates tokens
  (compute-bound, bursty)                                                (CUDA IPC / TCP)   (bandwidth-bound, steady)

This is the real meaning of "disaggregation": prefill and decode run in different
address spaces (different nodes, in production) and the KV cache is *transferred*
between them via a pluggable transport (dispec.transport). Here both processes share a
single GPU and we use CUDA-IPC zero-copy transfer; the same code runs multi-node by
swapping in TcpTransport.

The split-out `prefill_export` / `decode_from_kv` helpers also serve as the colocated
baseline used to prove the disaggregated output is identical.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.multiprocessing as mp

from dispec.engine.model_runner import ModelRunner, SeqMeta
from dispec.kv.paged_cache import PagedKVCache
from dispec.sampling import SamplingParams, sample
from dispec.transport.base import KVPayload
from dispec.transport.cuda_ipc import CudaIpcTransport

BLOCK_SIZE = 16


# -- stage helpers (usable in-process for the colocated baseline) --------------
def prefill_export(runner: ModelRunner, cache: PagedKVCache, sid: int,
                   prompt_ids: list[int], params: SamplingParams):
    """Prefill a prompt; return (k, v contiguous KV, first decoded token)."""
    n = len(prompt_ids)
    table = cache.manager.allocate(sid, n)
    dev = runner.device
    slots = cache.context_slots(table.block_ids, n)
    table.num_tokens = n
    hidden = runner.forward(torch.tensor(prompt_ids, device=dev),
                            torch.arange(n, device=dev), slots,
                            [SeqMeta(table.block_ids, n, n)], cache)
    first = int(sample(runner.logits(hidden[-1:]), params)[0])
    k, v = cache.export_contiguous(table.block_ids, n)
    cache.manager.free(sid)
    return k, v, first


def decode_from_kv(runner: ModelRunner, cache: PagedKVCache, sid: int,
                   prompt_ids: list[int], k, v, first_token: int,
                   max_new: int, params: SamplingParams, eos: int | None):
    """Import transferred KV and generate from `first_token`."""
    n = len(prompt_ids)
    dev = runner.device
    table = cache.manager.allocate(sid, n)
    cache.import_contiguous(table.block_ids, k, v)
    table.num_tokens = n
    out = [first_token]
    nxt, cur = first_token, n
    while len(out) < max_new and nxt != eos:
        blk, off = cache.manager.append_token(sid)
        slot = torch.tensor([blk * cache.block_size + off], device=dev)
        hidden = runner.forward(torch.tensor([nxt], device=dev),
                                torch.tensor([cur], device=dev), slot,
                                [SeqMeta(table.block_ids, 1, cur + 1)], cache)
        nxt = int(sample(runner.logits(hidden[-1:]), params)[0])
        out.append(nxt)
        cur += 1
    cache.manager.free(sid)
    return out


# -- worker process mains ------------------------------------------------------
@dataclass
class DisaggConfig:
    model_name: str
    num_blocks: int = 512
    max_new_tokens: int = 64
    eos_id: int | None = None
    temperature: float = 0.0


def _build(model_name, num_blocks):
    from dispec.models.loader import load_model
    model = load_model(model_name)
    runner = ModelRunner(model)
    cache = PagedKVCache.for_model(model.config, num_blocks, BLOCK_SIZE,
                                   runner.dtype, str(runner.device))
    return runner, cache


def _prefill_main(cfg: DisaggConfig, req_q, kv_q, ready):
    runner, cache = _build(cfg.model_name, cfg.num_blocks)
    transport = CudaIpcTransport(kv_q)
    params = SamplingParams(cfg.temperature)
    ready.set()
    sid = 0
    while True:
        item = req_q.get()
        if item is None:
            break
        rid, prompt_ids = item
        sid += 1
        k, v, first = prefill_export(runner, cache, sid, prompt_ids, params)
        transport.send(KVPayload(prompt_ids, k, v, meta={"rid": rid, "first_token": first}))


def _decode_main(cfg: DisaggConfig, kv_q, res_q, ready):
    runner, cache = _build(cfg.model_name, cfg.num_blocks)
    transport = CudaIpcTransport(kv_q)
    params = SamplingParams(cfg.temperature)
    ready.set()
    sid = 0
    while True:
        payload = transport.recv()
        if payload is None:
            break
        sid += 1
        t0 = time.perf_counter()
        out = decode_from_kv(runner, cache, sid, payload.tokens, payload.k, payload.v,
                             payload.meta["first_token"], cfg.max_new_tokens, params, cfg.eos_id)
        res_q.put({"rid": payload.meta["rid"], "out": out,
                   "decode_s": time.perf_counter() - t0, "kv_bytes": payload.nbytes()})


class DisaggregatedEngine:
    """Spawns prefill + decode worker processes wired by a CUDA-IPC KV transport."""

    def __init__(self, cfg: DisaggConfig):
        self.cfg = cfg
        ctx = mp.get_context("spawn")
        self.req_q = ctx.Queue()
        self.kv_q = ctx.Queue()
        self.res_q = ctx.Queue()
        pf_ready, dc_ready = ctx.Event(), ctx.Event()
        self.pf = ctx.Process(target=_prefill_main, args=(cfg, self.req_q, self.kv_q, pf_ready))
        self.dc = ctx.Process(target=_decode_main, args=(cfg, self.kv_q, self.res_q, dc_ready))
        self.pf.start()
        self.dc.start()
        pf_ready.wait()
        dc_ready.wait()

    def submit(self, rid: int, prompt_ids: list[int]) -> None:
        self.req_q.put((rid, prompt_ids))

    def collect(self, n: int) -> list[dict]:
        return [self.res_q.get() for _ in range(n)]

    def shutdown(self) -> None:
        self.req_q.put(None)   # stop prefill once it drains requests
        self.pf.join(timeout=30)
        self.kv_q.put(None)    # then stop decode once it drains KV
        self.dc.join(timeout=30)
