"""CUDA IPC transport: zero-copy KV handoff between processes on one node.

Wraps a ``torch.multiprocessing`` queue. Putting a CUDA tensor on such a queue ships a
CUDA IPC *handle* (not the data) — the receiving process maps the same physical GPU
memory, so the KV cache is handed over without a device copy. This is the on-node fast
path for P/D disaggregation (workers sharing a GPU / NVLink island).

Caveat (CUDA IPC semantics): the *sender* must keep the tensor alive until the receiver
has mapped it, otherwise the memory can be freed underneath the handle. Callers should
retain sent payloads until the consumer acks (the prefill worker does this).
"""

from __future__ import annotations

import torch.multiprocessing as mp

from dispec.transport.base import KVPayload, Transport


class CudaIpcTransport(Transport):
    def __init__(self, queue: "mp.Queue"):
        self._q = queue
        self._inflight: list[KVPayload] = []  # keep sent tensors alive (IPC requirement)

    @staticmethod
    def make_queue(ctx: "mp.context.BaseContext | None" = None) -> "mp.Queue":
        """Create a queue suitable for sharing CUDA tensors (use a spawn context)."""
        ctx = ctx or mp.get_context("spawn")
        return ctx.Queue()

    def send(self, payload: KVPayload) -> None:
        # CUDA tensors are shared by IPC handle (zero-copy); metadata is pickled.
        self._inflight.append(payload)
        self._q.put({"tokens": payload.tokens, "k": payload.k, "v": payload.v,
                     "meta": payload.meta})

    def recv(self) -> "KVPayload | None":
        obj = self._q.get()
        if obj is None:  # shutdown sentinel
            return None
        return KVPayload(obj["tokens"], obj["k"], obj["v"], obj.get("meta", {}))

    def send_sentinel(self) -> None:
        self._q.put(None)

    def release(self) -> None:
        """Drop references to sent payloads once the consumer has mapped them."""
        self._inflight.clear()
