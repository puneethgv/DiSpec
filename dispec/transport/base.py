"""KV-transfer transport abstraction for disaggregated serving.

In prefill/decode (P/D) disaggregation the prompt's KV cache is computed on a prefill
worker and must be moved to the decode worker. The transport is the pluggable channel
that does the move:

  - CudaIpcTransport: zero-copy on a single node (shares the GPU tensor across processes
    via CUDA IPC handles) — what you'd use when workers share a host/NVLink island.
  - TcpTransport: serialize + ship over a socket — the multi-node fallback (and what a
    NIXL/RDMA backend would replace for speed).

Keeping this behind one interface means the rest of the system (workers, router) is
agnostic to whether the peer is on the same GPU or another node.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


@dataclass
class KVPayload:
    """A prompt's contiguous KV cache plus the tokens it was computed from.

    k, v: (num_layers, seq_len, num_kv_heads, head_dim). Contiguous (not paged) so the
    layout is portable across workers whose block pools differ; the receiver writes it
    into its own paged cache. `meta` carries handoff state (request id, first decoded
    token, etc.) so the decode worker can resume without recomputing.
    """

    tokens: list[int]
    k: torch.Tensor
    v: torch.Tensor
    meta: dict = field(default_factory=dict)

    @property
    def seq_len(self) -> int:
        return len(self.tokens)

    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2


class Transport(ABC):
    """A one-directional channel that moves a KVPayload from sender to receiver."""

    @abstractmethod
    def send(self, payload: KVPayload) -> None:
        ...

    @abstractmethod
    def recv(self) -> KVPayload:
        ...

    def close(self) -> None:  # optional cleanup
        pass
