"""Transport round-trip tests for KV transfer.

TCP runs in-process over a localhost socket (CPU tensors), so it needs no GPU. The
CUDA-IPC path is exercised by the disaggregation integration test (needs GPU + spawned
processes); here we just check the TCP serialization preserves the KV payload exactly.
"""

import threading

import torch

from dispec.transport.base import KVPayload
from dispec.transport.tcp import TcpTransport


def test_tcp_roundtrip_preserves_kv():
    L, S, H, D = 4, 7, 2, 8
    payload = KVPayload(
        tokens=list(range(S)),
        k=torch.randn(L, S, H, D),
        v=torch.randn(L, S, H, D),
    )
    received = {}

    def server():
        t = TcpTransport.listen("127.0.0.1", 53111, device="cpu")
        received["p"] = t.recv()
        t.close()

    th = threading.Thread(target=server)
    th.start()
    # Give the listener a moment to bind/accept.
    import time
    time.sleep(0.2)
    client = TcpTransport.connect("127.0.0.1", 53111, device="cpu")
    client.send(payload)
    client.close()
    th.join(timeout=10)

    got = received["p"]
    assert got.tokens == payload.tokens
    assert torch.equal(got.k, payload.k)
    assert torch.equal(got.v, payload.v)
    assert got.nbytes() == payload.nbytes()
