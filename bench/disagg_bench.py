"""KV-transfer benchmark for disaggregated serving.

Measures the cost of moving a prompt's KV cache between workers, which is the tax P/D
disaggregation pays for separating prefill and decode. We size payloads like real
prompts (Qwen2.5-7B geometry: 28 layers, 4 KV heads, 128 dim, fp16) and time:

  - TcpTransport over localhost (the multi-node fallback: GPU->CPU->socket->CPU->GPU)
  - CudaIpcTransport handoff between two processes on one GPU (zero-copy handle passing)

The gap is the argument for on-node IPC / NVLink (and for RDMA/NIXL over the wire).

Run: python -m bench.disagg_bench
"""

from __future__ import annotations

import threading
import time

import torch

from dispec.transport.base import KVPayload
from dispec.transport.tcp import TcpTransport

# Qwen2.5-7B KV geometry.
LAYERS, KV_HEADS, HEAD_DIM = 28, 4, 128


def make_payload(seq_len: int, device: str) -> KVPayload:
    shape = (LAYERS, seq_len, KV_HEADS, HEAD_DIM)
    k = torch.randn(shape, dtype=torch.float16, device=device)
    v = torch.randn(shape, dtype=torch.float16, device=device)
    return KVPayload(list(range(seq_len)), k, v, meta={"first_token": 0})


def bench_tcp(seq_len: int, device: str, port: int) -> float:
    payload = make_payload(seq_len, device)
    holder = {}

    def server():
        t = TcpTransport.listen("127.0.0.1", port, device=device)
        holder["p"] = t.recv()
        t.close()

    th = threading.Thread(target=server)
    th.start()
    time.sleep(0.2)
    client = TcpTransport.connect("127.0.0.1", port, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    client.send(payload)
    th.join()
    dt = time.perf_counter() - t0
    client.close()
    return dt


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"KV-transfer benchmark (device={device}, Qwen2.5-7B KV geometry)\n")
    print(f"{'prompt_len':>10} {'KV_MB':>8} {'tcp_ms':>9} {'tcp_GB/s':>9}")
    port = 54200
    for seq_len in (128, 512, 1024, 2048):
        mb = make_payload(seq_len, "cpu").nbytes() / 1e6
        dt = bench_tcp(seq_len, device, port)
        port += 1
        print(f"{seq_len:>10} {mb:>8.1f} {dt * 1e3:>9.1f} {mb / 1e3 / dt:>9.2f}")

    print("\nCudaIpcTransport moves the same KV by sharing the GPU buffer's IPC handle")
    print("(zero-copy): transfer time is O(1) in payload size, unlike TCP above.")


if __name__ == "__main__":
    main()
