"""TCP transport: serialize the KV cache and ship it over a socket.

The multi-node fallback. It copies (GPU->CPU->wire->GPU), so it's the slow path a
production system would replace with RDMA/NIXL — but it makes the architecture
genuinely node-agnostic and lets us measure transfer cost honestly.
"""

from __future__ import annotations

import io
import socket
import struct

import torch

from dispec.transport.base import KVPayload, Transport


def _send_msg(sock: socket.socket, data: bytes) -> None:
    sock.sendall(struct.pack("!Q", len(data)))
    sock.sendall(data)


def _recv_n(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("peer closed during recv")
        buf.extend(chunk)
    return bytes(buf)


def _recv_msg(sock: socket.socket) -> bytes:
    (n,) = struct.unpack("!Q", _recv_n(sock, 8))
    return _recv_n(sock, n)


class TcpTransport(Transport):
    def __init__(self, sock: socket.socket, device: str = "cuda"):
        self._sock = sock
        self.device = device if torch.cuda.is_available() else "cpu"

    @classmethod
    def connect(cls, host: str, port: int, device: str = "cuda") -> "TcpTransport":
        s = socket.create_connection((host, port))
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return cls(s, device)

    @classmethod
    def listen(cls, host: str, port: int, device: str = "cuda") -> "TcpTransport":
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        srv.close()
        return cls(conn, device)

    def send(self, payload: KVPayload) -> None:
        buf = io.BytesIO()
        torch.save({"tokens": payload.tokens, "k": payload.k.cpu(),
                    "v": payload.v.cpu(), "meta": payload.meta}, buf)
        _send_msg(self._sock, buf.getvalue())

    def recv(self) -> KVPayload:
        obj = torch.load(io.BytesIO(_recv_msg(self._sock)), weights_only=False)
        return KVPayload(obj["tokens"], obj["k"].to(self.device), obj["v"].to(self.device),
                         obj.get("meta", {}))

    def close(self) -> None:
        self._sock.close()
