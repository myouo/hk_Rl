"""TCP transport framing tests."""

from __future__ import annotations

import socket
import struct
import threading
import time
from collections.abc import Callable

import pytest
from hkrl.transport.tcp import TcpTransport


def _start_server(handler: Callable[[socket.socket], None]) -> tuple[str, int, threading.Thread]:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    host, port = server.getsockname()

    def run() -> None:
        try:
            conn, _ = server.accept()
            with conn:
                handler(conn)
        finally:
            server.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return host, port, thread


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size

    while remaining > 0:
        chunk = conn.recv(remaining)
        if chunk == b"":
            raise ConnectionError("peer closed")
        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def test_tcp_transport_sends_and_receives_length_prefixed_frames() -> None:
    received: list[bytes] = []

    def handler(conn: socket.socket) -> None:
        header = _recv_exact(conn, 4)
        (length,) = struct.unpack("<I", header)
        received.append(_recv_exact(conn, length))
        response = b"step-response"
        conn.sendall(struct.pack("<I", len(response)) + response)

    host, port, thread = _start_server(handler)
    transport = TcpTransport(host, port)

    transport.connect(timeout_s=1.0)
    transport.send(b"step-request")

    assert transport.recv(timeout_s=1.0) == b"step-response"
    assert received == [b"step-request"]
    assert transport.is_connected()

    transport.close()
    assert not transport.is_connected()
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_tcp_transport_recv_timeout() -> None:
    def handler(_conn: socket.socket) -> None:
        time.sleep(0.2)

    host, port, thread = _start_server(handler)
    transport = TcpTransport(host, port)

    transport.connect(timeout_s=1.0)
    with pytest.raises(TimeoutError):
        transport.recv(timeout_s=0.01)

    transport.close()
    thread.join(timeout=1.0)
