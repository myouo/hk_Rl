"""TCP transport (MVP, cross-machine-capable).

Length-prefixed FlatBuffers frames over a TCP socket (docs/protocol.md §1).
Bind/connect localhost or LAN only (security: PRD §9.10). Registered as
``"tcp"`` in the transport registry.
"""

from __future__ import annotations

import socket
import struct
from contextlib import suppress

from hkrl.transport.base import Transport
from hkrl.utils.registry import register_transport

MAX_FRAME_BYTES = 16 * 1024 * 1024


@register_transport("tcp")
class TcpTransport(Transport):
    """Blocking TCP client to HKRLEnvMod's TcpServer.

    Frame = ``uint32 LE length`` + payload. Implements heartbeat/timeout/reconnect
    per docs/protocol.md §5.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        auth_token: str | None = None,
    ) -> None:
        if auth_token == "":
            raise ValueError("auth_token must not be empty")

        self.host = host
        self.port = port
        self.auth_token = auth_token
        self._sock: socket.socket | None = None

    def connect(self, timeout_s: float = 10.0) -> None:
        self.close()
        self._sock = socket.create_connection((self.host, self.port), timeout=timeout_s)
        if self.auth_token is not None:
            try:
                self._send_frame(self._sock, b"HKRL_AUTH\0" + self.auth_token.encode("utf-8"))
            except OSError as exc:
                self.close()
                raise ConnectionError("failed to send TCP auth token") from exc

    def send(self, frame: bytes) -> None:
        sock = self._require_socket()
        try:
            self._send_frame(sock, frame)
        except OSError as exc:
            self.close()
            raise ConnectionError("failed to send TCP frame") from exc

    def recv(self, timeout_s: float | None = None) -> bytes:
        sock = self._require_socket()
        previous_timeout = sock.gettimeout()
        sock.settimeout(timeout_s)
        try:
            header = self._recv_exact(4)
            (length,) = struct.unpack("<I", header)
            if length > MAX_FRAME_BYTES:
                self.close()
                raise ValueError("TCP frame too large")
            return self._recv_exact(length)
        finally:
            if self._sock is sock:
                sock.settimeout(previous_timeout)

    def is_connected(self) -> bool:
        return self._sock is not None

    def reconnect(self, timeout_s: float = 10.0) -> None:
        self.close()
        self.connect(timeout_s=timeout_s)

    def close(self) -> None:
        sock = self._sock
        self._sock = None
        if sock is None:
            return

        with suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)
        sock.close()

    def _require_socket(self) -> socket.socket:
        if self._sock is None:
            raise ConnectionError("TCP transport is not connected")
        return self._sock

    @staticmethod
    def _send_frame(sock: socket.socket, frame: bytes) -> None:
        if len(frame) > MAX_FRAME_BYTES:
            raise ValueError("TCP frame too large")
        sock.sendall(struct.pack("<I", len(frame)) + frame)

    def _recv_exact(self, size: int) -> bytes:
        sock = self._require_socket()
        chunks: list[bytes] = []
        remaining = size

        while remaining > 0:
            try:
                chunk = sock.recv(remaining)
            except TimeoutError as exc:
                raise TimeoutError("timed out receiving TCP frame") from exc
            except OSError as exc:
                self.close()
                raise ConnectionError("failed to receive TCP frame") from exc

            if chunk == b"":
                self.close()
                raise ConnectionError("TCP peer closed the connection")

            chunks.append(chunk)
            remaining -= len(chunk)

        return b"".join(chunks)
