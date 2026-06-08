"""TCP transport (MVP, cross-machine-capable).

Length-prefixed FlatBuffers frames over a TCP socket (docs/protocol.md §1).
Bind/connect localhost or LAN only (security: PRD §9.10). Registered as
``"tcp"`` in the transport registry.
"""

from __future__ import annotations

from hkrl.transport.base import Transport
from hkrl.utils.registry import register_transport


@register_transport("tcp")
class TcpTransport(Transport):
    """Blocking TCP client to HKRLEnvMod's TcpServer.

    Frame = ``uint32 LE length`` + payload. Implements heartbeat/timeout/reconnect
    per docs/protocol.md §5.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5555) -> None:
        self.host = host
        self.port = port
        # TODO(phase-1): socket, recv buffer, optional token auth handshake.

    def connect(self, timeout_s: float = 10.0) -> None:
        raise NotImplementedError  # TODO(phase-1)

    def send(self, frame: bytes) -> None:
        raise NotImplementedError  # TODO(phase-1): write uint32 len + frame

    def recv(self, timeout_s: float | None = None) -> bytes:
        raise NotImplementedError  # TODO(phase-1): read len, then exactly len bytes

    def is_connected(self) -> bool:
        raise NotImplementedError  # TODO(phase-1)

    def reconnect(self, timeout_s: float = 10.0) -> None:
        raise NotImplementedError  # TODO(phase-1)

    def close(self) -> None:
        raise NotImplementedError  # TODO(phase-1)
