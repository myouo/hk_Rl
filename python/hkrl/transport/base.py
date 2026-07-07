"""Transport abstraction (docs/protocol.md, invariant #3).

A Transport carries length-prefixed FlatBuffers frames between the worker and the
mod. Implementations MUST NOT interpret message contents beyond framing — decode
belongs in ``hkrl.protocol``. TCP is the supported live HKRLEnvMod transport;
``shared_memory`` is currently an explicit opt-in in-process prototype.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Transport(Protocol):
    """Bidirectional, framed message channel to one HKRLEnvMod env instance.

    Lifecycle: ``connect`` -> repeated ``send``/``recv`` -> ``close``. Thread-safety
    is implementation-defined; the GameWorker uses one Transport per env from a
    single thread on the hot path.
    """

    def connect(self, timeout_s: float = 10.0) -> None:
        """Establish or attach the transport backend. Blocking."""
        ...

    def send(self, frame: bytes) -> None:
        """Send one encoded StepRequest payload; transport adds its framing."""
        ...

    def recv(self, timeout_s: float | None = None) -> bytes:
        """Receive one StepResponse payload. Raises on timeout/disconnect."""
        ...

    def is_connected(self) -> bool:
        """Liveness check (heartbeat-aware where supported)."""
        ...

    def reconnect(self, timeout_s: float = 10.0) -> None:
        """Tear down and re-establish; used after timeout/crash (docs/protocol.md §5)."""
        ...

    def close(self) -> None:
        """Release resources. Idempotent."""
        ...
