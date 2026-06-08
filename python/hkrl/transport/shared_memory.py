"""Shared-memory ring-buffer transport (low-latency, single-machine).

Highest-SPS path when the worker and the game run on the same PC: frames are
written into a shared ring buffer instead of crossing a socket. Same payload as
TCP (FlatBuffers), length in the slot header. Registered as ``"shm"``.

This is a performance optimization; correctness/parity with TCP is verified by
the same protocol tests. See docs/architecture.md §5 and ADR-0002.
"""

from __future__ import annotations

from hkrl.transport.base import Transport
from hkrl.utils.registry import register_transport


@register_transport("shm")
class SharedMemoryTransport(Transport):
    """Two single-producer/single-consumer ring buffers (req out, resp in).

    Synchronization via OS shared memory + lightweight signaling. The mod side
    is implemented in ``mod/HKRLEnvMod/Transport`` (a future SHM server variant).
    """

    def __init__(self, name: str = "hkrl_env", req_slots: int = 8, resp_slots: int = 8) -> None:
        self.name = name
        self.req_slots = req_slots
        self.resp_slots = resp_slots
        # TODO(phase-8): multiprocessing.shared_memory segments + headers.

    def connect(self, timeout_s: float = 10.0) -> None:
        raise NotImplementedError  # TODO(phase-8)

    def send(self, frame: bytes) -> None:
        raise NotImplementedError  # TODO(phase-8)

    def recv(self, timeout_s: float | None = None) -> bytes:
        raise NotImplementedError  # TODO(phase-8)

    def is_connected(self) -> bool:
        raise NotImplementedError  # TODO(phase-8)

    def reconnect(self, timeout_s: float = 10.0) -> None:
        raise NotImplementedError  # TODO(phase-8)

    def close(self) -> None:
        raise NotImplementedError  # TODO(phase-8)
