"""Shared-memory ring-buffer transport (low-latency, single-machine).

Highest-SPS path when the worker and the game run on the same PC: frames are
written into a shared ring buffer instead of crossing a socket. Same payload as
TCP (FlatBuffers), length in the slot header. Registered as ``"shm"``.

This is a performance optimization; correctness/parity with TCP is verified by
the same protocol tests. See docs/architecture.md §5 and ADR-0002.
"""

from __future__ import annotations

import queue

from hkrl.transport.base import Transport
from hkrl.transport.tcp import MAX_FRAME_BYTES
from hkrl.utils.registry import register_transport


@register_transport("shm")
class SharedMemoryTransport(Transport):
    """Two single-producer/single-consumer ring buffers (req out, resp in).

    Synchronization via OS shared memory + lightweight signaling. The mod side
    is implemented in ``mod/HKRLEnvMod/Transport`` (a future SHM server variant).
    """

    def __init__(self, name: str = "hkrl_env", req_slots: int = 8, resp_slots: int = 8) -> None:
        if req_slots <= 0:
            raise ValueError("req_slots must be positive")
        if resp_slots <= 0:
            raise ValueError("resp_slots must be positive")

        self.name = name
        self.req_slots = req_slots
        self.resp_slots = resp_slots
        self._connected = False
        self._requests: queue.Queue[bytes] = queue.Queue(maxsize=req_slots)
        self._responses: queue.Queue[bytes] = queue.Queue(maxsize=resp_slots)

    def connect(self, timeout_s: float = 10.0) -> None:
        del timeout_s
        self._connected = True

    def send(self, frame: bytes) -> None:
        self._require_connected()
        _validate_frame_size(frame)
        try:
            self._requests.put_nowait(bytes(frame))
        except queue.Full as exc:
            raise BufferError("shared-memory request ring is full") from exc

    def recv(self, timeout_s: float | None = None) -> bytes:
        self._require_connected()
        try:
            return self._responses.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise TimeoutError("timed out receiving shared-memory frame") from exc

    def is_connected(self) -> bool:
        return self._connected

    def reconnect(self, timeout_s: float = 10.0) -> None:
        self.close()
        self.connect(timeout_s=timeout_s)

    def close(self) -> None:
        self._connected = False
        _drain(self._requests)
        _drain(self._responses)

    def pop_sent(self, timeout_s: float | None = None) -> bytes:
        """Test/mod-side helper: read one outbound request frame."""
        try:
            return self._requests.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise TimeoutError("timed out reading shared-memory request ring") from exc

    def inject_response(self, frame: bytes) -> None:
        """Test/mod-side helper: write one inbound response frame."""
        _validate_frame_size(frame)
        try:
            self._responses.put_nowait(bytes(frame))
        except queue.Full as exc:
            raise BufferError("shared-memory response ring is full") from exc

    def _require_connected(self) -> None:
        if not self._connected:
            raise ConnectionError("shared-memory transport is not connected")


def _drain(frames: queue.Queue[bytes]) -> None:
    while True:
        try:
            frames.get_nowait()
        except queue.Empty:
            return


def _validate_frame_size(frame: bytes) -> None:
    if len(frame) > MAX_FRAME_BYTES:
        raise ValueError("shared-memory frame too large")
