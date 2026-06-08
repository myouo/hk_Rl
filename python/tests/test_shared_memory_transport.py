"""Shared-memory transport ring-semantics tests."""

from __future__ import annotations

import pytest
from hkrl.transport.shared_memory import SharedMemoryTransport


def test_shared_memory_transport_sends_and_receives_frames() -> None:
    transport = SharedMemoryTransport(req_slots=2, resp_slots=2)
    transport.connect()

    transport.send(b"request")
    assert transport.pop_sent(timeout_s=0.01) == b"request"

    transport.inject_response(b"response")
    assert transport.recv(timeout_s=0.01) == b"response"
    assert transport.is_connected()


def test_shared_memory_transport_requires_connection() -> None:
    transport = SharedMemoryTransport()

    with pytest.raises(ConnectionError, match="not connected"):
        transport.send(b"request")
    with pytest.raises(ConnectionError, match="not connected"):
        transport.recv(timeout_s=0.01)


def test_shared_memory_transport_times_out_and_enforces_capacity() -> None:
    transport = SharedMemoryTransport(req_slots=1, resp_slots=1)
    transport.connect()

    with pytest.raises(TimeoutError, match="timed out"):
        transport.recv(timeout_s=0.01)

    transport.send(b"one")
    with pytest.raises(BufferError, match="request ring is full"):
        transport.send(b"two")

    transport.inject_response(b"one")
    with pytest.raises(BufferError, match="response ring is full"):
        transport.inject_response(b"two")


def test_shared_memory_transport_reconnect_and_close_clear_rings() -> None:
    transport = SharedMemoryTransport()
    transport.connect()
    transport.send(b"stale")
    transport.inject_response(b"stale")

    transport.reconnect()

    assert transport.is_connected()
    with pytest.raises(TimeoutError):
        transport.pop_sent(timeout_s=0.01)
    with pytest.raises(TimeoutError):
        transport.recv(timeout_s=0.01)

    transport.close()
    assert not transport.is_connected()


def test_shared_memory_transport_rejects_invalid_slots() -> None:
    with pytest.raises(ValueError, match="req_slots"):
        SharedMemoryTransport(req_slots=0)
    with pytest.raises(ValueError, match="resp_slots"):
        SharedMemoryTransport(resp_slots=0)
