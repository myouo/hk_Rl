"""Length-prefixed TCP intake for RolloutBatch uploads.

This is the learner-side batch channel, not the env action transport. It carries
whole rollout bundles asynchronously, preserving ADR-0004's local action loop.
"""

from __future__ import annotations

import json
import socket
import struct
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from hkrl.learner.learner_server import LearnerServer
from hkrl.training.batch_io import deserialize_rollout_batch, serialize_rollout_batch
from hkrl.training.rollout_buffer import RolloutBatch

BATCH_INTAKE_TYPE = "hkrl.rollout_batch.v2"
MAX_FRAME_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class BatchIntakeResult:
    """Result returned after one worker upload is handled."""

    accepted: bool
    peer: str
    policy_version: int


class BatchIntakeServer:
    """Blocking one-connection-at-a-time RolloutBatch intake server."""

    def __init__(
        self,
        learner: LearnerServer,
        bind: str,
        *,
        auth_token: str | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        if auth_token == "":
            raise ValueError("auth_token must not be empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        self.learner = learner
        self.bind = bind
        self.auth_token = auth_token
        self.timeout_s = timeout_s
        self._sock: socket.socket | None = None
        self.endpoint: str | None = None

    def __enter__(self) -> BatchIntakeServer:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def start(self) -> None:
        """Bind and listen if not already started."""
        if self._sock is not None:
            return

        host, port = split_endpoint(self.bind)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(self.timeout_s)
            sock.bind((host, port))
            sock.listen(16)
            actual_host, actual_port = sock.getsockname()[:2]
            self._sock = sock
            self.endpoint = f"{actual_host}:{actual_port}"
        except OSError:
            sock.close()
            raise

    def serve_once(self) -> BatchIntakeResult:
        """Accept one worker connection, submit its batch, and send an ACK."""
        self.start()
        assert self._sock is not None
        conn, peer = self._sock.accept()
        peer_label = f"{peer[0]}:{peer[1]}"
        with conn:
            conn.settimeout(self.timeout_s)
            accepted = self._handle_connection(conn)
        return BatchIntakeResult(
            accepted=accepted,
            peer=peer_label,
            policy_version=self.learner.policy_version,
        )

    def close(self) -> None:
        sock = self._sock
        self._sock = None
        self.endpoint = None
        if sock is None:
            return
        with suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)
        sock.close()

    def _handle_connection(self, conn: socket.socket) -> bool:
        try:
            header = _decode_json_frame(_recv_frame(conn))
            _validate_header(header, self.auth_token)
            batch = deserialize_rollout_batch(_recv_frame(conn))
            accepted = self.learner.submit(batch)
            _send_json_frame(
                conn,
                {
                    "ok": True,
                    "accepted": accepted,
                    "policy_version": self.learner.policy_version,
                },
            )
            return accepted
        except Exception as exc:
            with suppress(OSError):
                _send_json_frame(
                    conn,
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "policy_version": self.learner.policy_version,
                    },
                )
            raise


class BatchIntakeClient:
    """Small TCP client used by workers to upload completed RolloutBatches."""

    def __init__(
        self,
        endpoint: str,
        *,
        auth_token: str | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        if auth_token == "":
            raise ValueError("auth_token must not be empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.endpoint = endpoint
        self.auth_token = auth_token
        self.timeout_s = timeout_s

    def submit(self, batch: RolloutBatch) -> bool:
        host, port = split_endpoint(self.endpoint)
        with socket.create_connection((host, port), timeout=self.timeout_s) as sock:
            sock.settimeout(self.timeout_s)
            _send_json_frame(
                sock,
                {
                    "type": BATCH_INTAKE_TYPE,
                    "token": self.auth_token,
                },
            )
            _send_frame(sock, serialize_rollout_batch(batch))
            ack = _decode_json_frame(_recv_frame(sock))
        return _accepted_from_ack(ack)


def split_endpoint(endpoint: str) -> tuple[str, int]:
    """Parse ``host:port`` or ``[ipv6]:port`` endpoint strings."""
    if endpoint.startswith("["):
        host, sep, rest = endpoint[1:].partition("]")
        if sep != "]" or not rest.startswith(":"):
            raise ValueError(f"endpoint must be host:port, got {endpoint!r}")
        port_text = rest[1:]
    else:
        host, sep, port_text = endpoint.rpartition(":")
        if sep != ":":
            raise ValueError(f"endpoint must be host:port, got {endpoint!r}")

    if host == "":
        raise ValueError(f"endpoint host must not be empty: {endpoint!r}")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"endpoint port must be an integer: {endpoint!r}") from exc
    if not 0 <= port <= 65535:
        raise ValueError(f"endpoint port must be in [0, 65535], got {port}")
    return host, port


def _validate_header(header: dict[str, Any], auth_token: str | None) -> None:
    if header.get("type") != BATCH_INTAKE_TYPE:
        raise ValueError("invalid batch intake header type")
    token = header.get("token")
    if auth_token is None:
        return
    if token != auth_token:
        raise PermissionError("invalid batch intake auth token")


def _accepted_from_ack(ack: dict[str, Any]) -> bool:
    if not ack.get("ok"):
        raise RuntimeError(str(ack.get("error", "batch intake failed")))
    accepted = ack.get("accepted")
    if not isinstance(accepted, bool):
        raise ValueError("batch intake ACK must include accepted boolean")
    return accepted


def _send_json_frame(sock: socket.socket, payload: dict[str, Any]) -> None:
    _send_frame(sock, json.dumps(payload, sort_keys=True).encode("utf-8"))


def _decode_json_frame(payload: bytes) -> dict[str, Any]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON frame") from exc
    if not isinstance(decoded, dict):
        raise ValueError("JSON frame must be an object")
    return decoded


def _send_frame(sock: socket.socket, payload: bytes) -> None:
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError("batch frame too large")
    sock.sendall(struct.pack("<I", len(payload)) + payload)


def _recv_frame(sock: socket.socket) -> bytes:
    header = _recv_exact(sock, 4)
    (size,) = struct.unpack("<I", header)
    if size > MAX_FRAME_BYTES:
        raise ValueError("batch frame too large")
    return _recv_exact(sock, size)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if chunk == b"":
            raise ConnectionError("batch intake peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
