"""Authenticated HTTP serving for checkpoints and bounded live-tuning control.

Checkpoint files remain read-only. The sole write endpoint accepts a validated,
monotonic live-tuning snapshot and atomically stores it beside the registry. The
service is intended to bind to loopback and be reached through an SSH forward.
"""

from __future__ import annotations

import hmac
import json
import re
import socket
import threading
from contextlib import suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

from hkrl.learner.batch_intake import split_endpoint
from hkrl.utils.live_tuning import (
    LIVE_TUNING_REQUEST,
    LIVE_TUNING_STATUS,
    LiveTuning,
    atomic_write_json,
    load_live_tuning,
)

_CHECKPOINT_NAME = re.compile(r"checkpoint_v[0-9]{6,}\.pt\Z")
_COPY_CHUNK_BYTES = 1024 * 1024
_MAX_TUNING_BYTES = 64 * 1024


class CheckpointHttpServer:
    """Threaded checkpoint registry + narrow authenticated control endpoint."""

    def __init__(
        self,
        root: str | Path,
        bind: str = "127.0.0.1:5601",
        *,
        auth_token: str | None,
    ) -> None:
        if auth_token == "":
            raise ValueError("auth_token must not be empty")
        self.root = Path(root).expanduser().resolve()
        self.bind = bind
        self.auth_token = auth_token
        self._server: ThreadingHTTPServer | None = None
        self.endpoint: str | None = None
        self._serving = threading.Event()

    def __enter__(self) -> CheckpointHttpServer:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def start(self) -> None:
        """Create the registry root and start listening if needed."""
        if self._server is not None:
            return

        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise NotADirectoryError(f"checkpoint root is not a directory: {self.root}")

        host, port = split_endpoint(self.bind)
        handler = _handler_factory(self.root, self.auth_token)
        server_type = _IPv6ThreadingHTTPServer if ":" in host else _CheckpointHTTPServer
        server = server_type((host, port), handler)
        actual = server.server_address
        actual_host, actual_port = str(actual[0]), int(actual[1])
        self._server = server
        self.endpoint = (
            f"[{actual_host}]:{actual_port}"
            if ":" in actual_host
            else f"{actual_host}:{actual_port}"
        )

    def serve_forever(self) -> None:
        self.start()
        server = self._server
        assert server is not None
        self._serving.set()
        try:
            server.serve_forever(poll_interval=0.25)
        finally:
            self._serving.clear()

    def serve_in_thread(self) -> threading.Thread:
        """Start serving in a daemon thread (primarily for integration tests)."""
        self.start()
        thread = threading.Thread(target=self.serve_forever, daemon=True)
        thread.start()
        if not self._serving.wait(timeout=2.0):
            raise RuntimeError("checkpoint HTTP server thread did not start")
        return thread

    def close(self) -> None:
        server = self._server
        self._server = None
        self.endpoint = None
        if server is None:
            return
        if self._serving.is_set():
            with suppress(Exception):
                server.shutdown()
        server.server_close()


class _CheckpointHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class _IPv6ThreadingHTTPServer(_CheckpointHTTPServer):
    address_family = socket.AF_INET6


def _handler_factory(
    root: Path,
    auth_token: str | None,
) -> type[BaseHTTPRequestHandler]:
    tuning_lock = threading.Lock()

    class CheckpointRequestHandler(BaseHTTPRequestHandler):
        server_version = "HKRLCheckpointRegistry/2"
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            self._serve(send_body=True)

        def do_HEAD(self) -> None:
            self._serve(send_body=False)

        def do_POST(self) -> None:
            if not self._authorize():
                return
            if _request_path(self.path) != "/live-tuning":
                self.send_error(HTTPStatus.METHOD_NOT_ALLOWED)
                return
            self._store_live_tuning()

        def log_message(self, message_format: str, *args: Any) -> None:
            return

        def _serve(self, *, send_body: bool) -> None:
            if not self._authorize():
                return

            request_path = _request_path(self.path)
            control_name = {
                "/live-tuning": LIVE_TUNING_REQUEST,
                "/live-tuning/status": LIVE_TUNING_STATUS,
            }.get(request_path or "")
            path = (
                root / control_name
                if control_name is not None
                else _resolve_served_path(root, self.path)
            )
            if path is None or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            try:
                file_size = path.stat().st_size
                file_handle = path.open("rb")
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            with file_handle:
                self.send_response(HTTPStatus.OK)
                self.send_header(
                    "Content-Type",
                    (
                        "application/x-ndjson; charset=utf-8"
                        if path.name == "index.jsonl"
                        else "application/json; charset=utf-8"
                        if path.name in {LIVE_TUNING_REQUEST, LIVE_TUNING_STATUS}
                        else "application/octet-stream"
                    ),
                )
                self.send_header("Content-Length", str(file_size))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header(
                    "Cache-Control",
                    (
                        "no-store"
                        if path.name
                        in {
                            "index.jsonl",
                            LIVE_TUNING_REQUEST,
                            LIVE_TUNING_STATUS,
                        }
                        else "public, max-age=31536000, immutable"
                    ),
                )
                self.end_headers()
                if not send_body:
                    return

                while chunk := file_handle.read(_COPY_CHUNK_BYTES):
                    self.wfile.write(chunk)

        def _authorize(self) -> bool:
            if _authorized(self.headers.get("Authorization"), auth_token):
                return True
            self.send_response(HTTPStatus.UNAUTHORIZED)
            self.send_header("WWW-Authenticate", "Bearer")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False

        def _store_live_tuning(self) -> None:
            content_type = self.headers.get("Content-Type", "")
            if content_type.split(";", 1)[0].strip().lower() != "application/json":
                self.send_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
                return
            try:
                content_length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self.send_error(HTTPStatus.LENGTH_REQUIRED)
                return
            if not 0 < content_length <= _MAX_TUNING_BYTES:
                self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
            body = self.rfile.read(content_length)
            if len(body) != content_length:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            try:
                tuning = LiveTuning.model_validate(json.loads(body))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"invalid live tuning: {exc}"},
                )
                return

            with tuning_lock:
                try:
                    current = load_live_tuning(root / LIVE_TUNING_REQUEST)
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    self._send_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": f"stored live tuning is invalid: {exc}"},
                    )
                    return
                current_version = 0 if current is None else current.version
                if tuning.version != current_version + 1:
                    self._send_json(
                        HTTPStatus.CONFLICT,
                        {
                            "current_version": current_version,
                            "error": "live tuning version must increment by exactly one",
                        },
                    )
                    return
                atomic_write_json(
                    root / LIVE_TUNING_REQUEST,
                    tuning.checkpoint_payload(),
                )

            self._send_json(
                HTTPStatus.CREATED,
                {
                    "digest": tuning.digest,
                    "requested_version": tuning.version,
                },
            )

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            encoded = (
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(encoded)

    return CheckpointRequestHandler


def _authorized(header: str | None, auth_token: str | None) -> bool:
    if auth_token is None:
        return True
    if header is None or not header.startswith("Bearer "):
        return False
    return hmac.compare_digest(header[7:], auth_token)


def _resolve_served_path(root: Path, request_target: str) -> Path | None:
    try:
        decoded = unquote(urlsplit(request_target).path)
    except ValueError:
        return None
    relative = PurePosixPath(decoded.lstrip("/"))
    parts = relative.parts
    if (
        not parts
        or relative.is_absolute()
        or any(part in ("", ".", "..") for part in parts)
        or any("\\" in part for part in parts)
    ):
        return None
    if len(parts) == 1 and parts[0] == "index.jsonl":
        pass
    elif not _CHECKPOINT_NAME.fullmatch(parts[-1]):
        return None

    candidate = (root / Path(*parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _request_path(request_target: str) -> str | None:
    try:
        return unquote(urlsplit(request_target).path)
    except ValueError:
        return None
