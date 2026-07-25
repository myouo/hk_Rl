"""Authenticated, read-only HTTP serving for checkpoint registries.

The service exposes only ``index.jsonl`` and immutable ``checkpoint_v*.pt``
files from one registry root. It is intended to bind to loopback and be reached
through an SSH local-forward by GameWorkers.
"""

from __future__ import annotations

import hmac
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

_CHECKPOINT_NAME = re.compile(r"checkpoint_v[0-9]{6,}\.pt\Z")
_COPY_CHUNK_BYTES = 1024 * 1024


class CheckpointHttpServer:
    """Threaded, read-only checkpoint registry HTTP server."""

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
    class CheckpointRequestHandler(BaseHTTPRequestHandler):
        server_version = "HKRLCheckpointRegistry/1"
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            self._serve(send_body=True)

        def do_HEAD(self) -> None:
            self._serve(send_body=False)

        def do_POST(self) -> None:
            self.send_error(HTTPStatus.METHOD_NOT_ALLOWED)

        def log_message(self, message_format: str, *args: Any) -> None:
            return

        def _serve(self, *, send_body: bool) -> None:
            if not _authorized(self.headers.get("Authorization"), auth_token):
                self.send_response(HTTPStatus.UNAUTHORIZED)
                self.send_header("WWW-Authenticate", "Bearer")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            path = _resolve_served_path(root, self.path)
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
                        else "application/octet-stream"
                    ),
                )
                self.send_header("Content-Length", str(file_size))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header(
                    "Cache-Control",
                    (
                        "no-store"
                        if path.name == "index.jsonl"
                        else "public, max-age=31536000, immutable"
                    ),
                )
                self.end_headers()
                if not send_body:
                    return

                while chunk := file_handle.read(_COPY_CHUNK_BYTES):
                    self.wfile.write(chunk)

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
