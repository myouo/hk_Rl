"""Checkpoint pulling + verification on the worker (PRD §9.10).

Polls the learner's checkpoint registry, downloads new weights, and hash-verifies
before loading (never load an unverified checkpoint).
"""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

import torch

from hkrl.learner.checkpoint_registry import CheckpointMeta


class CheckpointClient:
    """Pulls and verifies policy checkpoints for hot-swapping."""

    def __init__(
        self,
        registry_endpoint: str,
        verify_hash: bool = True,
        auth_token: str | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        if auth_token == "":
            raise ValueError("auth_token must not be empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        self.registry_endpoint = registry_endpoint
        self.verify_hash = verify_hash
        self.auth_token = auth_token
        self.timeout_s = timeout_s
        self._current_version = -1
        self._parsed_endpoint = urlparse(registry_endpoint)
        self._kind: Literal["local", "http"]
        self._root: Path | None
        self._base_url: str | None
        if self._parsed_endpoint.scheme in ("", "file"):
            self._kind = "local"
            self._root = _resolve_local_endpoint(registry_endpoint)
            self._base_url = None
        elif self._parsed_endpoint.scheme in ("http", "https"):
            self._kind = "http"
            self._root = None
            self._base_url = registry_endpoint.rstrip("/") + "/"
        else:
            raise ValueError(
                f"unsupported checkpoint registry endpoint scheme {self._parsed_endpoint.scheme!r}"
            )

    @property
    def current_version(self) -> int:
        return self._current_version

    def latest_version(self) -> int:
        """Return the newest available checkpoint version (or -1)."""
        metas = self._load_index()
        if not metas:
            return -1
        return max(metas)

    def pull(self, version: int) -> dict[str, object]:
        """Download + hash-verify a checkpoint; return a loadable state dict.

        The checkpoint is loaded only after its content hash matches registry
        metadata when ``verify_hash`` is enabled.
        """
        metas = self._load_index()
        try:
            meta = metas[version]
        except KeyError as exc:
            raise KeyError(f"unknown checkpoint version {version}") from exc

        payload = self._read_checkpoint(meta)
        if self.verify_hash:
            actual_hash = _sha256_bytes(payload)
            if actual_hash != meta.sha256:
                raise ValueError(
                    f"checkpoint sha256 mismatch for version {version}: "
                    f"expected {meta.sha256}, got {actual_hash}"
                )

        state = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError(f"checkpoint version {version} did not contain a state dict")
        self._current_version = version
        return state

    def _load_index(self) -> dict[int, CheckpointMeta]:
        if self._kind == "local":
            assert self._root is not None
            return _load_index(self._root)

        assert self._base_url is not None
        try:
            payload = _http_get_bytes(
                urljoin(self._base_url, "index.jsonl"),
                auth_token=self.auth_token,
                timeout_s=self.timeout_s,
            )
        except HTTPError as exc:
            if exc.code == 404:
                return {}
            raise
        return _parse_index(payload.decode("utf-8").splitlines())

    def _read_checkpoint(self, meta: CheckpointMeta) -> bytes:
        if self._kind == "local":
            assert self._root is not None
            path = _checkpoint_path(self._root, meta.path)
            with open(path, "rb") as fh:
                return fh.read()

        assert self._base_url is not None
        return _http_get_bytes(
            _checkpoint_url(self._base_url, meta.path),
            auth_token=self.auth_token,
            timeout_s=self.timeout_s,
        )


def _resolve_local_endpoint(endpoint: str) -> Path:
    parsed = urlparse(endpoint)
    if parsed.scheme in ("", "file"):
        if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
            raise ValueError("file checkpoint endpoints must be local")
        raw_path = parsed.path if parsed.scheme == "file" else endpoint
        return Path(unquote(raw_path)).expanduser().resolve()
    raise ValueError(f"unsupported checkpoint registry endpoint scheme {parsed.scheme!r}")


def _load_index(root: Path) -> dict[int, CheckpointMeta]:
    index_path = root / "index.jsonl"
    if not index_path.exists():
        return {}

    with open(index_path, encoding="utf-8") as fh:
        return _parse_index(fh)


def _parse_index(lines: Any) -> dict[int, CheckpointMeta]:
    metas: dict[int, CheckpointMeta] = {}
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
            meta = _meta_from_payload(payload)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"invalid checkpoint index line {line_no}") from exc
        if meta.version in metas:
            raise ValueError(f"duplicate checkpoint version {meta.version}")
        metas[meta.version] = meta
    return metas


def _meta_from_payload(payload: dict[str, Any]) -> CheckpointMeta:
    path = str(payload["path"])
    _validate_checkpoint_path(path)
    return CheckpointMeta(
        version=int(payload["version"]),
        path=path,
        sha256=str(payload["sha256"]),
        policy_version=int(payload["policy_version"]),
        created_step=int(payload["created_step"]),
    )


def _checkpoint_path(root: Path, path: str) -> Path:
    _validate_checkpoint_path(path)
    checkpoint_path = Path(path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    resolved_root = root.resolve()
    resolved_checkpoint = checkpoint_path.expanduser().resolve()
    try:
        resolved_checkpoint.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("checkpoint path escapes registry root") from exc
    return resolved_checkpoint


def _checkpoint_url(base_url: str, path: str) -> str:
    _validate_checkpoint_path(path)
    parsed = urlparse(path)
    if parsed.scheme or parsed.netloc or path.startswith("/"):
        raise ValueError("remote checkpoint paths must be relative to the registry root")
    parts = [part for part in path.split("/") if part]
    if any(part == ".." for part in parts):
        raise ValueError("remote checkpoint path escapes registry root")
    quoted_path = "/".join(quote(part) for part in parts)
    url = urljoin(base_url, quoted_path)
    if not url.startswith(base_url):
        raise ValueError("remote checkpoint path escapes registry root")
    return url


def _validate_checkpoint_path(path: str) -> None:
    if Path(path) == Path("."):
        raise ValueError("checkpoint path must name a file")


def _http_get_bytes(url: str, *, auth_token: str | None, timeout_s: float) -> bytes:
    headers = {}
    if auth_token is not None:
        headers["Authorization"] = f"Bearer {auth_token}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout_s) as response:
        return response.read()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
