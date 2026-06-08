"""Checkpoint pulling + verification on the worker (PRD §9.10).

Polls the learner's checkpoint registry, downloads new weights, and hash-verifies
before loading (never load an unverified checkpoint).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import torch

from hkrl.learner.checkpoint_registry import CheckpointMeta


class CheckpointClient:
    """Pulls and verifies policy checkpoints for hot-swapping."""

    def __init__(self, registry_endpoint: str, verify_hash: bool = True) -> None:
        self.registry_endpoint = registry_endpoint
        self.verify_hash = verify_hash
        self._current_version = -1
        self._root = _resolve_local_endpoint(registry_endpoint)

    @property
    def current_version(self) -> int:
        return self._current_version

    def latest_version(self) -> int:
        """Return the newest available checkpoint version (or -1).

        Phase 6 starts with local/file registry endpoints. A future learner
        service can expose the same metadata over an authenticated LAN endpoint.
        """
        metas = _load_index(self._root)
        if not metas:
            return -1
        return max(metas)

    def pull(self, version: int) -> dict[str, object]:
        """Download + hash-verify a checkpoint; return a loadable state dict.

        The checkpoint is loaded only after its content hash matches registry
        metadata when ``verify_hash`` is enabled.
        """
        metas = _load_index(self._root)
        try:
            meta = metas[version]
        except KeyError as exc:
            raise KeyError(f"unknown checkpoint version {version}") from exc

        path = _checkpoint_path(self._root, meta.path)
        if self.verify_hash:
            actual_hash = _sha256_file(path)
            if actual_hash != meta.sha256:
                raise ValueError(
                    f"checkpoint sha256 mismatch for version {version}: "
                    f"expected {meta.sha256}, got {actual_hash}"
                )

        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError(f"checkpoint version {version} did not contain a state dict")
        self._current_version = version
        return state


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

    metas: dict[int, CheckpointMeta] = {}
    with open(index_path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                meta = _meta_from_payload(payload)
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"invalid checkpoint index line {line_no}") from exc
            metas[meta.version] = meta
    return metas


def _meta_from_payload(payload: dict[str, Any]) -> CheckpointMeta:
    return CheckpointMeta(
        version=int(payload["version"]),
        path=str(payload["path"]),
        sha256=str(payload["sha256"]),
        policy_version=int(payload["policy_version"]),
        created_step=int(payload["created_step"]),
    )


def _checkpoint_path(root: Path, path: str) -> Path:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    return checkpoint_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
