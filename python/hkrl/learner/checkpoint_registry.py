"""Versioned, hash-signed checkpoint registry (PRD §4.2, §9.10).

Stores policy checkpoints by version with a content hash so workers can verify
before loading. Also the policy registry the coordinator/evaluator reference.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from hkrl.learner.checkpoint_payload import validate_checkpoint_payload


@dataclass
class CheckpointMeta:
    version: int
    path: str
    sha256: str
    policy_version: int
    created_step: int


class CheckpointRegistry:
    """Append-only store of checkpoints keyed by version."""

    def __init__(self, root: str) -> None:
        self.root = str(Path(root).expanduser().resolve())
        self._root_path = Path(self.root)
        self._index_path = self._root_path / "index.jsonl"
        self._root_path.mkdir(parents=True, exist_ok=True)
        self._metas: dict[int, CheckpointMeta] = {}
        self._load_index()

    def publish(self, state: dict[str, object], policy_version: int, step: int) -> CheckpointMeta:
        """Persist a checkpoint, compute its hash, return its metadata.

        Checkpoint versions are registry-local and monotonic; ``policy_version``
        records the learner policy version carried by rollout batches.
        """
        validate_checkpoint_payload(state)
        version = self._next_version()
        _validate_checkpoint_numbers(version, policy_version, step)
        filename = f"checkpoint_v{version:06d}.pt"
        path = self._root_path / filename
        torch.save(state, path)
        meta = CheckpointMeta(
            version=version,
            path=filename,
            sha256=_sha256_file(path),
            policy_version=policy_version,
            created_step=step,
        )
        self._append_meta(meta)
        self._metas[version] = meta
        return meta

    def latest(self) -> CheckpointMeta | None:
        if not self._metas:
            return None
        return self._metas[max(self._metas)]

    def get(self, version: int) -> CheckpointMeta:
        try:
            return self._metas[version]
        except KeyError as exc:
            raise KeyError(f"unknown checkpoint version {version}") from exc

    def resolve_path(self, meta_or_path: CheckpointMeta | str) -> Path:
        """Resolve a checkpoint metadata path against this registry root."""
        path = meta_or_path.path if isinstance(meta_or_path, CheckpointMeta) else meta_or_path
        return _checkpoint_path(self._root_path, path)

    def _load_index(self) -> None:
        if not self._index_path.exists():
            return

        with open(self._index_path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                    meta = _meta_from_payload(payload)
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError(f"invalid checkpoint index line {line_no}") from exc
                _checkpoint_path(self._root_path, meta.path)
                if meta.version in self._metas:
                    raise ValueError(f"duplicate checkpoint version {meta.version}")
                self._metas[meta.version] = meta

    def _append_meta(self, meta: CheckpointMeta) -> None:
        with open(self._index_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(meta), sort_keys=True))
            fh.write("\n")

    def _next_version(self) -> int:
        return 1 if not self._metas else max(self._metas) + 1


def _meta_from_payload(payload: dict[str, Any]) -> CheckpointMeta:
    path = str(payload["path"])
    _validate_checkpoint_path(path)
    version = int(payload["version"])
    policy_version = int(payload["policy_version"])
    created_step = int(payload["created_step"])
    sha256 = str(payload["sha256"])
    _validate_checkpoint_numbers(version, policy_version, created_step)
    _validate_sha256(sha256)
    return CheckpointMeta(
        version=version,
        path=path,
        sha256=sha256,
        policy_version=policy_version,
        created_step=created_step,
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


def _validate_checkpoint_path(path: str) -> None:
    if Path(path) == Path("."):
        raise ValueError("checkpoint path must name a file")


def _validate_checkpoint_numbers(version: int, policy_version: int, created_step: int) -> None:
    if version <= 0:
        raise ValueError("checkpoint version must be positive")
    if policy_version < 0:
        raise ValueError("checkpoint policy_version must be non-negative")
    if created_step < 0:
        raise ValueError("checkpoint created_step must be non-negative")


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in value):
        raise ValueError("checkpoint sha256 must be 64 hex characters")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
