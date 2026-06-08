"""Versioned, hash-signed checkpoint registry (PRD §4.2, §9.10).

Stores policy checkpoints by version with a content hash so workers can verify
before loading. Also the policy registry the coordinator/evaluator reference.
"""

from __future__ import annotations

from dataclasses import dataclass


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
        self.root = root
        # TODO(phase-6): index file under root; load existing metas.

    def publish(self, state: dict[str, object], policy_version: int, step: int) -> CheckpointMeta:
        """Persist a checkpoint, compute its hash, return its metadata.

        TODO(phase-6): torch.save, sha256, append to index.
        """
        raise NotImplementedError

    def latest(self) -> CheckpointMeta | None:
        raise NotImplementedError  # TODO(phase-6)

    def get(self, version: int) -> CheckpointMeta:
        raise NotImplementedError  # TODO(phase-6)
