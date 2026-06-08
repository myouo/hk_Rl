"""Checkpoint pulling + verification on the worker (PRD §9.10).

Polls the learner's checkpoint registry, downloads new weights, and hash-verifies
before loading (never load an unverified checkpoint).
"""

from __future__ import annotations


class CheckpointClient:
    """Pulls and verifies policy checkpoints for hot-swapping."""

    def __init__(self, registry_endpoint: str, verify_hash: bool = True) -> None:
        self.registry_endpoint = registry_endpoint
        self.verify_hash = verify_hash
        self._current_version = -1

    def latest_version(self) -> int:
        """Return the newest available checkpoint version (or -1).

        TODO(phase-6): query registry endpoint.
        """
        raise NotImplementedError

    def pull(self, version: int) -> dict[str, object]:
        """Download + hash-verify a checkpoint; return a loadable state dict.

        TODO(phase-6): fetch bytes, verify signature/hash, torch.load.
        """
        raise NotImplementedError
