"""Checkpoint client tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hkrl.learner.checkpoint_registry import CheckpointRegistry
from hkrl.worker.checkpoint_client import CheckpointClient


def test_checkpoint_client_pulls_verified_local_checkpoint(tmp_path: Path) -> None:
    registry = CheckpointRegistry(str(tmp_path))
    registry.publish({"model_state_dict": {"weight": torch.tensor([3.0])}}, 5, 10)
    meta = registry.publish({"model_state_dict": {"weight": torch.tensor([4.0])}}, 6, 20)
    client = CheckpointClient(str(tmp_path))

    assert client.latest_version() == meta.version
    state = client.pull(meta.version)

    assert client.current_version == meta.version
    torch.testing.assert_close(state["model_state_dict"]["weight"], torch.tensor([4.0]))


def test_checkpoint_client_accepts_file_url(tmp_path: Path) -> None:
    registry = CheckpointRegistry(str(tmp_path))
    meta = registry.publish({"model_state_dict": {"weight": torch.tensor([1.0])}}, 1, 1)

    client = CheckpointClient(tmp_path.as_uri())

    assert client.latest_version() == meta.version


def test_checkpoint_client_rejects_hash_mismatch(tmp_path: Path) -> None:
    registry = CheckpointRegistry(str(tmp_path))
    meta = registry.publish({"model_state_dict": {"weight": torch.tensor([1.0])}}, 1, 1)
    with open(meta.path, "ab") as fh:
        fh.write(b"corruption")
    client = CheckpointClient(str(tmp_path))

    with pytest.raises(ValueError, match="sha256 mismatch"):
        client.pull(meta.version)


def test_checkpoint_client_rejects_unknown_version(tmp_path: Path) -> None:
    client = CheckpointClient(str(tmp_path))

    with pytest.raises(KeyError, match="unknown checkpoint version"):
        client.pull(99)


def test_checkpoint_client_rejects_unsupported_endpoint() -> None:
    with pytest.raises(ValueError, match="unsupported checkpoint registry endpoint"):
        CheckpointClient("https://example.invalid/checkpoints")
