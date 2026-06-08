"""Checkpoint registry tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from hkrl.learner.checkpoint_registry import CheckpointRegistry


def test_checkpoint_registry_publish_latest_get_and_reload(tmp_path: Path) -> None:
    registry = CheckpointRegistry(str(tmp_path))
    first = registry.publish(
        {"model_state_dict": {"weight": torch.tensor([1.0])}},
        policy_version=10,
        step=123,
    )
    second = registry.publish(
        {"model_state_dict": {"weight": torch.tensor([2.0])}},
        policy_version=11,
        step=456,
    )

    assert first.version == 1
    assert second.version == 2
    assert registry.latest() == second
    assert registry.get(1) == first
    assert first.path == "checkpoint_v000001.pt"
    first_path = registry.resolve_path(first)
    assert first_path.exists()
    assert first.sha256 == _sha256(first_path)

    loaded = torch.load(first_path, map_location="cpu", weights_only=True)
    torch.testing.assert_close(loaded["model_state_dict"]["weight"], torch.tensor([1.0]))

    reloaded = CheckpointRegistry(str(tmp_path))
    assert reloaded.latest() == second
    assert reloaded.get(1) == first


def test_checkpoint_registry_get_rejects_unknown_version(tmp_path: Path) -> None:
    registry = CheckpointRegistry(str(tmp_path))

    with pytest.raises(KeyError, match="unknown checkpoint version"):
        registry.get(99)


def test_checkpoint_registry_rejects_invalid_index(tmp_path: Path) -> None:
    (tmp_path / "index.jsonl").write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid checkpoint index line"):
        CheckpointRegistry(str(tmp_path))


def test_checkpoint_registry_accepts_legacy_absolute_paths_inside_root(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint_v000001.pt"
    torch.save({"model_state_dict": {}}, path)
    payload = {
        "version": 1,
        "path": str(path),
        "sha256": _sha256(path),
        "policy_version": 1,
        "created_step": 10,
    }
    (tmp_path / "index.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    registry = CheckpointRegistry(str(tmp_path))
    latest = registry.latest()

    assert latest is not None
    assert registry.resolve_path(latest) == path


def test_checkpoint_registry_rejects_paths_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    root.mkdir()
    outside = tmp_path / "outside.pt"
    payload = {
        "version": 1,
        "path": str(outside),
        "sha256": "0" * 64,
        "policy_version": 1,
        "created_step": 10,
    }
    (root / "index.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint path escapes registry root"):
        CheckpointRegistry(str(root))


def test_checkpoint_registry_rejects_empty_checkpoint_path(tmp_path: Path) -> None:
    payload = {
        "version": 1,
        "path": "",
        "sha256": "0" * 64,
        "policy_version": 1,
        "created_step": 10,
    }
    (tmp_path / "index.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid checkpoint index line"):
        CheckpointRegistry(str(tmp_path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        digest.update(fh.read())
    return digest.hexdigest()
