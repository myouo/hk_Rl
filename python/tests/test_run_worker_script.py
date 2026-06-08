"""run_worker script tests."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

from hkrl.learner.checkpoint_registry import CheckpointRegistry


def test_run_worker_dry_run_builds_summary(tmp_path: Path) -> None:
    registry = CheckpointRegistry(str(tmp_path / "checkpoints"))
    registry.publish({"model_state_dict": {}, "policy_version": 3}, policy_version=3, step=1)
    module = _load_script("run_worker.py")
    root = Path(__file__).parents[2]
    args = argparse.Namespace(
        config=str(root / "configs/train/remote_learner.yaml"),
        task=str(root / "configs/tasks/gruz_mother.yaml"),
        learner="127.0.0.1:5600",
        registry=str(tmp_path / "checkpoints"),
        steps=None,
        dry_run=True,
    )

    summary = module.run_from_args(args)

    assert summary == {
        "algorithm": "appo",
        "dry_run": True,
        "learner": "127.0.0.1:5600",
        "latest_checkpoint": 1,
        "model": "entity_attention_gru",
        "registry": str(tmp_path / "checkpoints"),
        "task_id": "gruz_mother",
    }


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
