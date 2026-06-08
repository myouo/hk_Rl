"""run_worker script tests."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from hkrl.learner.checkpoint_registry import CheckpointRegistry
from hkrl.training.batch_io import load_rollout_batch
from hkrl.training.rollout_buffer import RolloutBatch


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
        batch_dir=str(tmp_path / "batches"),
        worker_id="game-pc-1",
        steps=None,
        max_consecutive_failures=5,
        dry_run=True,
    )

    summary = module.run_from_args(args)

    assert summary == {
        "algorithm": "appo",
        "batch_dir": str(tmp_path / "batches"),
        "dry_run": True,
        "learner": "127.0.0.1:5600",
        "latest_checkpoint": 1,
        "max_consecutive_failures": 5,
        "model": "entity_attention_gru",
        "registry": str(tmp_path / "checkpoints"),
        "task_id": "gruz_mother",
        "worker_id": "game-pc-1",
    }


def test_run_worker_batch_spooler_writes_rollout_npz(tmp_path: Path) -> None:
    module = _load_script("run_worker.py")
    written: list[str] = []
    uploader = module._make_batch_uploader(str(tmp_path), "game/pc:1", written)
    assert uploader is not None

    uploader(_sample_batch(policy_version=4))

    assert len(written) == 1
    path = Path(written[0])
    assert path.name == "game_pc_1_00000001_v000004.npz"
    loaded = load_rollout_batch(path)
    assert loaded.policy_version == 4
    np.testing.assert_array_equal(loaded.rewards, np.array([[1.0]], dtype=np.float32))


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_batch(policy_version: int) -> RolloutBatch:
    return RolloutBatch(
        obs_global=np.zeros((1, 1, 2), dtype=np.float32),
        obs_player=np.zeros((1, 1, 3), dtype=np.float32),
        obs_entities=np.zeros((1, 1, 4, 5), dtype=np.float32),
        entity_mask=np.ones((1, 1, 4), dtype=bool),
        actions=np.zeros((1, 1, 2), dtype=np.int64),
        log_probs=np.zeros((1, 1), dtype=np.float32),
        values=np.zeros((1, 1), dtype=np.float32),
        advantages=np.ones((1, 1), dtype=np.float32),
        returns=np.ones((1, 1), dtype=np.float32),
        rewards=np.ones((1, 1), dtype=np.float32),
        dones=np.zeros((1, 1), dtype=bool),
        truncateds=np.zeros((1, 1), dtype=bool),
        action_masks=np.ones((1, 1, 6), dtype=bool),
        prev_actions=np.zeros((1, 1, 2), dtype=np.int64),
        rnn_states=None,
        episode_ids=np.ones((1, 1), dtype=np.uint64),
        task_ids=np.ones((1, 1), dtype=np.int64),
        policy_version=policy_version,
    )
