"""run_eval script tests."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from hkrl.learner.checkpoint_registry import CheckpointRegistry
from hkrl.models.mlp import MlpActorCritic
from hkrl.spaces import make_observation_space
from hkrl.utils.config import TaskConfig


def test_run_eval_builds_mlp_policy_from_checkpoint_registry(tmp_path: Path) -> None:
    module = _load_script("run_eval.py")
    root = Path(__file__).parents[2]
    task = TaskConfig(task_id="gruz_mother", scene="GG_Gruz_Mother")
    model = _mlp_for_task(task)
    registry = CheckpointRegistry(str(tmp_path / "checkpoints"))
    registry.publish(
        {"model_state_dict": model.state_dict(), "policy_version": 3},
        policy_version=3,
        step=12,
    )
    args = argparse.Namespace(
        policy="mlp",
        checkpoint=None,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        train_config=str(root / "configs/train/ppo_mlp.yaml"),
    )

    policy = module._build_policy(args, task)

    assert isinstance(policy, MlpActorCritic)


def test_run_eval_resolves_checkpoint_directory_argument(tmp_path: Path) -> None:
    module = _load_script("run_eval.py")
    registry = CheckpointRegistry(str(tmp_path))
    meta = registry.publish({"model_state_dict": {}}, policy_version=1, step=1)
    args = argparse.Namespace(checkpoint=str(tmp_path), checkpoint_dir=None)

    assert module._resolve_checkpoint_path(args) == Path(meta.path)


def test_run_eval_requires_checkpoint_for_mlp_policy() -> None:
    module = _load_script("run_eval.py")
    args = argparse.Namespace(checkpoint=None, checkpoint_dir=None)

    with pytest.raises(SystemExit, match="--checkpoint or --checkpoint-dir"):
        module._resolve_checkpoint_path(args)


def _mlp_for_task(task: TaskConfig) -> MlpActorCritic:
    observation_space = make_observation_space(
        max_entities=task.observation.max_entities,
        tier=task.observation.tier,
    )
    return MlpActorCritic(
        {
            "global": observation_space["global"].shape,
            "player": observation_space["player"].shape,
            "entities": observation_space["entities"].shape,
            "entity_mask": observation_space["entity_mask"].shape,
        },
        hidden=256,
        enable_macro=task.action.enable_macro_actions,
        n_macros=task.action.n_macro_actions,
    )


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
