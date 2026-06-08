"""run_learner script tests."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from hkrl.spaces import action_mask_layout, make_observation_space
from hkrl.training.batch_io import save_rollout_batch
from hkrl.training.rollout_buffer import RolloutBatch


def test_run_learner_builds_server_summary(tmp_path: Path) -> None:
    module = _load_script("run_learner.py")
    args = argparse.Namespace(
        config=str(Path(__file__).parents[2] / "configs/train/remote_learner.yaml"),
        bind="127.0.0.1:0",
        batch_dir=None,
        checkpoint_dir=str(tmp_path),
        max_staleness=2,
        publish_every_updates=1,
        max_entities=4,
        disable_macro_actions=False,
        n_macro_actions=11,
        task=None,
        tasks=None,
        tier="privileged",
    )

    summary = module.run_from_args(args)

    assert summary["algorithm"] == "appo"
    assert summary["accepted_batches"] == 0
    assert summary["batch_dir"] is None
    assert summary["bind"] == "127.0.0.1:0"
    assert summary["checkpoint_dir"] == str(tmp_path.resolve())
    assert summary["enable_macro_actions"] is True
    assert summary["latest_checkpoint"] is None
    assert summary["max_entities"] == 4
    assert summary["max_staleness"] == 2
    assert summary["model"] == "entity_attention_gru"
    assert summary["n_macro_actions"] == 11
    assert summary["publish_every_updates"] == 1
    assert summary["policy_version"] == 0
    assert summary["queued_batches"] == 0
    assert summary["rejected_batches"] == 0
    assert summary["submitted_batches"] == 0
    assert summary["task_ids"] == []
    assert summary["tier"] == "privileged"


def test_run_learner_ingests_batch_dir_and_updates(tmp_path: Path) -> None:
    module = _load_script("run_learner.py")
    config = tmp_path / "appo_mlp.yaml"
    config.write_text(
        "\n".join(
            [
                "algorithm: appo",
                "epochs: 1",
                "minibatch_size: 2",
                "learning_rate: 0.001",
                "entropy_coef: 0.0",
                "model:",
                "  name: mlp",
                "  rnn_hidden: 16",
            ]
        ),
        encoding="utf-8",
    )
    batch_dir = tmp_path / "batches"
    save_rollout_batch(batch_dir / "worker_00000001_v000000.npz", _learner_batch())
    args = argparse.Namespace(
        config=str(config),
        bind="127.0.0.1:0",
        batch_dir=str(batch_dir),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        max_staleness=2,
        publish_every_updates=1,
        max_entities=4,
        disable_macro_actions=False,
        n_macro_actions=11,
        task=None,
        tasks=None,
        tier="privileged",
    )

    summary = module.run_from_args(args)

    assert summary["accepted_batches"] == 1
    assert summary["batch_dir"] == str(batch_dir)
    assert summary["latest_checkpoint"] == 1
    assert summary["policy_version"] == 1
    assert summary["queued_batches"] == 0
    assert summary["rejected_batches"] == 0
    assert summary["submitted_batches"] == 1


def test_run_learner_uses_nested_config_defaults(tmp_path: Path) -> None:
    module = _load_script("run_learner.py")
    config = tmp_path / "remote.yaml"
    checkpoint_dir = tmp_path / "configured-checkpoints"
    config.write_text(
        "\n".join(
            [
                "algorithm: appo",
                "minibatch_size: 2",
                "model:",
                "  name: mlp",
                "  rnn_hidden: 16",
                "learner:",
                "  bind: 127.0.0.1:9999",
                "  max_staleness: 6",
                f"  checkpoint_dir: {checkpoint_dir}",
                "  publish_every_updates: 3",
            ]
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config=str(config),
        bind=None,
        batch_dir=None,
        checkpoint_dir=None,
        max_staleness=None,
        publish_every_updates=None,
        max_entities=4,
        disable_macro_actions=False,
        n_macro_actions=11,
        task=None,
        tasks=None,
        tier="privileged",
    )

    summary = module.run_from_args(args)

    assert summary["bind"] == "127.0.0.1:9999"
    assert summary["checkpoint_dir"] == str(checkpoint_dir.resolve())
    assert summary["max_staleness"] == 6
    assert summary["publish_every_updates"] == 3


def test_run_learner_rejects_wildcard_bind_for_localhost_scope(tmp_path: Path) -> None:
    module = _load_script("run_learner.py")
    config = tmp_path / "localhost.yaml"
    config.write_text(
        "\n".join(
            [
                "algorithm: appo",
                "minibatch_size: 2",
                "model:",
                "  name: mlp",
                "  rnn_hidden: 16",
                "learner:",
                "  bind: 0.0.0.0:5600",
                "security:",
                "  bind_scope: localhost",
            ]
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config=str(config),
        bind=None,
        batch_dir=None,
        checkpoint_dir=str(tmp_path),
        max_staleness=2,
        publish_every_updates=1,
        max_entities=4,
        disable_macro_actions=False,
        n_macro_actions=11,
        task=None,
        tasks=None,
        tier="privileged",
    )

    with pytest.raises(ValueError, match="loopback"):
        module.run_from_args(args)


def test_run_learner_infers_layout_from_task_configs(tmp_path: Path) -> None:
    module = _load_script("run_learner.py")
    root = Path(__file__).parents[2]
    args = argparse.Namespace(
        config=str(root / "configs/train/remote_learner.yaml"),
        bind="127.0.0.1:0",
        batch_dir=None,
        checkpoint_dir=str(tmp_path),
        max_staleness=2,
        publish_every_updates=1,
        max_entities=None,
        disable_macro_actions=False,
        n_macro_actions=None,
        task=None,
        tasks=[
            str(root / "configs/tasks/gruz_mother.yaml"),
            str(root / "configs/tasks/hornet_protector.yaml"),
        ],
        tier=None,
    )

    summary = module.run_from_args(args)

    assert summary["enable_macro_actions"] is True
    assert summary["max_entities"] == 64
    assert summary["n_macro_actions"] == 11
    assert summary["task_ids"] == ["gruz_mother", "hornet_protector_attuned"]
    assert summary["tier"] == "privileged"


def test_run_learner_rejects_incompatible_task_layouts() -> None:
    module = _load_script("run_learner.py")
    tasks = [
        module.TaskConfig(task_id="a", wire_id=1, scene="A", action={"n_macro_actions": 11}),
        module.TaskConfig(task_id="b", wire_id=2, scene="B", action={"n_macro_actions": 4}),
    ]

    with pytest.raises(ValueError, match="n_macro_actions"):
        module._validate_task_layouts(tasks)


def test_run_learner_mlp_model_uses_default_hidden_when_rnn_hidden_zero() -> None:
    module = _load_script("run_learner.py")
    cfg = module.load_train_config(Path(__file__).parents[2] / "configs/train/ppo_mlp.yaml")
    model = module._build_model(
        cfg,
        {
            "global": (2,),
            "player": (3,),
            "entities": (4, 5),
            "entity_mask": (4,),
        },
        max_entities=4,
        enable_macro=True,
        n_macros=11,
    )

    assert model.trunk[0].out_features == 256


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _learner_batch() -> RolloutBatch:
    observation_space = make_observation_space(max_entities=4, tier="privileged")
    time_steps = 4
    action_dim = 13
    mask_dim = len(action_mask_layout(enable_macro=True))
    actions = np.zeros((time_steps, 1, action_dim), dtype=np.int64)
    actions[:, :, 0] = np.arange(time_steps, dtype=np.int64).reshape(time_steps, 1) % 3
    actions[:, :, 1] = 1
    actions[:, :, 11] = 1
    actions[:, :, 12] = 0

    return RolloutBatch(
        obs_global=np.zeros((time_steps, 1, *observation_space["global"].shape), dtype=np.float32),
        obs_player=np.zeros((time_steps, 1, *observation_space["player"].shape), dtype=np.float32),
        obs_entities=np.zeros(
            (time_steps, 1, *observation_space["entities"].shape),
            dtype=np.float32,
        ),
        entity_mask=np.ones((time_steps, 1, *observation_space["entity_mask"].shape), dtype=bool),
        actions=actions,
        log_probs=np.full((time_steps, 1), -1.0, dtype=np.float32),
        values=np.zeros((time_steps, 1), dtype=np.float32),
        advantages=np.ones((time_steps, 1), dtype=np.float32),
        returns=np.ones((time_steps, 1), dtype=np.float32),
        rewards=np.ones((time_steps, 1), dtype=np.float32),
        dones=np.array([[False], [False], [False], [True]]),
        truncateds=np.zeros((time_steps, 1), dtype=bool),
        action_masks=np.ones((time_steps, 1, mask_dim), dtype=bool),
        prev_actions=np.zeros((time_steps, 1, action_dim), dtype=np.int64),
        rnn_states=None,
        episode_ids=np.ones((time_steps, 1), dtype=np.uint64),
        task_ids=np.ones((time_steps, 1), dtype=np.int64),
        policy_version=0,
    )
