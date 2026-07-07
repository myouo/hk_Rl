"""CLI smoke-loop tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hkrl.cli import (
    _build_model,
    _make_env_transport,
    build_argparser,
    run_ppo_training_loop,
    run_random_policy_smoke,
)
from hkrl.learner.checkpoint_registry import CheckpointRegistry
from hkrl.models.recurrent_policy import EntityAttentionRecurrentAC
from hkrl.spaces import make_observation_space
from hkrl.training.recurrent_ppo import RecurrentPPO
from hkrl.utils.config import TaskConfig, TrainConfig
from hkrl.utils.logging import JsonlSink
from hkrl.utils.registry import get


class FakeEnv:
    def __init__(self) -> None:
        self.reset_options: dict[str, Any] | None = None
        self.actions: list[Any] = []

    def reset(self, *, options: dict[str, Any]) -> tuple[dict[str, int], dict[str, Any]]:
        self.reset_options = options
        return {"step": 0}, {"action_mask": np.array([True, False])}

    def step(self, action: Any) -> tuple[dict[str, int], float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        step = len(self.actions)
        terminated = step == 2
        return {"step": step}, 1.5, terminated, False, {"action_mask": np.array([False, True])}


class FakePolicy:
    def __init__(self) -> None:
        self.masks: list[Any] = []

    def act(self, obs: Any, action_mask: Any | None = None) -> dict[str, int]:
        del obs
        self.masks.append(action_mask)
        return {"action": len(self.masks)}


def test_build_argparser_defaults_task_and_requires_config() -> None:
    parser = build_argparser()
    args = parser.parse_args(["--config", "configs/train/ppo_mlp.yaml", "--smoke"])

    assert args.task == "configs/tasks/gruz_mother.yaml"
    assert args.metrics_kind == "jsonl"
    assert args.smoke is True


def test_build_argparser_accepts_csv_metrics_kind() -> None:
    parser = build_argparser()
    args = parser.parse_args(
        [
            "--config",
            "configs/train/ppo_mlp.yaml",
            "--smoke",
            "--metrics",
            "runs/smoke.csv",
            "--metrics-kind",
            "csv",
        ]
    )

    assert args.metrics == "runs/smoke.csv"
    assert args.metrics_kind == "csv"


def test_build_argparser_accepts_env_endpoint_overrides() -> None:
    parser = build_argparser()
    args = parser.parse_args(
        [
            "--config",
            "configs/train/ppo_mlp.yaml",
            "--smoke",
            "--host",
            "127.0.0.2",
            "--port",
            "6000",
        ]
    )

    assert args.host == "127.0.0.2"
    assert args.port == 6000


def test_make_env_transport_uses_cli_endpoint_overrides() -> None:
    cfg = TrainConfig()
    args = build_argparser().parse_args(
        [
            "--config",
            "configs/train/ppo_mlp.yaml",
            "--host",
            "127.0.0.2",
            "--port",
            "6000",
        ]
    )

    transport = _make_env_transport(cfg, args)

    assert transport.host == "127.0.0.2"
    assert transport.port == 6000


def test_make_env_transport_rejects_invalid_endpoint_overrides() -> None:
    cfg = TrainConfig()
    args = build_argparser().parse_args(["--config", "configs/train/ppo_mlp.yaml", "--port", "0"])

    with pytest.raises(ValueError, match="port"):
        _make_env_transport(cfg, args)


def test_training_components_support_recurrent_ppo() -> None:
    cfg = TrainConfig(
        algorithm="recurrent_ppo",
        model={
            "name": "entity_attention_gru",
            "entity_hidden": 8,
            "attention_layers": 1,
            "attention_heads": 2,
            "rnn_hidden": 16,
        },
    )
    task = TaskConfig(task_id="gruz_mother", scene="GG_Gruz_Mother")
    observation_space = make_observation_space(
        max_entities=task.observation.max_entities,
        tier=task.observation.tier,
    )

    model = _build_model(
        cfg,
        {
            "global": observation_space["global"].shape,
            "player": observation_space["player"].shape,
            "entities": observation_space["entities"].shape,
            "entity_mask": observation_space["entity_mask"].shape,
        },
        enable_macro=task.action.enable_macro_actions,
        n_macros=task.action.n_macro_actions,
        max_entities=task.observation.max_entities,
    )
    algo = get("algo", cfg.algorithm)(model=model, config=cfg)

    assert isinstance(model, EntityAttentionRecurrentAC)
    assert isinstance(algo, RecurrentPPO)


def test_run_random_policy_smoke_steps_until_done_and_logs(tmp_path: Path) -> None:
    env = FakeEnv()
    policy = FakePolicy()
    sink = JsonlSink(tmp_path / "smoke.jsonl")

    summary = run_random_policy_smoke(
        env=env,
        policy=policy,
        sink=sink,
        task_id="gruz_mother",
        max_steps=4,
        reset_timeout_s=2.0,
    )
    sink.close()

    assert env.reset_options == {"reset_timeout_s": 2.0}
    assert env.actions == [{"action": 1}, {"action": 2}]
    np.testing.assert_array_equal(policy.masks[0], np.array([True, False]))
    np.testing.assert_array_equal(policy.masks[1], np.array([False, True]))
    assert summary == {
        "task_id": "gruz_mother",
        "steps": 2,
        "total_reward": 3.0,
        "terminated": True,
        "truncated": False,
    }
    assert (tmp_path / "smoke.jsonl").read_text(encoding="utf-8").count("\n") == 3


def test_run_random_policy_smoke_rejects_non_positive_steps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_steps"):
        run_random_policy_smoke(
            env=FakeEnv(),
            policy=FakePolicy(),
            sink=JsonlSink(tmp_path / "smoke.jsonl"),
            task_id="gruz_mother",
            max_steps=0,
        )


def test_run_ppo_training_loop_collects_updates_and_logs() -> None:
    worker = FakeWorker()
    algo = FakeAlgo()
    sink = MemorySink()

    summary = run_ppo_training_loop(worker=worker, algo=algo, sink=sink, updates=2)

    assert worker.collect_calls == 2
    assert worker.collected_versions == [0, 1]
    assert worker.policy_version == 2
    assert algo.buffers == [worker.buffer, worker.buffer]
    assert summary == {
        "updates": 2,
        "total_steps": 6,
        "last_metrics": {"policy_loss": -2.0, "value_loss": 0.5},
        "last_checkpoint": None,
        "last_checkpoint_version": None,
        "policy_version": 2,
    }
    assert sink.scalars == [
        ("policy_loss", -1.0, 1),
        ("value_loss", 0.5, 1),
        ("policy_loss", -2.0, 2),
        ("value_loss", 0.5, 2),
    ]
    assert len(sink.episodes) == 2
    assert sink.flushed


def test_run_ppo_training_loop_rejects_non_positive_updates() -> None:
    with pytest.raises(ValueError, match="updates"):
        run_ppo_training_loop(worker=FakeWorker(), algo=FakeAlgo(), sink=MemorySink(), updates=0)


def test_run_ppo_training_loop_publishes_registry_checkpoint(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    summary = run_ppo_training_loop(
        worker=FakeWorker(),
        algo=FakeAlgo(),
        sink=MemorySink(),
        updates=1,
        checkpoint_dir=tmp_path,
        model=model,
    )

    assert summary["policy_version"] == 1
    assert summary["last_checkpoint_version"] == 1
    assert summary["last_checkpoint"] is not None
    registry = CheckpointRegistry(str(tmp_path))
    meta = registry.latest()
    assert meta is not None
    assert meta.version == 1
    assert meta.policy_version == 1
    assert meta.created_step == 3

    checkpoint = torch.load(summary["last_checkpoint"], map_location="cpu", weights_only=True)
    assert checkpoint["policy_version"] == 1
    assert checkpoint["update"] == 1
    assert checkpoint["step"] == 3
    assert checkpoint["metrics"] == {"policy_loss": -1.0, "value_loss": 0.5}
    assert "model_state_dict" in checkpoint


class FakeBatch:
    def __init__(self, policy_version: int = 7) -> None:
        self.rewards = np.ones((3, 1), dtype=np.float32)
        self.policy_version = policy_version


class FakeWorker:
    def __init__(self) -> None:
        self.buffer = object()
        self.collect_calls = 0
        self.collected_versions: list[int] = []
        self.policy_version = 0

    def collect_rollout(self) -> FakeBatch:
        self.collect_calls += 1
        self.collected_versions.append(self.policy_version)
        return FakeBatch(policy_version=self.policy_version)


class FakeAlgo:
    def __init__(self) -> None:
        self.buffers: list[object] = []

    def update(self, buffer: object) -> dict[str, float]:
        self.buffers.append(buffer)
        return {"policy_loss": -float(len(self.buffers)), "value_loss": 0.5}


class MemorySink:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.episodes: list[dict[str, Any]] = []
        self.flushed = False

    def log_scalar(self, key: str, value: float, step: int) -> None:
        self.scalars.append((key, value, step))

    def log_episode(self, record: dict[str, Any]) -> None:
        self.episodes.append(record)

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        pass
