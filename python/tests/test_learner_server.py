"""Learner server tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from hkrl.learner.checkpoint_registry import CheckpointRegistry
from hkrl.learner.learner_server import LearnerServer
from hkrl.models.mlp import MlpActorCritic
from hkrl.spaces import N_AIM_Y, N_BUTTONS, N_DURATION, N_MOVEMENT_X
from hkrl.training.rollout_buffer import RolloutBatch, RolloutBuffer
from hkrl.utils.config import TrainConfig
from hkrl.utils.live_tuning import LiveTuning, atomic_write_json


def test_learner_server_submits_updates_and_publishes_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(123)
    model = MlpActorCritic(_obs_spec(), hidden=16, enable_macro=False)
    cfg = TrainConfig(
        algorithm="appo",
        epochs=1,
        minibatch_size=2,
        learning_rate=1.0e-2,
        entropy_coef=0.0,
    )
    registry = CheckpointRegistry(str(tmp_path))
    server = LearnerServer(
        model=model,
        config=cfg,
        registry=registry,
        bind="127.0.0.1:0",
        publish_every_updates=1,
    )
    batch = _rollout_batch(model, policy_version=0)

    assert server.submit(batch)
    metrics = server.update_once()

    assert server.accepted_batches == 1
    assert server.rejected_batches == 0
    assert server.policy_version == 1
    assert metrics["policy_version"] == 1.0
    assert registry.latest() is not None
    assert server.last_checkpoint == registry.latest()
    checkpoint = torch.load(
        registry.resolve_path(server.last_checkpoint),
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["policy_version"] == 1
    assert checkpoint["update"] == 1


def test_learner_server_rejects_stale_batch(tmp_path: Path) -> None:
    model = MlpActorCritic(_obs_spec(), hidden=16, enable_macro=False)
    server = LearnerServer(
        model=model,
        config=TrainConfig(algorithm="appo"),
        registry=CheckpointRegistry(str(tmp_path)),
        max_staleness=1,
    )
    server.policy_version = 3

    assert not server.submit(_rollout_batch(model, policy_version=1))
    assert server.rejected_batches == 1


def test_learner_server_serve_drains_queued_batch(tmp_path: Path) -> None:
    model = MlpActorCritic(_obs_spec(), hidden=16, enable_macro=False)
    server = LearnerServer(
        model=model,
        config=TrainConfig(algorithm="appo", epochs=1, minibatch_size=2),
        registry=CheckpointRegistry(str(tmp_path)),
    )

    assert server.submit(_rollout_batch(model, policy_version=0))
    server.serve()

    assert server.update_count == 1
    assert server.last_checkpoint is not None


def test_learner_server_aggregates_worker_batches_before_update(tmp_path: Path) -> None:
    model = MlpActorCritic(_obs_spec(), hidden=16, enable_macro=False)
    server = LearnerServer(
        model=model,
        config=TrainConfig(algorithm="appo", epochs=1, minibatch_size=4),
        registry=CheckpointRegistry(str(tmp_path)),
        batches_per_update=2,
    )

    assert server.submit(_rollout_batch(model, policy_version=0))
    assert not server.ready_to_update
    assert server.serve() is False
    assert server.update_count == 0

    assert server.submit(_rollout_batch(model, policy_version=0))
    assert server.ready_to_update
    assert server.serve() is True
    assert server.update_count == 1
    assert server.last_metrics["samples"] == 8.0


def test_learner_server_force_flushes_partial_batch_group(tmp_path: Path) -> None:
    model = MlpActorCritic(_obs_spec(), hidden=16, enable_macro=False)
    server = LearnerServer(
        model=model,
        config=TrainConfig(algorithm="appo", epochs=1, minibatch_size=2),
        registry=CheckpointRegistry(str(tmp_path)),
        batches_per_update=4,
    )

    assert server.submit(_rollout_batch(model, policy_version=0))
    assert server.serve(force=True) is True
    assert server.update_count == 1


def test_learner_server_applies_tuning_after_flushing_old_partial_queue(
    tmp_path: Path,
) -> None:
    model = MlpActorCritic(_obs_spec(), hidden=16, enable_macro=False)
    server = LearnerServer(
        model=model,
        config=TrainConfig(
            algorithm="appo",
            epochs=1,
            minibatch_size=2,
            learning_rate=3.0e-4,
        ),
        registry=CheckpointRegistry(str(tmp_path)),
        batches_per_update=4,
    )
    assert server.submit(_rollout_batch(model, policy_version=0))
    tuning = LiveTuning(
        version=1,
        learner={"learning_rate": 1.0e-4, "entropy_coef": 0.02},
        reward={"boss_damage": 1.0},
    )
    atomic_write_json(server.live_tuning_path, tuning.checkpoint_payload())

    status = server.reconcile_live_tuning()

    assert status is not None
    assert status["flushed_update"] is True
    assert server.update_count == 1
    assert server.tuning_version == 1
    assert server.cfg.learning_rate == 1.0e-4
    assert server.cfg.entropy_coef == 0.02
    assert all(group["lr"] == 1.0e-4 for group in server.algo.optimizer.param_groups)
    assert server.last_checkpoint is not None
    checkpoint = torch.load(
        server.registry.resolve_path(server.last_checkpoint),
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["tuning_version"] == 1
    assert checkpoint["live_tuning"]["reward"]["boss_damage"] == 1.0

    stale = _rollout_batch(model, policy_version=server.policy_version)
    assert stale.tuning_version == 0
    assert not server.submit(stale)
    current = _rollout_batch(model, policy_version=server.policy_version)
    current.tuning_version = 1
    assert server.submit(current)


def test_learner_server_recovers_missing_tuning_status_from_checkpoint(
    tmp_path: Path,
) -> None:
    model = MlpActorCritic(_obs_spec(), hidden=16, enable_macro=False)
    server = LearnerServer(
        model=model,
        config=TrainConfig(algorithm="ppo"),
        registry=CheckpointRegistry(str(tmp_path)),
    )
    tuning = LiveTuning(version=3, reward={"boss_damage": 1.0})
    assert server.restore_live_tuning(tuning.checkpoint_payload())
    server.last_checkpoint = server.registry.publish(
        server.checkpoint_payload(),
        policy_version=server.policy_version,
        step=server.update_count,
    )

    status = server.ensure_live_tuning_status()

    assert status is not None
    assert status["recovered_after_restart"] is True
    assert status["tuning_version"] == 3
    assert json.loads(server.live_tuning_status_path.read_text())["checkpoint_version"] == 1
    assert server.ensure_live_tuning_status() is None


def _rollout_batch(model: MlpActorCritic, policy_version: int) -> RolloutBatch:
    buffer = RolloutBuffer(
        capacity=4,
        num_envs=1,
        obs_spec={
            **_obs_spec(),
            "action": (12,),
            "action_mask": (_mask_dim(),),
        },
    )
    action_mask = np.ones((_mask_dim(),), dtype=bool)
    for step in range(4):
        obs = _numpy_obs(step)
        with torch.no_grad():
            action, log_prob, value, _ = model.act(
                _torch_obs(obs),
                action_mask=torch.as_tensor(action_mask[None, :]),
            )
        buffer.add(
            obs=obs,
            action=action.numpy(),
            log_prob=log_prob.numpy(),
            value=value.numpy(),
            reward=np.array([1.0 + step], dtype=np.float32),
            done=np.array([step == 3]),
            truncated=np.array([False]),
            action_mask=action_mask,
        )
    buffer.compute_returns(
        last_value=np.array([0.0], dtype=np.float32),
        gamma=0.99,
        gae_lambda=0.95,
    )
    return buffer.to_batch(policy_version=policy_version)


def _obs_spec() -> dict[str, tuple[int, ...]]:
    return {
        "global": (2,),
        "player": (3,),
        "entities": (4, 5),
        "entity_mask": (4,),
    }


def _numpy_obs(step: int) -> dict[str, np.ndarray]:
    return {
        "global": np.array([step, step + 0.5], dtype=np.float32),
        "player": np.ones((3,), dtype=np.float32) * step,
        "entities": np.ones((4, 5), dtype=np.float32) * (step + 1),
        "entity_mask": np.array([True, True, False, False]),
    }


def _torch_obs(obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        "global": torch.as_tensor(obs["global"][None, :]),
        "player": torch.as_tensor(obs["player"][None, :]),
        "entities": torch.as_tensor(obs["entities"][None, :, :]),
        "entity_mask": torch.as_tensor(obs["entity_mask"][None, :]),
    }


def _mask_dim() -> int:
    return N_MOVEMENT_X + N_AIM_Y + N_BUTTONS + N_DURATION
