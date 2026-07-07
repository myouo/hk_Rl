"""RolloutBatch TCP intake tests."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hkrl.learner.batch_intake import (
    BATCH_INTAKE_TYPE,
    BatchIntakeClient,
    BatchIntakeResult,
    BatchIntakeServer,
    _accepted_from_ack,
    _validate_header,
)
from hkrl.learner.checkpoint_registry import CheckpointRegistry
from hkrl.learner.learner_server import LearnerServer
from hkrl.models.mlp import MlpActorCritic
from hkrl.models.recurrent_policy import EntityAttentionRecurrentAC
from hkrl.spaces import (
    N_AIM_Y,
    N_BUTTONS,
    N_DURATION,
    N_MOVEMENT_X,
    make_action_space,
    make_observation_space,
)
from hkrl.training.rollout_buffer import RolloutBatch, RolloutBuffer
from hkrl.utils.config import TrainConfig
from hkrl.worker.game_worker import GameWorker


def test_batch_intake_client_submits_to_learner_server(tmp_path: Path) -> None:
    model = MlpActorCritic(_obs_spec(), hidden=16, enable_macro=False)
    server = LearnerServer(
        model=model,
        config=TrainConfig(algorithm="appo", epochs=1, minibatch_size=2),
        registry=CheckpointRegistry(str(tmp_path)),
        bind="127.0.0.1:0",
    )
    results: list[BatchIntakeResult] = []
    errors: list[BaseException] = []

    with BatchIntakeServer(
        server,
        "127.0.0.1:0",
        auth_token="secret",
        timeout_s=3.0,
    ) as intake:
        assert intake.endpoint is not None
        thread = threading.Thread(target=_serve_once, args=(intake, results, errors))
        thread.start()

        accepted = BatchIntakeClient(
            intake.endpoint,
            auth_token="secret",
            timeout_s=3.0,
        ).submit(_rollout_batch(model, policy_version=0))

        thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert errors == []
    assert accepted is True
    assert server.accepted_batches == 1
    assert server.rejected_batches == 0
    assert len(results) == 1
    assert results[0].accepted is True


def test_recurrent_game_worker_uploads_tcp_batch_for_learner_update(
    tmp_path: Path,
) -> None:
    torch.manual_seed(123)
    env = _WorkerIntakeEnv()
    obs_dims = _env_obs_dims(env)
    worker_model = _recurrent_model(obs_dims)
    learner_model = _recurrent_model(obs_dims)
    learner_model.load_state_dict(worker_model.state_dict())
    cfg = TrainConfig(
        algorithm="appo",
        rollout_steps=4,
        epochs=1,
        minibatch_size=2,
        entropy_coef=0.0,
    )
    server = LearnerServer(
        model=learner_model,
        config=cfg,
        registry=CheckpointRegistry(str(tmp_path)),
        bind="127.0.0.1:0",
        publish_every_updates=1,
    )
    results: list[BatchIntakeResult] = []
    errors: list[BaseException] = []

    with BatchIntakeServer(server, "127.0.0.1:0", timeout_s=3.0) as intake:
        assert intake.endpoint is not None
        client = BatchIntakeClient(intake.endpoint, timeout_s=3.0)
        thread = threading.Thread(target=_serve_once, args=(intake, results, errors))
        thread.start()
        worker = GameWorker(
            env=env,
            model=worker_model,
            config=cfg,
            learner_endpoint=intake.endpoint,
            batch_uploader=client.submit,
        )

        worker.run(total_steps=4)
        thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert errors == []
    assert len(results) == 1
    assert results[0].accepted is True
    assert server.accepted_batches == 1
    assert server.rejected_batches == 0
    assert worker.last_batch is not None
    assert worker.last_batch.rnn_states is not None
    assert worker.last_batch.rnn_states.shape == (4, 1, 1, 16)
    assert worker.learner_upload_submitted_batches == 1
    assert worker.learner_upload_accepted_batches == 1
    assert worker.learner_upload_failed_batches == 0

    server.serve()

    latest = server.registry.latest()
    assert latest is not None
    assert latest.version == 1
    assert server.policy_version == 1
    assert server.last_checkpoint == latest


def test_batch_intake_rejects_invalid_auth_token() -> None:
    with pytest.raises(PermissionError, match="auth token"):
        _validate_header({"type": BATCH_INTAKE_TYPE, "token": "wrong"}, "secret")


def test_batch_intake_rejects_legacy_envelope_type() -> None:
    with pytest.raises(ValueError, match="header type"):
        _validate_header({"type": "hkrl.rollout_batch.v1", "token": "secret"}, "secret")


def test_batch_intake_ack_requires_accepted_boolean() -> None:
    assert _accepted_from_ack({"ok": True, "accepted": False}) is False
    with pytest.raises(ValueError, match="accepted boolean"):
        _accepted_from_ack({"ok": True})
    with pytest.raises(RuntimeError, match="stale"):
        _accepted_from_ack({"ok": False, "error": "stale"})


def _serve_once(
    intake: BatchIntakeServer,
    results: list[BatchIntakeResult],
    errors: list[BaseException],
) -> None:
    try:
        results.append(intake.serve_once())
    except BaseException as exc:
        errors.append(exc)


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


def _env_obs_dims(env: Any) -> dict[str, tuple[int, ...]]:
    return {
        "global": env.observation_space["global"].shape,
        "player": env.observation_space["player"].shape,
        "entities": env.observation_space["entities"].shape,
        "entity_mask": env.observation_space["entity_mask"].shape,
    }


def _recurrent_model(obs_dims: dict[str, tuple[int, ...]]) -> EntityAttentionRecurrentAC:
    return EntityAttentionRecurrentAC(
        obs_dims,
        entity_hidden=8,
        attention_layers=1,
        attention_heads=2,
        rnn_hidden=16,
        enable_macro=False,
        max_entities=4,
    )


class _WorkerIntakeEnv:
    def __init__(self) -> None:
        self.observation_space = make_observation_space(max_entities=4, tier="privileged")
        self.action_space = make_action_space(enable_macro=False)
        self.reset_count = 0
        self.step_count = 0
        self.actions: list[dict[str, Any]] = []

    def reset(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.reset_count += 1
        return self._obs(), self._info()

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        self.step_count += 1
        return self._obs(), 1.0, self.step_count % 4 == 0, False, self._info()

    def _obs(self) -> dict[str, np.ndarray]:
        return {
            "global": np.zeros(self.observation_space["global"].shape, dtype=np.float32),
            "player": np.zeros(self.observation_space["player"].shape, dtype=np.float32),
            "entities": np.zeros(self.observation_space["entities"].shape, dtype=np.float32),
            "entity_mask": np.ones(self.observation_space["entity_mask"].shape, dtype=bool),
        }

    def _info(self) -> dict[str, Any]:
        return {
            "action_mask": np.ones((_mask_dim(),), dtype=bool),
            "episode_id": self.reset_count,
            "task_id": 0,
        }
