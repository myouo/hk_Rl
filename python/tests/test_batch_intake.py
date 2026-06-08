"""RolloutBatch TCP intake tests."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch
from hkrl.learner.batch_intake import BatchIntakeClient, BatchIntakeResult, BatchIntakeServer
from hkrl.learner.checkpoint_registry import CheckpointRegistry
from hkrl.learner.learner_server import LearnerServer
from hkrl.models.mlp import MlpActorCritic
from hkrl.spaces import N_AIM_Y, N_BUTTONS, N_DURATION, N_MOVEMENT_X
from hkrl.training.rollout_buffer import RolloutBatch, RolloutBuffer
from hkrl.utils.config import TrainConfig


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
