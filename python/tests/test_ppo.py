"""Synchronous PPO tests."""

from __future__ import annotations

import numpy as np
import torch
from hkrl.models.mlp import MlpActorCritic
from hkrl.spaces import N_AIM_Y, N_BUTTONS, N_DURATION, N_MOVEMENT_X
from hkrl.training.ppo import PPO
from hkrl.training.rollout_buffer import RolloutBuffer
from hkrl.utils.config import TrainConfig


def test_ppo_update_returns_metrics_and_changes_parameters() -> None:
    torch.manual_seed(123)
    model = MlpActorCritic(_obs_spec(), hidden=16, enable_macro=False)
    cfg = TrainConfig(
        algorithm="ppo",
        epochs=2,
        minibatch_size=2,
        learning_rate=1.0e-2,
        entropy_coef=0.0,
    )
    ppo = PPO(model, cfg)
    buffer = RolloutBuffer(
        capacity=4,
        num_envs=1,
        obs_spec={
            **_obs_spec(),
            "action": (12,),
            "action_mask": (N_MOVEMENT_X + N_AIM_Y + N_BUTTONS + N_DURATION,),
        },
    )

    action_mask = np.ones((N_MOVEMENT_X + N_AIM_Y + N_BUTTONS + N_DURATION,), dtype=bool)
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
        last_value=np.array([0.0], dtype=np.float32), gamma=0.99, gae_lambda=0.95
    )

    before = [param.detach().clone() for param in model.parameters()]
    metrics = ppo.update(buffer)

    for key in (
        "policy_loss",
        "value_loss",
        "action_entropy",
        "policy_kl",
        "explained_variance",
        "grad_norm",
    ):
        assert key in metrics
        assert np.isfinite(metrics[key])
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, model.parameters(), strict=True)
    )


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
