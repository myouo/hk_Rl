"""Synchronous PPO tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from hkrl.models.base import ActorCritic
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


def test_ppo_update_rejects_non_finite_rollout_values() -> None:
    model = MlpActorCritic(_obs_spec(), hidden=16, enable_macro=False)
    ppo = PPO(model, TrainConfig(algorithm="ppo", epochs=1, minibatch_size=1))
    buffer = RolloutBuffer(
        capacity=1,
        num_envs=1,
        obs_spec={
            **_obs_spec(),
            "action": (12,),
            "action_mask": (N_MOVEMENT_X + N_AIM_Y + N_BUTTONS + N_DURATION,),
        },
    )
    buffer.add(
        obs=_numpy_obs(0),
        action=np.zeros((12,), dtype=np.int64),
        log_prob=np.array([0.0], dtype=np.float32),
        value=np.array([0.0], dtype=np.float32),
        reward=np.array([0.0], dtype=np.float32),
        done=np.array([False]),
        truncated=np.array([False]),
        action_mask=np.ones((_mask_dim(),), dtype=bool),
    )
    buffer.compute_returns(np.array([0.0], dtype=np.float32), gamma=0.99, gae_lambda=0.95)
    buffer.returns[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        ppo.update(buffer)


def test_ppo_update_rejects_non_finite_model_outputs_before_step() -> None:
    model = _NaNLossActorCritic()
    ppo = PPO(model, TrainConfig(algorithm="ppo", epochs=1, minibatch_size=1))
    buffer = _single_step_buffer()
    before = model.weight.detach().clone()

    with pytest.raises(ValueError, match="model log_probs"):
        ppo.update(buffer)

    assert torch.equal(model.weight.detach(), before)


class _NaNLossActorCritic(ActorCritic):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
    ) -> None:
        return None

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        rnn_state: Any = None,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[None, torch.Tensor, None]:
        batch_size = obs["global"].shape[0]
        return None, self.weight.expand(batch_size), None

    def act(
        self,
        obs: dict[str, torch.Tensor],
        rnn_state: Any = None,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        batch_size = obs["global"].shape[0]
        value = self.weight.expand(batch_size)
        return torch.zeros((batch_size, 12), dtype=torch.long), value, value, None

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        rnn_state: Any = None,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_shape = actions.shape[:-1]
        finite = self.weight.expand(output_shape)
        nan = self.weight * torch.full(output_shape, float("nan"), device=actions.device)
        return nan, finite, finite


def _single_step_buffer() -> RolloutBuffer:
    buffer = RolloutBuffer(
        capacity=1,
        num_envs=1,
        obs_spec={
            **_obs_spec(),
            "action": (12,),
            "action_mask": (_mask_dim(),),
        },
    )
    buffer.add(
        obs=_numpy_obs(0),
        action=np.zeros((12,), dtype=np.int64),
        log_prob=np.array([0.0], dtype=np.float32),
        value=np.array([0.0], dtype=np.float32),
        reward=np.array([0.0], dtype=np.float32),
        done=np.array([False]),
        truncated=np.array([False]),
        action_mask=np.ones((_mask_dim(),), dtype=bool),
    )
    buffer.compute_returns(np.array([0.0], dtype=np.float32), gamma=0.99, gae_lambda=0.95)
    return buffer


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
