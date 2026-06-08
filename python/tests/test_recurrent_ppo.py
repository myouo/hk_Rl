"""Recurrent PPO tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hkrl.models.recurrent_policy import EntityAttentionRecurrentAC
from hkrl.spaces import N_AIM_Y, N_BUTTONS, N_DURATION, N_MOVEMENT_X
from hkrl.training.recurrent_buffer import RecurrentRolloutBuffer
from hkrl.training.recurrent_ppo import RecurrentPPO
from hkrl.utils.config import TrainConfig


def test_recurrent_ppo_update_returns_metrics_and_changes_parameters() -> None:
    torch.manual_seed(123)
    model = EntityAttentionRecurrentAC(
        _obs_spec(),
        entity_hidden=8,
        attention_layers=1,
        attention_heads=2,
        rnn_hidden=16,
        enable_macro=False,
    )
    cfg = TrainConfig(
        algorithm="recurrent_ppo",
        epochs=2,
        minibatch_size=4,
        learning_rate=1.0e-2,
        entropy_coef=0.0,
        sequence_length=2,
        burn_in=1,
        seed=123,
    )
    ppo = RecurrentPPO(model, cfg)
    buffer = RecurrentRolloutBuffer(
        capacity=4,
        num_envs=1,
        sequence_length=cfg.sequence_length,
        burn_in=cfg.burn_in,
        obs_spec={
            **_obs_spec(),
            "action": (12,),
            "action_mask": (_mask_dim(),),
        },
    )

    action_mask = np.ones((_mask_dim(),), dtype=bool)
    state = model.initial_state(batch_size=1)
    for step in range(4):
        obs = _numpy_obs(step)
        torch_obs = _torch_obs(obs)
        rnn_state = state
        action, log_prob, value, state = model.act(
            torch_obs,
            rnn_state=rnn_state,
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
            rnn_state=rnn_state,
        )
    buffer.compute_returns(
        last_value=np.array([0.0], dtype=np.float32),
        gamma=0.99,
        gae_lambda=0.95,
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


def test_recurrent_ppo_update_rejects_non_finite_sequence_values() -> None:
    model = EntityAttentionRecurrentAC(
        _obs_spec(),
        entity_hidden=8,
        attention_layers=1,
        attention_heads=2,
        rnn_hidden=16,
        enable_macro=False,
    )
    ppo = RecurrentPPO(
        model,
        TrainConfig(
            algorithm="recurrent_ppo",
            epochs=1,
            minibatch_size=1,
            sequence_length=1,
            burn_in=0,
        ),
    )
    buffer = RecurrentRolloutBuffer(
        capacity=1,
        num_envs=1,
        sequence_length=1,
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
        rnn_state=model.initial_state(batch_size=1),
    )
    buffer.compute_returns(np.array([0.0], dtype=np.float32), gamma=0.99, gae_lambda=0.95)
    buffer.advantages[0, 0] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        ppo.update(buffer)


def _obs_spec() -> dict[str, tuple[int, ...]]:
    return {
        "global": (2,),
        "player": (3,),
        "entities": (4, 6),
        "entity_mask": (4,),
    }


def _numpy_obs(step: int) -> dict[str, np.ndarray]:
    entities = np.zeros((4, 6), dtype=np.float32)
    entities[0, 0] = 1
    entities[0, 1] = 1
    entities[0, 2] = float(step)
    return {
        "global": np.array([step, step + 0.5], dtype=np.float32),
        "player": np.ones((3,), dtype=np.float32) * step,
        "entities": entities,
        "entity_mask": np.array([True, False, False, False]),
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
