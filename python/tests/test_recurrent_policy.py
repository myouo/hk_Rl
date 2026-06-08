"""Entity-attention recurrent policy tests."""

from __future__ import annotations

import pytest
import torch
from hkrl.models.recurrent_policy import EntityAttentionRecurrentAC
from hkrl.spaces import N_AIM_Y, N_BUTTONS, N_DURATION, N_MOVEMENT_X


def test_recurrent_policy_gru_act_and_evaluate_single_step() -> None:
    model = EntityAttentionRecurrentAC(
        _obs_dims(), entity_hidden=8, rnn_hidden=16, enable_macro=False
    )
    obs = _obs(batch_size=2)
    action_mask = torch.ones((2, _mask_dim()), dtype=torch.bool)

    state = model.initial_state(batch_size=2)
    action, log_prob, value, next_state = model.act(obs, state, action_mask=action_mask)
    eval_log_prob, entropy, eval_value = model.evaluate_actions(
        obs,
        action,
        rnn_state=state,
        action_mask=action_mask,
    )

    assert action.shape == (2, 12)
    assert log_prob.shape == (2,)
    assert value.shape == (2,)
    assert next_state.shape == (1, 2, 16)
    torch.testing.assert_close(eval_log_prob, log_prob)
    torch.testing.assert_close(eval_value, value)
    assert entropy.shape == (2,)


def test_recurrent_policy_lstm_initial_state() -> None:
    model = EntityAttentionRecurrentAC(
        _obs_dims(),
        entity_hidden=8,
        rnn_hidden=16,
        rnn_type="lstm",
        enable_macro=False,
    )

    state = model.initial_state(batch_size=3)

    assert isinstance(state, tuple)
    assert state[0].shape == (1, 3, 16)
    assert state[1].shape == (1, 3, 16)


def test_recurrent_policy_sequence_forward_and_evaluate() -> None:
    model = EntityAttentionRecurrentAC(
        _obs_dims(), entity_hidden=8, rnn_hidden=16, enable_macro=False
    )
    obs = _obs(batch_size=2, seq_len=3)
    action_mask = torch.ones((2, 3, _mask_dim()), dtype=torch.bool)

    dist, value, next_state = model(obs, action_mask=action_mask)
    action = dist.sample()
    log_prob, entropy, eval_value = model.evaluate_actions(
        obs,
        action,
        action_mask=action_mask,
    )

    assert action.shape == (2, 3, 12)
    assert value.shape == (2, 3)
    assert next_state.shape == (1, 2, 16)
    assert log_prob.shape == (2, 3)
    assert entropy.shape == (2, 3)
    torch.testing.assert_close(eval_value, value)


def test_recurrent_policy_accepts_prev_action_and_reward_context() -> None:
    model = EntityAttentionRecurrentAC(
        _obs_dims(), entity_hidden=8, rnn_hidden=16, enable_macro=False
    )
    obs = _obs(batch_size=2, seq_len=3)
    obs["prev_action"] = torch.zeros((2, 3, 12), dtype=torch.float32)
    obs["prev_action"][:, :, 0] = 2
    obs["prev_reward"] = torch.ones((2, 3), dtype=torch.float32)

    dist, value, next_state = model(obs)

    assert model.rnn.input_size == 8 * 4 + 1
    assert dist.sample().shape == (2, 3, 12)
    assert value.shape == (2, 3)
    assert next_state.shape == (1, 2, 16)


def test_recurrent_policy_rejects_bad_prev_action_shape() -> None:
    model = EntityAttentionRecurrentAC(
        _obs_dims(), entity_hidden=8, rnn_hidden=16, enable_macro=False
    )
    obs = _obs(batch_size=2)
    obs["prev_action"] = torch.zeros((2, 11), dtype=torch.float32)

    with pytest.raises(ValueError, match="prev_action"):
        model(obs)


def _obs_dims() -> dict[str, tuple[int, ...]]:
    return {
        "global": (2,),
        "player": (3,),
        "entities": (4, 6),
        "entity_mask": (4,),
    }


def _obs(batch_size: int, seq_len: int | None = None) -> dict[str, torch.Tensor]:
    prefix = (batch_size,) if seq_len is None else (batch_size, seq_len)
    entities = torch.zeros((*prefix, 4, 6), dtype=torch.float32)
    entities[..., 0, 0] = 1
    entities[..., 0, 1] = 1
    mask = torch.zeros((*prefix, 4), dtype=torch.bool)
    mask[..., 0] = True
    return {
        "global": torch.zeros((*prefix, 2), dtype=torch.float32),
        "player": torch.zeros((*prefix, 3), dtype=torch.float32),
        "entities": entities,
        "entity_mask": mask,
    }


def _mask_dim() -> int:
    return N_MOVEMENT_X + N_AIM_Y + N_BUTTONS + N_DURATION
