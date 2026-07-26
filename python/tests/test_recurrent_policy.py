"""Entity-attention recurrent policy tests."""

from __future__ import annotations

import pytest
import torch
from hkrl.models.recurrent_policy import EntityAttentionRecurrentAC
from hkrl.spaces import (
    ENTITY_FEATURE_INDEX,
    GLOBAL_FEATURE_INDEX,
    N_AIM_Y,
    N_BUTTONS,
    N_DURATION,
    N_MOVEMENT_X,
    PLAYER_FEATURE_INDEX,
)


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


def test_recurrent_policy_keeps_live_scale_hashes_finite_under_cpu_autocast() -> None:
    model = EntityAttentionRecurrentAC(
        _live_obs_dims(),
        entity_hidden=8,
        attention_layers=1,
        attention_heads=2,
        rnn_hidden=16,
        enable_macro=False,
    )
    obs = _live_scale_obs(device=torch.device("cpu"))

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        dist, value, _ = model(obs)
        actions = dist.mode()
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

    assert torch.isfinite(value).all()
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA AMP device")
def test_recurrent_policy_keeps_live_scale_hashes_finite_under_cuda_fp16() -> None:
    device = torch.device("cuda")
    model = EntityAttentionRecurrentAC(
        _live_obs_dims(),
        entity_hidden=8,
        attention_layers=1,
        attention_heads=2,
        rnn_hidden=16,
        enable_macro=False,
    ).to(device)
    obs = _live_scale_obs(device=device)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        dist, value, _ = model(obs)
        actions = dist.mode()
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

    assert torch.isfinite(value).all()
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()


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


def _live_obs_dims() -> dict[str, tuple[int, ...]]:
    return {
        "global": (9,),
        "player": (32,),
        "entities": (64, 24),
        "entity_mask": (64,),
    }


def _live_scale_obs(*, device: torch.device) -> dict[str, torch.Tensor]:
    global_state = torch.zeros((2, 3, 9), dtype=torch.float32, device=device)
    global_state[..., GLOBAL_FEATURE_INDEX["scene_hash"]] = -1_364_303_872
    global_state[..., GLOBAL_FEATURE_INDEX["arena_id"]] = -1_364_303_872
    player = torch.zeros((2, 3, 32), dtype=torch.float32, device=device)
    for name in (
        "actor_state_hash",
        "spell_fsm_state_hash",
        "dream_nail_fsm_state_hash",
        "nail_arts_fsm_state_hash",
    ):
        player[..., PLAYER_FEATURE_INDEX[name]] = 1_932_350_208
    entities = torch.zeros((2, 3, 64, 24), dtype=torch.float32, device=device)
    entities[..., 0, ENTITY_FEATURE_INDEX["entity_id"]] = 100_000
    entities[..., 0, ENTITY_FEATURE_INDEX["entity_type"]] = 1
    for name in ("prefab_hash", "fsm_name_hash", "fsm_state_hash"):
        entities[..., 0, ENTITY_FEATURE_INDEX[name]] = 2_082_767_232
    entity_mask = torch.zeros((2, 3, 64), dtype=torch.bool, device=device)
    entity_mask[..., 0] = True
    return {
        "global": global_state,
        "player": player,
        "entities": entities,
        "entity_mask": entity_mask,
    }
