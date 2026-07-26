"""MLP actor-critic baseline tests."""

from __future__ import annotations

import torch
from hkrl.models.mlp import MlpActorCritic
from hkrl.spaces import (
    ENTITY_FEATURE_INDEX,
    GLOBAL_FEATURE_INDEX,
    N_AIM_Y,
    N_BUTTONS,
    N_DURATION,
    N_MOVEMENT_X,
    PLAYER_FEATURE_INDEX,
)


def test_mlp_actor_critic_act_and_evaluate_actions() -> None:
    model = MlpActorCritic(_obs_dims(), hidden=16, enable_macro=False)
    obs = _obs(batch_size=3)
    action_mask = torch.ones(
        (3, N_MOVEMENT_X + N_AIM_Y + N_BUTTONS + N_DURATION),
        dtype=torch.bool,
    )

    action, log_prob, value, next_state = model.act(obs, action_mask=action_mask)
    eval_log_prob, entropy, eval_value = model.evaluate_actions(
        obs,
        action,
        action_mask=action_mask,
    )

    assert next_state is None
    assert action.shape == (3, 12)
    assert log_prob.shape == (3,)
    assert value.shape == (3,)
    assert entropy.shape == (3,)
    torch.testing.assert_close(eval_log_prob, log_prob)
    torch.testing.assert_close(eval_value, value)


def test_mlp_actor_critic_masks_padded_entities_before_flattening() -> None:
    model = MlpActorCritic(_obs_dims(max_entities=2, entity_dim=2), hidden=16, enable_macro=False)
    base = {
        "global": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        "player": torch.tensor([[0.3, 0.4, 0.5]], dtype=torch.float32),
        "entity_mask": torch.tensor([[True, False]]),
    }
    obs_a = {
        **base,
        "entities": torch.tensor([[[1.0, 2.0], [99.0, 99.0]]], dtype=torch.float32),
    }
    obs_b = {
        **base,
        "entities": torch.tensor([[[1.0, 2.0], [-99.0, -99.0]]], dtype=torch.float32),
    }

    dist_a, value_a, _ = model(obs_a)
    dist_b, value_b, _ = model(obs_b)

    torch.testing.assert_close(value_a, value_b)
    torch.testing.assert_close(dist_a.movement_x.logits, dist_b.movement_x.logits)


def test_mlp_actor_critic_drops_raw_int32_scale_hashes() -> None:
    model = MlpActorCritic(
        {
            "global": (9,),
            "player": (32,),
            "entities": (2, 24),
            "entity_mask": (2,),
        },
        hidden=16,
        enable_macro=False,
    )
    obs = {
        "global": torch.zeros((1, 9), dtype=torch.float32),
        "player": torch.zeros((1, 32), dtype=torch.float32),
        "entities": torch.zeros((1, 2, 24), dtype=torch.float32),
        "entity_mask": torch.tensor([[True, False]]),
    }
    obs["global"][:, GLOBAL_FEATURE_INDEX["scene_hash"]] = -1_364_303_872
    obs["global"][:, GLOBAL_FEATURE_INDEX["arena_id"]] = -1_364_303_872
    obs["player"][:, PLAYER_FEATURE_INDEX["actor_state_hash"]] = 1_932_350_208
    obs["entities"][:, 0, ENTITY_FEATURE_INDEX["entity_id"]] = 100_000
    obs["entities"][:, 0, ENTITY_FEATURE_INDEX["prefab_hash"]] = 2_082_767_232
    obs["entities"][:, 0, ENTITY_FEATURE_INDEX["fsm_name_hash"]] = 2_082_767_232
    obs["entities"][:, 0, ENTITY_FEATURE_INDEX["fsm_state_hash"]] = 2_082_767_232

    dist, value, _ = model(obs)

    assert torch.isfinite(dist.log_prob(dist.mode())).all()
    assert torch.isfinite(value).all()


def _obs_dims(max_entities: int = 4, entity_dim: int = 5) -> dict[str, tuple[int, ...]]:
    return {
        "global": (2,),
        "player": (3,),
        "entities": (max_entities, entity_dim),
        "entity_mask": (max_entities,),
    }


def _obs(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "global": torch.zeros((batch_size, 2), dtype=torch.float32),
        "player": torch.zeros((batch_size, 3), dtype=torch.float32),
        "entities": torch.zeros((batch_size, 4, 5), dtype=torch.float32),
        "entity_mask": torch.ones((batch_size, 4), dtype=torch.bool),
    }
