"""Model encoder tests."""

from __future__ import annotations

import pytest
import torch
from hkrl.models.encoders import EntityEncoder, GlobalEncoder, PlayerEncoder
from hkrl.spaces import ENTITY_FEATURE_INDEX, GLOBAL_FEATURE_INDEX, PLAYER_FEATURE_INDEX


def test_global_and_player_encoders_preserve_leading_batch_dims() -> None:
    global_encoder = GlobalEncoder(in_dim=3, hidden=8)
    player_encoder = PlayerEncoder(in_dim=5, hidden=8)

    assert global_encoder(torch.zeros((2, 3))).shape == (2, 8)
    assert player_encoder(torch.zeros((4, 2, 5))).shape == (4, 2, 8)


def test_entity_encoder_combines_feature_type_and_id_embeddings() -> None:
    encoder = EntityEncoder(feat_dim=6, hidden=8, n_types=4, n_ids=16)
    entities = torch.zeros((2, 3, 6), dtype=torch.float32)
    entity_type = torch.tensor([[1, 2, 255], [0, 1, 2]])
    entity_id = torch.tensor([[1, 2, 17], [3, 4, 5]])

    output = encoder(entities, entity_type=entity_type, entity_id=entity_id)
    changed_type = encoder(entities, entity_type=torch.zeros_like(entity_type), entity_id=entity_id)

    assert output.shape == (2, 3, 8)
    assert not torch.allclose(output, changed_type)


def test_entity_encoder_supports_no_id_embedding() -> None:
    encoder = EntityEncoder(feat_dim=2, hidden=4, n_types=3, n_ids=0)

    output = encoder(
        torch.zeros((1, 2, 2), dtype=torch.float32),
        entity_type=torch.tensor([[0, 1]]),
        entity_id=torch.tensor([[100, 101]]),
    )

    assert output.shape == (1, 2, 4)


def test_encoders_reject_non_positive_dims() -> None:
    with pytest.raises(ValueError, match="in_dim"):
        GlobalEncoder(in_dim=0)
    with pytest.raises(ValueError, match="hidden"):
        PlayerEncoder(in_dim=1, hidden=0)


def test_hash_columns_are_zeroed_before_continuous_mlps() -> None:
    global_encoder = GlobalEncoder(in_dim=9, hidden=8)
    player_encoder = PlayerEncoder(in_dim=32, hidden=8)
    entity_encoder = EntityEncoder(feat_dim=24, hidden=8, n_types=8, n_ids=16)
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            captured[name] = args[0].detach()

        return hook

    handles = [
        global_encoder.encoder.net[0].register_forward_pre_hook(capture("global")),
        player_encoder.encoder.net[0].register_forward_pre_hook(capture("player")),
        entity_encoder.feature_encoder.net[0].register_forward_pre_hook(capture("entity")),
    ]
    try:
        global_state = torch.zeros((1, 9), dtype=torch.float32)
        global_state[:, GLOBAL_FEATURE_INDEX["scene_hash"]] = -1_364_303_872
        global_state[:, GLOBAL_FEATURE_INDEX["arena_id"]] = -1_364_303_872
        player = torch.zeros((1, 32), dtype=torch.float32)
        for name in (
            "actor_state_hash",
            "spell_fsm_state_hash",
            "dream_nail_fsm_state_hash",
            "nail_arts_fsm_state_hash",
        ):
            player[:, PLAYER_FEATURE_INDEX[name]] = 1_932_350_208
        entities = torch.zeros((1, 1, 24), dtype=torch.float32)
        entities[:, :, ENTITY_FEATURE_INDEX["entity_id"]] = 100_000
        entities[:, :, ENTITY_FEATURE_INDEX["entity_type"]] = 1
        for name in ("prefab_hash", "fsm_name_hash", "fsm_state_hash"):
            entities[:, :, ENTITY_FEATURE_INDEX[name]] = 2_082_767_232

        global_encoder(global_state)
        player_encoder(player)
        entity_encoder(
            entities,
            entity_type=entities[..., ENTITY_FEATURE_INDEX["entity_type"]],
            entity_id=entities[..., ENTITY_FEATURE_INDEX["entity_id"]],
        )
    finally:
        for handle in handles:
            handle.remove()

    assert captured["global"][0, GLOBAL_FEATURE_INDEX["scene_hash"]] == 0
    assert captured["global"][0, GLOBAL_FEATURE_INDEX["arena_id"]] == 0
    for name in (
        "actor_state_hash",
        "spell_fsm_state_hash",
        "dream_nail_fsm_state_hash",
        "nail_arts_fsm_state_hash",
    ):
        assert captured["player"][0, PLAYER_FEATURE_INDEX[name]] == 0
    for name in ("entity_id", "entity_type", "prefab_hash", "fsm_name_hash", "fsm_state_hash"):
        assert captured["entity"][0, 0, ENTITY_FEATURE_INDEX[name]] == 0
