"""Model encoder tests."""

from __future__ import annotations

import pytest
import torch
from hkrl.models.encoders import EntityEncoder, GlobalEncoder, PlayerEncoder


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
