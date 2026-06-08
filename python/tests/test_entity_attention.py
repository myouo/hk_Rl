"""Entity attention tests."""

from __future__ import annotations

import pytest
import torch
from hkrl.models.entity_attention import EntityTransformerEncoder, PlayerCrossAttention


def test_entity_transformer_encoder_masks_padded_entities() -> None:
    encoder = EntityTransformerEncoder(dim=4, layers=1, heads=2)
    entity_embs = torch.randn((1, 2, 4), dtype=torch.float32)
    changed = entity_embs.clone()
    changed[:, 1] = 100.0
    mask = torch.tensor([[True, False]])

    out = encoder(entity_embs, mask)
    changed_out = encoder(changed, mask)

    assert out.shape == (1, 4)
    torch.testing.assert_close(out, changed_out)


def test_entity_transformer_encoder_handles_empty_rows() -> None:
    encoder = EntityTransformerEncoder(dim=4, layers=1, heads=2)
    out = encoder(torch.randn((2, 3, 4)), torch.zeros((2, 3), dtype=torch.bool))

    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()


def test_player_cross_attention_masks_padded_entities() -> None:
    attention = PlayerCrossAttention(dim=4, heads=2)
    player = torch.randn((1, 4), dtype=torch.float32)
    entity_embs = torch.randn((1, 2, 4), dtype=torch.float32)
    changed = entity_embs.clone()
    changed[:, 1] = -100.0
    mask = torch.tensor([[True, False]])

    out = attention(player, entity_embs, mask)
    changed_out = attention(player, changed, mask)

    assert out.shape == (1, 4)
    torch.testing.assert_close(out, changed_out)


def test_entity_attention_rejects_bad_shapes() -> None:
    encoder = EntityTransformerEncoder(dim=4, layers=1, heads=2)

    with pytest.raises(ValueError, match="entity_mask"):
        encoder(torch.zeros((1, 2, 4)), torch.ones((1, 3), dtype=torch.bool))
