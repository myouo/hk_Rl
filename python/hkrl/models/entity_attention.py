"""Masked attention over entity embeddings (docs/model_architecture.md §2-3).

Two interchangeable aggregators: a masked TransformerEncoder over the entity set,
or cross-attention with the player embedding as query. Both consume ``entity_mask``
as a key-padding mask so padded slots contribute nothing. Entities beyond top-k
are pre-aggregated mod-side into a single summary token (PRD §7.3).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EntityTransformerEncoder(nn.Module):
    """Self-attention encoder over entities; returns a pooled context vector.

    O(N^2) but N is bounded (max_entities=64). Pooling is masked mean/attention
    over valid slots.
    """

    def __init__(self, dim: int = 128, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if layers <= 0:
            raise ValueError("layers must be positive")
        if heads <= 0:
            raise ValueError("heads must be positive")

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, entity_embs: Tensor, entity_mask: Tensor) -> Tensor:
        """entity_embs: (B, N, dim); entity_mask: (B, N) bool (True=valid)."""
        valid_mask, safe_mask, safe_embs, empty_rows = _safe_attention_inputs(
            entity_embs,
            entity_mask,
        )
        encoded = self.encoder(safe_embs, src_key_padding_mask=~safe_mask)
        encoded = encoded * valid_mask.unsqueeze(-1).to(dtype=encoded.dtype)
        denom = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=encoded.dtype)
        pooled = encoded.sum(dim=1) / denom
        pooled = torch.where(empty_rows.unsqueeze(-1), torch.zeros_like(pooled), pooled)
        return self.norm(pooled)


class PlayerCrossAttention(nn.Module):
    """Cross-attention: query=player_emb, key/value=entity_embs (masked).

    Lighter alternative to the full encoder; directly answers "which entities
    matter to the player right now".
    """

    def __init__(self, dim: int = 128, heads: int = 4) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if heads <= 0:
            raise ValueError("heads must be positive")

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, player_emb: Tensor, entity_embs: Tensor, entity_mask: Tensor) -> Tensor:
        _, safe_mask, safe_embs, empty_rows = _safe_attention_inputs(entity_embs, entity_mask)
        query = player_emb.unsqueeze(1)
        context, _ = self.attention(
            query,
            safe_embs,
            safe_embs,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        context = context.squeeze(1)
        context = torch.where(empty_rows.unsqueeze(-1), torch.zeros_like(context), context)
        return self.norm(player_emb + context)


def _safe_attention_inputs(
    entity_embs: Tensor,
    entity_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if entity_embs.ndim != 3:
        raise ValueError("entity_embs must have shape (B, N, D)")
    if entity_mask.shape != entity_embs.shape[:2]:
        raise ValueError("entity_mask must have shape (B, N)")

    valid_mask = entity_mask.to(dtype=torch.bool, device=entity_embs.device)
    safe_mask = valid_mask.clone()
    empty_rows = ~safe_mask.any(dim=-1)
    safe_embs = entity_embs
    if bool(empty_rows.any().item()):
        safe_mask[empty_rows, 0] = True
        safe_embs = entity_embs.clone()
        safe_embs[empty_rows, 0] = 0.0
    return valid_mask, safe_mask, safe_embs, empty_rows
