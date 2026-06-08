"""Masked attention over entity embeddings (docs/model_architecture.md §2-3).

Two interchangeable aggregators: a masked TransformerEncoder over the entity set,
or cross-attention with the player embedding as query. Both consume ``entity_mask``
as a key-padding mask so padded slots contribute nothing. Entities beyond top-k
are pre-aggregated mod-side into a single summary token (PRD §7.3).
"""

from __future__ import annotations

from torch import Tensor, nn


class EntityTransformerEncoder(nn.Module):
    """Self-attention encoder over entities; returns a pooled context vector.

    O(N^2) but N is bounded (max_entities=64). Pooling is masked mean/attention
    over valid slots.
    """

    def __init__(self, dim: int = 128, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        # TODO(phase-5): nn.TransformerEncoder(batch_first=True) + masked pool.

    def forward(self, entity_embs: Tensor, entity_mask: Tensor) -> Tensor:
        """entity_embs: (B, N, dim); entity_mask: (B, N) bool (True=valid)."""
        raise NotImplementedError  # TODO(phase-5)


class PlayerCrossAttention(nn.Module):
    """Cross-attention: query=player_emb, key/value=entity_embs (masked).

    Lighter alternative to the full encoder; directly answers "which entities
    matter to the player right now".
    """

    def __init__(self, dim: int = 128, heads: int = 4) -> None:
        super().__init__()
        # TODO(phase-5): nn.MultiheadAttention(batch_first=True).

    def forward(self, player_emb: Tensor, entity_embs: Tensor, entity_mask: Tensor) -> Tensor:
        raise NotImplementedError  # TODO(phase-5)
