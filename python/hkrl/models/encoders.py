"""Feature encoders + embeddings (docs/model_architecture.md §2).

Encode the three observation parts into embeddings before attention/memory:
global context, player state, and each entity. Hashes (scene/fsm/prefab) and the
discrete entity_type/id go through learned embeddings, never raw ints
(docs/observation_schema.md §2).
"""

from __future__ import annotations

from torch import Tensor, nn


class GlobalEncoder(nn.Module):
    """MLP over GlobalState features -> global_emb."""

    def __init__(self, in_dim: int, hidden: int = 128) -> None:
        super().__init__()
        # TODO(phase-5): nn.Sequential MLP.

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError  # TODO(phase-5)


class PlayerEncoder(nn.Module):
    """MLP over PlayerState features -> player_emb."""

    def __init__(self, in_dim: int, hidden: int = 128) -> None:
        super().__init__()
        # TODO(phase-5): nn.Sequential MLP.

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError  # TODO(phase-5)


class EntityEncoder(nn.Module):
    """Per-entity embedding: type_emb (+ optional id_emb) + MLP(feat).

    Operates on (batch, max_entities, feat) and respects entity_mask downstream.
    """

    def __init__(
        self,
        feat_dim: int,
        hidden: int = 128,
        n_types: int = 256,
        n_ids: int = 0,
    ) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(n_types, hidden)
        self.id_embedding = nn.Embedding(n_ids, hidden) if n_ids > 0 else None
        # TODO(phase-5): feature MLP, combine type+feat(+id).

    def forward(
        self,
        entities: Tensor,
        entity_type: Tensor,
        entity_id: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError  # TODO(phase-5)
