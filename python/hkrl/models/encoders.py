"""Feature encoders + embeddings (docs/model_architecture.md §2).

Encode the three observation parts into embeddings before attention/memory:
global context, player state, and each entity. Hashes (scene/fsm/prefab) and the
discrete entity_type/id go through learned embeddings, never raw ints
(docs/observation_schema.md §2).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GlobalEncoder(nn.Module):
    """MLP over GlobalState features -> global_emb."""

    def __init__(self, in_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = _mlp(in_dim, hidden)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PlayerEncoder(nn.Module):
    """MLP over PlayerState features -> player_emb."""

    def __init__(self, in_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = _mlp(in_dim, hidden)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


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
        self.n_types = n_types
        self.n_ids = n_ids
        self.type_embedding = nn.Embedding(n_types, hidden)
        self.id_embedding = nn.Embedding(n_ids, hidden) if n_ids > 0 else None
        self.feature_mlp = _mlp(feat_dim, hidden)

    def forward(
        self,
        entities: Tensor,
        entity_type: Tensor,
        entity_id: Tensor | None = None,
    ) -> Tensor:
        type_index = entity_type.to(dtype=torch.long).clamp(min=0, max=self.n_types - 1)
        encoded = self.feature_mlp(entities) + self.type_embedding(type_index)

        if self.id_embedding is not None and entity_id is not None:
            id_index = torch.remainder(entity_id.to(dtype=torch.long), self.n_ids)
            encoded = encoded + self.id_embedding(id_index)
        return encoded


def _mlp(in_dim: int, hidden: int) -> nn.Sequential:
    if in_dim <= 0:
        raise ValueError("in_dim must be positive")
    if hidden <= 0:
        raise ValueError("hidden must be positive")

    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )
