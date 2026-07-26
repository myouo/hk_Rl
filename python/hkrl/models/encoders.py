"""Feature encoders + embeddings (docs/model_architecture.md §2).

Encode the three observation parts into embeddings before attention/memory:
global context, player state, and each entity. Hashes (scene/fsm/prefab) and the
discrete entity_type/id go through learned embeddings, never raw ints
(docs/observation_schema.md §2).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hkrl.spaces import ENTITY_FEATURE_INDEX, GLOBAL_FEATURE_INDEX, PLAYER_FEATURE_INDEX

HASH_EMBEDDING_BUCKETS = 4096


class GlobalEncoder(nn.Module):
    """Hash embeddings + MLP over continuous GlobalState features."""

    def __init__(self, in_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.encoder = _HashAwareFeatureEncoder(
            in_dim,
            hidden,
            hash_indices=(
                GLOBAL_FEATURE_INDEX["scene_hash"],
                GLOBAL_FEATURE_INDEX["arena_id"],
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class PlayerEncoder(nn.Module):
    """FSM/state hash embeddings + MLP over continuous PlayerState features."""

    def __init__(self, in_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.encoder = _HashAwareFeatureEncoder(
            in_dim,
            hidden,
            hash_indices=(
                PLAYER_FEATURE_INDEX["actor_state_hash"],
                PLAYER_FEATURE_INDEX["spell_fsm_state_hash"],
                PLAYER_FEATURE_INDEX["dream_nail_fsm_state_hash"],
                PLAYER_FEATURE_INDEX["nail_arts_fsm_state_hash"],
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


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
        self.feature_encoder = _HashAwareFeatureEncoder(
            feat_dim,
            hidden,
            hash_indices=(
                ENTITY_FEATURE_INDEX["prefab_hash"],
                ENTITY_FEATURE_INDEX["fsm_name_hash"],
                ENTITY_FEATURE_INDEX["fsm_state_hash"],
            ),
            excluded_indices=(
                ENTITY_FEATURE_INDEX["entity_id"],
                ENTITY_FEATURE_INDEX["entity_type"],
            ),
        )

    def forward(
        self,
        entities: Tensor,
        entity_type: Tensor,
        entity_id: Tensor | None = None,
    ) -> Tensor:
        type_index = entity_type.to(dtype=torch.long).clamp(min=0, max=self.n_types - 1)
        encoded = self.feature_encoder(entities) + self.type_embedding(type_index)

        if self.id_embedding is not None and entity_id is not None:
            id_index = torch.remainder(entity_id.to(dtype=torch.long), self.n_ids)
            encoded = encoded + self.id_embedding(id_index)
        return encoded


class _HashAwareFeatureEncoder(nn.Module):
    """Keep int32-scale hashes out of AMP linear layers and embed them by bucket."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        *,
        hash_indices: tuple[int, ...],
        excluded_indices: tuple[int, ...] = (),
        hash_buckets: int = HASH_EMBEDDING_BUCKETS,
    ) -> None:
        super().__init__()
        if hash_buckets <= 0:
            raise ValueError("hash_buckets must be positive")
        self.in_dim = in_dim
        self.hash_buckets = hash_buckets
        self.hash_indices = _indices_within_dim(hash_indices, in_dim)
        excluded = set(_indices_within_dim(excluded_indices, in_dim))
        excluded.update(self.hash_indices)

        continuous_mask = torch.ones((in_dim,), dtype=torch.float32)
        if excluded:
            continuous_mask[list(sorted(excluded))] = 0.0
        self.register_buffer("_continuous_mask", continuous_mask, persistent=False)
        self.net = _mlp(in_dim, hidden)
        self.hash_embeddings = nn.ModuleList(
            nn.Embedding(hash_buckets, hidden) for _ in self.hash_indices
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.in_dim:
            raise ValueError(f"feature dimension must be {self.in_dim}, got {features.shape[-1]}")
        mask = self._continuous_mask.to(device=features.device, dtype=features.dtype)
        encoded = self.net(features * mask)
        for feature_index, embedding in zip(
            self.hash_indices,
            self.hash_embeddings,
            strict=True,
        ):
            bucket = torch.remainder(
                features[..., feature_index].to(dtype=torch.long),
                self.hash_buckets,
            )
            encoded = encoded + embedding(bucket)
        return encoded


def without_raw_categorical_features(
    features: Tensor,
    indices: tuple[int, ...],
) -> Tensor:
    """Return continuous inputs with selected raw categorical columns zeroed."""

    valid_indices = _indices_within_dim(indices, features.shape[-1])
    if not valid_indices:
        return features
    mask = torch.ones((features.shape[-1],), dtype=features.dtype, device=features.device)
    mask[list(valid_indices)] = 0.0
    return features * mask


def _indices_within_dim(indices: tuple[int, ...], in_dim: int) -> tuple[int, ...]:
    if in_dim <= 0:
        raise ValueError("in_dim must be positive")
    if any(index < 0 for index in indices):
        raise ValueError("feature indices must be non-negative")
    return tuple(index for index in indices if index < in_dim)


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
