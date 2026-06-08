"""Policy/value models (docs/model_architecture.md, PRD §7).

    base                 ActorCritic ABC
    encoders             Global/Player/Entity encoders + type/id embeddings
    entity_attention     masked Transformer / cross-attention over entities
    recurrent_policy     GRU/LSTM memory + full ActorCritic assembly
    heads                per-component policy heads (mask-aware) + value head
    mlp                  non-recurrent baseline

Models register via @register_model and are selected from ModelConfig.name.
"""

from __future__ import annotations

from hkrl.models.base import ActorCritic

__all__ = ["ActorCritic"]
