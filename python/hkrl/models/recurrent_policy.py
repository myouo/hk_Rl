"""Entity-attention + recurrent ActorCritic (docs/model_architecture.md §2).

The project's high-ceiling model: encoders -> masked attention -> GRU/LSTM memory
-> hybrid heads + value. Threads ``entity_mask`` through attention and ``rnn_state``
through the recurrence. Trained with truncated BPTT (+ optional burn-in) by
training/recurrent_ppo.py.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from hkrl.models.base import ActorCritic, RnnState
from hkrl.utils.registry import register_model


@register_model("entity_attention_gru")
class EntityAttentionRecurrentAC(ActorCritic):
    """Encoders -> attention(entity_context) -> [global|player|context|prev_action|
    prev_reward] -> GRU/LSTM -> heads.

    ``rnn_type`` in {"gru", "lstm"}. ``initial_state`` returns the zeroed hidden
    (tuple for LSTM). Reset hidden at episode boundaries (and optionally between
    bosses in a linear sequence — config-controlled).
    """

    def __init__(
        self,
        obs_dims: dict[str, int],
        entity_hidden: int = 128,
        attention_layers: int = 2,
        attention_heads: int = 4,
        rnn_type: str = "gru",
        rnn_hidden: int = 256,
        enable_macro: bool = True,
        max_entities: int = 64,
    ) -> None:
        super().__init__()
        self.rnn_type = rnn_type
        self.rnn_hidden = rnn_hidden
        # TODO(phase-5): GlobalEncoder, PlayerEncoder, EntityEncoder,
        # EntityTransformerEncoder / PlayerCrossAttention, prev-action embedding,
        # GRU/LSTM, HybridPolicyHead, ValueHead.

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> RnnState:
        raise NotImplementedError  # TODO(phase-5): zeros((1, B, rnn_hidden)) [(h,c) for LSTM]

    def forward(
        self, obs: dict[str, Tensor], rnn_state: RnnState = None, action_mask: Tensor | None = None
    ) -> tuple[Any, Tensor, RnnState]:
        raise NotImplementedError  # TODO(phase-5)

    def act(
        self,
        obs: dict[str, Tensor],
        rnn_state: RnnState = None,
        action_mask: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, RnnState]:
        raise NotImplementedError  # TODO(phase-5)

    def evaluate_actions(
        self,
        obs: dict[str, Tensor],
        actions: Tensor,
        rnn_state: RnnState = None,
        action_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError  # TODO(phase-5): sequence forward for BPTT
