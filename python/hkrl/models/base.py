"""ActorCritic abstract base (docs/model_architecture.md §2).

All policies implement this interface so training code (PPO/RecurrentPPO/APPO) is
model-agnostic. The interface is mask- and recurrence-aware: ``entity_mask`` is
threaded through attention/pooling, and ``rnn_state`` carries recurrent memory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn

# A model's recurrent state is opaque to the training loop (tuple for LSTM, single
# tensor for GRU, or None for feedforward).
RnnState = Any


class ActorCritic(nn.Module, ABC):
    """Policy + value network over the entity-list observation.

    Conventions:
      * ``obs`` is a dict of batched tensors: global, player, entities, entity_mask.
      * ``action_mask`` (optional) zeroes invalid logits per head before sampling.
      * ``rnn_state`` is per-env recurrent memory; ``initial_state`` makes a zero one.
      * Action is composite (movement_x, aim_y, buttons, duration, [macro]) — heads
        return per-component distributions; log_prob/entropy sum across components.
    """

    @abstractmethod
    def initial_state(self, batch_size: int, device: torch.device | None = None) -> RnnState:
        """Return a zeroed recurrent state (or None for feedforward models)."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        obs: dict[str, Tensor],
        rnn_state: RnnState = None,
        action_mask: Tensor | None = None,
    ) -> tuple[Any, Tensor, RnnState]:
        """Return ``(policy_dists, value, next_rnn_state)`` for a batch/sequence."""
        raise NotImplementedError

    @abstractmethod
    def act(
        self,
        obs: dict[str, Tensor],
        rnn_state: RnnState = None,
        action_mask: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, RnnState]:
        """Sample for the rollout: return ``(action, log_prob, value, next_state)``."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_actions(
        self,
        obs: dict[str, Tensor],
        actions: Tensor,
        rnn_state: RnnState = None,
        action_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """For the PPO update: return ``(log_prob, entropy, value)`` of given actions."""
        raise NotImplementedError
