"""MLP Actor-Critic baseline (PRD Phase 3).

Non-recurrent, fixed-vector model for the single-boss baseline. Concatenates
global + player + (flattened, masked) entity features and runs an MLP trunk into
the hybrid heads. Useful as the ablation floor vs attention / attention+GRU.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from hkrl.models.base import ActorCritic, RnnState
from hkrl.utils.registry import register_model


@register_model("mlp")
class MlpActorCritic(ActorCritic):
    """Feedforward baseline. ``initial_state`` returns None (no recurrence)."""

    def __init__(
        self,
        obs_dims: dict[str, int],
        hidden: int = 256,
        enable_macro: bool = True,
    ) -> None:
        super().__init__()
        # TODO(phase-3): trunk MLP + HybridPolicyHead + ValueHead.

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> RnnState:
        return None

    def forward(
        self, obs: dict[str, Tensor], rnn_state: RnnState = None, action_mask: Tensor | None = None
    ) -> tuple[Any, Tensor, RnnState]:
        raise NotImplementedError  # TODO(phase-3)

    def act(
        self,
        obs: dict[str, Tensor],
        rnn_state: RnnState = None,
        action_mask: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, RnnState]:
        raise NotImplementedError  # TODO(phase-3)

    def evaluate_actions(
        self,
        obs: dict[str, Tensor],
        actions: Tensor,
        rnn_state: RnnState = None,
        action_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError  # TODO(phase-3)
