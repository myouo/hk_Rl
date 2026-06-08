"""Recurrent PPO over sequences (PRD Phase 5).

PPO trained with truncated BPTT on the RecurrentRolloutBuffer. Handles hidden
state propagation within a sequence, burn-in, and padded-timestep masking so the
loss only counts valid steps (docs/model_architecture.md §4).
"""

from __future__ import annotations

from hkrl.models.base import ActorCritic
from hkrl.training.recurrent_buffer import RecurrentRolloutBuffer
from hkrl.utils.config import TrainConfig
from hkrl.utils.registry import register_algo


@register_algo("recurrent_ppo")
class RecurrentPPO:
    """PPO learner for recurrent ActorCritic models."""

    def __init__(self, model: ActorCritic, config: TrainConfig) -> None:
        self.model = model
        self.cfg = config
        # TODO(phase-5): optimizer; sequence-aware update; torch.compile/AMP.

    def update(self, buffer: RecurrentRolloutBuffer) -> dict[str, float]:
        """Sequence-minibatch PPO update with masked loss; returns metrics.

        TODO(phase-5): iterate buffer.iter_sequences(); evaluate_actions over the
        sequence; apply loss_mask; clipped objective + value + entropy.
        """
        raise NotImplementedError
