"""Synchronous PPO (PRD Phase 3, ADR-0001).

Clipped-objective PPO over a flat RolloutBuffer. Model-agnostic via the
ActorCritic interface; logs the training metrics from docs/metrics.md
(policy_loss, value_loss, entropy, kl, explained_variance).
"""

from __future__ import annotations

from hkrl.models.base import ActorCritic
from hkrl.training.rollout_buffer import RolloutBuffer
from hkrl.utils.config import TrainConfig
from hkrl.utils.registry import register_algo


@register_algo("ppo")
class PPO:
    """Vanilla clipped PPO learner.

    Reserve performance levers on the update path: ``torch.compile`` and AMP
    (mixed precision) — see docs/model_architecture.md §5.
    """

    def __init__(self, model: ActorCritic, config: TrainConfig) -> None:
        self.model = model
        self.cfg = config
        # TODO(phase-3): optimizer (Adam), optional torch.compile, GradScaler.

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run ``epochs`` of minibatch updates; return training metrics.

        TODO(phase-3): compute advantages (normalized), clipped policy loss,
        value loss, entropy bonus; clip grad norm; return metrics dict.
        """
        raise NotImplementedError
