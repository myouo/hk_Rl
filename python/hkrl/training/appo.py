"""Asynchronous PPO / IMPALA-style learner (PRD Phase 6/8, §9.5).

For async multi-worker sampling where rollouts may be off-policy. Filters/
down-weights batches by ``policy_version`` and applies importance correction
(V-trace) to tolerate bounded staleness (docs/distributed_training.md §4).
"""

from __future__ import annotations

from hkrl.models.base import ActorCritic
from hkrl.training.rollout_buffer import RolloutBatch
from hkrl.utils.config import TrainConfig
from hkrl.utils.registry import register_algo


@register_algo("appo")
class APPO:
    """Async PPO with staleness handling.

    Accepts RolloutBatches from many workers, drops those older than a version
    threshold, and corrects for off-policyness.
    """

    def __init__(self, model: ActorCritic, config: TrainConfig, max_staleness: int = 4) -> None:
        self.model = model
        self.cfg = config
        self.max_staleness = max_staleness
        # TODO(phase-6): optimizer; V-trace; intake queue.

    def ingest(self, batch: RolloutBatch, current_version: int) -> bool:
        """Accept or reject a batch by staleness; returns True if used.

        TODO(phase-6): version filter + enqueue.
        """
        raise NotImplementedError

    def update(self) -> dict[str, float]:
        """Run an async update step over accepted batches; return metrics.

        TODO(phase-6): V-trace corrected PPO update.
        """
        raise NotImplementedError
