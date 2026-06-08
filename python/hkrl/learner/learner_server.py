"""Learner server (Remote GPU): collect batches, update, publish (PRD §8.1).

Receives RolloutBatches from workers, filters by ``policy_version``, runs the
configured algorithm's update, and publishes new checkpoints to the registry.
Only large-batch training happens here — never real-time inference (ADR-0004).
"""

from __future__ import annotations

from hkrl.learner.checkpoint_registry import CheckpointRegistry
from hkrl.models.base import ActorCritic
from hkrl.utils.config import TrainConfig


class LearnerServer:
    """Hosts the training loop and the inbound rollout endpoint."""

    def __init__(
        self,
        model: ActorCritic,
        config: TrainConfig,
        registry: CheckpointRegistry,
        bind: str = "0.0.0.0:5600",
    ) -> None:
        self.model = model
        self.cfg = config
        self.registry = registry
        self.bind = bind
        # TODO(phase-6): build algo from registry(config.algorithm); intake server.

    def serve(self) -> None:
        """Run the receive->filter->update->publish loop.

        TODO(phase-6): accept batches (LAN/token-auth), version-filter, update,
        publish checkpoint on a cadence; export training metrics.
        """
        raise NotImplementedError
