"""Learner server (Remote GPU): collect batches, update, publish (PRD §8.1).

Receives RolloutBatches from workers, filters by ``policy_version``, runs the
configured algorithm's update, and publishes new checkpoints to the registry.
Only large-batch training happens here — never real-time inference (ADR-0004).
"""

from __future__ import annotations

from typing import Any

from hkrl.learner.checkpoint_registry import CheckpointMeta, CheckpointRegistry
from hkrl.models.base import ActorCritic
from hkrl.training import appo as _appo  # noqa: F401
from hkrl.training import ppo as _ppo  # noqa: F401
from hkrl.training import recurrent_ppo as _recurrent_ppo  # noqa: F401
from hkrl.training.rollout_buffer import RolloutBatch
from hkrl.utils.config import TrainConfig
from hkrl.utils.registry import get


class LearnerServer:
    """Hosts the training loop and the inbound rollout endpoint."""

    def __init__(
        self,
        model: ActorCritic,
        config: TrainConfig,
        registry: CheckpointRegistry,
        bind: str = "127.0.0.1:5600",
        max_staleness: int = 4,
        publish_every_updates: int = 1,
    ) -> None:
        if publish_every_updates <= 0:
            raise ValueError("publish_every_updates must be positive")

        self.model = model
        self.cfg = config
        self.registry = registry
        self.bind = bind
        self.publish_every_updates = publish_every_updates
        self.algo = _build_algorithm(model, config, max_staleness=max_staleness)
        self.policy_version = int(getattr(self.algo, "current_version", 0))
        self.update_count = 0
        self.accepted_batches = 0
        self.rejected_batches = 0
        self.last_metrics: dict[str, float] = {}
        self.last_checkpoint: CheckpointMeta | None = None

    def submit(self, batch: RolloutBatch) -> bool:
        """Submit one worker rollout batch for the next learner update."""
        ingest = getattr(self.algo, "ingest", None)
        if ingest is None:
            raise TypeError(f"algorithm {self.cfg.algorithm!r} does not accept RolloutBatch intake")

        accepted = bool(ingest(batch, current_version=self.policy_version))
        if accepted:
            self.accepted_batches += 1
        else:
            self.rejected_batches += 1
        return accepted

    def update_once(self) -> dict[str, float]:
        """Run one learner update over queued batches and publish as configured."""
        metrics = self.algo.update()
        self.update_count += 1
        self.policy_version = int(getattr(self.algo, "current_version", self.policy_version + 1))
        self.last_metrics = {key: float(value) for key, value in metrics.items()}

        if self.update_count % self.publish_every_updates == 0:
            self.last_checkpoint = self.registry.publish(
                {
                    "model_state_dict": self.model.state_dict(),
                    "policy_version": self.policy_version,
                    "update": self.update_count,
                    "metrics": self.last_metrics,
                },
                policy_version=self.policy_version,
                step=self.update_count,
            )
        return self.last_metrics

    def serve(self) -> None:
        """Run the receive->filter->update->publish loop.

        The network listener is intentionally separate from the training core.
        For now this drains any in-process queued batches once; CLI/network
        wiring can call :meth:`submit` from an authenticated intake endpoint.
        """
        queued_batches = int(getattr(self.algo, "queued_batches", 0))
        if queued_batches > 0:
            self.update_once()


def _build_algorithm(model: ActorCritic, config: TrainConfig, *, max_staleness: int) -> Any:
    algo_cls = get("algo", config.algorithm)
    if config.algorithm == "appo":
        return algo_cls(model, config, max_staleness=max_staleness)
    return algo_cls(model, config)
