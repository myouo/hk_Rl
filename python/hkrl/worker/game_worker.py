"""GameWorker: the local sampling loop (PRD §8.1, invariant #1).

Runs entirely on the Game PC: local inference, env stepping, rollout buffering,
batch upload, checkpoint pulling, and crash/reconnect handling. The action loop
NEVER crosses the remote network.
"""

from __future__ import annotations

from hkrl.env import HKRLEnv
from hkrl.models.base import ActorCritic
from hkrl.utils.config import TrainConfig
from hkrl.worker.checkpoint_client import CheckpointClient


class GameWorker:
    """Owns one (or a few) HKRLEnv, a local policy, and a rollout buffer.

    Loop: ``act -> step -> buffer.add``; on full buffer upload a RolloutBatch; on a
    new checkpoint hot-swap weights (PRD Phase 6 milestone).
    """

    def __init__(
        self,
        env: HKRLEnv,
        model: ActorCritic,
        config: TrainConfig,
        checkpoint_client: CheckpointClient | None = None,
        learner_endpoint: str | None = None,
    ) -> None:
        self.env = env
        self.model = model
        self.cfg = config
        self.checkpoint_client = checkpoint_client
        self.learner_endpoint = learner_endpoint
        # TODO(phase-6): rollout buffer, uploader, heartbeat, policy_version.

    def run(self, total_steps: int | None = None) -> None:
        """Sampling loop. Handles reset failures, reconnect, and weight reloads.

        TODO(phase-2 local-only / phase-6 remote): implement obs->act->step->buffer;
        upload on full; reload on new checkpoint.
        """
        raise NotImplementedError

    def collect_rollout(self) -> object:
        """Fill one rollout and return a RolloutBatch.

        TODO(phase-3): single-buffer fill + GAE.
        """
        raise NotImplementedError
