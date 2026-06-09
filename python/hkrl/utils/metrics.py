"""Metric definitions and aggregation (docs/metrics.md, PRD §13).

Capability is judged by shaping-free metrics (win rate, damage ratio, time-to-kill,
invalid-action ratio, generalization, old-task regression) — NOT training reward.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

# Canonical metric keys (keep in sync with docs/metrics.md §1).
CORE_METRICS: tuple[str, ...] = (
    "episode_reward",
    "win_rate",
    "episode_length",
    "damage_dealt",
    "damage_taken",
    "heal_count",
    "heal_amount",
    "death_rate",
    "death_reason",
    "invalid_action_ratio",
    "time_to_kill",
    "action_entropy",
    "policy_kl",
    "value_loss",
    "policy_loss",
    "explained_variance",
    "sps",
    "reset_success_rate",
    "reset_duration",
    "worker_crash_count",
    "worker_learner_upload_submitted_batches",
    "worker_learner_upload_accepted_batches",
    "worker_learner_upload_rejected_batches",
    "worker_learner_upload_failed_batches",
    "worker_policy_lag_max",
    "worker_checkpoint_lag_max",
    "stale_policy_worker_count",
    "stale_checkpoint_worker_count",
    "recovering_worker_count",
    "per_boss_win_rate",
    "per_boss_damage_ratio",
)


@dataclass
class EpisodeStats:
    """Accumulated stats for one episode; serialized via logging.log_episode."""

    episode_id: int
    task_id: str
    won: bool = False
    reward: float = 0.0
    length: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    heal_count: int = 0
    heal_amount: float = 0.0
    invalid_actions: int = 0
    death_reason: int = 0
    extra: dict[str, float] = field(default_factory=dict)


class RunningMeter:
    """Windowed mean/EMA for streaming scalars."""

    def __init__(self, window: int = 100) -> None:
        if window <= 0:
            raise ValueError("window must be positive")

        self.window = window
        self.values: deque[float] = deque(maxlen=window)
        self.ema: float | None = None
        self.ema_alpha = 2.0 / (window + 1.0)

    def update(self, value: float) -> None:
        value = float(value)
        self.values.append(value)

        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.ema_alpha * value + (1.0 - self.ema_alpha) * self.ema

    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def ema_mean(self) -> float:
        """Return the exponential moving average, or 0.0 before any update."""
        return 0.0 if self.ema is None else self.ema
