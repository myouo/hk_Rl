"""Metric definitions and aggregation (docs/metrics.md, PRD §13).

Capability is judged by shaping-free metrics (win rate, damage ratio, time-to-kill,
invalid-action ratio, generalization, old-task regression) — NOT training reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Canonical metric keys (keep in sync with docs/metrics.md §1).
CORE_METRICS: tuple[str, ...] = (
    "episode_reward",
    "win_rate",
    "episode_length",
    "damage_dealt",
    "damage_taken",
    "heal_count",
    "invalid_action_ratio",
    "action_entropy",
    "policy_kl",
    "value_loss",
    "policy_loss",
    "explained_variance",
    "sps",
    "reset_success_rate",
    "reset_duration",
    "worker_crash_count",
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
    invalid_actions: int = 0
    death_reason: int = 0
    extra: dict[str, float] = field(default_factory=dict)


class RunningMeter:
    """Windowed mean/EMA for streaming scalars.

    TODO(phase-2): implement deque-backed window + EMA.
    """

    def __init__(self, window: int = 100) -> None:
        self.window = window

    def update(self, value: float) -> None:
        raise NotImplementedError  # TODO(phase-2)

    def mean(self) -> float:
        raise NotImplementedError  # TODO(phase-2)
