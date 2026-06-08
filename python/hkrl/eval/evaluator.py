"""Evaluator (PRD §4.2, §13).

Runs the policy on fixed seeds/tasks, isolated from training, and reports
shaping-free metrics: per-boss win rate, damage taken, time-to-kill, invalid
action ratio, generalization, and old-task regression. Guards against the
"reward up, win rate down" failure (docs/metrics.md §2, PRD §9.4).
"""

from __future__ import annotations

from collections.abc import Sequence

from hkrl.models.base import ActorCritic
from hkrl.utils.config import TaskConfig


class Evaluator:
    """Deterministic per-task evaluation harness."""

    def __init__(
        self,
        model: ActorCritic,
        tasks: Sequence[TaskConfig],
        seeds: Sequence[int],
    ) -> None:
        self.model = model
        self.tasks = list(tasks)
        self.seeds = list(seeds)
        # TODO(phase-3): build eval envs per task (train/eval isolated).

    def evaluate(self, episodes_per_task: int = 20) -> dict[str, dict[str, float]]:
        """Return ``{task_id: {win_rate, damage_taken, time_to_kill, ...}}``.

        TODO(phase-3): deterministic rollouts on fixed seeds; aggregate
        shaping-free metrics; optionally capture replays.
        """
        raise NotImplementedError

    def regression_report(
        self, baseline: dict[str, dict[str, float]], current: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Per-task win-rate delta vs a baseline (catastrophic-forgetting check).

        TODO(phase-7): implement diff.
        """
        raise NotImplementedError
