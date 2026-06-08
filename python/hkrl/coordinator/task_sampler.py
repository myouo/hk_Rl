"""Task / boss sampling (PRD §7, §9.7).

Balanced or weighted sampling across tasks to prevent catastrophic forgetting,
with optional old-task replay. The curriculum scheduler adjusts weights over time.
"""

from __future__ import annotations

from collections.abc import Sequence


class TaskSampler:
    """Samples a task_id per request according to a weighting policy.

    Default: balanced. ``replay_fraction`` reserves samples for previously-mastered
    tasks (old-task replay) to fight forgetting.
    """

    def __init__(self, task_ids: Sequence[str], replay_fraction: float = 0.2) -> None:
        self.task_ids = list(task_ids)
        self.replay_fraction = replay_fraction
        # TODO(phase-7): weights, mastered-set tracking, RNG (seeded).

    def sample(self) -> str:
        raise NotImplementedError  # TODO(phase-7)

    def update_weights(self, per_task_winrate: dict[str, float]) -> None:
        """Reweight toward weaker tasks; keep replay for mastered ones."""
        raise NotImplementedError  # TODO(phase-7)
