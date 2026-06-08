"""Task / boss sampling (PRD §7, §9.7).

Balanced or weighted sampling across tasks to prevent catastrophic forgetting,
with optional old-task replay. The curriculum scheduler adjusts weights over time.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class TaskSampler:
    """Samples a task_id per request according to a weighting policy.

    Default: balanced. ``replay_fraction`` reserves samples for previously-mastered
    tasks (old-task replay) to fight forgetting.
    """

    def __init__(
        self,
        task_ids: Sequence[str],
        replay_fraction: float = 0.2,
        mastered_winrate: float = 0.8,
        seed: int | None = None,
    ) -> None:
        self.task_ids = list(task_ids)
        if not self.task_ids:
            raise ValueError("task_ids must not be empty")
        if not 0.0 <= replay_fraction <= 1.0:
            raise ValueError("replay_fraction must be in [0, 1]")
        if not 0.0 <= mastered_winrate <= 1.0:
            raise ValueError("mastered_winrate must be in [0, 1]")

        self.replay_fraction = replay_fraction
        self.mastered_winrate = mastered_winrate
        self.weights = {task_id: 1.0 for task_id in self.task_ids}
        self.mastered_tasks: set[str] = set()
        self._rng = np.random.default_rng(seed)

    def sample(self) -> str:
        if self.mastered_tasks and self._rng.random() < self.replay_fraction:
            return self._weighted_sample(sorted(self.mastered_tasks))

        active_tasks = [task_id for task_id in self.task_ids if task_id not in self.mastered_tasks]
        if not active_tasks:
            active_tasks = self.task_ids
        return self._weighted_sample(active_tasks)

    def update_weights(self, per_task_winrate: dict[str, float]) -> None:
        """Reweight toward weaker tasks; keep replay for mastered ones."""
        for task_id in self.task_ids:
            winrate = float(np.clip(per_task_winrate.get(task_id, 0.0), 0.0, 1.0))
            self.weights[task_id] = max(0.05, 1.0 - winrate)
            if winrate >= self.mastered_winrate:
                self.mastered_tasks.add(task_id)
            else:
                self.mastered_tasks.discard(task_id)

    def _weighted_sample(self, task_ids: Sequence[str]) -> str:
        weights = np.asarray([self.weights[task_id] for task_id in task_ids], dtype=np.float64)
        total = float(weights.sum())
        probabilities: np.ndarray
        if total <= 0.0:
            probabilities = np.full((len(task_ids),), 1.0 / len(task_ids), dtype=np.float64)
        else:
            probabilities = weights / total
        return str(self._rng.choice(list(task_ids), p=probabilities))
