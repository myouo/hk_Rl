"""Curriculum scheduler (PRD §3.2 plan A, §7, §9.7).

Drives difficulty progression and task introduction order. Plan A first
(per-boss training + curriculum sampling), Plan B later (one episode across
multiple bosses). Promotes a task once its win rate clears a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CurriculumStage:
    task_ids: list[str]
    promote_winrate: float = 0.5
    min_episodes: int = 200


@dataclass
class Curriculum:
    """Ordered stages; advances when the active stage is mastered."""

    stages: list[CurriculumStage] = field(default_factory=list)
    index: int = 0

    def active_tasks(self) -> list[str]:
        if not self.stages:
            return []
        self.index = min(max(self.index, 0), len(self.stages) - 1)
        return list(self.stages[self.index].task_ids)

    def maybe_advance(self, per_task_winrate: dict[str, float], episodes: int) -> bool:
        """Advance to the next stage if criteria met; return True if advanced."""
        if not self.stages or self.index >= len(self.stages) - 1:
            return False

        stage = self.stages[self.index]
        if episodes < stage.min_episodes:
            return False
        if not stage.task_ids:
            return False

        mastered = all(
            per_task_winrate.get(task_id, 0.0) >= stage.promote_winrate
            for task_id in stage.task_ids
        )
        if not mastered:
            return False

        self.index += 1
        return True
