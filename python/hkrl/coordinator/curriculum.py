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
        raise NotImplementedError  # TODO(phase-7)

    def maybe_advance(self, per_task_winrate: dict[str, float], episodes: int) -> bool:
        """Advance to the next stage if criteria met; return True if advanced.

        TODO(phase-7): check promote_winrate + min_episodes for active stage.
        """
        raise NotImplementedError
