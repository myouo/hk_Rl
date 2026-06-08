"""Curriculum scheduler tests."""

from __future__ import annotations

from hkrl.coordinator.curriculum import Curriculum, CurriculumStage


def test_curriculum_empty_has_no_active_tasks_and_does_not_advance() -> None:
    curriculum = Curriculum()

    assert curriculum.active_tasks() == []
    assert not curriculum.maybe_advance({}, episodes=999)


def test_curriculum_advances_when_all_active_tasks_meet_gate() -> None:
    curriculum = Curriculum(
        stages=[
            CurriculumStage(["gruz", "hornet"], promote_winrate=0.6, min_episodes=10),
            CurriculumStage(["mantis"], promote_winrate=0.7, min_episodes=20),
        ]
    )

    assert curriculum.active_tasks() == ["gruz", "hornet"]
    assert not curriculum.maybe_advance({"gruz": 1.0, "hornet": 1.0}, episodes=9)
    assert not curriculum.maybe_advance({"gruz": 1.0, "hornet": 0.5}, episodes=10)
    assert curriculum.maybe_advance({"gruz": 0.6, "hornet": 0.8}, episodes=10)
    assert curriculum.index == 1
    assert curriculum.active_tasks() == ["mantis"]


def test_curriculum_does_not_advance_past_final_stage() -> None:
    curriculum = Curriculum(stages=[CurriculumStage(["gruz"], promote_winrate=0.1)])

    assert not curriculum.maybe_advance({"gruz": 1.0}, episodes=999)
    assert curriculum.index == 0


def test_curriculum_clamps_active_index() -> None:
    curriculum = Curriculum(stages=[CurriculumStage(["a"]), CurriculumStage(["b"])], index=99)

    assert curriculum.active_tasks() == ["b"]
    assert curriculum.index == 1
