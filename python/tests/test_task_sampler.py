"""Task sampler tests."""

from __future__ import annotations

import pytest
from hkrl.coordinator.task_sampler import TaskSampler


def test_task_sampler_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="task_ids"):
        TaskSampler([])
    with pytest.raises(ValueError, match="replay_fraction"):
        TaskSampler(["a"], replay_fraction=1.5)
    with pytest.raises(ValueError, match="mastered_winrate"):
        TaskSampler(["a"], mastered_winrate=-0.1)


def test_task_sampler_updates_weights_and_mastered_set() -> None:
    sampler = TaskSampler(["a", "b"], mastered_winrate=0.8, seed=0)

    sampler.update_weights({"a": 0.9, "b": 0.1})

    assert sampler.weights["b"] > sampler.weights["a"]
    assert sampler.mastered_tasks == {"a"}


def test_task_sampler_replays_mastered_tasks_when_requested() -> None:
    sampler = TaskSampler(["a", "b"], replay_fraction=1.0, mastered_winrate=0.8, seed=0)
    sampler.update_weights({"a": 1.0, "b": 0.0})

    assert [sampler.sample() for _ in range(5)] == ["a", "a", "a", "a", "a"]


def test_task_sampler_prefers_low_winrate_active_task() -> None:
    sampler = TaskSampler(["strong", "weak"], replay_fraction=0.0, seed=3)
    sampler.update_weights({"strong": 0.9, "weak": 0.1})

    samples = [sampler.sample() for _ in range(100)]

    assert samples.count("weak") > samples.count("strong")
