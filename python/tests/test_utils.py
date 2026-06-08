"""Utility tests."""

from __future__ import annotations

import random

import numpy as np
import pytest
from hkrl.utils.metrics import CORE_METRICS, EpisodeStats, RunningMeter
from hkrl.utils.seeding import seed_everything


def test_seed_everything_reproducibly_seeds_python_and_numpy() -> None:
    seed_everything(123)
    first = (random.random(), np.random.random())

    seed_everything(123)
    second = (random.random(), np.random.random())

    assert first == second


def test_seed_everything_rejects_negative_seed() -> None:
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_running_meter_tracks_windowed_mean_and_ema() -> None:
    meter = RunningMeter(window=3)

    assert meter.mean() == 0.0
    assert meter.ema_mean() == 0.0

    meter.update(1.0)
    meter.update(2.0)
    meter.update(3.0)
    meter.update(4.0)

    assert meter.mean() == 3.0
    assert meter.ema_mean() > 1.0


def test_running_meter_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError):
        RunningMeter(window=0)


def test_core_metrics_include_evaluator_outcomes() -> None:
    for key in (
        "heal_amount",
        "death_rate",
        "death_reason",
        "time_to_kill",
        "per_boss_win_rate",
        "per_boss_damage_ratio",
    ):
        assert key in CORE_METRICS


def test_episode_stats_tracks_heal_amount() -> None:
    stats = EpisodeStats(
        episode_id=3,
        task_id="boss",
        heal_count=2,
        heal_amount=4.0,
    )

    assert stats.heal_count == 2
    assert stats.heal_amount == 4.0
