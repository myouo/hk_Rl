"""Metric sink tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hkrl.utils.logging import JsonlSink, make_sink


def test_jsonl_sink_writes_scalar_and_episode_records(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "metrics.jsonl"
    sink = JsonlSink(path)

    sink.log_scalar("sps", 12.5, step=3)
    sink.log_episode({"episode": 7, "win_rate": 1.0})
    sink.flush()
    sink.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {"key": "sps", "step": 3, "type": "scalar", "value": 12.5},
        {"episode": 7, "type": "episode", "win_rate": 1.0},
    ]


def test_jsonl_sink_rejects_write_after_close(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path / "metrics.jsonl")
    sink.close()
    sink.close()

    with pytest.raises(RuntimeError, match="closed"):
        sink.log_scalar("reward", 1.0, step=1)


def test_make_sink_builds_jsonl_sink(tmp_path: Path) -> None:
    sink = make_sink("jsonl", path=tmp_path / "metrics.jsonl")

    assert isinstance(sink, JsonlSink)
    sink.close()


def test_make_sink_rejects_unknown_kind_and_kwargs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown metric sink kind"):
        make_sink("tensorboard", logdir=tmp_path)

    with pytest.raises(TypeError, match="requires path"):
        make_sink("jsonl")

    with pytest.raises(TypeError, match="unknown JsonlSink kwargs"):
        make_sink("jsonl", path=tmp_path / "metrics.jsonl", unexpected=True)
