"""Metric sink tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from hkrl.utils.logging import CsvSink, JsonlSink, make_sink


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


def test_csv_sink_writes_scalar_and_episode_records(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    sink = CsvSink(path)

    sink.log_scalar("sps", 12.5, step=3)
    sink.log_episode({"episode": 7, "win_rate": 1.0})
    sink.flush()
    sink.close()

    with path.open(encoding="utf-8", newline="") as fh:
        records = list(csv.DictReader(fh))

    assert records[0] == {
        "type": "scalar",
        "step": "3",
        "key": "sps",
        "value": "12.5",
        "record": "",
    }
    assert records[1]["type"] == "episode"
    assert json.loads(records[1]["record"]) == {
        "episode": 7,
        "type": "episode",
        "win_rate": 1.0,
    }


def test_csv_sink_supports_custom_fieldnames(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    sink = CsvSink(path, fieldnames=["type", "episode", "win_rate"])

    sink.log_episode({"episode": 7, "win_rate": 1.0, "ignored": True})
    sink.close()

    with path.open(encoding="utf-8", newline="") as fh:
        records = list(csv.DictReader(fh))
    assert records == [{"type": "episode", "episode": "7", "win_rate": "1.0"}]


def test_csv_sink_rejects_write_after_close(tmp_path: Path) -> None:
    sink = CsvSink(tmp_path / "metrics.csv")
    sink.close()
    sink.close()

    with pytest.raises(RuntimeError, match="closed"):
        sink.log_episode({"episode": 1})


def test_make_sink_builds_jsonl_sink(tmp_path: Path) -> None:
    sink = make_sink("jsonl", path=tmp_path / "metrics.jsonl")

    assert isinstance(sink, JsonlSink)
    sink.close()


def test_make_sink_builds_csv_sink(tmp_path: Path) -> None:
    sink = make_sink("csv", path=tmp_path / "metrics.csv")

    assert isinstance(sink, CsvSink)
    sink.close()


def test_make_sink_rejects_unknown_kind_and_kwargs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown metric sink kind"):
        make_sink("tensorboard", logdir=tmp_path)

    with pytest.raises(TypeError, match="requires path"):
        make_sink("jsonl")

    with pytest.raises(TypeError, match="unknown JsonlSink kwargs"):
        make_sink("jsonl", path=tmp_path / "metrics.jsonl", unexpected=True)
    with pytest.raises(TypeError, match="requires path"):
        make_sink("csv")
    with pytest.raises(TypeError, match="unknown CsvSink kwargs"):
        make_sink("csv", path=tmp_path / "metrics.csv", unexpected=True)
