"""Metric sink tests."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest
from hkrl.utils.logging import CsvSink, JsonlSink, StdoutSink, make_sink


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


def test_metric_sinks_write_numpy_episode_values(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "metrics.jsonl"
    csv_path = tmp_path / "metrics.csv"
    stdout = io.StringIO()
    record = {
        "episode": np.int64(7),
        "reward": np.float32(1.5),
        "values": np.array([1, 2], dtype=np.int64),
    }

    jsonl = JsonlSink(jsonl_path)
    csv_sink = CsvSink(csv_path)
    stdout_sink = StdoutSink(stream=stdout)
    for sink in (jsonl, csv_sink, stdout_sink):
        sink.log_episode(record)
        sink.flush()
        sink.close()

    assert json.loads(jsonl_path.read_text(encoding="utf-8")) == {
        "episode": 7,
        "reward": 1.5,
        "type": "episode",
        "values": [1, 2],
    }
    with csv_path.open(encoding="utf-8", newline="") as fh:
        row = next(csv.DictReader(fh))
    assert json.loads(row["record"])["values"] == [1, 2]
    assert json.loads(stdout.getvalue())["reward"] == 1.5


def test_metric_sinks_normalize_non_finite_values(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    sink = JsonlSink(path)

    sink.log_episode(
        {
            "nan": float("nan"),
            "pos_inf": float("inf"),
            "np_nan": np.float32("nan"),
            "array": np.array([1.0, np.inf], dtype=np.float32),
        }
    )
    sink.close()

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "array": [1.0, None],
        "nan": None,
        "np_nan": None,
        "pos_inf": None,
        "type": "episode",
    }


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


def test_stdout_sink_writes_json_lines() -> None:
    stream = io.StringIO()
    sink = StdoutSink(stream=stream)

    sink.log_scalar("sps", 12.5, step=3)
    sink.log_episode({"episode": 7})
    sink.flush()

    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert records == [
        {"key": "sps", "step": 3, "type": "scalar", "value": 12.5},
        {"episode": 7, "type": "episode"},
    ]


def test_stdout_sink_rejects_write_after_close() -> None:
    sink = StdoutSink(stream=io.StringIO())
    sink.close()
    sink.close()

    with pytest.raises(RuntimeError, match="closed"):
        sink.log_scalar("reward", 1.0, step=1)


def test_make_sink_builds_jsonl_sink(tmp_path: Path) -> None:
    sink = make_sink("jsonl", path=tmp_path / "metrics.jsonl")

    assert isinstance(sink, JsonlSink)
    sink.close()


def test_make_sink_builds_csv_sink(tmp_path: Path) -> None:
    sink = make_sink("csv", path=tmp_path / "metrics.csv")

    assert isinstance(sink, CsvSink)
    sink.close()


def test_make_sink_builds_stdout_sink() -> None:
    stream = io.StringIO()
    sink = make_sink("stdout", stream=stream)

    assert isinstance(sink, StdoutSink)
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
    with pytest.raises(TypeError, match="unknown StdoutSink kwargs"):
        make_sink("stdout", unexpected=True)
