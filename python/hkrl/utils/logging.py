"""Metric/log sinks with a pluggable backend.

Always: stdout + JSONL/CSV per-episode records (PRD §2.1). Optional: TensorBoard /
WandB via the ``logging`` extra. The training reward is recorded but is NOT the
decision metric — see docs/metrics.md.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import IO, Any, Protocol


class MetricSink(Protocol):
    """A destination for scalar metrics and episode records."""

    def log_scalar(self, key: str, value: float, step: int) -> None: ...

    def log_episode(self, record: dict[str, Any]) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class JsonlSink:
    """Append per-episode records as JSON lines; always available."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] | None = self.path.open("a", encoding="utf-8")

    def log_scalar(self, key: str, value: float, step: int) -> None:
        self._write(
            {
                "type": "scalar",
                "key": key,
                "value": float(value),
                "step": int(step),
            }
        )

    def log_episode(self, record: dict[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("type", "episode")
        self._write(payload)

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is None:
            return
        self._fh.close()
        self._fh = None

    def _write(self, payload: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("JsonlSink is closed")
        self._fh.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


class CsvSink:
    """Append metrics to CSV with a stable envelope schema.

    The default columns avoid header rewrites as episode payloads evolve. Scalar
    metrics use ``type,key,value,step``; episode records are stored as compact
    JSON in ``record``. Pass ``fieldnames`` for a custom wide CSV export.
    """

    DEFAULT_FIELDNAMES: tuple[str, ...] = ("type", "step", "key", "value", "record")

    def __init__(
        self, path: str | Path, fieldnames: list[str] | tuple[str, ...] | None = None
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = tuple(fieldnames or self.DEFAULT_FIELDNAMES)
        if not self.fieldnames:
            raise ValueError("fieldnames must not be empty")
        file_exists = self.path.exists() and self.path.stat().st_size > 0
        self._fh: IO[str] | None = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames, extrasaction="ignore")
        if not file_exists:
            self._writer.writeheader()

    def log_scalar(self, key: str, value: float, step: int) -> None:
        self._write(
            {
                "type": "scalar",
                "key": key,
                "value": float(value),
                "step": int(step),
                "record": "",
            }
        )

    def log_episode(self, record: dict[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("type", "episode")
        row = dict(payload)
        row["record"] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self._write(row)

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is None:
            return
        self._fh.close()
        self._fh = None

    def _write(self, row: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("CsvSink is closed")
        self._writer.writerow(row)


def make_sink(kind: str = "jsonl", **kwargs: Any) -> MetricSink:
    """Factory for a metric sink. Supports ``jsonl`` and ``csv``."""
    if kind == "jsonl":
        path = kwargs.pop("path", None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unknown JsonlSink kwargs: {unknown}")
        if path is None:
            raise TypeError("make_sink('jsonl') requires path=")
        return JsonlSink(path)
    if kind == "csv":
        path = kwargs.pop("path", None)
        fieldnames = kwargs.pop("fieldnames", None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unknown CsvSink kwargs: {unknown}")
        if path is None:
            raise TypeError("make_sink('csv') requires path=")
        return CsvSink(path, fieldnames=fieldnames)

    raise ValueError(f"unknown metric sink kind: {kind}")
