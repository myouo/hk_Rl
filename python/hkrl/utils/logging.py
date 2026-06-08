"""Metric/log sinks with a pluggable backend.

Always: stdout + JSONL/CSV per-episode records (PRD §2.1). Optional: TensorBoard /
WandB via the ``logging`` extra. The training reward is recorded but is NOT the
decision metric — see docs/metrics.md.
"""

from __future__ import annotations

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


def make_sink(kind: str = "jsonl", **kwargs: Any) -> MetricSink:
    """Factory for a metric sink. Currently supports ``jsonl``."""
    if kind == "jsonl":
        path = kwargs.pop("path", None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unknown JsonlSink kwargs: {unknown}")
        if path is None:
            raise TypeError("make_sink('jsonl') requires path=")
        return JsonlSink(path)

    raise ValueError(f"unknown metric sink kind: {kind}")
