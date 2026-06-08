"""Metric/log sinks with a pluggable backend.

Always: stdout + JSONL/CSV per-episode records (PRD §2.1). Optional: TensorBoard /
WandB via the ``logging`` extra. The training reward is recorded but is NOT the
decision metric — see docs/metrics.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


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
        # TODO(phase-2): open file handle, ensure parent dir.

    def log_scalar(self, key: str, value: float, step: int) -> None:
        raise NotImplementedError  # TODO(phase-2)

    def log_episode(self, record: dict[str, Any]) -> None:
        raise NotImplementedError  # TODO(phase-2)

    def flush(self) -> None:
        raise NotImplementedError  # TODO(phase-2)

    def close(self) -> None:
        raise NotImplementedError  # TODO(phase-2)


def make_sink(kind: str = "jsonl", **kwargs: Any) -> MetricSink:
    """Factory for a metric sink. kind in {jsonl, tensorboard, wandb, multi}.

    TODO(phase-2/P1): tensorboard/wandb backends behind the ``logging`` extra.
    """
    raise NotImplementedError
