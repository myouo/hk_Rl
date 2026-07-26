"""Validated aggregate configuration for a remote-learner experiment.

TrainConfig remains the source of algorithm/model hyperparameters. This module
adds one launch manifest that binds those settings to task files and the remote
learner/local GameWorker roles without duplicating values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field

from hkrl.utils.config import (
    StrictConfigModel,
    TaskConfig,
    TrainConfig,
    load_task_config,
    load_train_config,
    load_yaml,
    validate_task_collection,
)


class RemoteExperimentRole(StrictConfigModel):
    """Remote GPU learner and read-only checkpoint registry."""

    train_config: str = Field(min_length=1)
    learner_bind: str = Field(default="127.0.0.1:5600", min_length=1)
    registry_bind: str = Field(default="127.0.0.1:5601", min_length=1)


class LocalExperimentRole(StrictConfigModel):
    """Game-host inference worker; the learner endpoints are normally SSH forwards."""

    train_config: str = Field(min_length=1)
    env_host: str = Field(default="127.0.0.1", min_length=1)
    env_port: int = Field(default=5555, ge=1, le=65535)
    learner_endpoint: str = Field(default="127.0.0.1:5600", min_length=1)
    registry_endpoint: str = Field(
        default="http://127.0.0.1:5601/",
        min_length=1,
    )
    worker_id: str = Field(default="game-worker-0", min_length=1)
    inference_threads: int = Field(default=1, ge=1)
    time_scale: float = Field(default=1.0, gt=0.0)
    batch_dir: str = Field(default="runs/game-worker/batches", min_length=1)
    heartbeat_jsonl: str = Field(
        default="runs/game-worker/heartbeats.jsonl",
        min_length=1,
    )
    max_consecutive_failures: int = Field(default=3, ge=0)


class ExperimentConfig(StrictConfigModel):
    """One self-contained distributed experiment launch manifest."""

    name: str = Field(min_length=1)
    project_root: str = Field(default="../..", min_length=1)
    tasks: list[str] = Field(min_length=1)
    remote: RemoteExperimentRole
    local: LocalExperimentRole


@dataclass(frozen=True)
class ResolvedExperiment:
    """Absolute paths plus the validated typed config objects."""

    manifest_path: Path
    project_root: Path
    config: ExperimentConfig
    remote_train_path: Path
    local_train_path: Path
    task_paths: tuple[Path, ...]
    remote_train: TrainConfig
    local_train: TrainConfig
    tasks: tuple[TaskConfig, ...]
    batch_dir: Path
    heartbeat_jsonl: Path


def load_experiment(path: str | Path) -> ResolvedExperiment:
    """Load an aggregate manifest and fail before opening any service or game."""
    manifest_path = Path(path).expanduser().resolve()
    config = ExperimentConfig.model_validate(load_yaml(manifest_path))
    project_root = (manifest_path.parent / config.project_root).resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"experiment project_root does not exist: {project_root}")

    remote_train_path = _required_file(project_root, config.remote.train_config)
    local_train_path = _required_file(project_root, config.local.train_config)
    task_paths = tuple(_required_file(project_root, item) for item in config.tasks)
    remote_train = load_train_config(remote_train_path)
    local_train = load_train_config(local_train_path)
    tasks = tuple(load_task_config(item) for item in task_paths)
    validate_task_collection(tasks, context="experiment tasks")
    _validate_training_contract(remote_train, local_train)
    _validate_task_layouts(tasks)
    _split_endpoint(config.remote.learner_bind, name="remote.learner_bind", allow_zero=True)
    _split_endpoint(config.remote.registry_bind, name="remote.registry_bind", allow_zero=True)
    _split_endpoint(config.local.learner_endpoint, name="local.learner_endpoint")
    _validate_registry_endpoint(config.local.registry_endpoint)

    return ResolvedExperiment(
        manifest_path=manifest_path,
        project_root=project_root,
        config=config,
        remote_train_path=remote_train_path,
        local_train_path=local_train_path,
        task_paths=task_paths,
        remote_train=remote_train,
        local_train=local_train,
        tasks=tasks,
        batch_dir=_resolve_output(project_root, config.local.batch_dir),
        heartbeat_jsonl=_resolve_output(project_root, config.local.heartbeat_jsonl),
    )


def _required_file(project_root: Path, value: str) -> Path:
    path = _resolve_under_root(project_root, value)
    if not path.is_file():
        raise FileNotFoundError(f"experiment file does not exist: {path}")
    return path


def _resolve_output(project_root: Path, value: str) -> Path:
    return _resolve_under_root(project_root, value)


def _resolve_under_root(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _validate_training_contract(remote: TrainConfig, local: TrainConfig) -> None:
    """Reject worker/learner algorithm drift while permitting role-only settings."""
    excluded = {"transport", "learner", "coordinator", "security"}
    remote_core = remote.model_dump(exclude=excluded)
    local_core = local.model_dump(exclude=excluded)
    if remote_core != local_core:
        raise ValueError(
            "remote and local train configs disagree on algorithm/model/hyperparameters"
        )


def _validate_task_layouts(tasks: tuple[TaskConfig, ...]) -> None:
    base = tasks[0]
    for task in tasks[1:]:
        if task.observation != base.observation:
            raise ValueError("experiment tasks must share one observation layout")
        if (
            task.action.enable_macro_actions != base.action.enable_macro_actions
            or task.action.n_macro_actions != base.action.n_macro_actions
        ):
            raise ValueError("experiment tasks must share one action vector layout")


def _split_endpoint(
    endpoint: str,
    *,
    name: str,
    allow_zero: bool = False,
) -> tuple[str, int]:
    host, separator, raw_port = endpoint.rpartition(":")
    if not separator or not host.strip():
        raise ValueError(f"{name} must be host:port")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError(f"{name} must be host:port") from exc
    minimum = 0 if allow_zero else 1
    if not minimum <= port <= 65535:
        raise ValueError(f"{name} port must be in [{minimum}, 65535]")
    return host.strip(), port


def _validate_registry_endpoint(endpoint: str) -> None:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError("local.registry_endpoint must be an http(s) URL")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("local.registry_endpoint contains an invalid port") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("local.registry_endpoint port must be in [1, 65535]")
