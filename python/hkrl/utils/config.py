"""Typed configuration models + YAML loading/composition.

Pydantic models give validated, IDE-friendly config; YAML files in ``configs/``
provide values (see configs/base.yaml). Fields mirror PRD §12. Configs compose:
a train run references a task config and a model config.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from hkrl.spaces import DEFAULT_N_MACROS


class StrictConfigModel(BaseModel):
    """Config base that rejects unknown keys instead of silently ignoring typos."""

    model_config = ConfigDict(extra="forbid")


class RewardWeights(StrictConfigModel):
    """Reward term weights (docs/reward_design.md §3)."""

    boss_damage: float = 1.0
    player_damage: float = -8.0
    soul_gained: float = 0.5
    heal_amount: float = 2.0
    boss_kill: float = 100.0
    player_death: float = -100.0
    time_penalty: float = -0.001
    invalid_action: float = -0.01


class ObservationConfig(StrictConfigModel):
    max_entities: int = Field(default=64, ge=1)
    include_fsm_state: bool = True
    include_hitbox: bool = True
    tier: Literal["privileged", "reduced", "human_visible"] = "privileged"


class ActionConfig(StrictConfigModel):
    action_repeat: int = Field(default=2, ge=1, le=255)
    enable_macro_actions: bool = True
    n_macro_actions: int = Field(default=DEFAULT_N_MACROS, ge=0, le=DEFAULT_N_MACROS)


class TaskConfig(StrictConfigModel):
    """One boss/arena task (configs/tasks/*.yaml). Mirrors PRD §12.1."""

    task_id: str = Field(min_length=1)
    wire_id: int = Field(default=0, ge=0)
    scene: str = Field(min_length=1)
    difficulty: str = Field(default="attuned", min_length=1)
    time_limit_seconds: int = Field(default=180, ge=1)
    player: dict[str, Any] = Field(default_factory=dict)
    reward: RewardWeights = Field(default_factory=RewardWeights)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    action: ActionConfig = Field(default_factory=ActionConfig)


class ModelConfig(StrictConfigModel):
    """Selects + configures an ActorCritic (registry name + kwargs). PRD §12.2."""

    name: str = Field(default="entity_attention_gru", min_length=1)
    entity_hidden: int = Field(default=128, ge=1)
    attention_layers: int = Field(default=2, ge=1)
    attention_heads: int = Field(default=4, ge=1)
    rnn_type: Literal["gru", "lstm", "none"] = "gru"
    rnn_hidden: int = Field(default=256, ge=0)


class TransportConfig(StrictConfigModel):
    name: Literal["tcp", "shm"] = "tcp"
    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=5555, ge=1, le=65535)
    shm_name: str = Field(default="hkrl_env", min_length=1)
    req_slots: int = Field(default=8, ge=1)
    resp_slots: int = Field(default=8, ge=1)


class LearnerRuntimeConfig(StrictConfigModel):
    """Remote learner runtime settings (docs/distributed_training.md §5)."""

    bind: str = Field(default="127.0.0.1:5600", min_length=1)
    max_staleness: int = Field(default=4, ge=0)
    checkpoint_dir: str = Field(default="checkpoints", min_length=1)
    publish_every_updates: int = Field(default=1, ge=1)


class CoordinatorRuntimeConfig(StrictConfigModel):
    """Coordinator runtime settings for worker fleets."""

    bind: str = Field(default="127.0.0.1:5610", min_length=1)
    num_workers: int = Field(default=1, ge=1)
    heartbeat_timeout_s: float = Field(default=30.0, gt=0)


class SecurityConfig(StrictConfigModel):
    """Runtime security toggles (PRD §9.10)."""

    bind_scope: Literal["lan", "localhost"] = "lan"
    require_token: bool = False
    auth_token_env: str = Field(default="HKRL_AUTH_TOKEN", min_length=1)


class TrainConfig(StrictConfigModel):
    """Training hyperparameters (configs/train/*.yaml). Mirrors PRD §12.2."""

    algorithm: Literal["ppo", "recurrent_ppo", "appo"] = "recurrent_ppo"
    gamma: float = Field(default=0.995, ge=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    clip_range: float = Field(default=0.2, gt=0.0)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    rollout_steps: int = Field(default=2048, ge=1)
    minibatch_size: int = Field(default=256, ge=1)
    epochs: int = Field(default=4, ge=1)
    sequence_length: int = Field(default=32, ge=1)
    burn_in: int = Field(default=0, ge=0)
    entropy_coef: float = Field(default=0.01, ge=0.0)
    value_coef: float = Field(default=0.5, ge=0.0)
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    model: ModelConfig = Field(default_factory=ModelConfig)
    transport: TransportConfig = Field(default_factory=TransportConfig)
    learner: LearnerRuntimeConfig = Field(default_factory=LearnerRuntimeConfig)
    coordinator: CoordinatorRuntimeConfig = Field(default_factory=CoordinatorRuntimeConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    seed: int = Field(default=0, ge=0)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict (supports an optional top-level ``defaults``
    list for simple composition; deep-merge order = defaults then overrides).
    """
    return _load_yaml(Path(path).expanduser().resolve(), stack=())


def _load_yaml(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    if path in stack:
        cycle = " -> ".join(str(p) for p in (*stack, path))
        raise ValueError(f"cyclic config defaults: {cycle}")

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")

    defaults = data.pop("defaults", [])
    if defaults is None:
        defaults = []
    if not isinstance(defaults, list):
        raise ValueError(f"config defaults must be a list: {path}")

    merged: dict[str, Any] = {}
    for entry in defaults:
        if not isinstance(entry, str):
            raise ValueError(f"config default entries must be paths: {path}")

        default_path = (path.parent / entry).resolve()
        merged = _deep_merge(merged, _load_yaml(default_path, (*stack, path)))

    return _deep_merge(merged, data)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)

    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = value

    return merged


def load_train_config(path: str | Path) -> TrainConfig:
    """Load and validate a training config."""
    return TrainConfig.model_validate(load_yaml(path))


def load_task_config(path: str | Path) -> TaskConfig:
    """Load and validate a task config."""
    return TaskConfig.model_validate(load_yaml(path))


def validate_task_collection(
    tasks: Sequence[TaskConfig],
    *,
    context: str = "tasks",
) -> None:
    """Validate a multi-task set before worker/evaluator/coordinator startup.

    ``task_id`` names are used by dashboards/eval/curriculum; ``wire_id`` values
    are stored in rollout batches and sent over the mod protocol. Duplicates make
    multi-boss training ambiguous, so fail before opening live env connections.
    """
    duplicate_task_ids = _duplicate_task_ids(tasks)
    duplicate_wire_ids = _duplicate_wire_ids(tasks)
    if not duplicate_task_ids and not duplicate_wire_ids:
        return

    details: list[str] = []
    if duplicate_task_ids:
        details.append(f"duplicate task_id(s): {', '.join(duplicate_task_ids)}")
    if duplicate_wire_ids:
        formatted = ", ".join(
            f"{wire_id} ({', '.join(task_ids)})" for wire_id, task_ids in duplicate_wire_ids
        )
        details.append(f"duplicate wire_id(s): {formatted}")
    raise ValueError(f"{context} must have unique task_id and wire_id; {'; '.join(details)}")


def _duplicate_task_ids(tasks: Sequence[TaskConfig]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for task in tasks:
        task_id = task.task_id
        if task_id in seen and task_id not in duplicates:
            duplicates.append(task_id)
        seen.add(task_id)
    return duplicates


def _duplicate_wire_ids(tasks: Sequence[TaskConfig]) -> list[tuple[int, list[str]]]:
    wire_to_tasks: dict[int, list[str]] = {}
    for task in tasks:
        wire_to_tasks.setdefault(task.wire_id, []).append(task.task_id)
    return [(wire_id, task_ids) for wire_id, task_ids in wire_to_tasks.items() if len(task_ids) > 1]


def resolve_auth_token(
    config: TrainConfig,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve the TCP auth token required by a TrainConfig.

    Returns ``None`` when token auth is disabled. When
    ``security.require_token`` is true, the token must be present in
    ``security.auth_token_env`` and must not be empty.
    """
    if not config.security.require_token:
        return None

    env = os.environ if environ is None else environ
    token = env.get(config.security.auth_token_env)
    if token is None or token == "":
        raise ValueError(
            "security.require_token is true but environment variable "
            f"{config.security.auth_token_env!r} is not set"
        )
    return token


def validate_bind_address(bind: str, bind_scope: Literal["lan", "localhost"]) -> str:
    """Validate a runtime service bind address against the configured scope."""
    host, port = _split_bind(bind)
    if not 0 <= port <= 65535:
        raise ValueError(f"bind port must be in [0, 65535], got {port}")

    if bind_scope == "localhost":
        if not _is_loopback_host(host):
            raise ValueError(
                f"bind_scope='localhost' requires a loopback bind address, got {bind!r}"
            )
        return bind

    if bind_scope == "lan":
        if _is_unspecified_host(host):
            raise ValueError(
                f"bind_scope='lan' requires an explicit loopback or private LAN "
                f"bind address, got wildcard bind {bind!r}"
            )
        if _is_public_ip_literal(host):
            raise ValueError(f"bind_scope='lan' must not bind to public IP {host!r}")
        return bind

    raise ValueError(f"unsupported bind_scope {bind_scope!r}")


def validate_service_auth(bind: str, config: TrainConfig) -> None:
    """Require token auth for any service reachable beyond loopback."""
    host, _ = _split_bind(bind)
    if not _is_loopback_host(host) and not config.security.require_token:
        raise ValueError("non-loopback service bind requires security.require_token=true")


def _split_bind(bind: str) -> tuple[str, int]:
    if bind.startswith("["):
        host, sep, rest = bind[1:].partition("]")
        if sep != "]" or not rest.startswith(":"):
            raise ValueError(f"bind must be host:port, got {bind!r}")
        port_text = rest[1:]
    else:
        host, sep, port_text = bind.rpartition(":")
        if sep != ":" or not host:
            raise ValueError(f"bind must be host:port, got {bind!r}")

    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"bind port must be an integer, got {port_text!r}") from exc
    return host, port


def _is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _is_public_ip_literal(host: str) -> bool:
    try:
        address = ip_address(host)
    except ValueError:
        return False
    if address.is_unspecified:
        return False
    return address.is_global


def _is_unspecified_host(host: str) -> bool:
    try:
        return ip_address(host).is_unspecified
    except ValueError:
        return False
