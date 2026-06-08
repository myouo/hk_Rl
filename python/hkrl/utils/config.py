"""Typed configuration models + YAML loading/composition.

Pydantic models give validated, IDE-friendly config; YAML files in ``configs/``
provide values (see configs/base.yaml). Fields mirror PRD §12. Configs compose:
a train run references a task config and a model config.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
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
    max_entities: int = 64
    include_fsm_state: bool = True
    include_hitbox: bool = True
    tier: Literal["privileged", "reduced", "human_visible"] = "privileged"


class ActionConfig(StrictConfigModel):
    action_repeat: int = 2
    enable_macro_actions: bool = True
    n_macro_actions: int = Field(default=DEFAULT_N_MACROS, ge=0)


class TaskConfig(StrictConfigModel):
    """One boss/arena task (configs/tasks/*.yaml). Mirrors PRD §12.1."""

    task_id: str
    wire_id: int = Field(default=0, ge=0)
    scene: str
    difficulty: str = "attuned"
    time_limit_seconds: int = 180
    player: dict[str, Any] = Field(default_factory=dict)
    reward: RewardWeights = Field(default_factory=RewardWeights)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    action: ActionConfig = Field(default_factory=ActionConfig)


class ModelConfig(StrictConfigModel):
    """Selects + configures an ActorCritic (registry name + kwargs). PRD §12.2."""

    name: str = "entity_attention_gru"
    entity_hidden: int = 128
    attention_layers: int = 2
    attention_heads: int = 4
    rnn_type: Literal["gru", "lstm", "none"] = "gru"
    rnn_hidden: int = 256


class TransportConfig(StrictConfigModel):
    name: Literal["tcp", "shm"] = "tcp"
    host: str = "127.0.0.1"
    port: int = 5555


class LearnerRuntimeConfig(StrictConfigModel):
    """Remote learner runtime settings (docs/distributed_training.md §5)."""

    bind: str = "0.0.0.0:5600"
    max_staleness: int = 4
    checkpoint_dir: str = "checkpoints"
    publish_every_updates: int = 1


class CoordinatorRuntimeConfig(StrictConfigModel):
    """Coordinator runtime settings for worker fleets."""

    bind: str = "0.0.0.0:5610"
    num_workers: int = 1
    heartbeat_timeout_s: float = 30.0


class SecurityConfig(StrictConfigModel):
    """Runtime security toggles (PRD §9.10)."""

    bind_scope: Literal["lan", "localhost"] = "lan"
    require_token: bool = False
    auth_token_env: str = "HKRL_AUTH_TOKEN"


class TrainConfig(StrictConfigModel):
    """Training hyperparameters (configs/train/*.yaml). Mirrors PRD §12.2."""

    algorithm: Literal["ppo", "recurrent_ppo", "appo"] = "recurrent_ppo"
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    learning_rate: float = 3e-4
    rollout_steps: int = 2048
    minibatch_size: int = 256
    epochs: int = 4
    sequence_length: int = 32
    burn_in: int = 0
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    model: ModelConfig = Field(default_factory=ModelConfig)
    transport: TransportConfig = Field(default_factory=TransportConfig)
    learner: LearnerRuntimeConfig = Field(default_factory=LearnerRuntimeConfig)
    coordinator: CoordinatorRuntimeConfig = Field(default_factory=CoordinatorRuntimeConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    seed: int = 0


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
