"""Typed configuration models + YAML loading/composition.

Pydantic models give validated, IDE-friendly config; YAML files in ``configs/``
provide values (see configs/base.yaml). Fields mirror PRD §12. Configs compose:
a train run references a task config and a model config.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RewardWeights(BaseModel):
    """Reward term weights (docs/reward_design.md §3)."""

    boss_damage: float = 1.0
    player_damage: float = -8.0
    soul_gained: float = 0.5
    heal_amount: float = 2.0
    boss_kill: float = 100.0
    player_death: float = -100.0
    time_penalty: float = -0.001
    invalid_action: float = -0.01


class ObservationConfig(BaseModel):
    max_entities: int = 64
    include_fsm_state: bool = True
    include_hitbox: bool = True
    tier: str = "privileged"  # privileged | reduced | human_visible


class ActionConfig(BaseModel):
    action_repeat: int = 2
    enable_macro_actions: bool = True


class TaskConfig(BaseModel):
    """One boss/arena task (configs/tasks/*.yaml). Mirrors PRD §12.1."""

    task_id: str
    scene: str
    difficulty: str = "attuned"
    time_limit_seconds: int = 180
    player: dict[str, Any] = Field(default_factory=dict)
    reward: RewardWeights = Field(default_factory=RewardWeights)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    action: ActionConfig = Field(default_factory=ActionConfig)


class ModelConfig(BaseModel):
    """Selects + configures an ActorCritic (registry name + kwargs). PRD §12.2."""

    name: str = "entity_attention_gru"
    entity_hidden: int = 128
    attention_layers: int = 2
    attention_heads: int = 4
    rnn_type: str = "gru"  # gru | lstm | none
    rnn_hidden: int = 256


class TransportConfig(BaseModel):
    name: str = "tcp"  # tcp | shm
    host: str = "127.0.0.1"
    port: int = 5555


class TrainConfig(BaseModel):
    """Training hyperparameters (configs/train/*.yaml). Mirrors PRD §12.2."""

    algorithm: str = "recurrent_ppo"  # ppo | recurrent_ppo | appo
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
