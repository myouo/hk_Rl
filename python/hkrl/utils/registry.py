"""Component registry — config-driven extensibility (invariant #4).

Models, transports, reward functions, tasks, and algorithms register by name and
are selected from YAML config. Adding a component requires no edits to core code:
decorate it with the matching ``@register_*`` and reference its name in config.

Example::

    @register_model("entity_attention_gru")
    class EntityAttentionGRU(ActorCritic): ...

    model = build("model", cfg.model.name, **cfg.model.kwargs)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

# kind -> {name -> object}
_REGISTRIES: dict[str, dict[str, type]] = {
    "model": {},
    "transport": {},
    "reward": {},
    "task": {},
    "algo": {},
}


def _register(kind: str, name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        table = _REGISTRIES.setdefault(kind, {})
        if name in table:
            raise ValueError(f"{kind} '{name}' already registered")
        table[name] = cls
        return cls

    return deco


def register_model(name: str) -> Callable[[type[T]], type[T]]:
    """Register an :class:`hkrl.models.base.ActorCritic` subclass."""
    return _register("model", name)


def register_transport(name: str) -> Callable[[type[T]], type[T]]:
    """Register a :class:`hkrl.transport.base.Transport` implementation."""
    return _register("transport", name)


def register_reward(name: str) -> Callable[[type[T]], type[T]]:
    """Register a reward function (events -> scalar)."""
    return _register("reward", name)


def register_task(name: str) -> Callable[[type[T]], type[T]]:
    """Register a task definition."""
    return _register("task", name)


def register_algo(name: str) -> Callable[[type[T]], type[T]]:
    """Register a training algorithm."""
    return _register("algo", name)


def get(kind: str, name: str) -> type:
    """Look up a registered class. Raises KeyError with available names."""
    table = _REGISTRIES.get(kind, {})
    if name not in table:
        raise KeyError(f"unknown {kind} '{name}'; available: {sorted(table)}")
    return table[name]


def build(kind: str, name: str, /, *args: object, **kwargs: object) -> object:
    """Instantiate a registered component by name."""
    return get(kind, name)(*args, **kwargs)


def available(kind: str) -> list[str]:
    """List registered names for a kind."""
    return sorted(_REGISTRIES.get(kind, {}))
