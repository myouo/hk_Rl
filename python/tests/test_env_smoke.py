"""Env import/wiring smoke tests (no live game required).

Behavioral env tests need a live HKRLEnvMod connection and are marked
``integration`` (skipped by default). Here we only assert the class wires up.
"""

from __future__ import annotations

import pytest
from hkrl.utils.config import load_task_config


class DummyTransport:
    def __init__(self) -> None:
        self.closed = False

    def connect(self, timeout_s: float = 10.0) -> None:
        pass

    def send(self, frame: bytes) -> None:
        pass

    def recv(self, timeout_s: float | None = None) -> bytes:
        raise TimeoutError

    def is_connected(self) -> bool:
        return not self.closed

    def reconnect(self, timeout_s: float = 10.0) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_env_module_imports() -> None:
    from hkrl.env import HKRLEnv

    assert HKRLEnv is not None


def test_env_constructs_spaces_from_task_config() -> None:
    from hkrl.env import HKRLEnv

    task = load_task_config("../configs/tasks/gruz_mother.yaml")
    transport = DummyTransport()

    env = HKRLEnv(transport=transport, task=task)

    assert env.observation_space["entities"].shape[0] == task.observation.max_entities
    assert "macro" in env.action_space.spaces

    env.close()
    assert transport.closed


def test_registry_has_builtin_components() -> None:
    # Importing the packages registers their components by name.
    import hkrl.models.mlp
    import hkrl.models.recurrent_policy
    import hkrl.training.ppo
    import hkrl.transport.tcp  # noqa: F401
    from hkrl.utils import registry

    assert "mlp" in registry.available("model")
    assert "entity_attention_gru" in registry.available("model")
    assert "tcp" in registry.available("transport")
    assert "ppo" in registry.available("algo")


@pytest.mark.integration
def test_random_policy_episode() -> None:
    pytest.skip("requires live Hollow Knight + HKRLEnvMod (phase 1+)")
