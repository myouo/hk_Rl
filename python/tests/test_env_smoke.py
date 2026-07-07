"""Env protocol and validation tests (no live game required).

Live rollout tests need a running HKRLEnvMod connection and are marked
``integration`` (skipped by default).
"""

from __future__ import annotations

import socket
import struct
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import flatbuffers
import numpy as np
import pytest
from hkrl import protocol
from hkrl.schema.HKRL import EntityState as FbEntityState
from hkrl.schema.HKRL import GlobalState as FbGlobalState
from hkrl.schema.HKRL import Observation as FbObservation
from hkrl.schema.HKRL import PlayerState as FbPlayerState
from hkrl.schema.HKRL import RewardEvent as FbRewardEvent
from hkrl.schema.HKRL import StepRequest as FbStepRequest
from hkrl.schema.HKRL import StepResponse as FbStepResponse
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


class ScriptedTransport:
    def __init__(self, handlers: list[Callable[[Any], bytes]]) -> None:
        self.handlers = handlers
        self.requests: list[Any] = []
        self.frames: list[bytes] = []
        self.connected = False
        self.closed = False

    def connect(self, timeout_s: float = 10.0) -> None:
        del timeout_s
        self.connected = True
        self.closed = False

    def send(self, frame: bytes) -> None:
        assert FbStepRequest.StepRequest.StepRequestBufferHasIdentifier(frame, 0)
        self.frames.append(frame)
        self.requests.append(FbStepRequest.StepRequest.GetRootAs(frame, 0))

    def recv(self, timeout_s: float | None = None) -> bytes:
        del timeout_s
        if not self.handlers:
            raise AssertionError("unexpected recv without scripted response")
        return self.handlers.pop(0)(self.requests[-1])

    def is_connected(self) -> bool:
        return self.connected and not self.closed

    def reconnect(self, timeout_s: float = 10.0) -> None:
        self.close()
        self.connect(timeout_s=timeout_s)

    def close(self) -> None:
        self.closed = True
        self.connected = False


class TcpModStub:
    def __init__(
        self,
        *,
        auth_token: str | None = None,
        response_lifecycles: list[protocol.LifecycleState] | None = None,
    ) -> None:
        self.auth_token = auth_token
        self.response_lifecycles = response_lifecycles or [protocol.LifecycleState.RUNNING]
        self.auth_payloads: list[bytes] = []
        self.requests: list[Any] = []
        self.error: BaseException | None = None
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._thread: threading.Thread | None = None

    def start(self) -> tuple[str, int]:
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        host, port = self._server.getsockname()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return str(host), int(port)

    def join(self, *, timeout_s: float) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
        self._server.close()
        if self._thread is not None and self._thread.is_alive():
            raise AssertionError("TCP mod stub did not stop")

    def _run(self) -> None:
        try:
            self._server.settimeout(2.0)
            conn, _ = self._server.accept()
            with conn:
                conn.settimeout(2.0)
                self._serve(conn)
        except BaseException as exc:
            self.error = exc
        finally:
            self._server.close()

    def _serve(self, conn: socket.socket) -> None:
        expected_requests = len(self.response_lifecycles)
        authenticated = self.auth_token is None
        while len(self.requests) < expected_requests:
            payload = _recv_tcp_frame(conn)
            if payload.startswith(b"HKRL_AUTH\0"):
                self.auth_payloads.append(payload)
                token = payload.removeprefix(b"HKRL_AUTH\0").decode("utf-8")
                if token != self.auth_token:
                    raise AssertionError("invalid auth token")
                authenticated = True
                continue
            if not authenticated:
                raise AssertionError("request arrived before auth frame")

            assert FbStepRequest.StepRequest.StepRequestBufferHasIdentifier(payload, 0)
            request = FbStepRequest.StepRequest.GetRootAs(payload, 0)
            self.requests.append(request)
            lifecycle = self.response_lifecycles[len(self.requests) - 1]
            response = _build_response(
                request,
                lifecycle=lifecycle,
                reward_kind=(
                    protocol.RewardEventKind.DAMAGE_DEALT
                    if len(self.requests) == expected_requests
                    else None
                ),
                reward_amount=2.0,
                server_tick=100 + len(self.requests),
            )
            conn.sendall(struct.pack("<I", len(response)) + response)


def test_env_module_imports() -> None:
    from hkrl.env import HKRLEnv

    assert HKRLEnv is not None


def test_env_constructs_spaces_from_task_config() -> None:
    from hkrl.env import HKRLEnv

    task = load_task_config("../configs/tasks/gruz_mother.yaml").model_copy(update={"wire_id": 11})
    transport = DummyTransport()

    env = HKRLEnv(transport=transport, task=task)

    assert env.observation_space["entities"].shape[0] == task.observation.max_entities
    assert "macro" in env.action_space.spaces
    assert env.action_space["macro"].n == task.action.n_macro_actions + 1

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


def test_env_reset_polls_until_running_and_returns_space_observation() -> None:
    from hkrl.env import HKRLEnv

    task = load_task_config("../configs/tasks/gruz_mother.yaml").model_copy(update={"wire_id": 11})
    transport = ScriptedTransport(
        [
            lambda req: _build_response(req, lifecycle=protocol.LifecycleState.COUNTDOWN),
            lambda req: _build_response(req, lifecycle=protocol.LifecycleState.RUNNING),
        ]
    )
    env = HKRLEnv(transport=transport, task=task)

    obs, info = env.reset(options={"reset_timeout_s": 1.0, "recv_timeout_s": 0.1})

    assert [protocol.Command(req.Command()) for req in transport.requests] == [
        protocol.Command.RESET,
        protocol.Command.STEP,
    ]
    assert all(req.ActionRepeat() == 1 for req in transport.requests)
    assert all(req.TaskId() == 11 for req in transport.requests)
    assert all(req.TaskScene() == task.scene.encode() for req in transport.requests)
    assert env.observation_space.contains(obs)
    assert obs["entities"].shape == env.observation_space["entities"].shape
    assert int(np.sum(obs["entity_mask"])) == 1
    assert info["lifecycle_state"] is protocol.LifecycleState.RUNNING
    assert info["episode_id"] == 123
    assert info["task_id"] == 11
    assert info["task_name"] == "gruz_mother"


def test_env_step_sends_action_repeat_and_composes_reward() -> None:
    from hkrl.env import HKRLEnv

    task = load_task_config("../configs/tasks/gruz_mother.yaml").model_copy(update={"wire_id": 12})
    transport = ScriptedTransport(
        [
            lambda req: _build_response(req, lifecycle=protocol.LifecycleState.RUNNING),
            lambda req: _build_response(
                req,
                lifecycle=protocol.LifecycleState.RUNNING,
                reward_kind=protocol.RewardEventKind.DAMAGE_DEALT,
                reward_amount=3.0,
                server_tick=102,
            ),
        ]
    )
    env = HKRLEnv(transport=transport, task=task)
    env.reset(options={"reset_timeout_s": 1.0, "recv_timeout_s": 0.1})

    obs, reward, terminated, truncated, info = env.step(
        {
            "movement_x": 2,
            "aim_y": 1,
            "buttons": {"attack": True},
            "duration": 0,
            "macro": 0,
        }
    )

    step_request = transport.requests[-1]
    assert protocol.Command(step_request.Command()) is protocol.Command.STEP
    assert step_request.ActionRepeat() == task.action.action_repeat
    assert step_request.TaskId() == 12
    assert step_request.TaskScene() == task.scene.encode()
    assert step_request.EnableMacroActions() is task.action.enable_macro_actions
    assert step_request.NMacroActions() == task.action.n_macro_actions
    assert step_request.Action().Buttons() == 1 << 3
    assert env.observation_space.contains(obs)
    assert reward == pytest.approx(3.0 + task.reward.time_penalty * task.action.action_repeat)
    assert terminated is False
    assert truncated is False
    assert info["reward_events"][0].kind is protocol.RewardEventKind.DAMAGE_DEALT


def test_env_reset_and_step_over_real_tcp_transport_with_auth() -> None:
    from hkrl.env import HKRLEnv
    from hkrl.transport.tcp import TcpTransport

    task = load_task_config(_repo_root() / "configs/tasks/gruz_mother.yaml").model_copy(
        update={"wire_id": 13}
    )
    server = TcpModStub(
        auth_token="secret",
        response_lifecycles=[
            protocol.LifecycleState.COUNTDOWN,
            protocol.LifecycleState.RUNNING,
            protocol.LifecycleState.RUNNING,
        ],
    )
    host, port = server.start()
    env = HKRLEnv(transport=TcpTransport(host, port, auth_token="secret"), task=task)

    try:
        obs, info = env.reset(options={"reset_timeout_s": 1.0, "recv_timeout_s": 0.5})
        _, reward, terminated, truncated, step_info = env.step(
            {
                "movement_x": 2,
                "aim_y": 1,
                "buttons": {"attack": True},
                "duration": 0,
                "macro": 1,
            }
        )
    finally:
        env.close()
        server.join(timeout_s=1.0)

    assert server.auth_payloads == [b"HKRL_AUTH\0secret"]
    assert [protocol.Command(req.Command()) for req in server.requests] == [
        protocol.Command.RESET,
        protocol.Command.STEP,
        protocol.Command.STEP,
    ]
    assert [req.TickId() for req in server.requests] == [0, 1, 2]
    assert all(req.TaskId() == 13 for req in server.requests)
    assert all(req.TaskScene() == task.scene.encode() for req in server.requests)
    assert all(
        req.EnableMacroActions() is task.action.enable_macro_actions for req in server.requests
    )
    assert all(req.NMacroActions() == task.action.n_macro_actions for req in server.requests)
    assert server.requests[-1].ActionRepeat() == task.action.action_repeat
    assert server.requests[-1].Action().Buttons() == 1 << 3
    assert server.requests[-1].Action().MacroId() == 0
    assert env.observation_space.contains(obs)
    assert info["lifecycle_state"] is protocol.LifecycleState.RUNNING
    assert step_info["lifecycle_state"] is protocol.LifecycleState.RUNNING
    assert reward == pytest.approx(2.0 + task.reward.time_penalty)
    assert terminated is False
    assert truncated is False
    assert server.error is None


def test_env_step_uses_server_tick_delta_for_reward_dt() -> None:
    from hkrl.env import HKRLEnv

    task = load_task_config("../configs/tasks/gruz_mother.yaml").model_copy(update={"wire_id": 12})
    transport = ScriptedTransport(
        [
            lambda req: _build_response(req, lifecycle=protocol.LifecycleState.RUNNING),
            lambda req: _build_response(
                req,
                lifecycle=protocol.LifecycleState.RUNNING,
                reward_kind=protocol.RewardEventKind.DAMAGE_DEALT,
                reward_amount=3.0,
                server_tick=101,
            ),
        ]
    )
    env = HKRLEnv(transport=transport, task=task)
    env.reset(options={"reset_timeout_s": 1.0, "recv_timeout_s": 0.1})

    _, reward, _, _, _ = env.step(env.action_space.sample())

    assert task.action.action_repeat == 2
    assert reward == pytest.approx(3.0 + task.reward.time_penalty)


def test_env_set_task_sends_wire_id_and_rebuilds_spaces() -> None:
    from hkrl.env import HKRLEnv

    initial_task = load_task_config("../configs/tasks/gruz_mother.yaml")
    next_task = load_task_config("../configs/tasks/hornet_protector.yaml")
    transport = ScriptedTransport(
        [
            lambda req: _build_response(req, lifecycle=protocol.LifecycleState.COUNTDOWN),
            lambda req: _build_response(req, lifecycle=protocol.LifecycleState.RUNNING),
        ]
    )
    env = HKRLEnv(transport=transport, task=initial_task)

    obs, info = env.set_task(
        next_task,
        options={"reset_timeout_s": 1.0, "recv_timeout_s": 0.1},
    )

    assert [protocol.Command(req.Command()) for req in transport.requests] == [
        protocol.Command.SET_TASK,
        protocol.Command.STEP,
    ]
    assert all(req.TaskId() == next_task.wire_id for req in transport.requests)
    assert all(req.TaskScene() == next_task.scene.encode() for req in transport.requests)
    assert env.task.task_id == "hornet_protector_attuned"
    assert env.observation_space["entities"].shape[0] == next_task.observation.max_entities
    assert env.observation_space.contains(obs)
    assert info["task_id"] == next_task.wire_id
    assert info["task_name"] == next_task.task_id
    assert info["lifecycle_state"] is protocol.LifecycleState.RUNNING


def test_env_step_requires_completed_reset() -> None:
    from hkrl.env import EnvProtocolError, HKRLEnv

    task = load_task_config("../configs/tasks/gruz_mother.yaml")
    env = HKRLEnv(transport=DummyTransport(), task=task)

    with pytest.raises(EnvProtocolError, match="reset must complete"):
        env.step(env.action_space.sample())


def test_env_control_commands_send_pause_resume_and_timescale() -> None:
    from hkrl.env import HKRLEnv

    task = load_task_config("../configs/tasks/gruz_mother.yaml").model_copy(update={"wire_id": 5})
    transport = ScriptedTransport(
        [
            lambda req: _build_response(
                req,
                lifecycle=protocol.LifecycleState.RUNNING,
                include_observation=False,
            ),
            lambda req: _build_response(
                req,
                lifecycle=protocol.LifecycleState.RUNNING,
                include_observation=False,
            ),
            lambda req: _build_response(
                req,
                lifecycle=protocol.LifecycleState.RUNNING,
                include_observation=False,
            ),
            lambda req: _build_response(
                req,
                lifecycle=protocol.LifecycleState.RUNNING,
                include_observation=False,
            ),
        ]
    )
    env = HKRLEnv(transport=transport, task=task)

    pause_info = env.pause(timeout_s=0.1)
    env.resume(timeout_s=0.1)
    ping_info = env.ping(timeout_s=0.1)
    env.set_timescale(2.5, timeout_s=0.1)

    assert [protocol.Command(req.Command()) for req in transport.requests] == [
        protocol.Command.PAUSE,
        protocol.Command.RESUME,
        protocol.Command.PING,
        protocol.Command.SET_TIMESCALE,
    ]
    assert all(req.ActionRepeat() == 1 for req in transport.requests)
    assert all(req.TaskId() == 5 for req in transport.requests)
    assert transport.requests[0].TimeScale() == 0.0
    assert transport.requests[1].TimeScale() == 0.0
    assert transport.requests[2].TimeScale() == 0.0
    assert transport.requests[3].TimeScale() == pytest.approx(2.5)
    assert pause_info["task_id"] == 5
    assert ping_info["server_tick"] == 102


def test_env_set_timescale_rejects_non_positive_scale() -> None:
    from hkrl.env import HKRLEnv

    task = load_task_config("../configs/tasks/gruz_mother.yaml")
    env = HKRLEnv(transport=DummyTransport(), task=task)

    with pytest.raises(ValueError, match="scale"):
        env.set_timescale(0.0)


def test_env_reset_surfaces_status_code() -> None:
    from hkrl.env import EnvProtocolError, HKRLEnv

    task = load_task_config("../configs/tasks/gruz_mother.yaml")
    transport = ScriptedTransport(
        [
            lambda req: _build_response(
                req,
                lifecycle=protocol.LifecycleState.WAIT_BOSS_READY,
                error_code=protocol.StatusCode.BOSS_NOT_FOUND,
                include_observation=False,
            )
        ]
    )
    env = HKRLEnv(transport=transport, task=task)

    with pytest.raises(EnvProtocolError, match="BOSS_NOT_FOUND"):
        env.reset(options={"reset_timeout_s": 1.0, "recv_timeout_s": 0.1})


def test_env_rejects_mismatched_response_env_id() -> None:
    from hkrl.env import EnvProtocolError, HKRLEnv

    task = load_task_config("../configs/tasks/gruz_mother.yaml")
    transport = ScriptedTransport(
        [
            lambda req: _build_response(
                req,
                lifecycle=protocol.LifecycleState.RUNNING,
                response_env_id=req.EnvId() + 1,
            )
        ]
    )
    env = HKRLEnv(transport=transport, task=task)

    with pytest.raises(EnvProtocolError, match="env_id mismatch"):
        env.reset(options={"reset_timeout_s": 1.0, "recv_timeout_s": 0.1})


def test_env_rejects_mismatched_response_tick_id() -> None:
    from hkrl.env import EnvProtocolError, HKRLEnv

    task = load_task_config("../configs/tasks/gruz_mother.yaml")
    transport = ScriptedTransport(
        [
            lambda req: _build_response(
                req,
                lifecycle=protocol.LifecycleState.RUNNING,
                response_tick_id=req.TickId() + 1,
            )
        ]
    )
    env = HKRLEnv(transport=transport, task=task)

    with pytest.raises(EnvProtocolError, match="tick_id mismatch"):
        env.reset(options={"reset_timeout_s": 1.0, "recv_timeout_s": 0.1})


def test_env_surfaces_unbound_error_response_before_tick_mismatch() -> None:
    from hkrl.env import EnvProtocolError, HKRLEnv

    task = load_task_config("../configs/tasks/gruz_mother.yaml")
    transport = ScriptedTransport(
        [
            lambda req: _build_response(req, lifecycle=protocol.LifecycleState.RUNNING),
            lambda req: _build_response(
                req,
                lifecycle=protocol.LifecycleState.RUNNING,
                error_code=protocol.StatusCode.INTERNAL_ERROR,
                include_observation=False,
                response_env_id=0,
                response_tick_id=0,
            ),
        ]
    )
    env = HKRLEnv(transport=transport, task=task)
    env.reset(options={"reset_timeout_s": 1.0, "recv_timeout_s": 0.1})

    with pytest.raises(EnvProtocolError, match="step failed with INTERNAL_ERROR"):
        env.step(env.action_space.sample())


def test_env_rejects_non_running_step_lifecycle() -> None:
    from hkrl.env import EnvProtocolError, HKRLEnv

    task = load_task_config("../configs/tasks/gruz_mother.yaml")
    transport = ScriptedTransport(
        [
            lambda req: _build_response(req, lifecycle=protocol.LifecycleState.RUNNING),
            lambda req: _build_response(
                req,
                lifecycle=protocol.LifecycleState.WAIT_BOSS_READY,
            ),
        ]
    )
    env = HKRLEnv(transport=transport, task=task)
    env.reset(options={"reset_timeout_s": 1.0, "recv_timeout_s": 0.1})

    with pytest.raises(EnvProtocolError, match="WAIT_BOSS_READY"):
        env.step(env.action_space.sample())


def test_env_rejects_mismatched_entity_mask() -> None:
    from hkrl.env import EnvProtocolError, HKRLEnv

    task = load_task_config("../configs/tasks/gruz_mother.yaml")
    transport = ScriptedTransport(
        [
            lambda req: _build_response(
                req,
                lifecycle=protocol.LifecycleState.RUNNING,
                include_entity_mask=False,
            )
        ]
    )
    env = HKRLEnv(transport=transport, task=task)

    with pytest.raises(EnvProtocolError, match="entity_mask length"):
        env.reset(options={"reset_timeout_s": 1.0, "recv_timeout_s": 0.1})


def test_env_rejects_mismatched_action_mask() -> None:
    from hkrl.env import EnvProtocolError, HKRLEnv

    task = load_task_config("../configs/tasks/gruz_mother.yaml")
    transport = ScriptedTransport(
        [
            lambda req: _build_response(
                req,
                lifecycle=protocol.LifecycleState.RUNNING,
                action_mask_len=1,
            )
        ]
    )
    env = HKRLEnv(transport=transport, task=task)

    with pytest.raises(EnvProtocolError, match="action_mask length"):
        env.reset(options={"reset_timeout_s": 1.0, "recv_timeout_s": 0.1})


def test_env_rejects_empty_action_mask_groups() -> None:
    from hkrl.env import EnvProtocolError, HKRLEnv

    task = load_task_config("../configs/tasks/gruz_mother.yaml")
    action_mask = [True] * 31
    action_mask[:3] = [False, False, False]
    transport = ScriptedTransport(
        [
            lambda req: _build_response(
                req,
                lifecycle=protocol.LifecycleState.RUNNING,
                action_mask_values=action_mask,
            )
        ]
    )
    env = HKRLEnv(transport=transport, task=task)

    with pytest.raises(EnvProtocolError, match="no valid movement_x"):
        env.reset(options={"reset_timeout_s": 1.0, "recv_timeout_s": 0.1})


def test_env_rejects_non_finite_observation_values() -> None:
    from hkrl.env import EnvProtocolError, HKRLEnv

    task = load_task_config("../configs/tasks/gruz_mother.yaml")
    transport = ScriptedTransport(
        [
            lambda req: _build_response(
                req,
                lifecycle=protocol.LifecycleState.RUNNING,
                entity_pos_x=float("nan"),
            )
        ]
    )
    env = HKRLEnv(transport=transport, task=task)

    with pytest.raises(EnvProtocolError, match="non-finite"):
        env.reset(options={"reset_timeout_s": 1.0, "recv_timeout_s": 0.1})


def test_env_rejects_entity_hp_above_max_hp() -> None:
    from hkrl.env import EnvProtocolError, HKRLEnv

    task = load_task_config("../configs/tasks/gruz_mother.yaml")
    transport = ScriptedTransport(
        [
            lambda req: _build_response(
                req,
                lifecycle=protocol.LifecycleState.RUNNING,
                entity_hp=31,
                entity_max_hp=30,
            )
        ]
    )
    env = HKRLEnv(transport=transport, task=task)

    with pytest.raises(EnvProtocolError, match="hp exceeds max_hp"):
        env.reset(options={"reset_timeout_s": 1.0, "recv_timeout_s": 0.1})


def test_env_rejects_boss_task_without_boss_entity() -> None:
    from hkrl.env import EnvProtocolError, HKRLEnv

    task = load_task_config("../configs/tasks/gruz_mother.yaml")
    transport = ScriptedTransport(
        [
            lambda req: _build_response(
                req,
                lifecycle=protocol.LifecycleState.RUNNING,
                entity_type=protocol.EntityType.PROJECTILE,
            )
        ]
    )
    env = HKRLEnv(transport=transport, task=task)

    with pytest.raises(EnvProtocolError, match="no boss entity"):
        env.reset(options={"reset_timeout_s": 1.0, "recv_timeout_s": 0.1})


@pytest.mark.integration
def test_random_policy_episode() -> None:
    pytest.skip("requires live Hollow Knight + HKRLEnvMod TCP connection")


def _build_response(
    request: Any,
    *,
    lifecycle: protocol.LifecycleState,
    error_code: protocol.StatusCode = protocol.StatusCode.OK,
    include_observation: bool = True,
    include_entity_mask: bool = True,
    reward_kind: protocol.RewardEventKind | None = None,
    reward_amount: float = 0.0,
    entity_type: protocol.EntityType = protocol.EntityType.BOSS,
    entity_hp: int = 20,
    entity_max_hp: int = 30,
    entity_pos_x: float = 0.0,
    server_tick: int | None = None,
    action_mask_len: int = 31,
    action_mask_values: list[bool] | None = None,
    response_env_id: int | None = None,
    response_tick_id: int | None = None,
) -> bytes:
    builder = flatbuffers.Builder(512)
    action_mask = _build_bool_vector(
        builder,
        FbStepResponse.StepResponseStartActionMaskVector,
        [True] * action_mask_len if action_mask_values is None else action_mask_values,
    )
    observation = _build_observation(
        builder,
        include_entity_mask=include_entity_mask,
        entity_type=entity_type,
        entity_hp=entity_hp,
        entity_max_hp=entity_max_hp,
        entity_pos_x=entity_pos_x,
    )
    reward_events = 0
    if reward_kind is not None:
        reward_event = _build_reward_event(builder, reward_kind, reward_amount)
        reward_events = _build_offset_vector(
            builder, FbStepResponse.StepResponseStartRewardEventsVector, [reward_event]
        )

    FbStepResponse.StepResponseStart(builder)
    FbStepResponse.StepResponseAddSchemaVersion(builder, protocol.SCHEMA_VERSION)
    FbStepResponse.StepResponseAddEnvId(
        builder, request.EnvId() if response_env_id is None else response_env_id
    )
    FbStepResponse.StepResponseAddTickId(
        builder, request.TickId() if response_tick_id is None else response_tick_id
    )
    FbStepResponse.StepResponseAddServerTick(
        builder, request.TickId() + 100 if server_tick is None else server_tick
    )
    if include_observation:
        FbStepResponse.StepResponseAddObservation(builder, observation)
    if reward_events:
        FbStepResponse.StepResponseAddRewardEvents(builder, reward_events)
    FbStepResponse.StepResponseAddActionMask(builder, action_mask)
    FbStepResponse.StepResponseAddLifecycleState(builder, lifecycle)
    FbStepResponse.StepResponseAddErrorCode(builder, error_code)
    root = FbStepResponse.StepResponseEnd(builder)
    builder.Finish(root, file_identifier=protocol.FILE_IDENTIFIER)
    return bytes(builder.Output())


def _build_observation(
    builder: flatbuffers.Builder,
    *,
    include_entity_mask: bool,
    entity_type: protocol.EntityType,
    entity_hp: int,
    entity_max_hp: int,
    entity_pos_x: float,
) -> int:
    global_state = _build_global_state(builder)
    player_state = _build_player_state(builder)
    entity_state = _build_entity_state(
        builder,
        entity_type=entity_type,
        entity_hp=entity_hp,
        entity_max_hp=entity_max_hp,
        entity_pos_x=entity_pos_x,
    )
    entities = _build_offset_vector(
        builder, FbObservation.ObservationStartEntitiesVector, [entity_state]
    )
    entity_mask = _build_bool_vector(
        builder, FbObservation.ObservationStartEntityMaskVector, [True]
    )

    FbObservation.ObservationStart(builder)
    FbObservation.ObservationAddGlobal(builder, global_state)
    FbObservation.ObservationAddPlayer(builder, player_state)
    FbObservation.ObservationAddEntities(builder, entities)
    if include_entity_mask:
        FbObservation.ObservationAddEntityMask(builder, entity_mask)
    return FbObservation.ObservationEnd(builder)


def _build_global_state(builder: flatbuffers.Builder) -> int:
    FbGlobalState.GlobalStateStart(builder)
    FbGlobalState.GlobalStateAddEpisodeId(builder, 123)
    return FbGlobalState.GlobalStateEnd(builder)


def _build_player_state(builder: flatbuffers.Builder) -> int:
    FbPlayerState.PlayerStateStart(builder)
    FbPlayerState.PlayerStateAddHp(builder, 8)
    FbPlayerState.PlayerStateAddMaxHp(builder, 9)
    FbPlayerState.PlayerStateAddSoul(builder, 33)
    FbPlayerState.PlayerStateAddMaxSoul(builder, 99)
    FbPlayerState.PlayerStateAddCanAttack(builder, True)
    return FbPlayerState.PlayerStateEnd(builder)


def _build_entity_state(
    builder: flatbuffers.Builder,
    *,
    entity_type: protocol.EntityType,
    entity_hp: int,
    entity_max_hp: int,
    entity_pos_x: float,
) -> int:
    FbEntityState.EntityStateStart(builder)
    FbEntityState.EntityStateAddEntityId(builder, 1)
    FbEntityState.EntityStateAddEntityType(builder, entity_type)
    FbEntityState.EntityStateAddPosX(builder, entity_pos_x)
    FbEntityState.EntityStateAddHp(builder, entity_hp)
    FbEntityState.EntityStateAddMaxHp(builder, entity_max_hp)
    return FbEntityState.EntityStateEnd(builder)


def _build_reward_event(
    builder: flatbuffers.Builder,
    kind: protocol.RewardEventKind,
    amount: float,
) -> int:
    FbRewardEvent.RewardEventStart(builder)
    FbRewardEvent.RewardEventAddKind(builder, kind)
    FbRewardEvent.RewardEventAddEntityId(builder, 1)
    FbRewardEvent.RewardEventAddAmount(builder, amount)
    return FbRewardEvent.RewardEventEnd(builder)


def _build_offset_vector(
    builder: flatbuffers.Builder,
    start_vector: Callable[[flatbuffers.Builder, int], int],
    offsets: list[int],
) -> int:
    start_vector(builder, len(offsets))
    for offset in reversed(offsets):
        builder.PrependUOffsetTRelative(offset)
    return builder.EndVector()


def _build_bool_vector(
    builder: flatbuffers.Builder,
    start_vector: Callable[[flatbuffers.Builder, int], int],
    values: list[bool],
) -> int:
    start_vector(builder, len(values))
    for value in reversed(values):
        builder.PrependBool(value)
    return builder.EndVector()


def _recv_tcp_frame(conn: socket.socket) -> bytes:
    header = _recv_exact(conn, 4)
    (length,) = struct.unpack("<I", header)
    return _recv_exact(conn, length)


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = conn.recv(remaining)
        if chunk == b"":
            raise ConnectionError("TCP peer closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _repo_root() -> Path:
    return Path(__file__).parents[2]
