"""Wire protocol constants and (de)serialization boundary.

Implements docs/protocol.md. The message *layout* is owned by
``schema/hkrl.fbs`` and the generated bindings under ``hkrl.schema``; this module
holds the version constant, command/enum mirrors for ergonomic Python use, and
the encode/decode helpers that wrap the FlatBuffers bindings.

IMPORTANT: ``SCHEMA_VERSION`` MUST match the constant mirrored in
``mod/HKRLEnvMod/Transport/Protocol.cs``. Bump both on every schema change and
record it in CHANGELOG.md.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import flatbuffers

from hkrl.spaces import DEFAULT_N_MACROS

if TYPE_CHECKING:
    import numpy as np

# Mirrors the schema_version carried in every StepRequest/StepResponse and the
# C# Protocol.SCHEMA_VERSION. See schema/README.md evolution rules.
SCHEMA_VERSION: int = 3

# FlatBuffers file_identifier (must equal the one in hkrl.fbs).
FILE_IDENTIFIER: bytes = b"HKRL"


class Command(enum.IntEnum):
    """Mirror of HKRL.Command (schema/hkrl.fbs)."""

    STEP = 0
    RESET = 1
    PAUSE = 2
    RESUME = 3
    SET_TASK = 4
    SET_TIMESCALE = 5
    PING = 6


class LifecycleState(enum.IntEnum):
    """Mirror of HKRL.LifecycleState. See docs/episode_lifecycle.md."""

    IDLE = 0
    RESET_REQUESTED = 1
    FREEZE_INPUT = 2
    CLEAR_EVENTS = 3
    LOAD_SCENE = 4
    WAIT_SCENE_READY = 5
    WAIT_PLAYER_READY = 6
    WAIT_BOSS_READY = 7
    RESTORE_PLAYER_STATE = 8
    CLEAR_PROJECTILES = 9
    COUNTDOWN = 10
    RUNNING = 11
    TERMINATING = 12
    REPORT_DONE = 13
    CLEANUP = 14


class StatusCode(enum.IntEnum):
    """Mirror of HKRL.StatusCode (StepResponse.error_code)."""

    OK = 0
    RESET_TIMEOUT = 1
    SCENE_LOAD_FAILED = 2
    BOSS_NOT_FOUND = 3
    PLAYER_NOT_READY = 4
    INTERNAL_ERROR = 5
    SCHEMA_MISMATCH = 6
    NOT_RUNNING = 7


class EntityType(enum.IntEnum):
    """Mirror of HKRL.EntityType."""

    PLAYER = 0
    BOSS = 1
    ENEMY = 2
    PROJECTILE = 3
    HAZARD = 4
    PLATFORM = 5
    PICKUP = 6
    EFFECT = 7
    UNKNOWN = 255


class RewardEventKind(enum.IntEnum):
    """Mirror of HKRL.RewardEventKind. Payload semantics: docs/reward_design.md."""

    DAMAGE_DEALT = 0
    DAMAGE_TAKEN = 1
    HEAL = 2
    SOUL_GAINED = 3
    BOSS_KILLED = 4
    PLAYER_DEATH = 5
    SCENE_CHANGED = 6
    INVALID_ACTION = 7
    STAGGER = 8


@dataclass(slots=True)
class RewardEvent:
    """Decoded reward event (Python-friendly view of HKRL.RewardEvent)."""

    kind: RewardEventKind
    entity_id: int = 0
    amount: float = 0.0
    aux_int: int = 0
    aux_int2: int = 0


@dataclass(slots=True)
class DecodedObservation:
    """Numpy-friendly decoded observation snapshot.

    Field arrays are documented in docs/observation_schema.md. Decoders fill
    these from the zero-copy FlatBuffers buffer; normalization happens later in
    ``hkrl.spaces`` / wrappers, not here.
    """

    global_state: np.ndarray  # GlobalState features
    player_state: np.ndarray  # PlayerState features
    entities: np.ndarray  # (max_entities, entity_feat_dim)
    entity_mask: np.ndarray  # (max_entities,) bool


@dataclass(slots=True)
class DecodedStepResponse:
    """Decoded StepResponse view used by env/worker code."""

    schema_version: int
    env_id: int
    tick_id: int
    server_tick: int
    observation: DecodedObservation | None
    reward_events: list[RewardEvent]
    action_mask: np.ndarray
    terminated: bool
    truncated: bool
    lifecycle_state: LifecycleState
    error_code: StatusCode
    info: str | None = None


def encode_step_request(
    *,
    command: Command = Command.STEP,
    action: dict[str, Any] | None = None,
    env_id: int = 0,
    tick_id: int = 0,
    action_repeat: int = 1,
    policy_version: int = 0,
    client_time: float = 0.0,
    task_id: int = 0,
    time_scale: float = 0.0,
    enable_macro_actions: bool = True,
    n_macro_actions: int = DEFAULT_N_MACROS,
) -> bytes:
    """Encode a StepRequest FlatBuffers payload.

    Framing is handled by ``Transport`` implementations; this function returns
    only the FlatBuffers payload carrying file_identifier ``HKRL``.
    """
    if not 1 <= action_repeat <= 255:
        raise ValueError("action_repeat must be in [1, 255]")
    if tick_id < 0:
        raise ValueError("tick_id must be non-negative")
    if not isinstance(enable_macro_actions, bool):
        raise ValueError("enable_macro_actions must be a bool")
    if not isinstance(n_macro_actions, int) or isinstance(n_macro_actions, bool):
        raise ValueError("n_macro_actions must be an integer")
    if n_macro_actions < 0:
        raise ValueError("n_macro_actions must be non-negative")

    action_fields = _action_fields(action)
    builder = flatbuffers.Builder(256)

    action_offset = _build_action(builder, action_fields)

    fb = _generated("StepRequest")
    fb.StepRequestStart(builder)
    fb.StepRequestAddSchemaVersion(builder, SCHEMA_VERSION)
    fb.StepRequestAddEnvId(builder, int(env_id))
    fb.StepRequestAddTickId(builder, int(tick_id))
    fb.StepRequestAddCommand(builder, int(command))
    fb.StepRequestAddAction(builder, action_offset)
    fb.StepRequestAddActionRepeat(builder, int(action_repeat))
    fb.StepRequestAddPolicyVersion(builder, int(policy_version))
    fb.StepRequestAddClientTime(builder, float(client_time))
    fb.StepRequestAddTaskId(builder, int(task_id))
    fb.StepRequestAddTimeScale(builder, float(time_scale))
    fb.StepRequestAddEnableMacroActions(builder, bool(enable_macro_actions))
    fb.StepRequestAddNMacroActions(builder, int(n_macro_actions))
    root = fb.StepRequestEnd(builder)
    builder.Finish(root, file_identifier=FILE_IDENTIFIER)
    return bytes(builder.Output())


def decode_step_response(frame: bytes) -> DecodedStepResponse:
    """Decode a StepResponse FlatBuffers payload.

    Verifies ``file_identifier`` and ``schema_version`` before reading; raises on
    ``StatusCode.SCHEMA_MISMATCH`` divergence.
    """
    import numpy as np

    fb = _generated("StepResponse")
    if not fb.StepResponse.StepResponseBufferHasIdentifier(frame, 0):
        raise ValueError("StepResponse missing HKRL file identifier")

    response = fb.StepResponse.GetRootAs(frame, 0)
    schema_version = int(response.SchemaVersion())
    error_code = StatusCode(int(response.ErrorCode()))
    if schema_version != SCHEMA_VERSION or error_code == StatusCode.SCHEMA_MISMATCH:
        raise ValueError(
            f"schema mismatch: local={SCHEMA_VERSION}, remote={schema_version}, "
            f"error_code={error_code.name}"
        )

    reward_events = [
        _decode_reward_event(response.RewardEvents(i)) for i in range(response.RewardEventsLength())
    ]
    action_mask = np.array(
        [bool(response.ActionMask(i)) for i in range(response.ActionMaskLength())],
        dtype=bool,
    )

    info = response.Info()
    return DecodedStepResponse(
        schema_version=schema_version,
        env_id=int(response.EnvId()),
        tick_id=int(response.TickId()),
        server_tick=int(response.ServerTick()),
        observation=_decode_observation(response.Observation()),
        reward_events=reward_events,
        action_mask=action_mask,
        terminated=bool(response.Terminated()),
        truncated=bool(response.Truncated()),
        lifecycle_state=LifecycleState(int(response.LifecycleState())),
        error_code=error_code,
        info=None if info is None else info.decode("utf-8"),
    )


def _generated(module_name: str) -> Any:
    try:
        import hkrl.schema  # noqa: F401  # registers the top-level HKRL alias

        return __import__(f"hkrl.schema.HKRL.{module_name}", fromlist=[module_name])
    except ModuleNotFoundError as exc:
        raise RuntimeError("FlatBuffers bindings missing; run `make gen-schema`") from exc


def _build_action(builder: flatbuffers.Builder, fields: dict[str, int]) -> int:
    fb = _generated("Action")
    fb.ActionStart(builder)
    fb.ActionAddMovementX(builder, fields["movement_x"])
    fb.ActionAddAimY(builder, fields["aim_y"])
    fb.ActionAddButtons(builder, fields["buttons"])
    fb.ActionAddDurationIdx(builder, fields["duration_idx"])
    fb.ActionAddMacroId(builder, fields["macro_id"])
    return int(fb.ActionEnd(builder))


def _action_fields(action: dict[str, Any] | None) -> dict[str, int]:
    action = action or {}
    movement_x = int(action.get("movement_x", 1))
    aim_y = int(action.get("aim_y", 1))
    duration_idx = int(action.get("duration_idx", action.get("duration", 0)))
    macro_id = int(action["macro_id"]) if "macro_id" in action else int(action.get("macro", 0)) - 1
    buttons = _buttons_to_mask(action.get("buttons", 0))

    if not 0 <= movement_x <= 2:
        raise ValueError("movement_x must be in [0, 2]")
    if not 0 <= aim_y <= 2:
        raise ValueError("aim_y must be in [0, 2]")
    if not 0 <= duration_idx <= 3:
        raise ValueError("duration index must be in [0, 3]")
    if not -(2**15) <= macro_id < 2**15:
        raise ValueError("macro_id must fit int16")

    return {
        "movement_x": movement_x,
        "aim_y": aim_y,
        "buttons": buttons,
        "duration_idx": duration_idx,
        "macro_id": macro_id,
    }


def _buttons_to_mask(buttons: Any) -> int:
    from hkrl.spaces import BUTTON_BITS, N_BUTTONS

    if isinstance(buttons, int):
        if not 0 <= buttons < (1 << N_BUTTONS):
            raise ValueError("buttons bitmask has bits outside BUTTON_BITS")
        return buttons

    if isinstance(buttons, dict):
        mask = 0
        for name, enabled in buttons.items():
            if name not in BUTTON_BITS:
                raise ValueError(f"unknown button name: {name}")
            if enabled not in (False, True):
                raise ValueError("button mapping values must be boolean")
            if enabled:
                mask |= 1 << BUTTON_BITS[name]
        return mask

    try:
        values = list(buttons)
    except TypeError as exc:
        raise ValueError("buttons must be an int bitmask, mapping, or sequence") from exc

    if len(values) != N_BUTTONS:
        raise ValueError(f"buttons sequence must have length {N_BUTTONS}")

    mask = 0
    for idx, enabled in enumerate(values):
        if enabled not in (0, 1, False, True):
            raise ValueError("buttons sequence values must be binary")
        if enabled:
            mask |= 1 << idx
    return mask


def _decode_reward_event(event: Any) -> RewardEvent:
    return RewardEvent(
        kind=RewardEventKind(int(event.Kind())),
        entity_id=int(event.EntityId()),
        amount=float(event.Amount()),
        aux_int=int(event.AuxInt()),
        aux_int2=int(event.AuxInt2()),
    )


def _decode_observation(observation: Any | None) -> DecodedObservation | None:
    if observation is None:
        return None

    import numpy as np

    global_state = observation.Global()
    player_state = observation.Player()
    if global_state is None or player_state is None:
        return None

    entities = []
    for idx in range(observation.EntitiesLength()):
        entity = observation.Entities(idx)
        if entity is not None:
            entities.append(_entity_features(entity))

    entity_mask = np.array(
        [bool(observation.EntityMask(i)) for i in range(observation.EntityMaskLength())],
        dtype=bool,
    )
    return DecodedObservation(
        global_state=np.array(_global_features(global_state), dtype=np.float32),
        player_state=np.array(_player_features(player_state), dtype=np.float32),
        entities=np.array(entities, dtype=np.float32),
        entity_mask=entity_mask,
    )


def _global_features(global_state: Any) -> list[float]:
    return [
        float(global_state.SceneHash()),
        float(global_state.ArenaId()),
        float(global_state.TaskId()),
        float(global_state.Difficulty()),
        float(global_state.TimeInEpisode()),
        float(global_state.TimeScale()),
        float(global_state.FixedDeltaTime()),
        float(global_state.StageIndex()),
        float(global_state.EpisodeId()),
    ]


def _player_features(player_state: Any) -> list[float]:
    return [
        float(player_state.PosX()),
        float(player_state.PosY()),
        float(player_state.VelX()),
        float(player_state.VelY()),
        float(player_state.Hp()),
        float(player_state.MaxHp()),
        float(player_state.Soul()),
        float(player_state.MaxSoul()),
        float(player_state.Facing()),
        float(player_state.OnGround()),
        float(player_state.WallSliding()),
        float(player_state.Jumping()),
        float(player_state.Falling()),
        float(player_state.Dashing()),
        float(player_state.ShadowDashing()),
        float(player_state.Invulnerable()),
        float(player_state.InvulnTimer()),
        float(player_state.AttackLockTimer()),
        float(player_state.CastLockTimer()),
        float(player_state.FocusState()),
        float(player_state.DashCooldown()),
        float(player_state.DoubleJumpAvailable()),
        float(player_state.CanAttack()),
        float(player_state.CanCast()),
        float(player_state.CanFocus()),
    ]


def _entity_features(entity: Any) -> list[float]:
    return [
        float(entity.EntityId()),
        float(entity.EntityType()),
        float(entity.Team()),
        float(entity.PrefabHash()),
        float(entity.FsmNameHash()),
        float(entity.FsmStateHash()),
        float(entity.PosX()),
        float(entity.PosY()),
        float(entity.RelX()),
        float(entity.RelY()),
        float(entity.VelX()),
        float(entity.VelY()),
        float(entity.Hp()),
        float(entity.MaxHp()),
        float(entity.HurtboxCenterX()),
        float(entity.HurtboxCenterY()),
        float(entity.HurtboxSizeX()),
        float(entity.HurtboxSizeY()),
        float(entity.HitboxActive()),
        float(entity.Damage()),
        float(entity.Ttl()),
        float(entity.Phase()),
        float(entity.ThreatScore()),
        float(entity.Flags()),
    ]
