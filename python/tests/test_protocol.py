"""Protocol-level invariants and FlatBuffers encode/decode helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import flatbuffers
import numpy as np
import pytest
from hkrl import protocol
from hkrl.schema.HKRL import (
    EntityState as FbEntityState,
)
from hkrl.schema.HKRL import (
    GlobalState as FbGlobalState,
)
from hkrl.schema.HKRL import (
    Observation as FbObservation,
)
from hkrl.schema.HKRL import (
    PlayerState as FbPlayerState,
)
from hkrl.schema.HKRL import (
    RewardEvent as FbRewardEvent,
)
from hkrl.schema.HKRL import (
    StepRequest as FbStepRequest,
)
from hkrl.schema.HKRL import (
    StepResponse as FbStepResponse,
)
from hkrl.spaces import BUTTON_BITS


def test_schema_version_is_positive() -> None:
    assert isinstance(protocol.SCHEMA_VERSION, int)
    assert protocol.SCHEMA_VERSION >= 1


def test_file_identifier_is_four_bytes() -> None:
    # FlatBuffers file_identifier must be exactly 4 bytes (matches hkrl.fbs).
    assert protocol.FILE_IDENTIFIER == b"HKRL"
    assert len(protocol.FILE_IDENTIFIER) == 4


def test_enum_mirrors_have_expected_members() -> None:
    assert protocol.Command.STEP == 0
    assert protocol.LifecycleState.RUNNING.name == "RUNNING"
    assert protocol.StatusCode.OK == 0
    assert protocol.EntityType.BOSS == 1


def test_encode_step_request_builds_schema_payload() -> None:
    frame = protocol.encode_step_request(
        command=protocol.Command.SET_TASK,
        action={
            "movement_x": 2,
            "aim_y": 0,
            "buttons": {"attack": True, "dash": True},
            "duration": 3,
            "macro": 2,
        },
        env_id=3,
        tick_id=42,
        action_repeat=4,
        policy_version=7,
        client_time=12.5,
        task_id=11,
        time_scale=2.0,
    )

    assert FbStepRequest.StepRequest.StepRequestBufferHasIdentifier(frame, 0)
    request = FbStepRequest.StepRequest.GetRootAs(frame, 0)

    assert request.SchemaVersion() == protocol.SCHEMA_VERSION
    assert request.EnvId() == 3
    assert request.TickId() == 42
    assert request.Command() == protocol.Command.SET_TASK
    assert request.ActionRepeat() == 4
    assert request.PolicyVersion() == 7
    assert request.ClientTime() == 12.5
    assert request.TaskId() == 11
    assert request.TimeScale() == 2.0

    action = request.Action()
    assert action is not None
    assert action.MovementX() == 2
    assert action.AimY() == 0
    assert action.Buttons() == (1 << BUTTON_BITS["attack"]) | (1 << BUTTON_BITS["dash"])
    assert action.DurationIdx() == 3
    assert action.MacroId() == 1


def test_decode_step_response_decodes_observation_events_and_mask() -> None:
    frame = _build_step_response()

    decoded = protocol.decode_step_response(frame)

    assert decoded.schema_version == protocol.SCHEMA_VERSION
    assert decoded.env_id == 5
    assert decoded.tick_id == 42
    assert decoded.server_tick == 4242
    assert decoded.terminated is False
    assert decoded.truncated is False
    assert decoded.lifecycle_state is protocol.LifecycleState.RUNNING
    assert decoded.error_code is protocol.StatusCode.OK
    assert decoded.info == '{"debug": true}'
    np.testing.assert_array_equal(decoded.action_mask, np.array([True, False, True]))

    assert decoded.reward_events == [
        protocol.RewardEvent(
            kind=protocol.RewardEventKind.DAMAGE_DEALT,
            entity_id=99,
            amount=2.5,
            aux_int=1,
            aux_int2=2,
        )
    ]

    assert decoded.observation is not None
    np.testing.assert_allclose(
        decoded.observation.global_state, [123, 7, 11, 2, 1.5, 2.0, 0.02, 1, 77]
    )
    np.testing.assert_allclose(
        decoded.observation.player_state[:8],
        [10.0, 20.0, 1.0, -1.0, 8.0, 9.0, 33.0, 99.0],
    )
    assert decoded.observation.player_state[9] == 1.0  # on_ground
    assert decoded.observation.player_state[22] == 1.0  # can_attack
    assert decoded.observation.entities.shape == (1, 24)
    np.testing.assert_allclose(
        decoded.observation.entities[0, :14],
        [99, protocol.EntityType.BOSS, 1, 1001, 2002, 3003, 15, 25, 5, 5, -1, 0.5, 20, 30],
    )
    np.testing.assert_array_equal(decoded.observation.entity_mask, np.array([True]))


def test_decode_step_response_rejects_schema_mismatch() -> None:
    frame = _build_step_response(
        schema_version=protocol.SCHEMA_VERSION + 1,
        error_code=protocol.StatusCode.SCHEMA_MISMATCH,
    )

    with pytest.raises(ValueError, match="schema mismatch"):
        protocol.decode_step_response(frame)


def test_mod_step_request_schema_mismatch_maps_to_status_code() -> None:
    root = Path(__file__).parents[2]
    codec = (root / "mod/HKRLEnvMod/Transport/MessageCodec.cs").read_text(encoding="utf-8")
    controller = (root / "mod/HKRLEnvMod/Env/StepController.cs").read_text(encoding="utf-8")

    assert "SchemaMismatchException" in codec
    assert "throw new SchemaMismatchException" in codec
    assert "catch (SchemaMismatchException" in controller
    assert "HKRL.StatusCode.SchemaMismatch" in controller


def test_mod_step_controller_honors_action_repeat_contract() -> None:
    root = Path(__file__).parents[2]
    controller = (root / "mod/HKRLEnvMod/Env/StepController.cs").read_text(encoding="utf-8")

    assert "_repeatRequest" in controller
    assert "_repeatTicksRemaining = request.ActionRepeat - 1" in controller
    assert "ApplyRepeatedStep" in controller
    assert "BufferedTerminalEvent" in controller


def test_mod_step_controller_reports_wire_invalid_actions() -> None:
    root = Path(__file__).parents[2]
    controller = (root / "mod/HKRLEnvMod/Env/StepController.cs").read_text(encoding="utf-8")

    assert "ReportInvalidAction(request.Action)" in controller
    assert "HKRL.RewardEventKind.InvalidAction" in controller
    assert "PrimitiveInput.ButtonMask" in controller
    assert "action.DurationIdx > 3" in controller


def test_mod_step_response_mask_uses_player_state() -> None:
    root = Path(__file__).parents[2]
    controller = (root / "mod/HKRLEnvMod/Env/StepController.cs").read_text(encoding="utf-8")

    assert "ToPlayerActionState(observation.Player)" in controller
    assert "soul: player.Soul" in controller
    assert "canAttack: player.CanAttack" in controller


def _build_step_response(
    *,
    schema_version: int = protocol.SCHEMA_VERSION,
    error_code: protocol.StatusCode = protocol.StatusCode.OK,
) -> bytes:
    builder = flatbuffers.Builder(512)

    info = builder.CreateString('{"debug": true}')
    reward_event = _build_reward_event(builder)
    reward_events = _build_offset_vector(
        builder, FbStepResponse.StepResponseStartRewardEventsVector, [reward_event]
    )

    action_mask = _build_bool_vector(
        builder,
        FbStepResponse.StepResponseStartActionMaskVector,
        [True, False, True],
    )
    observation = _build_observation(builder)

    FbStepResponse.StepResponseStart(builder)
    FbStepResponse.StepResponseAddSchemaVersion(builder, schema_version)
    FbStepResponse.StepResponseAddEnvId(builder, 5)
    FbStepResponse.StepResponseAddTickId(builder, 42)
    FbStepResponse.StepResponseAddServerTick(builder, 4242)
    FbStepResponse.StepResponseAddObservation(builder, observation)
    FbStepResponse.StepResponseAddRewardEvents(builder, reward_events)
    FbStepResponse.StepResponseAddActionMask(builder, action_mask)
    FbStepResponse.StepResponseAddTerminated(builder, False)
    FbStepResponse.StepResponseAddTruncated(builder, False)
    FbStepResponse.StepResponseAddLifecycleState(builder, protocol.LifecycleState.RUNNING)
    FbStepResponse.StepResponseAddErrorCode(builder, error_code)
    FbStepResponse.StepResponseAddInfo(builder, info)
    root = FbStepResponse.StepResponseEnd(builder)
    builder.Finish(root, file_identifier=protocol.FILE_IDENTIFIER)
    return bytes(builder.Output())


def _build_reward_event(builder: flatbuffers.Builder) -> int:
    FbRewardEvent.RewardEventStart(builder)
    FbRewardEvent.RewardEventAddKind(builder, protocol.RewardEventKind.DAMAGE_DEALT)
    FbRewardEvent.RewardEventAddEntityId(builder, 99)
    FbRewardEvent.RewardEventAddAmount(builder, 2.5)
    FbRewardEvent.RewardEventAddAuxInt(builder, 1)
    FbRewardEvent.RewardEventAddAuxInt2(builder, 2)
    return FbRewardEvent.RewardEventEnd(builder)


def _build_observation(builder: flatbuffers.Builder) -> int:
    global_state = _build_global_state(builder)
    player_state = _build_player_state(builder)
    entity_state = _build_entity_state(builder)
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
    FbObservation.ObservationAddEntityMask(builder, entity_mask)
    return FbObservation.ObservationEnd(builder)


def _build_global_state(builder: flatbuffers.Builder) -> int:
    FbGlobalState.GlobalStateStart(builder)
    FbGlobalState.GlobalStateAddSceneHash(builder, 123)
    FbGlobalState.GlobalStateAddArenaId(builder, 7)
    FbGlobalState.GlobalStateAddTaskId(builder, 11)
    FbGlobalState.GlobalStateAddDifficulty(builder, 2)
    FbGlobalState.GlobalStateAddTimeInEpisode(builder, 1.5)
    FbGlobalState.GlobalStateAddTimeScale(builder, 2.0)
    FbGlobalState.GlobalStateAddFixedDeltaTime(builder, 0.02)
    FbGlobalState.GlobalStateAddStageIndex(builder, 1)
    FbGlobalState.GlobalStateAddEpisodeId(builder, 77)
    return FbGlobalState.GlobalStateEnd(builder)


def _build_player_state(builder: flatbuffers.Builder) -> int:
    FbPlayerState.PlayerStateStart(builder)
    FbPlayerState.PlayerStateAddPosX(builder, 10.0)
    FbPlayerState.PlayerStateAddPosY(builder, 20.0)
    FbPlayerState.PlayerStateAddVelX(builder, 1.0)
    FbPlayerState.PlayerStateAddVelY(builder, -1.0)
    FbPlayerState.PlayerStateAddHp(builder, 8)
    FbPlayerState.PlayerStateAddMaxHp(builder, 9)
    FbPlayerState.PlayerStateAddSoul(builder, 33)
    FbPlayerState.PlayerStateAddMaxSoul(builder, 99)
    FbPlayerState.PlayerStateAddFacing(builder, 1)
    FbPlayerState.PlayerStateAddOnGround(builder, True)
    FbPlayerState.PlayerStateAddInvulnTimer(builder, 0.25)
    FbPlayerState.PlayerStateAddAttackLockTimer(builder, 0.5)
    FbPlayerState.PlayerStateAddCastLockTimer(builder, 0.75)
    FbPlayerState.PlayerStateAddFocusState(builder, 1)
    FbPlayerState.PlayerStateAddDashCooldown(builder, 0.1)
    FbPlayerState.PlayerStateAddDoubleJumpAvailable(builder, True)
    FbPlayerState.PlayerStateAddCanAttack(builder, True)
    FbPlayerState.PlayerStateAddCanCast(builder, True)
    FbPlayerState.PlayerStateAddCanFocus(builder, False)
    return FbPlayerState.PlayerStateEnd(builder)


def _build_entity_state(builder: flatbuffers.Builder) -> int:
    FbEntityState.EntityStateStart(builder)
    FbEntityState.EntityStateAddEntityId(builder, 99)
    FbEntityState.EntityStateAddEntityType(builder, protocol.EntityType.BOSS)
    FbEntityState.EntityStateAddTeam(builder, 1)
    FbEntityState.EntityStateAddPrefabHash(builder, 1001)
    FbEntityState.EntityStateAddFsmNameHash(builder, 2002)
    FbEntityState.EntityStateAddFsmStateHash(builder, 3003)
    FbEntityState.EntityStateAddPosX(builder, 15.0)
    FbEntityState.EntityStateAddPosY(builder, 25.0)
    FbEntityState.EntityStateAddRelX(builder, 5.0)
    FbEntityState.EntityStateAddRelY(builder, 5.0)
    FbEntityState.EntityStateAddVelX(builder, -1.0)
    FbEntityState.EntityStateAddVelY(builder, 0.5)
    FbEntityState.EntityStateAddHp(builder, 20)
    FbEntityState.EntityStateAddMaxHp(builder, 30)
    FbEntityState.EntityStateAddHurtboxCenterX(builder, 15.0)
    FbEntityState.EntityStateAddHurtboxCenterY(builder, 25.0)
    FbEntityState.EntityStateAddHurtboxSizeX(builder, 2.0)
    FbEntityState.EntityStateAddHurtboxSizeY(builder, 4.0)
    FbEntityState.EntityStateAddHitboxActive(builder, True)
    FbEntityState.EntityStateAddDamage(builder, 2)
    FbEntityState.EntityStateAddTtl(builder, 0.5)
    FbEntityState.EntityStateAddPhase(builder, 1)
    FbEntityState.EntityStateAddThreatScore(builder, 0.75)
    FbEntityState.EntityStateAddFlags(builder, 3)
    return FbEntityState.EntityStateEnd(builder)


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
