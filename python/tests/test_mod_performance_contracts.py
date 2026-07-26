"""Source-level guardrails for the Unity main-thread hot path."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]
MOD = ROOT / "mod" / "HKRLEnvMod"


def _source(relative_path: str) -> str:
    return (MOD / relative_path).read_text(encoding="utf-8")


def test_mod_version_has_one_msbuild_source_and_runtime_reads_the_assembly() -> None:
    version_props = _source("Version.props")
    project = _source("HKRLEnvMod.csproj")
    mod_source = _source("HKRLEnvMod.cs")

    assert "<HKRLModVersion>0.8.0</HKRLModVersion>" in version_props
    assert (
        "<IncludeSourceRevisionInInformationalVersion>false"
        "</IncludeSourceRevisionInInformationalVersion>"
    ) in version_props
    assert "<Deterministic>true</Deterministic>" in version_props
    assert '<Import Project="Version.props" />' in project
    assert "Assembly.GetName().Version?.ToString(3)" in mod_source
    assert 'GetVersion() => "0.1.0"' not in mod_source


def test_entity_observers_do_not_rescan_every_scene_transform_per_step() -> None:
    projectile = _source("Observation/ProjectileObserver.cs")
    hazard = _source("Observation/HazardObserver.cs")

    assert "FindObjectsOfType<Transform>" not in projectile
    assert "CandidateRefreshIntervalSeconds" in projectile
    assert "FindObjectsOfType<global::DamageHero>()" in projectile
    assert "FindObjectsOfType<global::DamageEnemies>()" in projectile
    assert "_sceneHandle" in hazard
    assert hazard.count("FindObjectsOfType<Collider2D>()") == 1


def test_input_bridge_resolves_reflection_once_without_per_frame_invoke() -> None:
    injector = _source("Action/InputInjector.cs")

    assert "Delegate.CreateDelegate" in injector
    assert "InputHandler.Instance" in injector
    assert "InputManager.OnUpdate += OnInputManagerUpdate" in injector
    assert "InputManager.CurrentTick" in injector
    assert "method.Invoke(" not in injector
    assert "new object[]" not in injector


def test_observation_hot_path_reuses_entity_buffers_and_direct_player_state() -> None:
    entity_observer = _source("Observation/EntityObserver.cs")
    registry = _source("Observation/EntityRegistry.cs")
    boss_locator = _source("Observation/BossLocator.cs")
    player = _source("Observation/PlayerObserver.cs")

    assert "_entities.Clear();" in entity_observer
    assert "_aliveInstanceIds.Clear();" in entity_observer
    assert "_deadInstanceIds.Clear();" in registry
    assert "BossBuffer.Clear();" in boss_locator
    assert "SeenInstanceIds.Clear();" in boss_locator
    assert "global::PlayerData.instance" in player
    assert "global::HeroControllerStates states = hero.cState;" in player
    assert 'ReadBool(hero, false, "cState.attacking")' not in player


def test_response_codec_reuses_builder_and_emits_one_size_prefixed_frame() -> None:
    codec = _source("Transport/MessageCodec.cs")

    assert "[ThreadStatic]" in codec
    assert "builder.Clear();" in codec
    assert "FinishSizePrefixedStepResponseBuffer" in codec
    assert "AddLengthPrefix" not in codec


def test_phase_zero_snapshot_logging_is_disabled_while_client_is_active() -> None:
    mod_source = _source("HKRLEnvMod.cs")

    assert "_server?.HasClient != true" in mod_source
    assert "Phase0LogIntervalSeconds = 10.0f" in mod_source


def test_save_bootstrap_accepts_a_loaded_bench_before_final_readiness() -> None:
    scene_controller = _source("Env/SceneController.cs")
    start = scene_controller.index("private void ProgressPendingLoad()")
    end = scene_controller.index(
        "private void BeginTargetSceneTransition",
        start,
    )
    bootstrap = scene_controller[start:end]

    assert "PlayerObserver.IsReadyForControl" in bootstrap
    assert "TryReleaseLoadedBench()" in bootstrap
    assert 'activeScene.name.StartsWith("Menu_"' in bootstrap
    assert "gameManager.IsInSceneTransition" in bootstrap

    release_start = scene_controller.index("private void TryReleaseLoadedBench")
    release = scene_controller[release_start:]
    assert "fsm.FsmName," in release
    assert '"Bench Control"' in release
    assert "_loadedBenchFsm.ActiveStateName" in release
    assert '"Resting"' in release
    assert '_loadedBenchFsm.SendEvent("GET UP")' in release
    assert "playerData.atBench = false" not in release
    assert "body.isKinematic" not in release


def test_godhome_entry_avoids_workshop_fsm_mutation_and_global_broadcasts() -> None:
    scene_controller = _source("Env/SceneController.cs")
    start = scene_controller.index("private void PrepareGodhomeChallenge")
    end = scene_controller.index("private void PrepareBossSceneSetup", start)
    challenge = scene_controller[start:end]

    assert "gameManager.TimePasses()" in challenge
    assert "gameManager.ResetSemiPersistentItems()" in challenge
    assert "PlayMakerFSM.BroadcastEvent" not in challenge
    assert "hero.EnterWithoutInput(true)" not in challenge
    assert "hero.AcceptInput()" not in challenge
    assert "BeginWorkshopChallenge" not in scene_controller


def test_reset_never_injects_policy_actions_or_mutates_boss_state() -> None:
    scene_controller = _source("Env/SceneController.cs")
    reset_manager = _source("Env/ResetManager.cs")
    step_controller = _source("Env/StepController.cs")
    boss_locator = _source("Observation/BossLocator.cs")

    assert "FindConfiguredBosses()" in scene_controller
    assert "FindActiveBosses()" in scene_controller
    assert "ApplyBossEntryPrelude" not in step_controller
    assert "_actions.DisableInput()" in step_controller
    assert "ReleaseOwnedBossSetupEvent()" in scene_controller
    assert "_scene.CancelPendingLoad()" in reset_manager
    assert "request.Command == HKRL.Command.Step" in step_controller
    assert "&& state == HKRL.LifecycleState.Running" in step_controller
    assert "Resources.FindObjectsOfTypeAll<global::HealthManager>()" in boss_locator
    assert "health.gameObject.scene.handle != scene.handle" in boss_locator
    assert "_nextCandidateScanAt = now + 0.5f" in boss_locator


def test_direct_scene_transition_resets_only_the_entry_handshake() -> None:
    scene_controller = _source("Env/SceneController.cs")
    lifecycle = _source("Env/SceneEntryLifecycle.cs")

    marker = "SceneEntryLifecycle.MarkTransitionPending(gameManager)"
    assert marker in scene_controller
    assert scene_controller.index(marker) < scene_controller.index(
        "gameManager.BeginSceneTransition("
    )
    assert '"hasFinishedEnteringScene"' in lifecycle
    assert "FinishedEntryField.SetValue(gameManager, false)" in lifecycle
    assert "BossSceneController" not in lifecycle
    assert "HealthManager" not in lifecycle
    assert "PlayMakerFSM" not in lifecycle
    assert ".transform" not in lifecycle
    assert "GetComponent<Rigidbody2D>" not in lifecycle
