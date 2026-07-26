"""Config loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hkrl.utils.config import (
    TaskConfig,
    load_task_config,
    load_train_config,
    load_yaml,
    resolve_auth_token,
    validate_bind_address,
    validate_service_auth,
    validate_task_collection,
)


def _write_yaml(path: Path, data: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_load_yaml_composes_defaults_with_deep_override(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "nested" / "child.yaml"
    child.parent.mkdir()

    _write_yaml(
        base,
        {
            "algorithm": "recurrent_ppo",
            "model": {"name": "entity_attention_gru", "rnn_hidden": 256},
            "transport": {"host": "127.0.0.1", "port": 5555},
        },
    )
    _write_yaml(
        child,
        {
            "defaults": ["../base.yaml"],
            "algorithm": "ppo",
            "model": {"name": "mlp"},
        },
    )

    assert load_yaml(child) == {
        "algorithm": "ppo",
        "model": {"name": "mlp", "rnn_hidden": 256},
        "transport": {"host": "127.0.0.1", "port": 5555},
    }


def test_load_yaml_rejects_default_cycles(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"

    _write_yaml(first, {"defaults": ["second.yaml"], "algorithm": "ppo"})
    _write_yaml(second, {"defaults": ["first.yaml"], "algorithm": "recurrent_ppo"})

    with pytest.raises(ValueError, match="cyclic config defaults"):
        load_yaml(first)


def test_load_train_config_composes_repo_defaults() -> None:
    config = load_train_config(Path("../configs/train/ppo_mlp.yaml"))

    assert config.algorithm == "ppo"
    assert config.gamma == 0.995
    assert config.transport.port == 5555
    assert config.transport.shm_name == "hkrl_env"
    assert config.transport.req_slots == 8
    assert config.transport.resp_slots == 8
    assert config.model.name == "mlp"


def test_windows_ssh_role_configs_keep_live_action_loop_local() -> None:
    remote = load_train_config(Path("../configs/train/ssh_remote_learner.yaml"))
    worker = load_train_config(Path("../configs/train/windows_game_worker.yaml"))

    assert remote.algorithm == worker.algorithm == "appo"
    assert remote.model == worker.model
    assert remote.learner.bind == "127.0.0.1:5600"
    assert remote.learner.device == "cuda"
    assert worker.learner.bind == "127.0.0.1:5600"
    assert worker.transport.name == "tcp"
    assert worker.transport.host == "127.0.0.1"
    assert worker.transport.port == 5555
    assert remote.security.bind_scope == worker.security.bind_scope == "localhost"
    assert remote.security.require_token is worker.security.require_token is True


def test_load_task_config_preserves_wire_id() -> None:
    gruz = load_task_config(Path("../configs/tasks/gruz_mother.yaml"))
    hornet = load_task_config(Path("../configs/tasks/hornet_protector.yaml"))
    mantis = load_task_config(Path("../configs/tasks/mantis_lords.yaml"))

    assert gruz.task_id == "gruz_mother"
    assert gruz.wire_id == 0
    assert gruz.action.n_macro_actions == 11
    assert hornet.wire_id == 1
    assert mantis.wire_id == 2


def test_hitless_arena_configs_select_recurrent_primitive_training() -> None:
    task = load_task_config(Path("../configs/tasks/gruz_mother_hitless_speed.yaml"))
    train = load_train_config(Path("../configs/train/arena_hitless_gru.yaml"))

    assert task.scene == "GG_Gruz_Mother"
    assert task.time_limit_seconds == 45
    assert task.action.enable_macro_actions is False
    assert task.action.n_macro_actions == 0
    assert task.action.expose_action_combinations is True
    assert task.arena.auto_reset_on_terminal is True
    assert task.arena.objective == "hitless_speedrun"
    assert task.arena.target_kill_time_seconds == 30.0
    assert train.algorithm == "recurrent_ppo"
    assert train.sequence_length == 64
    assert train.model.rnn_type == "gru"


def test_validate_task_collection_rejects_duplicate_task_identity() -> None:
    tasks = [
        TaskConfig(task_id="same", wire_id=1, scene="A"),
        TaskConfig(task_id="same", wire_id=2, scene="B"),
        TaskConfig(task_id="other", wire_id=1, scene="C"),
    ]

    with pytest.raises(ValueError, match="multi-task config"):
        validate_task_collection(tasks, context="multi-task config")

    with pytest.raises(ValueError) as exc_info:
        validate_task_collection(tasks)
    message = str(exc_info.value)
    assert "duplicate task_id" in message
    assert "same" in message
    assert "duplicate wire_id" in message
    assert "1 (same, other)" in message


def test_mod_scene_controller_uses_config_scene_with_wire_id_fallback() -> None:
    root = Path(__file__).parents[2]
    source = (root / "mod/HKRLEnvMod/Env/SceneController.cs").read_text(encoding="utf-8")
    reset_manager = (root / "mod/HKRLEnvMod/Env/ResetManager.cs").read_text(encoding="utf-8")

    assert "LoadTaskScene(int taskId, string? sceneName = null)" in source
    assert "ResolveSceneName(taskId, sceneName)" in source
    assert "return configuredSceneName.Trim()" in source
    assert "gameManager.BeginSceneTransition(" in source
    assert "SceneEntryLifecycle.MarkTransitionPending(gameManager)" in source
    assert "gameManager.LoadGameFromUI(_saveSlot)" in source
    assert 'GodhomeEntryGateName = "door_dreamEnter"' in source
    assert "SceneManager.LoadScene(_targetSceneName)" not in source
    assert "_loadFailed = true" in source
    assert "PlayerObserver.IsReadyForControl(hero)" in source
    assert "scene.handle != _sourceSceneHandle" in source
    assert "BossSceneController.SetupEvent = ConfigureBossScene" in source
    assert "controller.HasTransitionedIn" in source
    assert "PlayMakerFSM.BroadcastEvent" not in source
    assert '"Bench Control"' in source
    assert '_loadedBenchFsm.SendEvent("GET UP")' in source
    assert 'StaticVariableList.SetValue<string>("bossSceneToLoad"' in source
    assert "hero.EnterWithoutInput(true)" in source
    assert "hero.enterWithoutInput = true" not in source
    assert "hero.ClearMPSendEvents()" in source
    assert "gameManager.ResetSemiPersistentItems()" in source
    assert "GameManager.SceneLoadVisualizations.GodsAndGlory" in source
    assert '0 => "GG_Gruz_Mother"' in source
    assert '1 => "GG_Hornet_1"' in source
    assert '2 => "GG_Mantis_Lords"' in source
    assert "_ => string.Empty" in source
    assert "Unknown HKRL task id" in source
    assert "HasValidTarget" in source
    assert "BeginReset(int taskId, string? sceneName = null)" in reset_manager
    assert "_scene.LoadTaskScene(taskId, sceneName)" in reset_manager
    assert "!_scene.HasValidTarget" in reset_manager
    assert "HKRL.StatusCode.SceneLoadFailed" in reset_manager
    assert "ReadyStabilityGate" in reset_manager
    assert "_scene.CancelPendingLoad()" in reset_manager
    load_start = source.index("public void LoadTaskScene")
    load_end = source.index("public bool IsSceneReady", load_start)
    assert "== _targetSceneName" not in source[load_start:load_end]
    assert "BossLocator.FindActiveBosses()" in source
    assert "BossLocator.FindConfiguredBosses()" in source


def test_mod_player_readiness_requires_restored_game_physics() -> None:
    root = Path(__file__).parents[2]
    observer = (root / "mod/HKRLEnvMod/Observation/PlayerObserver.cs").read_text(encoding="utf-8")
    readiness = (root / "mod/HKRLEnvMod/Env/EpisodeReadiness.cs").read_text(encoding="utf-8")

    assert 'ReadBool(hero, true, "cState.transitioning")' in observer
    assert 'ReadText(hero, string.Empty, "transitionState")' in observer
    assert "body?.gravityScale ?? 0.0f" in observer
    assert "collider?.enabled ?? false" in observer
    assert 'ReadBool(hero, false, "CanJump")' in observer
    assert 'ReadyHeroTransitionState = "WAITING_TO_TRANSITION"' in readiness
    assert "gravityScale > 0.0f" in readiness
    assert "hasCollider" in readiness
    assert "ReadyStabilityGate" in readiness
    assert "groundedJumpReady" in readiness


def test_mod_action_readiness_uses_side_effect_free_game_queries() -> None:
    root = Path(__file__).parents[2]
    observer = (root / "mod/HKRLEnvMod/Observation/PlayerObserver.cs").read_text(encoding="utf-8")

    assert '"CanDoubleJump"' in observer
    assert '"CanDash"' in observer
    assert '"CanDreamNail"' in observer
    assert '"CanNailCharge"' in observer
    assert "playerData?.hasSpell" in observer
    assert '"cState.doubleJumpAvailable"' not in observer
    assert 'ReadBool(hero, false, "CanNailArt")' not in observer


def test_mod_runtime_config_supports_bootstrap_save_slot() -> None:
    root = Path(__file__).parents[2]
    runtime = (root / "mod/HKRLEnvMod/RuntimeConfiguration.cs").read_text(encoding="utf-8")
    driver = (root / "mod/HKRLEnvMod/HKRLEnvMod.cs").read_text(encoding="utf-8")

    assert 'SaveSlotEnv = "HKRL_SAVE_SLOT"' in runtime
    assert "minimum: 1" in runtime
    assert "maximum: 4" in runtime
    assert "new SceneController(runtime.SaveSlot)" in driver


def test_load_train_config_preserves_distributed_runtime_settings() -> None:
    config = load_train_config(Path("../configs/train/remote_learner.yaml"))

    assert config.algorithm == "appo"
    assert config.learner.bind == "127.0.0.1:5600"
    assert config.learner.device == "auto"
    assert config.learner.max_staleness == 4
    assert config.learner.checkpoint_dir == "checkpoints/"
    assert config.learner.publish_every_updates == 4
    assert config.coordinator.bind == "127.0.0.1:5610"
    assert config.coordinator.num_workers == 4
    assert config.security.bind_scope == "lan"
    assert config.security.require_token is True
    assert config.security.auth_token_env == "HKRL_AUTH_TOKEN"


def test_load_train_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    _write_yaml(
        config,
        {
            "algorithm": "ppo",
            "model": {"name": "mlp", "unknown_model_key": 1},
            "unexpected_top_level": True,
        },
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_train_config(config)


def test_load_train_config_rejects_invalid_learner_device(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    _write_yaml(config, {"learner": {"device": "gpu"}})

    with pytest.raises(ValueError, match="String should match pattern"):
        load_train_config(config)


def test_load_train_config_rejects_appo_checkpoint_cadence_past_staleness(
    tmp_path: Path,
) -> None:
    config = tmp_path / "bad.yaml"
    _write_yaml(
        config,
        {
            "algorithm": "appo",
            "learner": {
                "max_staleness": 2,
                "publish_every_updates": 4,
            },
        },
    )

    with pytest.raises(ValueError, match=r"publish_every_updates.*max_staleness"):
        load_train_config(config)


def test_load_train_config_rejects_unknown_enum_values(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    _write_yaml(config, {"algorithm": "dqn", "transport": {"name": "udp"}})

    with pytest.raises(ValueError, match="Input should be"):
        load_train_config(config)


def test_load_train_config_rejects_invalid_numeric_ranges(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    _write_yaml(
        config,
        {
            "gamma": 1.5,
            "rollout_steps": 0,
            "transport": {"port": 70000},
            "learner": {"max_staleness": -1},
            "coordinator": {"heartbeat_timeout_s": 0},
        },
    )

    with pytest.raises(ValueError, match=r"less than or equal to 1|greater than or equal to 1"):
        load_train_config(config)


def test_load_train_config_rejects_zero_env_transport_port(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    _write_yaml(config, {"transport": {"port": 0}})

    with pytest.raises(ValueError, match="greater than or equal to 1"):
        load_train_config(config)


def test_load_task_config_rejects_invalid_action_repeat(tmp_path: Path) -> None:
    config = tmp_path / "task.yaml"
    _write_yaml(
        config,
        {
            "task_id": "bad",
            "scene": "Scene",
            "time_limit_seconds": 0,
            "observation": {"max_entities": 0},
            "action": {"action_repeat": 256},
        },
    )

    with pytest.raises(
        ValueError,
        match=r"less than or equal to 255|greater than or equal to 1",
    ):
        load_task_config(config)


def test_load_task_config_rejects_unsupported_macro_count(tmp_path: Path) -> None:
    config = tmp_path / "task.yaml"
    _write_yaml(
        config,
        {
            "task_id": "bad",
            "scene": "Scene",
            "action": {"n_macro_actions": 12},
        },
    )

    with pytest.raises(ValueError, match="less than or equal to 11"):
        load_task_config(config)


def test_load_config_rejects_empty_required_strings(tmp_path: Path) -> None:
    train_config = tmp_path / "train.yaml"
    _write_yaml(
        train_config,
        {
            "model": {"name": ""},
            "transport": {"host": "", "shm_name": ""},
            "learner": {"bind": "", "checkpoint_dir": ""},
            "coordinator": {"bind": ""},
            "security": {"auth_token_env": ""},
        },
    )

    with pytest.raises(ValueError, match="String should have at least 1 character"):
        load_train_config(train_config)

    task_config = tmp_path / "task.yaml"
    _write_yaml(task_config, {"task_id": "", "scene": ""})

    with pytest.raises(ValueError, match="String should have at least 1 character"):
        load_task_config(task_config)


def test_resolve_auth_token_uses_configured_environment() -> None:
    config = load_train_config(Path("../configs/train/remote_learner.yaml"))

    assert resolve_auth_token(config, {"HKRL_AUTH_TOKEN": "secret"}) == "secret"


def test_resolve_auth_token_requires_non_empty_token() -> None:
    config = load_train_config(Path("../configs/train/remote_learner.yaml"))

    with pytest.raises(ValueError, match="HKRL_AUTH_TOKEN"):
        resolve_auth_token(config, {})
    with pytest.raises(ValueError, match="HKRL_AUTH_TOKEN"):
        resolve_auth_token(config, {"HKRL_AUTH_TOKEN": ""})


def test_resolve_auth_token_returns_none_when_disabled() -> None:
    config = load_train_config(Path("../configs/train/ppo_mlp.yaml"))

    assert resolve_auth_token(config, {}) is None


def test_validate_bind_address_accepts_scoped_binds() -> None:
    assert validate_bind_address("127.0.0.1:5600", "localhost") == "127.0.0.1:5600"
    assert validate_bind_address("[::1]:5600", "localhost") == "[::1]:5600"
    assert validate_bind_address("192.168.1.20:5600", "lan") == "192.168.1.20:5600"
    assert validate_bind_address("10.0.0.4:5600", "lan") == "10.0.0.4:5600"


def test_validate_bind_address_rejects_out_of_scope_binds() -> None:
    with pytest.raises(ValueError, match="loopback"):
        validate_bind_address("0.0.0.0:5600", "localhost")
    with pytest.raises(ValueError, match="wildcard"):
        validate_bind_address("0.0.0.0:5600", "lan")
    with pytest.raises(ValueError, match="wildcard"):
        validate_bind_address("[::]:5600", "lan")
    with pytest.raises(ValueError, match="public IP"):
        validate_bind_address("8.8.8.8:5600", "lan")
    with pytest.raises(ValueError, match="host:port"):
        validate_bind_address("127.0.0.1", "localhost")


def test_validate_service_auth_requires_token_for_non_loopback_bind() -> None:
    local = load_train_config(Path("../configs/train/ppo_mlp.yaml"))
    remote = load_train_config(Path("../configs/train/remote_learner.yaml"))

    validate_service_auth("127.0.0.1:5600", local)
    validate_service_auth("192.168.1.20:5600", remote)
    with pytest.raises(ValueError, match="require_token"):
        validate_service_auth("192.168.1.20:5600", local)
