"""Offline contracts for the versioned Hall of Gods compatibility sweep."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
from hkrl.godhome import GodhomeBossCatalog, load_godhome_catalog

ROOT = Path(__file__).parents[2]
CATALOG_PATH = ROOT / "configs/godhome_bosses.yaml"


def test_catalog_has_all_44_distinct_hall_of_gods_fights() -> None:
    catalog = load_godhome_catalog(CATALOG_PATH)
    by_id = {boss.boss_id: boss for boss in catalog.bosses}

    assert catalog.catalog_version == 1
    assert len(catalog.bosses) == 44
    assert by_id["gruz_mother"].scene == "GG_Gruz_Mother"
    assert by_id["sisters_of_battle"].scene == "GG_Mantis_Lords_V"
    assert by_id["winged_nosk"].scene == "GG_Nosk_Hornet"
    assert by_id["pure_vessel"].scene == "GG_Hollow_Knight"
    assert by_id["absolute_radiance"].scene == "GG_Radiance"
    assert by_id["nightmare_king_grimm"].variant_of == "troupe_master_grimm"


def test_catalog_builds_primitive_only_probe_tasks() -> None:
    catalog = load_godhome_catalog(CATALOG_PATH)
    task = catalog.make_task(catalog.bosses[0])

    assert task.task_id == "godhome_probe_gruz_mother"
    assert task.wire_id == 2000
    assert task.scene == "GG_Gruz_Mother"
    assert task.observation.tier == "privileged"
    assert task.observation.max_entities == 64
    assert task.action.action_repeat == 1
    assert not task.action.enable_macro_actions
    assert task.action.n_macro_actions == 0
    assert not task.action.expose_action_combinations


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda data: data["bosses"][1].update(boss_id=data["bosses"][0]["boss_id"]),
            "duplicate boss_id",
        ),
        (
            lambda data: data["bosses"][1].update(wire_id=data["bosses"][0]["wire_id"]),
            "duplicate wire_id",
        ),
        (
            lambda data: data["bosses"][1].update(scene=data["bosses"][0]["scene"]),
            "duplicate scene",
        ),
        (
            lambda data: data["bosses"][0].update(variant_of="missing_boss"),
            "unknown boss",
        ),
    ],
)
def test_catalog_rejects_ambiguous_or_dangling_identity(
    mutate: object,
    match: str,
) -> None:
    data = load_godhome_catalog(CATALOG_PATH).model_dump()
    mutate(data)  # type: ignore[operator]

    with pytest.raises(ValueError, match=match):
        GodhomeBossCatalog.model_validate(data)


def test_sweep_selection_is_ordered_restartable_and_strict() -> None:
    module = _load_script()
    catalog = load_godhome_catalog(CATALOG_PATH)

    selected = module.select_bosses(
        catalog,
        boss_ids=["absolute_radiance", "pure_vessel"],
        start_at="pure_vessel",
        max_bosses=None,
    )
    assert [boss.boss_id for boss in selected] == [
        "pure_vessel",
        "absolute_radiance",
    ]

    with pytest.raises(ValueError, match="unknown Boss"):
        module.select_bosses(
            catalog,
            boss_ids=["radiance"],
            start_at=None,
            max_bosses=None,
        )


def test_sweep_hash_matches_csharp_and_build_scene_preflight(tmp_path: Path) -> None:
    module = _load_script()
    build = tmp_path / "globalgamemanagers"
    build.write_bytes(b"\x00Assets/Scenes/GG_Gruz_Mother.unity\x00GG_Radiance\x00GG_Workshop\x00")

    assert module.fnv1a_32("GG_Gruz_Mother") == -1364303844
    assert module.wire_scene_hash("GG_Gruz_Mother") == -1364303872
    assert module.load_build_scenes(build) == {
        "GG_Gruz_Mother",
        "GG_Radiance",
        "GG_Workshop",
    }


def test_sweep_validates_lifecycle_controls_and_boss_activity() -> None:
    module = _load_script()
    catalog = load_godhome_catalog(CATALOG_PATH)
    boss = catalog.bosses[0]
    entity_a = module.BossTelemetry(10, 101, 5.0, 2.0, 0.0, 0.0, 90.0, 90.0)
    entity_b = module.BossTelemetry(10, 202, 5.5, 2.0, 1.0, 0.0, 90.0, 90.0)
    first = _sample(module, boss, entity_a, label="first", episode_id=7)
    second = _sample(module, boss, entity_b, label="second", episode_id=7)

    assert (
        module.validate_reset(
            first,
            boss=boss,
            expected_scene_hash=module.wire_scene_hash(boss.scene),
            label="initial",
        )
        == []
    )
    assert (
        module.validate_hero(
            {
                "movement_left": True,
                "movement_right": True,
                "jump_input_seen": True,
                "jump_takeoff": True,
                "gravity_seen": True,
                "jump_landed": True,
                "attack_input_seen": True,
                "attack_state_seen": True,
                "invalid_action_seen": False,
            }
        )
        == []
    )
    activity = module.summarize_boss_activity([first, second])
    assert activity["position_changed"]
    assert activity["velocity_seen"]
    assert activity["fsm_changed"]
    assert activity["post_ack_activity_observed"]
    assert activity["positive_hp_observed"]
    assert activity["full_health_observed"]
    assert activity["full_health_max_hp"] == [90.0]
    assert module._bosses_demonstrate_activity(
        second.boss_entities,
        {entity_a.stable_id: entity_a},
    )


def test_sweep_accepts_deferred_boss_hp_only_after_natural_activation() -> None:
    module = _load_script()
    boss = load_godhome_catalog(CATALOG_PATH).bosses[0]
    dormant = module.BossTelemetry(
        10,
        101,
        5.0,
        -11.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    active = module.BossTelemetry(
        11,
        202,
        5.0,
        2.0,
        0.0,
        0.0,
        90.0,
        90.0,
    )
    reset = _sample(module, boss, dormant, label="reset", episode_id=7)
    activated = _sample(module, boss, active, label="activated", episode_id=7)

    assert (
        module.validate_reset(
            reset,
            boss=boss,
            expected_scene_hash=module.wire_scene_hash(boss.scene),
            label="initial",
        )
        == []
    )
    activity = module.summarize_boss_activity([reset, activated])
    assert activity["entity_set_changed"]
    assert activity["post_ack_activity_observed"]
    assert activity["full_health_observed"]
    assert activity["full_health_sample_label"] == "activated"


def test_sweep_health_gate_ignores_zero_hp_transition_placeholder() -> None:
    module = _load_script()
    boss = load_godhome_catalog(CATALOG_PATH).bosses[0]
    placeholder = module.BossTelemetry(
        10,
        101,
        5.0,
        -11.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    combat_boss = module.BossTelemetry(
        11,
        202,
        5.0,
        2.0,
        0.0,
        0.0,
        90.0,
        90.0,
    )
    sample = replace(
        _sample(
            module,
            boss,
            placeholder,
            label="mixed",
            episode_id=7,
        ),
        boss_entities=(placeholder, combat_boss),
    )

    assert module._boss_health_ready(sample)
    activity = module.summarize_boss_activity([sample])
    assert activity["full_health_observed"]
    assert activity["full_health_max_hp"] == [90.0]


def test_sweep_uses_bounded_hero_primitives_for_dormant_boss_activation() -> None:
    source = (ROOT / "scripts/live_godhome_sweep.py").read_text(encoding="utf-8")

    assert "def probe_boss_activation" in source
    assert '"activation_strategy": "paced_right_neutral_jump_dash"' in source
    assert 'buttons.append("jump_hold")' in source
    assert 'buttons.append("dash")' in source
    assert "movement = 2 if index % 2 == 0 else 1" in source
    assert "duration=0" in source
    assert "max_steps: int = 320" in source


def test_movement_contract_uses_directional_displacement_at_step_boundary() -> None:
    source = (ROOT / "scripts/live_godhome_sweep.py").read_text(encoding="utf-8")

    assert 'best_dx["left"] < -POSITION_EPSILON' in source
    assert 'best_dx["right"] > POSITION_EPSILON' in source
    assert "for round_index in range(3)" in source
    assert 'self._wait_for_button("attack", max_ticks=400)' in source
    assert "left_dx < -POSITION_EPSILON and left_velocity_seen" not in source
    assert "right_dx > POSITION_EPSILON and right_velocity_seen" not in source


def test_sweep_source_cannot_mutate_boss_or_simulation_state() -> None:
    source = (ROOT / "scripts/live_godhome_sweep.py").read_text(encoding="utf-8")

    assert ".pause(" not in source
    assert ".resume(" not in source
    assert ".set_timescale(" not in source
    assert "BossSceneController" not in source
    assert "HealthManager" not in source
    assert "PlayMakerFSM" not in source
    assert "SetValue" not in source
    assert '"boss_mutation_allowed": False' in source
    assert '"simulation_control_allowed": False' in source


def test_report_renders_entry_and_reload_lifecycle_evidence() -> None:
    module = _load_script()
    report = module.render_report(
        {
            "counts": {
                "verified": 1,
                "failed": 0,
                "remaining": 0,
                "selected": 1,
            },
            "schema_version": 6,
            "catalog_version": 1,
            "tested_mod": {
                "version": "0.8.0",
                "dll_sha256": "a" * 64,
            },
            "results": [
                {
                    "display_name": "Boss",
                    "scene": "GG_Boss",
                    "status": "verified",
                    "failures": [],
                    "reset": {
                        "initial_duration_s": 1.0,
                        "same_scene_duration_s": 1.1,
                    },
                    "boss_activity": {
                        "fsm_changed": True,
                        "full_health_observed": True,
                        "activation_steps": 0,
                    },
                    "post_reset_boss_activity": {
                        "position_changed": True,
                        "full_health_observed": True,
                        "activation_steps": 12,
                    },
                    "hero": {
                        "movement_left": True,
                        "movement_right": True,
                        "jump_takeoff": True,
                        "gravity_seen": True,
                        "jump_landed": True,
                        "attack_state_seen": True,
                    },
                }
            ],
        }
    )

    assert "FSM+full-HP@0/position+full-HP@12" in report
    assert "L/R/J/G/L/A" in report
    assert "HKRLEnvMod v0.8.0" in report
    assert "a" * 64 in report


def test_resume_evidence_is_bound_to_the_same_binary_and_game_build(
    tmp_path: Path,
) -> None:
    module = _load_script()
    evidence = tmp_path / "sweep.json"
    tested_mod = {
        "id": "HKRLEnvMod",
        "version": "0.8.0",
        "dll_name": "HKRLEnvMod.dll",
        "dll_size_bytes": 10,
        "dll_sha256": "a" * 64,
    }
    catalog = {
        "name": "godhome_bosses.yaml",
        "size_bytes": 20,
        "sha256": "b" * 64,
    }
    installed_build = {
        "name": "globalgamemanagers",
        "size_bytes": 30,
        "sha256": "c" * 64,
    }
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 6,
                "tested_mod": tested_mod,
                "catalog": catalog,
                "installed_build": installed_build,
                "results": [{"boss_id": "gruz_mother", "status": "verified"}],
            }
        ),
        encoding="utf-8",
    )

    resumed = module._load_resume_results(
        evidence,
        expected_tested_mod=tested_mod,
        expected_catalog=catalog,
        expected_installed_build=installed_build,
    )
    assert set(resumed) == {"gruz_mother"}

    mismatched_mod = dict(tested_mod, dll_sha256="d" * 64)
    with pytest.raises(ValueError, match="tested_mod"):
        module._load_resume_results(
            evidence,
            expected_tested_mod=mismatched_mod,
            expected_catalog=catalog,
            expected_installed_build=installed_build,
        )


def _sample(
    module: ModuleType,
    boss: object,
    entity: object,
    *,
    label: str,
    episode_id: int,
) -> object:
    return module.ProbeSnapshot(
        label=label,
        lifecycle="RUNNING",
        server_tick=100,
        episode_id=episode_id,
        scene_hash=module.wire_scene_hash(boss.scene),
        task_id=boss.wire_id,
        player_x=1.0,
        player_y=2.0,
        player_vx=0.0,
        player_vy=0.0,
        player_hp=9,
        player_max_hp=9,
        on_ground=True,
        jumping=False,
        falling=False,
        action_flags=0,
        applied_input_buttons=0,
        boss_entities=(entity,),
        reward_events=(),
    )


def _load_script() -> ModuleType:
    path = ROOT / "scripts/live_godhome_sweep.py"
    spec = importlib.util.spec_from_file_location("live_godhome_sweep", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
