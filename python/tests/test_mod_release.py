"""Deterministic HKRLEnvMod release-package contracts."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from hkrl.utils import mod_release
from hkrl.utils.mod_release import (
    ModReleaseError,
    build_mod_release,
    load_mod_release_metadata,
    sha256_file,
    validate_live_evidence,
    validate_walk_evidence,
    verify_mod_release,
)

ROOT = Path(__file__).parents[2]


def test_release_metadata_aligns_version_project_and_protocol() -> None:
    metadata = load_mod_release_metadata(ROOT)

    assert metadata.mod_version == "0.8.0"
    assert metadata.release_tag == "v0.8.0"
    assert metadata.protocol_schema_version == 6
    assert metadata.target_framework == "net472"
    assert metadata.flatbuffers_runtime_version == "23.5.26"
    assert metadata.archive_stem == "HKRLEnvMod-v0.8.0-schema6"


def test_release_package_is_deterministic_complete_and_secret_free(
    tmp_path: Path,
) -> None:
    metadata = load_mod_release_metadata(ROOT)
    inputs = _release_inputs(tmp_path, metadata.mod_version)

    first = build_mod_release(
        repo_root=ROOT,
        **inputs,
        output_dir=tmp_path / "first",
        allow_dirty=True,
        require_tag=False,
    )
    second = build_mod_release(
        repo_root=ROOT,
        **inputs,
        output_dir=tmp_path / "second",
        allow_dirty=True,
        require_tag=False,
    )

    assert first.archive_sha256 == second.archive_sha256
    assert first.archive.read_bytes() == second.archive.read_bytes()
    manifest = verify_mod_release(
        first.archive,
        repo_root=ROOT,
        allow_dirty=True,
        require_tag=False,
    )
    assert manifest["mod_version"] == "0.8.0"
    assert manifest["protocol_schema_version"] == 6

    with zipfile.ZipFile(first.archive) as bundle:
        names = set(bundle.namelist())
    assert names == {
        "HKRLEnvMod/Google.FlatBuffers.dll",
        "HKRLEnvMod/HKRLEnvMod.dll",
        "HKRLEnvMod/LICENSE",
        "HKRLEnvMod/README.md",
        "HKRLEnvMod/evidence/godhome-all-boss-sweep-v0.8.0.json",
        "HKRLEnvMod/evidence/godhome-all-boss-sweep-v0.8.0.md",
        "HKRLEnvMod/evidence/walk-smoothness-post-fix.json",
        "HKRLEnvMod/manifest.json",
    }
    assert all("hkrl-runtime.conf" not in name for name in names)


def test_release_rejects_dirty_or_untagged_official_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = load_mod_release_metadata(ROOT)
    inputs = _release_inputs(tmp_path, metadata.mod_version)
    monkeypatch.setattr(
        mod_release,
        "_git_state",
        lambda _root: ("1" * 40, True, set()),
    )

    with pytest.raises(ModReleaseError, match="dirty worktree"):
        build_mod_release(
            repo_root=ROOT,
            **inputs,
            output_dir=tmp_path / "dirty",
        )

    monkeypatch.setattr(
        mod_release,
        "_git_state",
        lambda _root: ("1" * 40, False, set()),
    )
    with pytest.raises(ModReleaseError, match=r"not tagged v0\.8\.0"):
        build_mod_release(
            repo_root=ROOT,
            **inputs,
            output_dir=tmp_path / "untagged",
        )


def test_live_acceptance_rejects_binary_drift_or_failed_boss(
    tmp_path: Path,
) -> None:
    metadata = load_mod_release_metadata(ROOT)
    inputs = _release_inputs(tmp_path, metadata.mod_version)
    dll_digest = sha256_file(inputs["dll_path"])
    payload = json.loads(Path(inputs["evidence_json_path"]).read_text(encoding="utf-8"))

    validate_live_evidence(
        payload,
        metadata=metadata,
        dll_sha256=dll_digest,
        dll_size_bytes=Path(inputs["dll_path"]).stat().st_size,
    )
    with pytest.raises(ModReleaseError, match="DLL hash"):
        validate_live_evidence(
            payload,
            metadata=metadata,
            dll_sha256="f" * 64,
            dll_size_bytes=Path(inputs["dll_path"]).stat().st_size,
        )

    payload["results"][12]["status"] = "failed"
    with pytest.raises(ModReleaseError, match=r"result\[12\] is not verified"):
        validate_live_evidence(
            payload,
            metadata=metadata,
            dll_sha256=dll_digest,
            dll_size_bytes=Path(inputs["dll_path"]).stat().st_size,
        )

    payload["results"][11]["boss_meta"]["simultaneous_boss_count"] = 1
    with pytest.raises(ModReleaseError, match="at least two simultaneous Bosses"):
        validate_live_evidence(
            payload,
            metadata=metadata,
            dll_sha256=dll_digest,
            dll_size_bytes=Path(inputs["dll_path"]).stat().st_size,
        )

    walk_payload = json.loads(Path(inputs["walk_evidence_path"]).read_text(encoding="utf-8"))
    validate_walk_evidence(
        walk_payload,
        metadata=metadata,
        dll_sha256=dll_digest,
        dll_size_bytes=Path(inputs["dll_path"]).stat().st_size,
    )
    walk_payload["speed_retention"] = 0.5
    walk_payload["smooth"] = False
    with pytest.raises(ModReleaseError, match="walk benchmark did not pass"):
        validate_walk_evidence(
            walk_payload,
            metadata=metadata,
            dll_sha256=dll_digest,
            dll_size_bytes=Path(inputs["dll_path"]).stat().st_size,
        )


def _release_inputs(tmp_path: Path, version: str) -> dict[str, Path]:
    dll = tmp_path / "HKRLEnvMod.dll"
    flatbuffers = tmp_path / "Google.FlatBuffers.dll"
    evidence = tmp_path / "evidence.json"
    report = tmp_path / "evidence.md"
    walk_evidence = tmp_path / "walk-smoothness.json"
    dll.write_bytes(b"deterministic-mod-binary")
    flatbuffers.write_bytes(b"deterministic-flatbuffers-runtime")
    dll_digest = sha256_file(dll)
    boss_ids = [f"boss_{index:02d}" for index in range(44)]
    results = [
        {
            "boss_id": boss_id,
            "status": "verified",
            "failures": [],
            "build_scene_present": True,
            "hero": {
                "movement_left": True,
                "movement_right": True,
                "jump_input_seen": True,
                "jump_takeoff": True,
                "gravity_seen": True,
                "jump_landed": True,
                "attack_input_seen": True,
                "attack_state_seen": True,
                "invalid_action_seen": False,
            },
            "boss_activity": {
                "post_ack_activity_observed": True,
                "full_health_observed": True,
                "full_health_max_hp": [100.0],
            },
            "post_reset_boss_activity": {
                "post_ack_activity_observed": True,
                "full_health_observed": True,
                "full_health_max_hp": [100.0],
            },
            "reset": {
                "initial_episode_id": index * 2 + 1,
                "same_scene_episode_id": index * 2 + 2,
            },
            "initial_snapshot": {"reward_events": []},
            "post_reset_snapshot": {"reward_events": []},
            "boss_meta": (
                {
                    "sample_label": "initial_reset",
                    "simultaneous_boss_count": 2,
                    "expected_min_bosses": 2,
                    "unique_stable_ids": True,
                    "metadata_fields": sorted(mod_release.MULTI_BOSS_META_FIELDS),
                    "bosses": [
                        _boss_meta_row(stable_id=101, x=90.0),
                        _boss_meta_row(stable_id=102, x=110.0),
                    ],
                    "failures": [],
                    "valid": True,
                }
                if boss_id == "boss_11"
                else {
                    "sample_label": "initial_reset",
                    "simultaneous_boss_count": 1,
                    "expected_min_bosses": 1,
                    "unique_stable_ids": True,
                    "metadata_fields": sorted(mod_release.MULTI_BOSS_META_FIELDS),
                    "bosses": [_boss_meta_row(stable_id=index + 1, x=100.0)],
                    "failures": [],
                    "valid": True,
                }
            ),
        }
        for index, boss_id in enumerate(boss_ids)
    ]
    results[11]["boss_id"] = "oblobbles"
    boss_ids[11] = "oblobbles"
    evidence.write_text(
        json.dumps(
            {
                "probe_schema": "hkrl.godhome_sweep.v2",
                "schema_version": 6,
                "boss_mutation_allowed": False,
                "simulation_control_allowed": False,
                "tested_mod": {
                    "id": "HKRLEnvMod",
                    "version": version,
                    "dll_name": dll.name,
                    "dll_size_bytes": dll.stat().st_size,
                    "dll_sha256": dll_digest,
                },
                "installed_build": {
                    "name": "globalgamemanagers",
                    "size_bytes": 123,
                    "sha256": "a" * 64,
                },
                "catalog": {
                    "name": "godhome_bosses.yaml",
                    "size_bytes": 456,
                    "sha256": "b" * 64,
                },
                "counts": {
                    "selected": 44,
                    "completed": 44,
                    "verified": 44,
                    "failed": 0,
                    "remaining": 0,
                },
                "selected_boss_ids": boss_ids,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report.write_text(
        f"# Godhome all-Boss live compatibility sweep\n\n"
        f"- Tested Mod: HKRLEnvMod v{version} (`{dll_digest}`)\n",
        encoding="utf-8",
    )
    walk_evidence.write_text(
        json.dumps(
            {
                "schema": "hkrl.walk_smoothness.v1",
                "boss_mutation_allowed": False,
                "simulation_control_allowed": False,
                "smooth": True,
                "speed_retention": 0.98,
                "min_speed_retention": 0.9,
                "tested_mod": {
                    "id": "HKRLEnvMod",
                    "version": version,
                    "dll_name": dll.name,
                    "dll_size_bytes": dll.stat().st_size,
                    "dll_sha256": dll_digest,
                },
                "continuous": {
                    "commanded_ticks": 24,
                    "decisions": 1,
                    "direction": "right",
                    "fixed_delta_time_s": 0.02,
                    "event_kinds": [],
                    "start_hp": 9,
                    "end_hp": 9,
                },
                "stepped": {
                    "commanded_ticks": 24,
                    "decisions": 12,
                    "direction": "right",
                    "fixed_delta_time_s": 0.02,
                    "event_kinds": [],
                    "start_hp": 9,
                    "end_hp": 9,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "dll_path": dll,
        "flatbuffers_dll_path": flatbuffers,
        "evidence_json_path": evidence,
        "evidence_report_path": report,
        "walk_evidence_path": walk_evidence,
    }


def _boss_meta_row(*, stable_id: int, x: float) -> dict[str, object]:
    return {
        "stable_id": stable_id,
        "entity_type": 1,
        "team": 1,
        "prefab_hash": 1000 + stable_id,
        "fsm_name_hash": 2000,
        "fsm_state_hash": 3000,
        "x": x,
        "y": 14.0,
        "rel_x": x - 100.0,
        "rel_y": 8.0,
        "vx": 0.0,
        "vy": 0.0,
        "hp": 450.0,
        "max_hp": 450.0,
        "hurtbox_center_x": x,
        "hurtbox_center_y": 14.0,
        "hurtbox_size_x": 3.0,
        "hurtbox_size_y": 3.0,
        "hitbox_active": False,
        "phase": 0,
        "threat_score": 100.0,
        "flags": 16,
    }
