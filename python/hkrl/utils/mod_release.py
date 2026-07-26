"""Build and verify a self-describing HKRLEnvMod release archive.

The archive is deliberately small: the two runtime DLLs, install/license
documentation, a manifest, and the live Hall of Gods acceptance evidence.  It
never packages ``hkrl-runtime.conf`` because that file may contain an auth
token and is machine-specific.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree

MOD_ID = "HKRLEnvMod"
MOD_NAME = "HKRL Environment Server"
MANIFEST_VERSION = 1
SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
HERO_ACCEPTANCE_KEYS = (
    "movement_left",
    "movement_right",
    "jump_input_seen",
    "jump_takeoff",
    "gravity_seen",
    "jump_landed",
    "attack_input_seen",
    "attack_state_seen",
)


class ModReleaseError(ValueError):
    """Raised when release metadata or evidence is not self-consistent."""


@dataclass(frozen=True)
class ModReleaseMetadata:
    """Version contract assembled from the C# project and wire constants."""

    mod_version: str
    protocol_schema_version: int
    target_framework: str
    flatbuffers_runtime_version: str

    @property
    def release_tag(self) -> str:
        return f"v{self.mod_version}"

    @property
    def archive_stem(self) -> str:
        return f"{MOD_ID}-v{self.mod_version}-schema{self.protocol_schema_version}"


@dataclass(frozen=True)
class ModPackageResult:
    """Paths and digest produced by :func:`build_mod_release`."""

    archive: Path
    manifest: Path
    checksum: Path
    archive_sha256: str


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest for *path* without loading it whole."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(path: str | Path) -> dict[str, Any]:
    """Return a machine-path-free file identity suitable for evidence."""

    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise ModReleaseError(f"release input is not a file: {resolved}")
    return {
        "name": resolved.name,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def load_mod_release_metadata(repo_root: str | Path) -> ModReleaseMetadata:
    """Read and cross-check every source that participates in Mod compatibility."""

    root = Path(repo_root).resolve()
    mod_root = root / "mod" / MOD_ID
    props_path = mod_root / "Version.props"
    project_path = mod_root / f"{MOD_ID}.csproj"
    python_protocol = root / "python" / "hkrl" / "protocol.py"
    csharp_protocol = mod_root / "Transport" / "Protocol.cs"

    props = _parse_xml(props_path)
    project = _parse_xml(project_path)
    version = _required_xml_text(props, "HKRLModVersion", props_path)
    if not SEMVER.fullmatch(version):
        raise ModReleaseError(
            f"{props_path}: HKRLModVersion must be MAJOR.MINOR.PATCH; got {version!r}"
        )

    import_paths = {
        element.attrib.get("Project")
        for element in project.iter()
        if _local_name(element.tag) == "Import"
    }
    if "Version.props" not in import_paths:
        raise ModReleaseError(f"{project_path}: Version.props is not imported")

    target_framework = _required_xml_text(project, "TargetFramework", project_path)
    flatbuffers_version: str | None = None
    for element in project.iter():
        if (
            _local_name(element.tag) == "PackageReference"
            and element.attrib.get("Include") == "Google.FlatBuffers"
        ):
            flatbuffers_version = element.attrib.get("Version")
            break
    if not flatbuffers_version:
        raise ModReleaseError(f"{project_path}: Google.FlatBuffers PackageReference has no Version")

    python_schema = _extract_int_constant(
        python_protocol,
        r"^SCHEMA_VERSION\s*:\s*int\s*=\s*([0-9]+)\s*$",
        "SCHEMA_VERSION",
    )
    csharp_schema = _extract_int_constant(
        csharp_protocol,
        r"\bSchemaVersion\s*=\s*([0-9]+)\s*;",
        "SchemaVersion",
    )
    if python_schema != csharp_schema:
        raise ModReleaseError(f"protocol schema drift: Python={python_schema}, C#={csharp_schema}")

    return ModReleaseMetadata(
        mod_version=version,
        protocol_schema_version=python_schema,
        target_framework=target_framework,
        flatbuffers_runtime_version=flatbuffers_version,
    )


def validate_live_evidence(
    payload: dict[str, Any],
    *,
    metadata: ModReleaseMetadata,
    dll_sha256: str,
    dll_size_bytes: int,
) -> None:
    """Reject a Hall of Gods report that cannot certify the packaged DLL."""

    errors: list[str] = []
    if payload.get("schema_version") != metadata.protocol_schema_version:
        errors.append("protocol schema does not match the release")
    if payload.get("boss_mutation_allowed") is not False:
        errors.append("Boss mutation must be explicitly disabled")
    if payload.get("simulation_control_allowed") is not False:
        errors.append("simulation-state mutation must be explicitly disabled")

    tested_mod = payload.get("tested_mod")
    if not isinstance(tested_mod, dict):
        errors.append("tested_mod fingerprint is missing")
    else:
        if tested_mod.get("id") != MOD_ID:
            errors.append("tested_mod.id is not HKRLEnvMod")
        if tested_mod.get("version") != metadata.mod_version:
            errors.append("tested Mod version does not match the release")
        if tested_mod.get("dll_sha256") != dll_sha256:
            errors.append("tested Mod DLL hash does not match the packaged DLL")
        if tested_mod.get("dll_size_bytes") != dll_size_bytes:
            errors.append("tested Mod DLL size does not match the packaged DLL")

    installed_build = payload.get("installed_build")
    if not _valid_fingerprint(installed_build, expected_name="globalgamemanagers"):
        errors.append("installed Hollow Knight build fingerprint is invalid")
    catalog = payload.get("catalog")
    if not _valid_fingerprint(catalog, expected_name="godhome_bosses.yaml"):
        errors.append("Godhome catalog fingerprint is invalid")

    counts = payload.get("counts")
    expected_counts = {
        "selected": 44,
        "completed": 44,
        "verified": 44,
        "failed": 0,
        "remaining": 0,
    }
    if not isinstance(counts, dict) or any(
        counts.get(key) != value for key, value in expected_counts.items()
    ):
        errors.append("Hall of Gods counts are not a clean 44/44 pass")

    selected = payload.get("selected_boss_ids")
    results = payload.get("results")
    if (
        not isinstance(selected, list)
        or len(selected) != 44
        or len(set(selected)) != 44
        or not all(isinstance(item, str) and item for item in selected)
    ):
        errors.append("selected_boss_ids must contain 44 unique IDs")
        selected = []
    if not isinstance(results, list) or len(results) != 44:
        errors.append("evidence must contain exactly 44 result rows")
        results = []

    result_ids: list[str] = []
    for index, result in enumerate(results):
        prefix = f"result[{index}]"
        if not isinstance(result, dict):
            errors.append(f"{prefix} is not an object")
            continue
        boss_id = result.get("boss_id")
        if not isinstance(boss_id, str) or not boss_id:
            errors.append(f"{prefix} has no Boss ID")
        else:
            result_ids.append(boss_id)
        if result.get("status") != "verified":
            errors.append(f"{prefix} is not verified")
        if result.get("failures") != []:
            errors.append(f"{prefix} contains failures")
        if result.get("build_scene_present") is not True:
            errors.append(f"{prefix} did not pass the installed-scene preflight")

        hero = result.get("hero")
        if not isinstance(hero, dict) or any(
            hero.get(key) is not True for key in HERO_ACCEPTANCE_KEYS
        ):
            errors.append(f"{prefix} does not prove all Hero primitive controls")
        elif hero.get("invalid_action_seen") is not False:
            errors.append(f"{prefix} emitted an invalid Hero action")

        entry = result.get("boss_activity")
        reload_activity = result.get("post_reset_boss_activity")
        for label, activity in (("entry", entry), ("reload", reload_activity)):
            if not isinstance(activity, dict):
                errors.append(f"{prefix} has no {label} Boss activity")
                continue
            if activity.get("post_ack_activity_observed") is not True:
                errors.append(f"{prefix} has no natural {label} Boss activity")
            if activity.get("full_health_observed") is not True:
                errors.append(f"{prefix} has no full-health {label} Boss state")
        if isinstance(entry, dict) and isinstance(reload_activity, dict):
            entry_hp = entry.get("full_health_max_hp")
            reload_hp = reload_activity.get("full_health_max_hp")
            if (
                not isinstance(entry_hp, list)
                or not entry_hp
                or entry_hp != reload_hp
                or not all(_is_positive_number(value) for value in entry_hp)
            ):
                errors.append(f"{prefix} has inconsistent entry/reload Boss HP")

        reset = result.get("reset")
        if not isinstance(reset, dict):
            errors.append(f"{prefix} has no reset evidence")
        else:
            first_episode = reset.get("initial_episode_id")
            second_episode = reset.get("same_scene_episode_id")
            if (
                not isinstance(first_episode, int)
                or isinstance(first_episode, bool)
                or not isinstance(second_episode, int)
                or isinstance(second_episode, bool)
                or second_episode <= first_episode
            ):
                errors.append(f"{prefix} did not advance episode_id on reload")

        for snapshot_key in ("initial_snapshot", "post_reset_snapshot"):
            snapshot = result.get(snapshot_key)
            if not isinstance(snapshot, dict) or snapshot.get("reward_events") != []:
                errors.append(f"{prefix} has contaminated {snapshot_key} reward events")

    if len(set(result_ids)) != len(result_ids):
        errors.append("result Boss IDs are not unique")
    if selected and result_ids != selected:
        errors.append("result order/identity does not match selected_boss_ids")

    if errors:
        raise ModReleaseError("invalid live Mod evidence: " + "; ".join(errors))


def validate_walk_evidence(
    payload: dict[str, Any],
    *,
    metadata: ModReleaseMetadata,
    dll_sha256: str,
    dll_size_bytes: int,
) -> None:
    """Validate the binary-bound response-boundary movement regression."""

    errors: list[str] = []
    if payload.get("schema") != "hkrl.walk_smoothness.v1":
        errors.append("unexpected walk-smoothness schema")
    if payload.get("boss_mutation_allowed") is not False:
        errors.append("walk benchmark must disable Boss mutation")
    if payload.get("simulation_control_allowed") is not False:
        errors.append("walk benchmark must disable simulation mutation")
    if payload.get("smooth") is not True:
        errors.append("walk benchmark did not pass")

    tested_mod = payload.get("tested_mod")
    if not isinstance(tested_mod, dict):
        errors.append("walk benchmark tested_mod fingerprint is missing")
    else:
        if tested_mod.get("id") != MOD_ID:
            errors.append("walk benchmark tested_mod.id is invalid")
        if tested_mod.get("version") != metadata.mod_version:
            errors.append("walk benchmark Mod version does not match")
        if tested_mod.get("dll_sha256") != dll_sha256:
            errors.append("walk benchmark DLL hash does not match")
        if tested_mod.get("dll_size_bytes") != dll_size_bytes:
            errors.append("walk benchmark DLL size does not match")

    retention = payload.get("speed_retention")
    threshold = payload.get("min_speed_retention")
    retention_value = _positive_float(retention)
    threshold_value = _positive_float(threshold)
    if (
        retention_value is None
        or threshold_value is None
        or threshold_value < 0.9
        or threshold_value > 1.0
        or retention_value < threshold_value
    ):
        errors.append("walk speed retention is below the release threshold")

    continuous = payload.get("continuous")
    stepped = payload.get("stepped")
    if not isinstance(continuous, dict) or not isinstance(stepped, dict):
        errors.append("walk benchmark trial rows are missing")
    else:
        if continuous.get("commanded_ticks") != stepped.get("commanded_ticks"):
            errors.append("walk trials did not command the same tick count")
        if continuous.get("direction") != stepped.get("direction"):
            errors.append("walk trials used different directions")
        if continuous.get("decisions") != 1:
            errors.append("continuous walk reference is not one STEP")
        if (
            not isinstance(stepped.get("decisions"), int)
            or isinstance(stepped.get("decisions"), bool)
            or stepped["decisions"] <= 1
        ):
            errors.append("segmented walk trial has too few decisions")
        for label, trial in (("continuous", continuous), ("stepped", stepped)):
            fixed_delta = trial.get("fixed_delta_time_s")
            if (
                not isinstance(fixed_delta, (int, float))
                or isinstance(fixed_delta, bool)
                or abs(float(fixed_delta) - 0.02) > 1e-6
            ):
                errors.append(f"{label} walk trial changed fixed_delta_time")
            if trial.get("event_kinds") != []:
                errors.append(f"{label} walk trial contains reward events")
            start_hp = trial.get("start_hp")
            if (
                not isinstance(start_hp, int)
                or isinstance(start_hp, bool)
                or start_hp <= 0
                or trial.get("end_hp") != start_hp
            ):
                errors.append(f"{label} walk trial HP is contaminated")

    if errors:
        raise ModReleaseError("invalid walk-smoothness evidence: " + "; ".join(errors))


def build_mod_release(
    *,
    repo_root: str | Path,
    dll_path: str | Path,
    flatbuffers_dll_path: str | Path,
    evidence_json_path: str | Path,
    evidence_report_path: str | Path,
    walk_evidence_path: str | Path,
    output_dir: str | Path,
    allow_dirty: bool = False,
    require_tag: bool = True,
) -> ModPackageResult:
    """Create a deterministic, self-verifying Mod release ZIP."""

    root = Path(repo_root).resolve()
    metadata = load_mod_release_metadata(root)
    dll = _required_file(dll_path, "HKRLEnvMod DLL")
    flatbuffers_dll = _required_file(flatbuffers_dll_path, "FlatBuffers runtime DLL")
    evidence_path = _required_file(evidence_json_path, "live evidence JSON")
    report_path = _required_file(evidence_report_path, "live evidence report")
    walk_path = _required_file(walk_evidence_path, "walk-smoothness evidence")
    if dll.name != f"{MOD_ID}.dll":
        raise ModReleaseError(f"unexpected Mod DLL name: {dll.name}")
    if flatbuffers_dll.name != "Google.FlatBuffers.dll":
        raise ModReleaseError(f"unexpected FlatBuffers runtime DLL name: {flatbuffers_dll.name}")

    git_sha, git_dirty, tags = _git_state(root)
    if git_dirty and not allow_dirty:
        raise ModReleaseError("refusing to package a dirty worktree")
    if require_tag and metadata.release_tag not in tags:
        raise ModReleaseError(f"release commit is not tagged {metadata.release_tag}")

    dll_digest = sha256_file(dll)
    evidence = _load_json_object(evidence_path)
    validate_live_evidence(
        evidence,
        metadata=metadata,
        dll_sha256=dll_digest,
        dll_size_bytes=dll.stat().st_size,
    )
    walk_evidence = _load_json_object(walk_path)
    validate_walk_evidence(
        walk_evidence,
        metadata=metadata,
        dll_sha256=dll_digest,
        dll_size_bytes=dll.stat().st_size,
    )
    report_text = report_path.read_text(encoding="utf-8")
    if f"HKRLEnvMod v{metadata.mod_version}" not in report_text or dll_digest not in report_text:
        raise ModReleaseError(
            "live evidence report does not identify the tested Mod version and DLL hash"
        )

    source_files: list[tuple[str, str, bytes]] = [
        (
            f"{MOD_ID}/{MOD_ID}.dll",
            "mod_binary",
            dll.read_bytes(),
        ),
        (
            f"{MOD_ID}/Google.FlatBuffers.dll",
            "runtime_dependency",
            flatbuffers_dll.read_bytes(),
        ),
        (
            f"{MOD_ID}/README.md",
            "install_documentation",
            _required_file(root / "mod" / "README.md", "Mod README").read_bytes(),
        ),
        (
            f"{MOD_ID}/LICENSE",
            "license",
            _required_file(root / "LICENSE", "license").read_bytes(),
        ),
        (
            f"{MOD_ID}/evidence/godhome-all-boss-sweep-v{metadata.mod_version}.json",
            "live_acceptance_json",
            evidence_path.read_bytes(),
        ),
        (
            f"{MOD_ID}/evidence/godhome-all-boss-sweep-v{metadata.mod_version}.md",
            "live_acceptance_report",
            report_path.read_bytes(),
        ),
        (
            f"{MOD_ID}/evidence/walk-smoothness-post-fix.json",
            "walk_smoothness_json",
            walk_path.read_bytes(),
        ),
    ]
    source_files.sort(key=lambda item: item[0])
    file_rows = [
        {
            "path": archive_path,
            "role": role,
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for archive_path, role, content in source_files
    ]
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "release_kind": "hkrl_mod",
        "mod_id": MOD_ID,
        "mod_name": MOD_NAME,
        "mod_version": metadata.mod_version,
        "release_tag": metadata.release_tag,
        "protocol_schema_version": metadata.protocol_schema_version,
        "target_framework": metadata.target_framework,
        "flatbuffers_runtime_version": metadata.flatbuffers_runtime_version,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "live_acceptance": {
            "verified_bosses": 44,
            "selected_bosses": 44,
            "tested_dll_sha256": dll_digest,
            "boss_mutation_allowed": False,
            "simulation_control_allowed": False,
            "walk_speed_retention": walk_evidence["speed_retention"],
        },
        "files": file_rows,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    archive_entries = [
        *source_files,
        (f"{MOD_ID}/manifest.json", "manifest", manifest_bytes),
    ]
    archive_entries.sort(key=lambda item: item[0])

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"{metadata.archive_stem}.zip"
    with tempfile.NamedTemporaryFile(
        prefix=archive.name + ".",
        suffix=".tmp",
        dir=destination,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as bundle:
            for archive_path, _role, content in archive_entries:
                info = zipfile.ZipInfo(archive_path, date_time=ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                bundle.writestr(info, content, compresslevel=9)
        temporary_path.replace(archive)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    manifest_sidecar = destination / f"{metadata.archive_stem}.manifest.json"
    manifest_sidecar.write_bytes(manifest_bytes)
    archive_digest = sha256_file(archive)
    checksum = destination / f"{metadata.archive_stem}.zip.sha256"
    checksum.write_text(f"{archive_digest}  {archive.name}\n", encoding="utf-8")
    verify_mod_release(
        archive,
        repo_root=root,
        allow_dirty=allow_dirty,
        require_tag=require_tag,
    )
    return ModPackageResult(
        archive=archive,
        manifest=manifest_sidecar,
        checksum=checksum,
        archive_sha256=archive_digest,
    )


def verify_mod_release(
    archive_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    allow_dirty: bool = False,
    require_tag: bool = True,
) -> dict[str, Any]:
    """Verify archive shape, every hash, metadata, and live acceptance semantics."""

    archive = _required_file(archive_path, "Mod release archive")
    metadata = None if repo_root is None else load_mod_release_metadata(repo_root)
    try:
        with zipfile.ZipFile(archive, mode="r") as bundle:
            infos = bundle.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ModReleaseError("release archive contains duplicate entries")
            for entry_name in names:
                _validate_archive_path(entry_name)

            manifest_name = f"{MOD_ID}/manifest.json"
            if manifest_name not in names:
                raise ModReleaseError("release archive has no manifest.json")
            manifest = json.loads(bundle.read(manifest_name))
            if not isinstance(manifest, dict):
                raise ModReleaseError("release manifest is not an object")
            _validate_manifest_header(manifest, metadata=metadata, allow_dirty=allow_dirty)
            if repo_root is not None:
                git_sha, git_dirty, tags = _git_state(Path(repo_root).resolve())
                if manifest.get("git_sha") != git_sha:
                    raise ModReleaseError("release manifest Git SHA does not match the checkout")
                if manifest.get("git_dirty") != git_dirty:
                    raise ModReleaseError("release manifest git_dirty does not match the checkout")
                if require_tag and manifest.get("release_tag") not in tags:
                    raise ModReleaseError(f"checkout is not tagged {manifest.get('release_tag')}")

            file_rows = manifest.get("files")
            if not isinstance(file_rows, list) or not file_rows:
                raise ModReleaseError("release manifest files list is empty")
            declared_names: list[str] = []
            roles: dict[str, str] = {}
            for index, row in enumerate(file_rows):
                if not isinstance(row, dict):
                    raise ModReleaseError(f"manifest files[{index}] is not an object")
                declared_name = row.get("path")
                role = row.get("role")
                digest = row.get("sha256")
                size = row.get("size_bytes")
                if not isinstance(declared_name, str):
                    raise ModReleaseError(f"manifest files[{index}] has no path")
                _validate_archive_path(declared_name)
                if not isinstance(role, str) or not role:
                    raise ModReleaseError(f"manifest files[{index}] has no role")
                if role in roles:
                    raise ModReleaseError(f"manifest declares duplicate role {role}")
                if not isinstance(digest, str) or not FULL_SHA256.fullmatch(digest):
                    raise ModReleaseError(f"manifest files[{index}] has invalid sha256")
                if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                    raise ModReleaseError(f"manifest files[{index}] has invalid size")
                content = bundle.read(declared_name)
                if len(content) != size:
                    raise ModReleaseError(f"size mismatch for {declared_name}")
                if hashlib.sha256(content).hexdigest() != digest:
                    raise ModReleaseError(f"sha256 mismatch for {declared_name}")
                declared_names.append(declared_name)
                roles[role] = declared_name

            if len(declared_names) != len(set(declared_names)):
                raise ModReleaseError("manifest declares duplicate file paths")
            if set(names) != {manifest_name, *declared_names}:
                raise ModReleaseError("archive entries do not exactly match the manifest")
            required_roles = {
                "mod_binary",
                "runtime_dependency",
                "install_documentation",
                "license",
                "live_acceptance_json",
                "live_acceptance_report",
                "walk_smoothness_json",
            }
            if set(roles) != required_roles:
                raise ModReleaseError("release manifest has missing or duplicate file roles")

            dll_content = bundle.read(roles["mod_binary"])
            dll_digest = hashlib.sha256(dll_content).hexdigest()
            evidence = json.loads(bundle.read(roles["live_acceptance_json"]))
            if not isinstance(evidence, dict):
                raise ModReleaseError("live acceptance JSON is not an object")
            evidence_metadata = ModReleaseMetadata(
                mod_version=str(manifest["mod_version"]),
                protocol_schema_version=int(manifest["protocol_schema_version"]),
                target_framework=str(manifest["target_framework"]),
                flatbuffers_runtime_version=str(manifest["flatbuffers_runtime_version"]),
            )
            validate_live_evidence(
                evidence,
                metadata=evidence_metadata,
                dll_sha256=dll_digest,
                dll_size_bytes=len(dll_content),
            )
            walk_evidence = json.loads(bundle.read(roles["walk_smoothness_json"]))
            if not isinstance(walk_evidence, dict):
                raise ModReleaseError("walk-smoothness JSON is not an object")
            validate_walk_evidence(
                walk_evidence,
                metadata=evidence_metadata,
                dll_sha256=dll_digest,
                dll_size_bytes=len(dll_content),
            )
            acceptance = manifest.get("live_acceptance")
            if not isinstance(acceptance, dict) or acceptance != {
                "verified_bosses": 44,
                "selected_bosses": 44,
                "tested_dll_sha256": dll_digest,
                "boss_mutation_allowed": False,
                "simulation_control_allowed": False,
                "walk_speed_retention": walk_evidence["speed_retention"],
            }:
                raise ModReleaseError(
                    "manifest live acceptance summary does not match the packaged DLL"
                )
            report = bundle.read(roles["live_acceptance_report"]).decode("utf-8")
            if (
                f"HKRLEnvMod v{evidence_metadata.mod_version}" not in report
                or dll_digest not in report
            ):
                raise ModReleaseError("packaged acceptance report does not identify the tested DLL")
    except (KeyError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ModReleaseError(f"invalid Mod release archive: {exc}") from exc
    return manifest


def _validate_manifest_header(
    manifest: dict[str, Any],
    *,
    metadata: ModReleaseMetadata | None,
    allow_dirty: bool,
) -> None:
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ModReleaseError("unsupported Mod release manifest version")
    if manifest.get("release_kind") != "hkrl_mod":
        raise ModReleaseError("unexpected release kind")
    if manifest.get("mod_id") != MOD_ID or manifest.get("mod_name") != MOD_NAME:
        raise ModReleaseError("unexpected Mod identity")
    version = manifest.get("mod_version")
    if not isinstance(version, str) or not SEMVER.fullmatch(version):
        raise ModReleaseError("invalid Mod version")
    if manifest.get("release_tag") != f"v{version}":
        raise ModReleaseError("release tag does not match Mod version")
    schema = manifest.get("protocol_schema_version")
    if not isinstance(schema, int) or isinstance(schema, bool) or schema <= 0:
        raise ModReleaseError("invalid protocol schema version")
    git_sha = manifest.get("git_sha")
    if not isinstance(git_sha, str) or not FULL_GIT_SHA.fullmatch(git_sha):
        raise ModReleaseError("invalid release git SHA")
    dirty = manifest.get("git_dirty")
    if not isinstance(dirty, bool):
        raise ModReleaseError("invalid git_dirty flag")
    if dirty and not allow_dirty:
        raise ModReleaseError("release manifest was built from a dirty worktree")
    if metadata is not None:
        expected = {
            "mod_version": metadata.mod_version,
            "release_tag": metadata.release_tag,
            "protocol_schema_version": metadata.protocol_schema_version,
            "target_framework": metadata.target_framework,
            "flatbuffers_runtime_version": metadata.flatbuffers_runtime_version,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ModReleaseError(f"manifest {key} does not match the source")


def _git_state(root: Path) -> tuple[str, bool, set[str]]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        sha = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=normal"))
        tags = set(run("tag", "--points-at", "HEAD").splitlines())
    except subprocess.CalledProcessError as exc:
        raise ModReleaseError(f"cannot inspect Git release state: {exc}") from exc
    if not FULL_GIT_SHA.fullmatch(sha):
        raise ModReleaseError(f"Git HEAD is not a full commit SHA: {sha!r}")
    return sha, dirty, tags


def _valid_fingerprint(
    value: object,
    *,
    expected_name: str | None = None,
) -> bool:
    if not isinstance(value, dict):
        return False
    name = value.get("name")
    digest = value.get("sha256")
    return (
        isinstance(name, str)
        and bool(name)
        and (expected_name is None or name == expected_name)
        and _valid_positive_size(value.get("size_bytes"))
        and isinstance(digest, str)
        and FULL_SHA256.fullmatch(digest) is not None
    )


def _valid_positive_size(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_positive_number(value: object) -> bool:
    return _positive_float(value) is not None


def _positive_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def _required_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise ModReleaseError(f"{label} is not a file: {resolved}")
    return resolved


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModReleaseError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModReleaseError(f"JSON root is not an object: {path}")
    return payload


def _parse_xml(path: Path) -> ElementTree.Element:
    try:
        return ElementTree.parse(path).getroot()
    except (ElementTree.ParseError, OSError) as exc:
        raise ModReleaseError(f"cannot parse {path}: {exc}") from exc


def _required_xml_text(
    root: ElementTree.Element,
    local_name: str,
    path: Path,
) -> str:
    for element in root.iter():
        if _local_name(element.tag) == local_name and element.text:
            value = element.text.strip()
            if value:
                return value
    raise ModReleaseError(f"{path}: missing {local_name}")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _extract_int_constant(path: Path, pattern: str, name: str) -> int:
    match = re.search(pattern, path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    if match is None:
        raise ModReleaseError(f"{path}: cannot find {name}")
    return int(match.group(1))


def _validate_archive_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or ".." in path.parts
        or "." in path.parts
        or str(path) != value
        or not path.parts
        or path.parts[0] != MOD_ID
    ):
        raise ModReleaseError(f"unsafe or non-normalized archive path: {value!r}")
