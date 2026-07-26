"""Linux game-host launcher tests."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[2]
LINUX_SCRIPTS = ROOT / "scripts" / "linux"


def test_prepare_game_pc_inspects_proton_install(tmp_path: Path) -> None:
    game_root = _fake_game(tmp_path, runtime="proton")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "conda",
        '#!/usr/bin/env bash\nif [[ "$*" == *"python -c"* ]]; then printf \'3.10.20\\n\'; fi\n',
    )
    _write_executable(fake_bin / "dotnet", "#!/usr/bin/env bash\nexit 0\n")

    result = _run(
        LINUX_SCRIPTS / "prepare_game_pc.sh",
        "--game-root",
        str(game_root),
        env={
            "HKRL_CONDA_BIN": str(fake_bin / "conda"),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
    )

    summary = json.loads(result.stdout)
    assert summary["game_runtime"] == "proton"
    assert summary["game_executable"].endswith("hollow_knight.exe")
    assert summary["modding_api_detected"] is True
    assert summary["python_env_ready"] is True
    assert summary["python_version"] == "3.10.20"


def test_prepare_game_pc_rejects_missing_modding_api(tmp_path: Path) -> None:
    game_root = _fake_game(tmp_path, runtime="native")
    (game_root / "hollow_knight_Data" / "Managed" / "MMHOOK_Assembly-CSharp.dll").unlink()

    result = _run(
        LINUX_SCRIPTS / "prepare_game_pc.sh",
        "--game-root",
        str(game_root),
        check=False,
    )

    assert result.returncode != 0
    assert "Modding API" in result.stderr


def test_build_mod_does_not_implicitly_update_conda_env(tmp_path: Path) -> None:
    game_root = _fake_game(tmp_path, runtime="native")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    conda_log = tmp_path / "conda.log"
    _write_executable(
        fake_bin / "conda",
        """#!/usr/bin/env bash
printf '%s\n' "$*" >> "${HKRL_TEST_CONDA_LOG}"
if [[ "$*" == *"flatc --version"* ]]; then printf 'flatc version 23.5.26\n'; fi
""",
    )
    _write_executable(fake_bin / "dotnet", "#!/usr/bin/env bash\nexit 99\n")
    _write_executable(fake_bin / "pgrep", "#!/usr/bin/env bash\nexit 1\n")

    result = _run(
        LINUX_SCRIPTS / "prepare_game_pc.sh",
        "--game-root",
        str(game_root),
        "--build-and-install-mod",
        env={
            "HKRL_CONDA_BIN": str(fake_bin / "conda"),
            "HKRL_TEST_CONDA_LOG": str(conda_log),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
        check=False,
    )

    assert result.returncode != 0
    assert "env update" not in conda_log.read_text(encoding="utf-8")


def test_prepare_game_pc_uses_a_posix_compatible_live_process_guard() -> None:
    source = (LINUX_SCRIPTS / "prepare_game_pc.sh").read_text(encoding="utf-8")
    guard = next(line for line in source.splitlines() if "if pgrep -f" in line)

    assert "[h]ollow_knight" in guard
    assert "(?:" not in guard


def test_linux_ssh_tunnel_dry_run_is_loopback_only() -> None:
    result = _run(
        LINUX_SCRIPTS / "start_ssh_tunnel.sh",
        "--remote",
        "gpu@example",
        "--ssh-port",
        "30262",
        "--dry-run",
    )

    summary = json.loads(result.stdout)
    assert summary["remote"] == "gpu@example"
    assert summary["learner_forward"] == "127.0.0.1:5600 -> 127.0.0.1:5600"
    assert summary["registry_forward"] == "127.0.0.1:5601 -> 127.0.0.1:5601"
    assert "real-time" in summary["note"]


def test_linux_game_worker_dry_run_resolves_native_install(tmp_path: Path) -> None:
    game_root = _fake_game(tmp_path, runtime="native")
    mod_dir = game_root / "hollow_knight_Data" / "Managed" / "Mods" / "HKRLEnvMod"
    mod_dir.mkdir(parents=True)
    (mod_dir / "HKRLEnvMod.dll").write_bytes(b"mod")

    result = _run(
        LINUX_SCRIPTS / "start_game_worker.sh",
        "--game-root",
        str(game_root),
        "--worker-id",
        "linux-test-0",
        "--save-slot",
        "3",
        "--time-scale",
        "3",
        "--dry-run",
        env={"HKRL_PYTHON_BIN": sys.executable},
    )

    summary = json.loads(result.stdout)
    assert summary["game_runtime"] == "native"
    assert summary["worker_id"] == "linux-test-0"
    assert summary["env_endpoint"] == "127.0.0.1:5555"
    assert summary["inference_threads"] == 1
    assert summary["save_slot"] == 3
    assert summary["time_scale"] == 3.0
    assert summary["runtime_config"].endswith("HKRLEnvMod/hkrl-runtime.conf")


def test_linux_game_worker_preserves_runtime_save_slot(tmp_path: Path) -> None:
    game_root = _fake_game(tmp_path, runtime="native")
    mod_dir = game_root / "hollow_knight_Data" / "Managed" / "Mods" / "HKRLEnvMod"
    mod_dir.mkdir(parents=True)
    (mod_dir / "HKRLEnvMod.dll").write_bytes(b"mod")
    (mod_dir / "hkrl-runtime.conf").write_text(
        "HKRL_HOST=127.0.0.1\n"
        "HKRL_PORT=5555\n"
        "HKRL_SAVE_SLOT=4\n"
        "HKRL_AUTH_TOKEN=not-read-by-dry-run\n",
        encoding="utf-8",
    )

    result = _run(
        LINUX_SCRIPTS / "start_game_worker.sh",
        "--game-root",
        str(game_root),
        "--dry-run",
        env={"HKRL_PYTHON_BIN": sys.executable},
    )

    summary = json.loads(result.stdout)
    assert summary["save_slot"] == 4


def test_linux_game_worker_rejects_implicit_save_slot_one(tmp_path: Path) -> None:
    game_root = _fake_game(tmp_path, runtime="native")
    mod_dir = game_root / "hollow_knight_Data" / "Managed" / "Mods" / "HKRLEnvMod"
    mod_dir.mkdir(parents=True)
    (mod_dir / "HKRLEnvMod.dll").write_bytes(b"mod")

    result = _run(
        LINUX_SCRIPTS / "start_game_worker.sh",
        "--game-root",
        str(game_root),
        "--dry-run",
        env={"HKRL_PYTHON_BIN": sys.executable},
        check=False,
    )

    assert result.returncode != 0
    assert "--save-slot is required" in result.stderr


def test_select_steam_beta_updates_both_sections_atomically(tmp_path: Path) -> None:
    module = _load_script(LINUX_SCRIPTS / "select_steam_beta.py")
    manifest = tmp_path / "appmanifest_367520.acf"
    original = (
        '"AppState"\n'
        "{\n"
        '\t"UserConfig"\n'
        "\t{\n"
        '\t\t"language"\t\t"english"\n'
        "\t}\n"
        '\t"MountedConfig"\n'
        "\t{\n"
        "\t}\n"
        "}\n"
    )
    manifest.write_text(original, encoding="utf-8")

    updated = module._set_beta_key(original, "1.5.78.11833")
    manifest.write_text(updated, encoding="utf-8")

    assert updated.count('"betakey"\t\t"1.5.78.11833"') == 2
    assert module._set_beta_key(updated, "1.5.78.11833") == updated
    assert '"language"\t\t"english"' in updated


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--remote", "-bad", "--dry-run"], "--remote"),
        (["--remote", "host", "--local-learner-port", "0", "--dry-run"], "port"),
        (
            [
                "--remote",
                "host",
                "--local-learner-port",
                "5600",
                "--local-registry-port",
                "5600",
                "--dry-run",
            ],
            "different",
        ),
    ],
)
def test_linux_ssh_tunnel_rejects_invalid_args(
    args: list[str],
    message: str,
) -> None:
    result = _run(
        LINUX_SCRIPTS / "start_ssh_tunnel.sh",
        *args,
        check=False,
    )

    assert result.returncode != 0
    assert message in result.stderr


def _fake_game(tmp_path: Path, *, runtime: str) -> Path:
    game_root = tmp_path / "Steam Library" / "steamapps" / "common" / "Hollow Knight"
    managed = game_root / "hollow_knight_Data" / "Managed"
    managed.mkdir(parents=True)
    for name in (
        "Assembly-CSharp.dll",
        "UnityEngine.dll",
        "UnityEngine.CoreModule.dll",
        "UnityEngine.IMGUIModule.dll",
        "UnityEngine.Physics2DModule.dll",
        "MMHOOK_Assembly-CSharp.dll",
        "PlayMaker.dll",
    ):
        (managed / name).write_bytes(b"assembly")

    executable = (
        game_root / "hollow_knight.exe"
        if runtime == "proton"
        else game_root / "hollow_knight.x86_64"
    )
    api_native_library = (
        managed / "unityscenerepacker.dll"
        if runtime == "proton"
        else managed / "libunityscenerepacker.so"
    )
    api_native_library.write_bytes(b"native-api-library")
    _write_executable(executable, "#!/usr/bin/env bash\nexit 0\n")
    return game_root


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run(
    script: Path,
    *args: str,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    inherited_names = ("HOME", "LANG", "LC_ALL", "PATH", "TMPDIR")
    merged_env = {name: os.environ[name] for name in inherited_names if name in os.environ}
    if env:
        merged_env.update(env)
    return subprocess.run(
        [str(script), *args],
        cwd=ROOT,
        env=merged_env,
        text=True,
        capture_output=True,
        check=check,
    )


def _load_script(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
