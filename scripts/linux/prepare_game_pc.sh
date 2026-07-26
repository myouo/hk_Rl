#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/linux/prepare_game_pc.sh [options]

Options:
  --game-root PATH                 Explicit Hollow Knight installation root
  --python-env NAME                Conda worker environment (default: hkrl)
  --mod-build-env NAME             Conda flatc environment (default: hkrl-mod-build)
  --install-python-environment     Create/update the local worker environment
  --install-mod-build-environment  Create/update the flatc environment
  --build-and-install-mod          Generate schema, build, back up, and install mod DLLs
  -h, --help                       Show this help

With no install flags, this performs a non-mutating readiness check.
EOF
}

GAME_ROOT=""
PYTHON_ENV="hkrl"
MOD_BUILD_ENV="hkrl-mod-build"
INSTALL_PYTHON=0
INSTALL_MOD_BUILD=0
BUILD_AND_INSTALL=0

while (($# > 0)); do
  case "$1" in
    --game-root)
      [[ $# -ge 2 ]] || hkrl_die "--game-root requires a value"
      GAME_ROOT="$2"
      shift 2
      ;;
    --python-env)
      [[ $# -ge 2 ]] || hkrl_die "--python-env requires a value"
      PYTHON_ENV="$2"
      shift 2
      ;;
    --mod-build-env)
      [[ $# -ge 2 ]] || hkrl_die "--mod-build-env requires a value"
      MOD_BUILD_ENV="$2"
      shift 2
      ;;
    --install-python-environment)
      INSTALL_PYTHON=1
      shift
      ;;
    --install-mod-build-environment)
      INSTALL_MOD_BUILD=1
      shift
      ;;
    --build-and-install-mod)
      BUILD_AND_INSTALL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      hkrl_die "unknown option: $1"
      ;;
  esac
done

[[ -n "${PYTHON_ENV}" ]] || hkrl_die "--python-env must not be empty"
[[ -n "${MOD_BUILD_ENV}" ]] || hkrl_die "--mod-build-env must not be empty"

REPO_ROOT="$(hkrl_repo_root)"
RESOLVED_GAME_ROOT="$(hkrl_resolve_game_root "${GAME_ROOT}")"
MANAGED_DIR="${RESOLVED_GAME_ROOT}/hollow_knight_Data/Managed"
IFS='|' read -r GAME_RUNTIME GAME_EXECUTABLE \
  <<< "$(hkrl_detect_game_executable "${RESOLVED_GAME_ROOT}")"
hkrl_validate_supported_steam_branch "${RESOLVED_GAME_ROOT}"
hkrl_require_modding_api "${MANAGED_DIR}" "${GAME_RUNTIME}"

CONDA_BIN="$(hkrl_find_conda)"
command -v dotnet >/dev/null 2>&1 || hkrl_die "dotnet was not found"

if ((INSTALL_PYTHON)); then
  "${CONDA_BIN}" env update \
    --name "${PYTHON_ENV}" \
    --file "${REPO_ROOT}/environment.yml" \
    --prune
fi

if ((INSTALL_MOD_BUILD)); then
  "${CONDA_BIN}" env update \
    --name "${MOD_BUILD_ENV}" \
    --file "${REPO_ROOT}/environment-mod-build.yml" \
    --prune
fi

MOD_INSTALLED_TO=""
if ((BUILD_AND_INSTALL)); then
  # pgrep uses POSIX ERE; non-capturing groups (?:...) are invalid and make
  # pgrep return "no match", which would incorrectly permit a live DLL swap.
  if pgrep -f '[h]ollow_knight(\.exe|\.x86_64)?([[:space:]]|$)' >/dev/null 2>&1; then
    hkrl_die "Hollow Knight is running; close it before replacing mod DLLs"
  fi

  "${CONDA_BIN}" run --name "${MOD_BUILD_ENV}" flatc --version \
    | grep -F "23.5.26" >/dev/null \
    || hkrl_die \
      "C# schema generation requires flatc 23.5.26; use --install-mod-build-environment"

  SCHEMA_OUTPUT="${REPO_ROOT}/mod/HKRLEnvMod/Schema"
  mkdir -p "${SCHEMA_OUTPUT}"
  "${CONDA_BIN}" run --name "${MOD_BUILD_ENV}" flatc \
    --csharp \
    -o "${SCHEMA_OUTPUT}" \
    "${REPO_ROOT}/schema/hkrl.fbs"

  dotnet build "${REPO_ROOT}/mod/HKRLEnvMod/HKRLEnvMod.csproj" \
    -c Release \
    -p:HollowKnightManaged="${MANAGED_DIR}" \
    -p:TreatWarningsAsErrors=true

  MOD_INSTALLED_TO="${MANAGED_DIR}/Mods/HKRLEnvMod"
  mkdir -p "${MOD_INSTALLED_TO}"
  BUILD_OUTPUT="${REPO_ROOT}/mod/HKRLEnvMod/bin/Release"
  TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  for filename in HKRLEnvMod.dll Google.FlatBuffers.dll; do
    source_path="${BUILD_OUTPUT}/${filename}"
    destination="${MOD_INSTALLED_TO}/${filename}"
    [[ -f "${source_path}" ]] || hkrl_die \
      "expected build output was not found: ${source_path}"
    if [[ -f "${destination}" ]]; then
      cp -p "${destination}" "${destination}.bak-${TIMESTAMP}"
    fi
    install -m 0644 "${source_path}" "${destination}"
  done
fi

PYTHON_READY=0
PYTHON_VERSION=""
if PYTHON_VERSION="$("${CONDA_BIN}" run --name "${PYTHON_ENV}" \
  python -c 'import platform; print(platform.python_version())' 2>/dev/null)"; then
  PYTHON_READY=1
  PYTHON_VERSION="$(printf '%s' "${PYTHON_VERSION}" | tail -n 1)"
fi

python3 - \
  "${REPO_ROOT}" \
  "${RESOLVED_GAME_ROOT}" \
  "${MANAGED_DIR}" \
  "${GAME_EXECUTABLE}" \
  "${GAME_RUNTIME}" \
  "${CONDA_BIN}" \
  "${PYTHON_READY}" \
  "${PYTHON_VERSION}" \
  "${MOD_INSTALLED_TO}" <<'PY'
import json
import sys

(
    repo_root,
    game_root,
    managed_dir,
    game_executable,
    game_runtime,
    conda,
    python_ready,
    python_version,
    mod_installed_to,
) = sys.argv[1:]
print(
    json.dumps(
        {
            "conda": conda,
            "game_executable": game_executable,
            "game_runtime": game_runtime,
            "hollow_knight_root": game_root,
            "managed_dir": managed_dir,
            "mod_installed_to": mod_installed_to or None,
            "modding_api_detected": True,
            "python_env_ready": python_ready == "1",
            "python_version": python_version or None,
            "repo_root": repo_root,
        },
        sort_keys=True,
    )
)
PY
