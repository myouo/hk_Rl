#!/usr/bin/env bash

# Shared Linux game-host helpers. Source this file; do not execute it.

hkrl_die() {
  printf 'error: %s\n' "$*" >&2
  return 1
}

hkrl_repo_root() {
  local source_dir
  source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  cd "${source_dir}/../.." && pwd -P
}

hkrl_validate_port() {
  local value="$1"
  local name="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || hkrl_die "${name} must be an integer"
  ((value >= 1 && value <= 65535)) \
    || hkrl_die "${name} must be in [1, 65535]"
}

hkrl_resolve_repo_file() {
  local repo_root="$1"
  local requested="$2"
  local candidate
  if [[ "${requested}" = /* ]]; then
    candidate="${requested}"
  else
    candidate="${repo_root}/${requested}"
  fi
  [[ -f "${candidate}" ]] || hkrl_die "required file does not exist: ${candidate}"
  (
    cd "$(dirname "${candidate}")"
    printf '%s/%s\n' "$(pwd -P)" "$(basename "${candidate}")"
  )
}

hkrl_resolve_game_root() {
  local requested="${1:-}"
  local -a candidates=()
  local steam_root library_file library_path candidate

  [[ -z "${requested}" ]] || candidates+=("${requested}")
  [[ -z "${HKRL_GAME_ROOT:-}" ]] || candidates+=("${HKRL_GAME_ROOT}")
  candidates+=(
    "${HOME}/.local/share/Steam/steamapps/common/Hollow Knight"
    "${HOME}/.steam/steam/steamapps/common/Hollow Knight"
    "${HOME}/.steam/root/steamapps/common/Hollow Knight"
  )

  for steam_root in \
    "${HOME}/.local/share/Steam" \
    "${HOME}/.steam/steam" \
    "${HOME}/.steam/root"; do
    library_file="${steam_root}/steamapps/libraryfolders.vdf"
    [[ -f "${library_file}" ]] || continue
    while IFS= read -r library_path; do
      [[ -n "${library_path}" ]] || continue
      library_path="${library_path//\\\\/\\}"
      candidates+=("${library_path}/steamapps/common/Hollow Knight")
    done < <(awk -F'"' '$2 ~ /^[[:space:]]*path[[:space:]]*$/ { print $4 }' \
      "${library_file}")
  done

  for candidate in "${candidates[@]}"; do
    [[ -f "${candidate}/hollow_knight_Data/Managed/Assembly-CSharp.dll" ]] \
      || continue
    (
      cd "${candidate}"
      pwd -P
    )
    return 0
  done

  hkrl_die \
    "Hollow Knight is not fully installed; pass --game-root or finish Steam app 367520"
}

hkrl_detect_game_executable() {
  local game_root="$1"
  local candidate
  for candidate in hollow_knight.x86_64 hollow_knight; do
    if [[ -f "${game_root}/${candidate}" ]]; then
      printf 'native|%s/%s\n' "${game_root}" "${candidate}"
      return 0
    fi
  done
  if [[ -f "${game_root}/hollow_knight.exe" ]]; then
    printf 'proton|%s/hollow_knight.exe\n' "${game_root}"
    return 0
  fi
  hkrl_die "no native or Proton Hollow Knight executable found under ${game_root}"
}

hkrl_validate_supported_steam_branch() {
  local game_root="$1"
  local steamapps_dir manifest state branch
  steamapps_dir="$(cd "${game_root}/../.." 2>/dev/null && pwd -P)" || return 0
  manifest="${steamapps_dir}/appmanifest_367520.acf"
  [[ -f "${manifest}" ]] || return 0

  state="$(awk -F'"' '$2 == "StateFlags" { print $4; exit }' "${manifest}")"
  branch="$(awk -F'"' 'tolower($2) == "betakey" { print $4; exit }' "${manifest}")"
  [[ "${branch}" == "1.5.78.11833" ]] || hkrl_die \
    "Steam branch ${branch:-public} is unsupported by Modding API; select 1.5.78.11833"
  [[ "${state}" == "4" ]] || hkrl_die \
    "Steam is still applying Hollow Knight 1.5.78.11833; wait for the download to finish"
}

hkrl_require_modding_api() {
  local managed_dir="$1"
  local game_runtime="${2:-}"
  local assembly
  local -a required=(
    Assembly-CSharp.dll
    UnityEngine.dll
    UnityEngine.CoreModule.dll
    UnityEngine.IMGUIModule.dll
    UnityEngine.Physics2DModule.dll
    MMHOOK_Assembly-CSharp.dll
    PlayMaker.dll
  )
  for assembly in "${required[@]}"; do
    [[ -f "${managed_dir}/${assembly}" ]] || hkrl_die \
      "missing ${managed_dir}/${assembly}; install/enable HK Modding API with Lumafly"
  done

  case "${game_runtime}" in
    proton)
      [[ -f "${managed_dir}/unityscenerepacker.dll" ]] || hkrl_die \
        "Proton requires the Windows Modding API package (missing unityscenerepacker.dll)"
      ;;
    native)
      [[ -f "${managed_dir}/libunityscenerepacker.so" ]] || hkrl_die \
        "native Linux requires the Linux Modding API package (missing libunityscenerepacker.so)"
      ;;
    "")
      ;;
    *)
      hkrl_die "unsupported game runtime for Modding API validation: ${game_runtime}"
      ;;
  esac
}

hkrl_find_conda() {
  if [[ -n "${HKRL_CONDA_BIN:-}" ]]; then
    [[ -x "${HKRL_CONDA_BIN}" ]] || hkrl_die \
      "HKRL_CONDA_BIN is not executable: ${HKRL_CONDA_BIN}"
    printf '%s\n' "${HKRL_CONDA_BIN}"
    return 0
  fi
  command -v conda 2>/dev/null || hkrl_die \
    "conda was not found; set HKRL_CONDA_BIN or install Miniconda/Miniforge"
}
