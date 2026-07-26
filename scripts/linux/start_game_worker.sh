#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/linux/start_game_worker.sh [options]

Options:
  --config PATH                 Worker train config
  --task PATH                   Primary task config
  --tasks PATH...               Complete round-robin task list
  --worker-id ID                Stable worker id
  --game-root PATH              Hollow Knight installation root
  --launch-game                 Launch native Hollow Knight or Steam app 367520
  --env-port PORT               Mod environment port (default: 5555)
  --learner-port PORT           Local SSH learner forward (default: 5600)
  --registry-port PORT          Local SSH registry forward (default: 5601)
  --steps N                     Finite sample count; 0 means continuous
  --python-env NAME             Conda worker environment (default: hkrl)
  --inference-threads N         PyTorch CPU inference threads (default: 1)
  --game-startup-timeout N      Seconds to wait for the mod (default: 90)
  --skip-live-env-check         Skip PING preflight
  --dry-run                     Validate paths and print a secret-free plan
  -h, --help                    Show this help
EOF
}

REPO_ROOT="$(hkrl_repo_root)"
CONFIG="configs/train/linux_game_worker.yaml"
TASK="configs/tasks/gruz_mother.yaml"
TASKS=()
WORKER_ID="$(hostname -s 2>/dev/null || printf 'linux')-game-0"
GAME_ROOT=""
LAUNCH_GAME=0
ENV_PORT=5555
LEARNER_PORT=5600
REGISTRY_PORT=5601
STEPS=0
PYTHON_ENV="hkrl"
INFERENCE_THREADS=1
GAME_STARTUP_TIMEOUT=90
SKIP_LIVE_ENV_CHECK=0
DRY_RUN=0

while (($# > 0)); do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || hkrl_die "--config requires a value"
      CONFIG="$2"
      shift 2
      ;;
    --task)
      [[ $# -ge 2 ]] || hkrl_die "--task requires a value"
      TASK="$2"
      shift 2
      ;;
    --tasks)
      shift
      (($# > 0)) || hkrl_die "--tasks requires at least one path"
      while (($# > 0)) && [[ "$1" != --* ]]; do
        TASKS+=("$1")
        shift
      done
      ;;
    --worker-id)
      [[ $# -ge 2 ]] || hkrl_die "--worker-id requires a value"
      WORKER_ID="$2"
      shift 2
      ;;
    --game-root)
      [[ $# -ge 2 ]] || hkrl_die "--game-root requires a value"
      GAME_ROOT="$2"
      shift 2
      ;;
    --launch-game)
      LAUNCH_GAME=1
      shift
      ;;
    --env-port)
      ENV_PORT="$2"
      shift 2
      ;;
    --learner-port)
      LEARNER_PORT="$2"
      shift 2
      ;;
    --registry-port)
      REGISTRY_PORT="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --python-env)
      PYTHON_ENV="$2"
      shift 2
      ;;
    --inference-threads)
      INFERENCE_THREADS="$2"
      shift 2
      ;;
    --game-startup-timeout)
      GAME_STARTUP_TIMEOUT="$2"
      shift 2
      ;;
    --skip-live-env-check)
      SKIP_LIVE_ENV_CHECK=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
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

[[ -n "${WORKER_ID//[[:space:]]/}" ]] || hkrl_die "--worker-id must not be empty"
for item in \
  "${ENV_PORT}:env port" \
  "${LEARNER_PORT}:learner port" \
  "${REGISTRY_PORT}:registry port"; do
  hkrl_validate_port "${item%%:*}" "${item#*:}"
done
[[ "${STEPS}" =~ ^[0-9]+$ ]] || hkrl_die "--steps must be non-negative"
[[ "${INFERENCE_THREADS}" =~ ^[0-9]+$ ]] && ((INFERENCE_THREADS >= 1)) \
  || hkrl_die "--inference-threads must be positive"
[[ "${GAME_STARTUP_TIMEOUT}" =~ ^[0-9]+$ ]] && ((GAME_STARTUP_TIMEOUT >= 1)) \
  || hkrl_die "--game-startup-timeout must be positive"

CONFIG_PATH="$(hkrl_resolve_repo_file "${REPO_ROOT}" "${CONFIG}")"
TASK_PATH="$(hkrl_resolve_repo_file "${REPO_ROOT}" "${TASK}")"
TASK_PATHS=()
for task_item in "${TASKS[@]}"; do
  TASK_PATHS+=("$(hkrl_resolve_repo_file "${REPO_ROOT}" "${task_item}")")
done

RESOLVED_GAME_ROOT="$(hkrl_resolve_game_root "${GAME_ROOT}")"
MANAGED_DIR="${RESOLVED_GAME_ROOT}/hollow_knight_Data/Managed"
hkrl_validate_supported_steam_branch "${RESOLVED_GAME_ROOT}"
IFS='|' read -r GAME_RUNTIME GAME_EXECUTABLE \
  <<< "$(hkrl_detect_game_executable "${RESOLVED_GAME_ROOT}")"
hkrl_require_modding_api "${MANAGED_DIR}" "${GAME_RUNTIME}"
MOD_DIR="${MANAGED_DIR}/Mods/HKRLEnvMod"
[[ -f "${MOD_DIR}/HKRLEnvMod.dll" ]] || hkrl_die \
  "HKRLEnvMod is not installed under ${MOD_DIR}; run prepare_game_pc.sh"
RUNTIME_CONFIG="${MOD_DIR}/hkrl-runtime.conf"

if [[ -n "${HKRL_PYTHON_BIN:-}" ]]; then
  [[ -x "${HKRL_PYTHON_BIN}" ]] || hkrl_die \
    "HKRL_PYTHON_BIN is not executable: ${HKRL_PYTHON_BIN}"
  PYTHON_CMD=("${HKRL_PYTHON_BIN}")
  PYTHON_LABEL="${HKRL_PYTHON_BIN}"
else
  CONDA_BIN="$(hkrl_find_conda)"
  "${CONDA_BIN}" run --name "${PYTHON_ENV}" python --version >/dev/null 2>&1 \
    || hkrl_die "Conda environment ${PYTHON_ENV} is unavailable"
  PYTHON_CMD=("${CONDA_BIN}" run --no-capture-output --name "${PYTHON_ENV}" python)
  PYTHON_LABEL="${CONDA_BIN} run --name ${PYTHON_ENV} python"
fi

WORKER_BASE=(
  "${REPO_ROOT}/scripts/run_worker.py"
  --config "${CONFIG_PATH}"
  --task "${TASK_PATH}"
  --env-host 127.0.0.1
  --env-port "${ENV_PORT}"
  --learner "127.0.0.1:${LEARNER_PORT}"
  --registry "http://127.0.0.1:${REGISTRY_PORT}/"
  --worker-id "${WORKER_ID}"
  --batch-dir "${REPO_ROOT}/runs/linux-worker/batches"
  --heartbeat-jsonl "${REPO_ROOT}/runs/linux-worker/heartbeats.jsonl"
  --inference-threads "${INFERENCE_THREADS}"
)
if ((${#TASK_PATHS[@]} > 0)); then
  WORKER_BASE+=(--tasks "${TASK_PATHS[@]}")
fi

if ((DRY_RUN)); then
  python3 - \
    "${CONFIG_PATH}" \
    "${TASK_PATH}" \
    "${WORKER_ID}" \
    "${RESOLVED_GAME_ROOT}" \
    "${GAME_RUNTIME}" \
    "${GAME_EXECUTABLE}" \
    "${RUNTIME_CONFIG}" \
    "${PYTHON_LABEL}" \
    "${ENV_PORT}" \
    "${LEARNER_PORT}" \
    "${REGISTRY_PORT}" \
    "${INFERENCE_THREADS}" <<'PY'
import json
import sys

(
    config,
    task,
    worker_id,
    game_root,
    game_runtime,
    game_executable,
    runtime_config,
    python,
    env_port,
    learner_port,
    registry_port,
    inference_threads,
) = sys.argv[1:]
print(
    json.dumps(
        {
            "config": config,
            "env_endpoint": f"127.0.0.1:{env_port}",
            "game_executable": game_executable,
            "game_root": game_root,
            "game_runtime": game_runtime,
            "inference_threads": int(inference_threads),
            "learner_endpoint": f"127.0.0.1:{learner_port}",
            "python": python,
            "registry_endpoint": f"http://127.0.0.1:{registry_port}/",
            "runtime_config": runtime_config,
            "task": task,
            "worker_id": worker_id,
        },
        sort_keys=True,
    )
)
PY
  exit 0
fi

[[ -n "${HKRL_AUTH_TOKEN:-}" ]] || hkrl_die \
  "HKRL_AUTH_TOKEN is not set; source the same token used by the remote learner"
[[ "${HKRL_AUTH_TOKEN}" != *$'\n'* && "${HKRL_AUTH_TOKEN}" != *$'\r'* ]] \
  || hkrl_die "HKRL_AUTH_TOKEN must be a single line"
HKRL_SAVE_SLOT="${HKRL_SAVE_SLOT:-1}"
[[ "${HKRL_SAVE_SLOT}" =~ ^[1-4]$ ]] \
  || hkrl_die "HKRL_SAVE_SLOT must be an integer in [1, 4]"

mkdir -p "${MOD_DIR}" "${REPO_ROOT}/runs/linux-worker"
umask 077
runtime_temp="$(mktemp "${MOD_DIR}/.hkrl-runtime.conf.XXXXXX")"
cleanup_runtime_temp() {
  [[ ! -e "${runtime_temp}" ]] || rm -f "${runtime_temp}"
}
trap cleanup_runtime_temp EXIT
{
  printf 'HKRL_HOST=127.0.0.1\n'
  printf 'HKRL_PORT=%s\n' "${ENV_PORT}"
  printf 'HKRL_SAVE_SLOT=%s\n' "${HKRL_SAVE_SLOT}"
  printf 'HKRL_AUTH_TOKEN=%s\n' "${HKRL_AUTH_TOKEN}"
} > "${runtime_temp}"
chmod 0600 "${runtime_temp}"
mv -f "${runtime_temp}" "${RUNTIME_CONFIG}"
runtime_temp=""
trap - EXIT

export HKRL_HOST=127.0.0.1
export HKRL_PORT="${ENV_PORT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${INFERENCE_THREADS}"
export MKL_NUM_THREADS="${INFERENCE_THREADS}"

port_open() {
  "${PYTHON_CMD[@]}" -c \
    'import socket,sys; s=socket.socket(); s.settimeout(0.25); sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)' \
    "${ENV_PORT}" >/dev/null 2>&1
}

if ! port_open && ((LAUNCH_GAME)); then
  if [[ "${GAME_RUNTIME}" == "native" ]]; then
    [[ -x "${GAME_EXECUTABLE}" ]] || chmod u+x "${GAME_EXECUTABLE}"
    (
      cd "${RESOLVED_GAME_ROOT}"
      nohup "${GAME_EXECUTABLE}" \
        > "${REPO_ROOT}/runs/linux-worker/game.log" 2>&1 &
    )
  else
    command -v steam >/dev/null 2>&1 || hkrl_die \
      "steam was not found for Proton launch"
    nohup steam -applaunch 367520 \
      > "${REPO_ROOT}/runs/linux-worker/steam-launch.log" 2>&1 &
  fi

  deadline=$((SECONDS + GAME_STARTUP_TIMEOUT))
  until port_open; do
    ((SECONDS < deadline)) || hkrl_die \
      "HKRLEnvMod did not listen on 127.0.0.1:${ENV_PORT} before timeout"
    sleep 0.5
  done
fi

cd "${REPO_ROOT}"
DRY_OUTPUT="$("${PYTHON_CMD[@]}" "${WORKER_BASE[@]}" --dry-run)"
printf '%s\n' "${DRY_OUTPUT}"
LATEST_CHECKPOINT="$(printf '%s\n' "${DRY_OUTPUT}" | "${PYTHON_CMD[@]}" -c \
  'import json,sys; rows=[x for x in sys.stdin.read().splitlines() if x.strip()]; print(json.loads(rows[-1]).get("latest_checkpoint") or -1)')"
((LATEST_CHECKPOINT >= 1)) || hkrl_die \
  "remote registry has no startup checkpoint"

if ((SKIP_LIVE_ENV_CHECK == 0)); then
  "${PYTHON_CMD[@]}" "${REPO_ROOT}/scripts/check_env.py" \
    --config "${CONFIG_PATH}" \
    --task "${TASK_PATH}" \
    --host 127.0.0.1 \
    --port "${ENV_PORT}"
fi

WORKER_ARGS=("${WORKER_BASE[@]}")
if ((STEPS > 0)); then
  WORKER_ARGS+=(--steps "${STEPS}")
fi
exec "${PYTHON_CMD[@]}" "${WORKER_ARGS[@]}"
