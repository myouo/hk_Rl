#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/linux/start_training_stack.sh --remote USER@HOST [options]

Options:
  --ssh-port PORT          Remote SSH port (default: 22)
  --identity-file PATH     SSH private key
  --auth-file PATH         0600 shell file containing HKRL_AUTH_TOKEN
  --game-root PATH         Hollow Knight installation root
  --task PATH              Primary task
  --worker-id ID           Stable worker id
  --steps N                Finite sample count; 0 means continuous
  --env-port PORT          Local mod port (default: 5555)
  --learner-port PORT      Local learner forward (default: 5600)
  --registry-port PORT     Local registry forward (default: 5601)
  --no-launch-game         Require an already-running game
  --dry-run                Print tunnel and worker plans without mutations
  -h, --help               Show this help
EOF
}

REMOTE=""
SSH_PORT=22
IDENTITY_FILE=""
AUTH_FILE="${HOME}/.config/hkrl/worker.env"
GAME_ROOT=""
TASK="configs/tasks/gruz_mother.yaml"
WORKER_ID="$(hostname -s 2>/dev/null || printf 'linux')-game-0"
STEPS=0
ENV_PORT=5555
LEARNER_PORT=5600
REGISTRY_PORT=5601
LAUNCH_GAME=1
DRY_RUN=0

while (($# > 0)); do
  case "$1" in
    --remote)
      REMOTE="$2"
      shift 2
      ;;
    --ssh-port)
      SSH_PORT="$2"
      shift 2
      ;;
    --identity-file)
      IDENTITY_FILE="$2"
      shift 2
      ;;
    --auth-file)
      AUTH_FILE="$2"
      shift 2
      ;;
    --game-root)
      GAME_ROOT="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --worker-id)
      WORKER_ID="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
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
    --no-launch-game)
      LAUNCH_GAME=0
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

[[ -n "${REMOTE}" ]] || hkrl_die "--remote is required"
TUNNEL_ARGS=(
  --remote "${REMOTE}"
  --ssh-port "${SSH_PORT}"
  --local-learner-port "${LEARNER_PORT}"
  --local-registry-port "${REGISTRY_PORT}"
)
[[ -z "${IDENTITY_FILE}" ]] || TUNNEL_ARGS+=(--identity-file "${IDENTITY_FILE}")
WORKER_ARGS=(
  --game-root "${GAME_ROOT}"
  --task "${TASK}"
  --worker-id "${WORKER_ID}"
  --steps "${STEPS}"
  --env-port "${ENV_PORT}"
  --learner-port "${LEARNER_PORT}"
  --registry-port "${REGISTRY_PORT}"
)
((LAUNCH_GAME == 0)) || WORKER_ARGS+=(--launch-game)

if ((DRY_RUN)); then
  "${SCRIPT_DIR}/start_ssh_tunnel.sh" "${TUNNEL_ARGS[@]}" --dry-run
  "${SCRIPT_DIR}/start_game_worker.sh" "${WORKER_ARGS[@]}" --dry-run
  exit 0
fi

if [[ -z "${HKRL_AUTH_TOKEN:-}" ]]; then
  [[ -f "${AUTH_FILE}" ]] || hkrl_die \
    "auth file does not exist and HKRL_AUTH_TOKEN is unset: ${AUTH_FILE}"
  mode="$(stat -c '%a' "${AUTH_FILE}")"
  ((8#${mode} & 077 == 0)) || hkrl_die \
    "auth file must not be accessible by group/others: ${AUTH_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${AUTH_FILE}"
  set +a
fi
[[ -n "${HKRL_AUTH_TOKEN:-}" ]] || hkrl_die \
  "HKRL_AUTH_TOKEN is empty after loading ${AUTH_FILE}"

tunnel_pid=""
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "${tunnel_pid}" ]]; then
    kill "${tunnel_pid}" 2>/dev/null || true
    wait "${tunnel_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"${SCRIPT_DIR}/start_ssh_tunnel.sh" "${TUNNEL_ARGS[@]}" &
tunnel_pid="$!"

deadline=$((SECONDS + 20))
until python3 - "${REGISTRY_PORT}" <<'PY' >/dev/null 2>&1
import sys
import urllib.error
import urllib.request

try:
    urllib.request.urlopen(
        f"http://127.0.0.1:{int(sys.argv[1])}/index.jsonl",
        timeout=0.5,
    )
except urllib.error.HTTPError as exc:
    raise SystemExit(0 if exc.code == 401 else 1)
except Exception:
    raise SystemExit(1)
PY
do
  kill -0 "${tunnel_pid}" 2>/dev/null || hkrl_die \
    "SSH tunnel exited before registry became ready"
  ((SECONDS < deadline)) || hkrl_die \
    "SSH registry forward did not become ready before timeout"
  sleep 0.5
done

"${SCRIPT_DIR}/start_game_worker.sh" "${WORKER_ARGS[@]}"
