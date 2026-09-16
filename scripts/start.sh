#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# shellcheck source-path=scripts
# shellcheck source=lifecycle_common.sh
# shellcheck disable=SC1091  # lifecycle_common.sh lives in scripts/ and is followed via shellcheck -x in CI
source ./scripts/lifecycle_common.sh

# shellcheck disable=SC2317  # trap EXIT target; reached indirectly on every exit
cleanup_tx() {
  local rc=$?
  if (( rc != 0 )); then
    echo "[INFO] lifecycle start failed rc=$rc; stopping only processes started by this invocation"
    cleanup_new_services
    echo "[INFO] lifecycle failure cleanup complete; original rc=$rc preserved"
    exit "$rc"
  fi
  echo "[INFO] lifecycle start committed; pre-existing and newly started services remain resident"
  exit 0
}
trap cleanup_tx EXIT

REST_PORT="${MAGE_FLOW_REST_PORT:-8090}"
T2I_PORT="${MAGE_FLOW_T2I_INTERNAL_PORT:-8101}"
EDIT_PORT="${MAGE_FLOW_EDIT_INTERNAL_PORT:-8102}"
T2I_URL="http://127.0.0.1:${T2I_PORT}"
EDIT_URL="http://127.0.0.1:${EDIT_PORT}"
export MAGE_FLOW_T2I_INTERNAL_URL="$T2I_URL"
export MAGE_FLOW_EDIT_INTERNAL_URL="$EDIT_URL"

python scripts/token_store.py prepare-dir .runtime

python scripts/runtime_config.py validate-ports \
  --rest "$REST_PORT" --t2i "$T2I_PORT" --edit "$EDIT_PORT"

if [[ -z "${MAGE_FLOW_API_TOKEN:-}" ]]; then
  echo '[FAIL] MAGE_FLOW_API_TOKEN is not set'
  echo 'Suggested action / Hướng xử lý: export a strong temporary token before starting the coordinator.'
  exit 1
fi

python scripts/runtime_config.py validate-token --value="$MAGE_FLOW_API_TOKEN"

T2I_BEFORE_PID="$(cat .runtime/t2i.pid 2>/dev/null || true)"
bash scripts/start_t2i.sh
T2I_AFTER_PID="$(cat .runtime/t2i.pid 2>/dev/null || true)"
if [[ -n "$T2I_AFTER_PID" && "$T2I_BEFORE_PID" != "$T2I_AFTER_PID" ]]; then
  record_new_service .runtime/t2i.pid scripts/stop_t2i.sh
fi

EDIT_BEFORE_PID="$(cat .runtime/edit.pid 2>/dev/null || true)"
bash scripts/start_edit.sh
EDIT_AFTER_PID="$(cat .runtime/edit.pid 2>/dev/null || true)"
if [[ -n "$EDIT_AFTER_PID" && "$EDIT_BEFORE_PID" != "$EDIT_AFTER_PID" ]]; then
  record_new_service .runtime/edit.pid scripts/stop_edit.sh
fi

if [[ -f .runtime/server.pid ]]; then
  if python scripts/process_identity.py check \
       --pid-file .runtime/server.pid \
       --kind coordinator \
       --project-root "$PROJECT_ROOT" \
       --port "$REST_PORT" \
       --token-env MAGE_FLOW_API_TOKEN; then
    echo '[PASS] REST_COORDINATOR_ALREADY_RUNNING'
    exit 0
  fi
  echo "[WARN] coordinator pid=$(cat .runtime/server.pid) failed identity/current-token check; restarting it"
  bash scripts/stop.sh || true
  rm -f .runtime/server.pid
fi

echo "[INFO] Starting authenticated REST coordinator on 127.0.0.1:${REST_PORT}"
nohup python -m uvicorn server.app:app \
  --host 127.0.0.1 \
  --port "$REST_PORT" \
  > .runtime/server.log 2>&1 &
echo $! > .runtime/server.pid
record_new_service .runtime/server.pid scripts/stop.sh
echo "[INFO] pid=$(cat .runtime/server.pid)"
echo '[PASS] REST_COORDINATOR_START_REQUESTED'
echo '[INFO] Overall /ready becomes true only after both GPU workers report ready.'
exit 0
