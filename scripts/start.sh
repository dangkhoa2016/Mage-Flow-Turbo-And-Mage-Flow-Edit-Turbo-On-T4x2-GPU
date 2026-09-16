#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

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

./scripts/start_t2i.sh

./scripts/start_edit.sh

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
echo "[INFO] pid=$(cat .runtime/server.pid)"
echo '[PASS] REST_COORDINATOR_START_REQUESTED'
echo '[INFO] Overall /ready becomes true only after both GPU workers report ready.'
