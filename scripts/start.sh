#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p .runtime

if [[ -z "${MAGE_FLOW_API_TOKEN:-}" ]]; then
  echo '[FAIL] MAGE_FLOW_API_TOKEN is not set'
  echo 'Suggested action / Hướng xử lý: export a strong temporary token before starting the coordinator.'
  exit 1
fi

./scripts/start_t2i.sh

./scripts/start_edit.sh

if [[ -f .runtime/server.pid ]] && kill -0 "$(cat .runtime/server.pid)" 2>/dev/null; then
  echo '[PASS] REST_COORDINATOR_ALREADY_RUNNING'
  exit 0
fi

echo '[INFO] Starting authenticated REST coordinator on 127.0.0.1:8090'
MAGE_FLOW_T2I_INTERNAL_URL="${MAGE_FLOW_T2I_INTERNAL_URL:-http://127.0.0.1:8101}" \
MAGE_FLOW_EDIT_INTERNAL_URL="${MAGE_FLOW_EDIT_INTERNAL_URL:-http://127.0.0.1:8102}" \
nohup python -m uvicorn server.app:app --host 127.0.0.1 --port 8090 > .runtime/server.log 2>&1 &
echo $! > .runtime/server.pid
echo "[INFO] pid=$(cat .runtime/server.pid)"
echo '[PASS] REST_COORDINATOR_START_REQUESTED'
echo '[INFO] Overall /ready becomes true only after both GPU workers report ready.'