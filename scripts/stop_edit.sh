#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ -f .runtime/edit.pid ]]; then
  EDIT_PORT="${MAGE_FLOW_EDIT_INTERNAL_PORT:-8102}"
  python scripts/runtime_config.py validate-port \
    --name MAGE_FLOW_EDIT_INTERNAL_PORT --value "$EDIT_PORT"
  bash scripts/stop_process.sh \
    --pid-file .runtime/edit.pid \
    --kind edit_worker \
    --project-root "$PROJECT_ROOT" \
    --port "$EDIT_PORT" \
    --device cuda:1
else
  echo '[PASS] EDIT_WORKER_ALREADY_STOPPED'
fi
