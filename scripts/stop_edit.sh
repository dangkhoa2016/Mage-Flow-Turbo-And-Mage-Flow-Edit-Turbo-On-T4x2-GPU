#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ -f .runtime/edit.pid ]]; then
  bash scripts/stop_process.sh \
    --pid-file .runtime/edit.pid \
    --kind edit_worker \
    --project-root "$PROJECT_ROOT" \
    --port "${MAGE_FLOW_EDIT_INTERNAL_PORT:-8102}" \
    --device cuda:1
else
  echo '[PASS] EDIT_WORKER_ALREADY_STOPPED'
fi
