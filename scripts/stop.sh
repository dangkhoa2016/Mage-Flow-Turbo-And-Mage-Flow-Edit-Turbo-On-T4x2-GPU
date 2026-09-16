#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ -f .runtime/server.pid ]]; then
  bash scripts/stop_process.sh \
    --pid-file .runtime/server.pid \
    --kind coordinator \
    --project-root "$PROJECT_ROOT" \
    --port "${MAGE_FLOW_REST_PORT:-8090}"
else
  echo '[PASS] REST_COORDINATOR_ALREADY_STOPPED'
fi

./scripts/stop_edit.sh

./scripts/stop_t2i.sh