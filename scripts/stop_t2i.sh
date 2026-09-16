#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ -f .runtime/t2i.pid ]]; then
  bash scripts/stop_process.sh \
    --pid-file .runtime/t2i.pid \
    --kind t2i_worker \
    --project-root "$PROJECT_ROOT" \
    --port "${MAGE_FLOW_T2I_INTERNAL_PORT:-8101}" \
    --device cuda:0
else
  echo '[PASS] T2I_WORKER_ALREADY_STOPPED'
fi
