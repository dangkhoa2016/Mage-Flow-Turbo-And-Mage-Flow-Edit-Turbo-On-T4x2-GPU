#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ -f .runtime/t2i.pid ]]; then
  T2I_PORT="${MAGE_FLOW_T2I_INTERNAL_PORT:-8101}"
  python scripts/runtime_config.py validate-port \
    --name MAGE_FLOW_T2I_INTERNAL_PORT --value "$T2I_PORT"
  MODEL_PATH="$(python scripts/runtime_config.py resolve-model-path --kind t2i)"
  bash scripts/stop_process.sh \
    --pid-file .runtime/t2i.pid \
    --kind t2i_worker \
    --project-root "$PROJECT_ROOT" \
    --port "$T2I_PORT" \
    --device cuda:0 \
    --model-path "$MODEL_PATH"
else
  echo '[PASS] T2I_WORKER_ALREADY_STOPPED'
fi
