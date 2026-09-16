#!/usr/bin/env bash
set -euo pipefail

# CUDA libraries are at /usr/local/nvidia/lib64 on Kaggle GPU sessions.
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Use SDPA attention when flash-attn is not installed (upstream env override).
if [[ -z "${VF_HF_ATTN_IMPL:-}" ]]; then
  export VF_HF_ATTN_IMPL=sdpa
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUNTIME_ROOT="${MAGE_FLOW_RUNTIME_ROOT:-/kaggle/working/mage-flow-t4x2-runtime}"
MODEL_PATH="$(python scripts/runtime_config.py resolve-model-path --kind edit)"
DEVICE="${MAGE_FLOW_EDIT_DEVICE:-cuda:1}"
RUNTIME_PYTHON="$RUNTIME_ROOT/.venv/bin/python"
PORT="${MAGE_FLOW_EDIT_INTERNAL_PORT:-8102}"
URL="http://127.0.0.1:${PORT}"

python scripts/runtime_config.py validate-port \
  --name MAGE_FLOW_EDIT_INTERNAL_PORT --value "$PORT"

mkdir -p .runtime

printf '%s\n' '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
printf '%s\n' '[STAGE] Start Mage-Flow Edit Turbo worker'
printf '%s\n' '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

echo "[INFO] runtime_python=$RUNTIME_PYTHON"
echo "[INFO] model_path=$MODEL_PATH"
echo "[INFO] device=$DEVICE"
echo "[INFO] internal_url=$URL"

if [[ "$DEVICE" != "cuda:1" ]]; then
  echo "[FAIL] Edit worker must use cuda:1; got $DEVICE"
  exit 1
fi
if [[ ! -x "$RUNTIME_PYTHON" ]]; then
  echo "[FAIL] validated runtime Python not found: $RUNTIME_PYTHON"
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[FAIL] Edit model mount not found: $MODEL_PATH"
  exit 1
fi

process_identity_check() {
  python scripts/process_identity.py check \
    --pid-file .runtime/edit.pid \
    --kind edit_worker \
    --project-root "$PROJECT_ROOT" \
    --port "$PORT" \
    --device "$DEVICE" \
    --model-path "$MODEL_PATH" \
    "$@" >/dev/null 2>&1
}

EXISTING=0
if [[ -f .runtime/edit.pid ]]; then
  if process_identity_check; then
    echo "[INFO] Existing verified Edit worker pid=$(cat .runtime/edit.pid); checking readiness..."
    EXISTING=1
  else
    echo "[WARN] existing edit.pid does not match the Edit identity; cleaning it"
    bash scripts/stop_edit.sh || true
  fi
fi

if [[ "$EXISTING" == "0" ]]; then
  rm -f .runtime/edit.pid
  echo '[INFO] Starting localhost-only Edit runtime worker...'
  PYTHONUNBUFFERED=1 nohup "$RUNTIME_PYTHON" \
    server/workers/edit_runtime_server.py \
    --host 127.0.0.1 \
    --port "$PORT" \
    --model-path "$MODEL_PATH" \
    --device "$DEVICE" \
    > .runtime/edit.log 2>&1 &
  echo $! > .runtime/edit.pid
  echo "[INFO] pid=$(cat .runtime/edit.pid)"
fi

EDIT_START_TIMEOUT="${MAGE_FLOW_EDIT_START_TIMEOUT_SECONDS:-900}"
python scripts/runtime_config.py validate-timeout \
  --name MAGE_FLOW_EDIT_START_TIMEOUT_SECONDS \
  --value "$EDIT_START_TIMEOUT" \
  --default 900 \
  --upper-bound 7200
DEADLINE=$((SECONDS + EDIT_START_TIMEOUT))
LAST_REPORT=0
while (( SECONDS < DEADLINE )); do
  if process_identity_check --health-url "$URL" --model mage-flow-edit-turbo; then
    echo '[PASS] EDIT_WORKER_READY'
    exit 0
  fi

  if [[ -f .runtime/edit.pid ]] && ! kill -0 "$(cat .runtime/edit.pid)" 2>/dev/null; then
    echo '[FAIL] Edit worker exited before readiness.'
    tail -n 80 .runtime/edit.log || true
    rm -f .runtime/edit.pid
    exit 1
  fi

  if (( SECONDS - LAST_REPORT >= 15 )); then
    echo "[HEARTBEAT] waiting for Edit worker elapsed=${SECONDS}s"
    tail -n 8 .runtime/edit.log 2>/dev/null || true
    LAST_REPORT=$SECONDS
  fi
  sleep 2
done

echo '[FAIL] Edit worker readiness timeout'
tail -n 100 .runtime/edit.log || true
bash scripts/stop_edit.sh || true
exit 1
