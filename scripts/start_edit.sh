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

RUNTIME_ROOT="${MAGE_FLOW_RUNTIME_ROOT:-/kaggle/working/mage-flow-v5-t4x2-c1-concurrency-source-20260912}"
MODEL_PATH="${MAGE_FLOW_EDIT_MODEL_PATH:-/kaggle/input/models/dangkhoa2016/mage-flow-community-mage-flow-edit-turbo/pytorch/default/1}"
DEVICE="${MAGE_FLOW_EDIT_DEVICE:-cuda:1}"
RUNTIME_PYTHON="$RUNTIME_ROOT/.venv/bin/python"
PORT="${MAGE_FLOW_EDIT_INTERNAL_PORT:-8102}"
URL="http://127.0.0.1:${PORT}"

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

if [[ -f .runtime/edit.pid ]] && kill -0 "$(cat .runtime/edit.pid)" 2>/dev/null; then
  echo "[INFO] Existing Edit worker pid=$(cat .runtime/edit.pid); checking readiness..."
else
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

DEADLINE=$((SECONDS + ${MAGE_FLOW_EDIT_START_TIMEOUT_SECONDS:-900}))
LAST_REPORT=0
while (( SECONDS < DEADLINE )); do
  if [[ -f .runtime/edit.pid ]] && ! kill -0 "$(cat .runtime/edit.pid)" 2>/dev/null; then
    echo '[FAIL] Edit worker exited before readiness.'
    tail -n 80 .runtime/edit.log || true
    rm -f .runtime/edit.pid
    exit 1
  fi

  if python - "$URL" "$MODEL_PATH" <<'PY' >/dev/null 2>&1
import json, os, sys, urllib.request
with urllib.request.urlopen(sys.argv[1] + '/health', timeout=2) as r:
    data = json.load(r)
assert data['ready'] is True
assert data['device'] == 'cuda:1'
assert data['model'] == 'mage-flow-edit-turbo'
assert os.path.realpath(data['model_path']) == os.path.realpath(sys.argv[2])
PY
  then
    echo '[PASS] EDIT_WORKER_READY'
    exit 0
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
if [[ -f .runtime/edit.pid ]]; then
  pid="$(cat .runtime/edit.pid)"
  kill "$pid" 2>/dev/null || true
  rm -f .runtime/edit.pid
fi
exit 1
