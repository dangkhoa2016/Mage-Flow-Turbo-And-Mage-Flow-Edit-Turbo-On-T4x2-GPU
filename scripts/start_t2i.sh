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
MODEL_PATH="${MAGE_FLOW_T2I_MODEL_PATH:-/kaggle/input/models/dangkhoa2016/mage-flow-community-mage-flow-turbo/pytorch/default/1}"
DEVICE="${MAGE_FLOW_T2I_DEVICE:-cuda:0}"
RUNTIME_PYTHON="$RUNTIME_ROOT/.venv/bin/python"
PORT="${MAGE_FLOW_T2I_INTERNAL_PORT:-8101}"
URL="http://127.0.0.1:${PORT}"

mkdir -p .runtime

printf '%s\n' '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
printf '%s\n' '[STAGE] Start Mage-Flow Turbo T2I worker'
printf '%s\n' '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

echo "[INFO] runtime_python=$RUNTIME_PYTHON"
echo "[INFO] model_path=$MODEL_PATH"
echo "[INFO] device=$DEVICE"
echo "[INFO] internal_url=$URL"

if [[ "$DEVICE" != "cuda:0" ]]; then
  echo "[FAIL] T2I worker must use cuda:0; got $DEVICE"
  exit 1
fi
if [[ ! -x "$RUNTIME_PYTHON" ]]; then
  echo "[FAIL] validated runtime Python not found: $RUNTIME_PYTHON"
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[FAIL] T2I model mount not found: $MODEL_PATH"
  exit 1
fi

if [[ -f .runtime/t2i.pid ]] && kill -0 "$(cat .runtime/t2i.pid)" 2>/dev/null; then
  echo "[INFO] Existing T2I worker pid=$(cat .runtime/t2i.pid); checking readiness..."
else
  rm -f .runtime/t2i.pid
  echo '[INFO] Starting localhost-only T2I runtime worker...'
  PYTHONUNBUFFERED=1 nohup "$RUNTIME_PYTHON" \
    server/workers/t2i_runtime_server.py \
    --host 127.0.0.1 \
    --port "$PORT" \
    --model-path "$MODEL_PATH" \
    --device "$DEVICE" \
    > .runtime/t2i.log 2>&1 &
  echo $! > .runtime/t2i.pid
  echo "[INFO] pid=$(cat .runtime/t2i.pid)"
fi

DEADLINE=$((SECONDS + ${MAGE_FLOW_T2I_START_TIMEOUT_SECONDS:-900}))
LAST_REPORT=0
while (( SECONDS < DEADLINE )); do
  if [[ -f .runtime/t2i.pid ]] && ! kill -0 "$(cat .runtime/t2i.pid)" 2>/dev/null; then
    echo '[FAIL] T2I worker exited before readiness.'
    tail -n 80 .runtime/t2i.log || true
    rm -f .runtime/t2i.pid
    exit 1
  fi

  if python - "$URL" "$MODEL_PATH" <<'PY' >/dev/null 2>&1
import json, os, sys, urllib.request
with urllib.request.urlopen(sys.argv[1] + '/health', timeout=2) as r:
    data = json.load(r)
assert data['ready'] is True
assert data['device'] == 'cuda:0'
assert data['model'] == 'mage-flow-turbo'
assert os.path.realpath(data['model_path']) == os.path.realpath(sys.argv[2])
PY
  then
    echo '[PASS] T2I_WORKER_READY'
    exit 0
  fi

  if (( SECONDS - LAST_REPORT >= 15 )); then
    echo "[HEARTBEAT] waiting for T2I worker elapsed=${SECONDS}s"
    tail -n 8 .runtime/t2i.log 2>/dev/null || true
    LAST_REPORT=$SECONDS
  fi
  sleep 2
done

echo '[FAIL] T2I worker readiness timeout'
tail -n 100 .runtime/t2i.log || true
if [[ -f .runtime/t2i.pid ]]; then
  pid="$(cat .runtime/t2i.pid)"
  kill "$pid" 2>/dev/null || true
  rm -f .runtime/t2i.pid
fi
exit 1
