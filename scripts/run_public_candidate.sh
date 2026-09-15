#!/usr/bin/env bash
set -euo pipefail

# CUDA libraries are at /usr/local/nvidia/lib64 on Kaggle GPU sessions.
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUNTIME_ROOT="${MAGE_FLOW_RUNTIME_ROOT:-/kaggle/working/mage-flow-v5-t4x2-c1-concurrency-source-20260912}"
RUNTIME_PYTHON="$RUNTIME_ROOT/.venv/bin/python"
MAGE_SOURCE="$RUNTIME_ROOT/vendor/Mage"
EXPECTED_MAGE_COMMIT="${EXPECTED_MAGE_COMMIT:-76bec2bb3818863f470de7e867c2dc7f1d0bfd83}"
T2I_MODEL="${MAGE_FLOW_T2I_MODEL_PATH:-/kaggle/input/models/dangkhoa2016/mage-flow-community-mage-flow-turbo/pytorch/default/1}"
EDIT_MODEL="${MAGE_FLOW_EDIT_MODEL_PATH:-/kaggle/input/models/dangkhoa2016/mage-flow-community-mage-flow-edit-turbo/pytorch/default/1}"
EDIT_SOURCE="${MAGE_FLOW_EDIT_SOURCE_IMAGE:-$EDIT_MODEL/assets/dog.jpg}"

T2I_URL="http://127.0.0.1:8101"
EDIT_URL="http://127.0.0.1:8102"
REST_URL="http://127.0.0.1:8090"

RUNTIME_DIR=".runtime"
ACCEPT_DIR="$RUNTIME_DIR/acceptance"
TOKEN_FILE="$RUNTIME_DIR/api_token"

bar() { printf '%s\n' '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'; }
fail() { echo "[FAIL] $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }

mkdir -p "$RUNTIME_DIR" "$ACCEPT_DIR"
chmod 700 "$RUNTIME_DIR" 2>/dev/null || true

bar
echo '[STAGE] Mage-Flow T4 x2 public candidate orchestrator'
bar

is_endpoint_healthy() {
  local url="$1" model="$2" device="$3" model_path="$4"
  python scripts/service_health.py "$url" "$model" "$device" "$model_path" >/dev/null 2>&1
}

stop_stale_if_unhealthy() {
  local pid_file="$1" url="$2" model="$3" device="$4" model_path="$5" stop_script="$6"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    if ! is_endpoint_healthy "$url" "$model" "$device" "$model_path"; then
      echo "[WARN] stale/misconfigured worker pid=$(cat "$pid_file") is not healthy on $url; stopping it"
      bash "$stop_script" || true
    fi
  else
    rm -f "$pid_file"
  fi
}

# ---------------- path / commit / token validation ----------------
[[ -d "$RUNTIME_ROOT" ]] || fail "validated runtime root missing: $RUNTIME_ROOT"
[[ -d "$MAGE_SOURCE" ]] || fail "Mage source missing: $MAGE_SOURCE"
[[ -x "$RUNTIME_PYTHON" ]] || fail "validated runtime Python missing: $RUNTIME_PYTHON"
[[ -d "$T2I_MODEL" ]] || fail "T2I model mount missing: $T2I_MODEL"
[[ -d "$EDIT_MODEL" ]] || fail "Edit model mount missing: $EDIT_MODEL"
[[ -f "$EDIT_SOURCE" ]] || fail "Edit source sample image missing: $EDIT_SOURCE"
info "runtime_root=$RUNTIME_ROOT"
info "t2i_model=$T2I_MODEL"
info "edit_model=$EDIT_MODEL"

ACTUAL_COMMIT="$(git -C "$MAGE_SOURCE" rev-parse HEAD 2>/dev/null || true)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_MAGE_COMMIT" ]]; then
  fail "Mage source commit mismatch: expected $EXPECTED_MAGE_COMMIT, found ${ACTUAL_COMMIT:-<unresolved>}"
fi
info "mage_commit=$ACTUAL_COMMIT"

# Persistent per-session API token; never printed.
prepare_api_token() {
  local token_file="$1"
  local runtime_dir
  runtime_dir="$(dirname "$token_file")"
  mkdir -p "$runtime_dir"
  chmod 700 "$runtime_dir" 2>/dev/null || true
  if [[ -s "$token_file" ]]; then
    chmod 600 "$token_file" 2>/dev/null || true
    MAGE_FLOW_API_TOKEN="$(cat "$token_file")"
    export MAGE_FLOW_API_TOKEN
    info "api_token=<reused persisted temporary token>"
    return 0
  fi
  MAGE_FLOW_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
  export MAGE_FLOW_API_TOKEN
  umask 077
  printf '%s' "$MAGE_FLOW_API_TOKEN" > "$token_file"
  chmod 600 "$token_file" 2>/dev/null || true
  info "api_token=<generated temporary token>"
}

prepare_api_token "$TOKEN_FILE"
unset TOKEN_VALUE

# ---------------- hardware gate (torch via validated runtime) ----------------
echo '[STAGE] Validate NVIDIA T4 x2 topology'
rc=0
"$RUNTIME_PYTHON" - <<'PY' || rc=$?
import sys
try:
    import torch
except Exception as exc:
    print(f"[FAIL] torch import failed in validated runtime: {exc}")
    sys.exit(2)
if not torch.cuda.is_available():
    print("[HOLD] GPU_ACCELERATOR_REQUIRED: CUDA unavailable")
    sys.exit(2)
count = torch.cuda.device_count()
if count < 2:
    print(f"[HOLD] GPU_ACCELERATOR_REQUIRED: expected >=2 CUDA devices, found {count}")
    sys.exit(2)
names = [torch.cuda.get_device_name(i) for i in range(count)]
if "T4" not in names[0].upper() or "T4" not in names[1].upper():
    print(f"[FAIL] expected NVIDIA T4 x2, detected {names[:2]}")
    sys.exit(3)
print(f"[PASS] T4_x2 cuda:0={names[0]} cuda:1={names[1]}")
PY
if (( rc == 0 )); then
  :
elif (( rc == 2 )); then
  echo "[HOLD] GPU_ACCELERATOR_REQUIRED"
  exit 2
else
  fail "hardware gate rejected (rc=$rc)"
fi

# ---------------- stale PID / port detection ----------------
stop_stale_if_unhealthy "$RUNTIME_DIR/t2i.pid" "$T2I_URL" "mage-flow-turbo" "cuda:0" "$T2I_MODEL" "scripts/stop_t2i.sh"
stop_stale_if_unhealthy "$RUNTIME_DIR/edit.pid" "$EDIT_URL" "mage-flow-edit-turbo" "cuda:1" "$EDIT_MODEL" "scripts/stop_edit.sh"
if [[ -f "$RUNTIME_DIR/server.pid" ]] && kill -0 "$(cat "$RUNTIME_DIR/server.pid")" 2>/dev/null; then
  if ! python - "$REST_URL" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1] + "/health", timeout=2) as r:
        json.load(r)
except Exception:
    sys.exit(1)
PY
  then
    echo "[WARN] stale coordinator pid=$(cat "$RUNTIME_DIR/server.pid") is not healthy; stopping it"
    bash scripts/stop.sh || true
  fi
else
  rm -f "$RUNTIME_DIR/server.pid"
fi

# ---------------- workers (idempotent: reuse healthy resident processes) ----------------
T2I_BEFORE_PID="$(cat "$RUNTIME_DIR/t2i.pid" 2>/dev/null || true)"
bash scripts/start_t2i.sh
T2I_AFTER_PID="$(cat "$RUNTIME_DIR/t2i.pid" 2>/dev/null || true)"
if [[ -n "$T2I_BEFORE_PID" && "$T2I_BEFORE_PID" == "$T2I_AFTER_PID" ]]; then
  info "existing healthy T2I worker -> REUSE (pid=$T2I_AFTER_PID)"
  T2I_LOADED=0
else
  info "T2I worker loaded once (new pid=$T2I_AFTER_PID)"
  T2I_LOADED=1
fi

EDIT_BEFORE_PID="$(cat "$RUNTIME_DIR/edit.pid" 2>/dev/null || true)"
bash scripts/start_edit.sh
EDIT_AFTER_PID="$(cat "$RUNTIME_DIR/edit.pid" 2>/dev/null || true)"
if [[ -n "$EDIT_BEFORE_PID" && "$EDIT_BEFORE_PID" == "$EDIT_AFTER_PID" ]]; then
  info "existing healthy Edit worker -> REUSE (pid=$EDIT_AFTER_PID)"
  EDIT_LOADED=0
else
  info "Edit worker loaded once (new pid=$EDIT_AFTER_PID)"
  EDIT_LOADED=1
fi

# ---------------- coordinator (reuse healthy resident process) ----------------
if is_endpoint_healthy "$T2I_URL" "mage-flow-turbo" "cuda:0" "$T2I_MODEL"; then
  echo '[PASS] T2I_WORKER_READY'
else
  fail "T2I worker not healthy after start"
fi
if is_endpoint_healthy "$EDIT_URL" "mage-flow-edit-turbo" "cuda:1" "$EDIT_MODEL"; then
  echo '[PASS] EDIT_WORKER_READY'
else
  fail "Edit worker not healthy after start"
fi

if [[ -f "$RUNTIME_DIR/server.pid" ]] && kill -0 "$(cat "$RUNTIME_DIR/server.pid")" 2>/dev/null; then
  if python - "$REST_URL" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1] + "/health", timeout=2) as r:
        json.load(r)
except Exception:
    sys.exit(1)
PY
  then
    echo "[PASS] REST_COORDINATOR_ALREADY_RUNNING pid=$(cat "$RUNTIME_DIR/server.pid")"
  else
    fail "coordinator pid alive but /health failed"
  fi
else
  rm -f "$RUNTIME_DIR/server.pid"
  echo '[INFO] Starting authenticated REST coordinator on 127.0.0.1:8090'
  MAGE_FLOW_T2I_INTERNAL_URL="${MAGE_FLOW_T2I_INTERNAL_URL:-$T2I_URL}" \
  MAGE_FLOW_EDIT_INTERNAL_URL="${MAGE_FLOW_EDIT_INTERNAL_URL:-$EDIT_URL}" \
  nohup python -m uvicorn server.app:app --host 127.0.0.1 --port 8090 \
    > "$RUNTIME_DIR/server.log" 2>&1 &
  echo $! > "$RUNTIME_DIR/server.pid"
  echo "[INFO] pid=$(cat "$RUNTIME_DIR/server.pid")"
fi

# ---------------- readiness ----------------
python scripts/wait_ready.py

# ---------------- PIDs before acceptance ----------------
python - "$ACCEPT_DIR/pids_before.json" <<'PY'
import json, os, sys
def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return "none"
root = ".runtime"
data = {
    "t2i_pid": read(os.path.join(root, "t2i.pid")),
    "edit_pid": read(os.path.join(root, "edit.pid")),
    "coordinator_pid": read(os.path.join(root, "server.pid")),
}
with open(sys.argv[1], "w") as f:
    json.dump(data, f, indent=2)
print(f"[INFO] pids_before {data}")
PY

# ---------------- live public acceptance (workers stay resident) ----------------
if [[ -n "${MAGE_FLOW_ACCEPTANCE_EDIT_SOURCE:-}" ]]; then
  EDIT_SOURCE="$MAGE_FLOW_ACCEPTANCE_EDIT_SOURCE"
fi
python scripts/acceptance.py \
  --base-url "$REST_URL" \
  --output-dir "$ACCEPT_DIR" \
  --edit-source "$EDIT_SOURCE"

# ---------------- PIDs after acceptance + reveal ----------------
python - "$ACCEPT_DIR/pids_after.json" <<'PY'
import json, os, sys
def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return "none"
root = ".runtime"
data = {
    "t2i_pid": read(os.path.join(root, "t2i.pid")),
    "edit_pid": read(os.path.join(root, "edit.pid")),
    "coordinator_pid": read(os.path.join(root, "server.pid")),
}
with open(sys.argv[1], "w") as f:
    json.dump(data, f, indent=2)
print(f"[INFO] pids_after {data}")
PY

{
  echo "T2I_LOADED=$T2I_LOADED"
  echo "EDIT_LOADED=$EDIT_LOADED"
  echo "MODEL_RELOAD_COUNT_DURING_IDEMPOTENCY_TEST=$((T2I_LOADED + EDIT_LOADED))"
  echo "RUNTIME_EXTRACTION_COUNT=0"
  echo "T2I_PID=$T2I_AFTER_PID"
  echo "EDIT_PID=$EDIT_AFTER_PID"
  echo "COORDINATOR_PID=$(cat "$RUNTIME_DIR/server.pid")"
} > "$ACCEPT_DIR/orchestrator-reveal.txt"

bar
echo '[PASS] PUBLIC_CANDIDATE_ORCHESTRATOR_COMPLETE'
echo '[INFO] T2I/Edit/coordinator remain resident; do not stop them during the GPU session.'
bar
