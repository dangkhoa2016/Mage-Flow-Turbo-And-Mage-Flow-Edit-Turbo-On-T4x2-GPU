#!/usr/bin/env bash
set -euo pipefail

# CUDA libraries are at /usr/local/nvidia/lib64 on Kaggle GPU sessions.
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUNTIME_ROOT="${MAGE_FLOW_RUNTIME_ROOT:-/kaggle/working/mage-flow-v5-t4x2-c1-concurrency-source-20260912}"
RUNTIME_PYTHON="$RUNTIME_ROOT/.venv/bin/python"
MAGE_SOURCE="$RUNTIME_ROOT/vendor/Mage"

# The public-candidate expected Mage commit is an immutable project constant
# owned by scripts/validate_vendor_source.py, the single production authority.
# PUBLIC_SOURCE_AUTHORITY: vendor checkout must resolve to that exact commit.

T2I_MODEL="$(python scripts/runtime_config.py resolve-model-path --kind t2i)"
EDIT_MODEL="$(python scripts/runtime_config.py resolve-model-path --kind edit)"
EDIT_SOURCE="${MAGE_FLOW_EDIT_SOURCE_IMAGE:-$EDIT_MODEL/assets/dog.jpg}"

# Single authoritative configuration model: ports are read once and everything
# else is derived from them, so startup and health/routing cannot diverge.
T2I_PORT="${MAGE_FLOW_T2I_INTERNAL_PORT:-8101}"
EDIT_PORT="${MAGE_FLOW_EDIT_INTERNAL_PORT:-8102}"
REST_PORT="${MAGE_FLOW_REST_PORT:-8090}"
T2I_URL="http://127.0.0.1:${T2I_PORT}"
EDIT_URL="http://127.0.0.1:${EDIT_PORT}"
REST_URL="http://127.0.0.1:${REST_PORT}"
export MAGE_FLOW_T2I_INTERNAL_URL="$T2I_URL"
export MAGE_FLOW_EDIT_INTERNAL_URL="$EDIT_URL"

python scripts/runtime_config.py validate-ports \
  --rest "$REST_PORT" --t2i "$T2I_PORT" --edit "$EDIT_PORT"

RUNTIME_DIR=".runtime"
ACCEPT_DIR="$RUNTIME_DIR/acceptance"
TOKEN_FILE="$RUNTIME_DIR/api_token"

bar() { printf '%s\n' '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'; }
fail() { echo "[FAIL] $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }

# ---------------- transactional startup behaviour ----------------
# Track processes started by THIS invocation so a failed run cleans up only
# those, never healthy pre-existing (reused) processes.
declare -a NEW_SERVICES=()
record_new_service() {
  local pid_file="$1" stop_script="$2"
  NEW_SERVICES+=("$pid_file|$stop_script")
}
cleanup_new_services() {
  local entry pid_file stop_script
  for entry in "${NEW_SERVICES[@]:-}"; do
    [[ -n "$entry" ]] || continue
    pid_file="${entry%%|*}"
    stop_script="${entry##*|}"
    if [[ -f "$pid_file" ]]; then
      echo "[INFO] transactional cleanup: stopping newly started process in $pid_file"
      bash "$stop_script" || true
    fi
  done
}
on_exit() {
  local rc=$?
  if (( rc == 0 )); then
    return 0
  fi
  echo "[INFO] orchestrator terminated with rc=$rc; leaving pre-existing reused processes resident"
  cleanup_new_services
}
trap on_exit EXIT

# ---------------- path / commit / token validation ----------------
python scripts/token_store.py prepare-dir "$RUNTIME_DIR" "$ACCEPT_DIR"

[[ -d "$RUNTIME_ROOT" ]] || fail "validated runtime root missing: $RUNTIME_ROOT"
[[ -d "$MAGE_SOURCE" ]] || fail "Mage source missing: $MAGE_SOURCE"
[[ -x "$RUNTIME_PYTHON" ]] || fail "validated runtime Python missing: $RUNTIME_PYTHON"
[[ -d "$T2I_MODEL" ]] || fail "T2I model mount missing: $T2I_MODEL"
[[ -d "$EDIT_MODEL" ]] || fail "Edit model mount missing: $EDIT_MODEL"
[[ -f "$EDIT_SOURCE" ]] || fail "Edit source sample image missing: $EDIT_SOURCE"
info "runtime_root=$RUNTIME_ROOT"
info "t2i_model=$T2I_MODEL"
info "edit_model=$EDIT_MODEL"

python scripts/validate_vendor_source.py "$MAGE_SOURCE"
info "mage_source=$MAGE_SOURCE"

# Persistent per-session API token; fail-closed permissions, never printed.
prepare_api_token() {
  local token_file="$1"
  local runtime_dir
  runtime_dir="$(dirname "$token_file")"
  python scripts/token_store.py prepare-dir "$runtime_dir"
  MAGE_FLOW_API_TOKEN="$(python scripts/token_store.py ensure-token "$token_file")"
  export MAGE_FLOW_API_TOKEN
  python scripts/runtime_config.py validate-token --value="$MAGE_FLOW_API_TOKEN"
  info "api_token=<temporary session token; not printed>"
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
is_endpoint_healthy() {
  local url="$1" model="$2" device="$3" model_path="$4"
  python scripts/service_health.py "$url" "$model" "$device" "$model_path" >/dev/null 2>&1
}

stop_stale_if_unhealthy() {
  local pid_file="$1" url="$2" model="$3" device="$4" model_path="$5" stop_script="$6" kind="$7" port="$8"
  if [[ ! -f "$pid_file" ]]; then
    return 0
  fi
  if python scripts/process_identity.py check \
       --pid-file "$pid_file" --kind "$kind" --project-root "$PROJECT_ROOT" --port "$port" >/dev/null 2>&1; then
    if ! is_endpoint_healthy "$url" "$model" "$device" "$model_path"; then
      echo "[WARN] stale/misconfigured worker pid=$(cat "$pid_file" 2>/dev/null || true) is not healthy on $url; stopping it"
      bash "$stop_script" || true
    fi
  else
    echo "[WARN] pid file $pid_file does not match process identity; removing stale pid file without killing"
    rm -f "$pid_file"
  fi
}

stop_stale_if_unhealthy "$RUNTIME_DIR/t2i.pid" "$T2I_URL" "mage-flow-turbo" "cuda:0" "$T2I_MODEL" "scripts/stop_t2i.sh" "t2i_worker" "$T2I_PORT"
stop_stale_if_unhealthy "$RUNTIME_DIR/edit.pid" "$EDIT_URL" "mage-flow-edit-turbo" "cuda:1" "$EDIT_MODEL" "scripts/stop_edit.sh" "edit_worker" "$EDIT_PORT"

if [[ -f "$RUNTIME_DIR/server.pid" ]] && ! python scripts/process_identity.py check \
     --pid-file "$RUNTIME_DIR/server.pid" --kind coordinator \
     --project-root "$PROJECT_ROOT" --port "$REST_PORT" >/dev/null 2>&1; then
  echo "[WARN] stale coordinator pid=$(cat "$RUNTIME_DIR/server.pid" 2>/dev/null || true) failed identity check; stopping it"
  bash scripts/stop.sh || true
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
  record_new_service "$RUNTIME_DIR/t2i.pid" "scripts/stop_t2i.sh"
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
  record_new_service "$RUNTIME_DIR/edit.pid" "scripts/stop_edit.sh"
fi

# ---------------- coordinator (reuse only with current-token + identity proof) ----------------
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

COORDINATOR_REUSED=0
if [[ -f "$RUNTIME_DIR/server.pid" ]]; then
  if python scripts/process_identity.py check \
       --pid-file "$RUNTIME_DIR/server.pid" \
       --kind coordinator \
       --project-root "$PROJECT_ROOT" \
       --port "$REST_PORT" \
       --health-url "$REST_URL" \
       --token-file "$TOKEN_FILE"; then
    echo "[PASS] REST_COORDINATOR_ALREADY_RUNNING pid=$(cat "$RUNTIME_DIR/server.pid")"
    COORDINATOR_REUSED=1
  else
    echo "[WARN] coordinator pid=$(cat "$RUNTIME_DIR/server.pid") failed current-token/identity check; restarting it"
    bash scripts/stop.sh || true
    rm -f "$RUNTIME_DIR/server.pid"
  fi
fi
if [[ "$COORDINATOR_REUSED" == "0" ]]; then
  echo "[INFO] Starting authenticated REST coordinator on ${REST_URL}"
  nohup python -m uvicorn server.app:app \
    --host 127.0.0.1 \
    --port "$REST_PORT" \
    > "$RUNTIME_DIR/server.log" 2>&1 &
  echo $! > "$RUNTIME_DIR/server.pid"
  echo "[INFO] pid=$(cat "$RUNTIME_DIR/server.pid")"
  record_new_service "$RUNTIME_DIR/server.pid" "scripts/stop.sh"
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
