#!/usr/bin/env bash
# Shared identity-safe graceful stop for Mage-Flow runtime processes.
#
# Usage:
#   stop_process.sh --pid-file FILE --kind t2i_worker|edit_worker|coordinator \
#     --project-root ROOT [--port N] [--device D] [--model-path MP]
#
# Behaviour:
#   * reads the PID file with the strict shared guard (process_identity.py);
#   * refuses to signal a PID whose /proc identity does not match the service;
#   * sends SIGTERM, waits a bounded interval, then re-verifies identity before
#     escalating to SIGKILL (never kills a new process that reused the PID);
#   * only removes the PID file once the verified process has actually exited.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PID_FILE=""
KIND=""
PORT=""
DEVICE=""
MODEL_PATH=""
TERM_GRACE_STEP=1
TERM_GRACE_MAX="${MAGE_FLOW_STOP_TERM_GRACE_SECONDS:-20}"
KILL_GRACE_MAX="${MAGE_FLOW_STOP_KILL_GRACE_SECONDS:-10}"

while (( $# > 0 )); do
  case "$1" in
    --pid-file) PID_FILE="$2"; shift 2;;
    --kind) KIND="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --model-path) MODEL_PATH="$2"; shift 2;;
    --project-root) CWD_ROOT="$2"; shift 2;;
    *) echo "[FAIL] unknown argument: $1" >&2; exit 1;;
  esac
done

[[ -n "$PID_FILE" ]] || { echo "[FAIL] --pid-file is required" >&2; exit 1; }
[[ -n "$KIND" ]] || { echo "[FAIL] --kind is required" >&2; exit 1; }
CWD_ROOT="${CWD_ROOT:-$PROJECT_ROOT}"

python scripts/runtime_config.py validate-timeout \
  --name MAGE_FLOW_STOP_TERM_GRACE_SECONDS \
  --value "$TERM_GRACE_MAX" \
  --default 20 \
  --upper-bound 7200
python scripts/runtime_config.py validate-timeout \
  --name MAGE_FLOW_STOP_KILL_GRACE_SECONDS \
  --value "$KILL_GRACE_MAX" \
  --default 10 \
  --upper-bound 7200
if [[ -n "$PORT" ]]; then
  python scripts/runtime_config.py validate-port \
    --name MAGE_FLOW_STOP_PROCESS_PORT --value "$PORT"
fi

identity_check() {
  local extra=()
  [[ -n "$PORT" ]] && extra+=( --port "$PORT" )
  [[ -n "$DEVICE" ]] && extra+=( --device "$DEVICE" )
  [[ -n "$MODEL_PATH" ]] && extra+=( --model-path "$MODEL_PATH" )
  python scripts/process_identity.py check \
    --pid-file "$PID_FILE" --kind "$KIND" --project-root "$CWD_ROOT" \
    "${extra[@]}" >/dev/null 2>&1
}

read_pid() {
  python scripts/process_identity.py read-pid --pid-file "$PID_FILE" 2>/dev/null || true
}

alive() {
  # A zombie keeps /proc/<pid> briefly; its cmdline is empty, so treat it as
  # already exited rather than "still running" (and never escalate against it).
  [[ -d "/proc/$1" ]] && [[ -s "/proc/$1/cmdline" ]]
}

pid="$(read_pid)"
if [[ -z "$pid" ]]; then
  echo "[INFO] PID file does not hold a valid process id; cleaning stale file without killing"
  rm -f "$PID_FILE"
  exit 0
fi

if ! identity_check; then
  echo "[WARN] pid=$pid does not match the expected $KIND identity; removing stale pid file without killing"
  rm -f "$PID_FILE"
  exit 0
fi

echo "[INFO] stopping $KIND pid=$pid with SIGTERM"
kill -- "$pid" 2>/dev/null || true

elapsed=0
while (( elapsed < TERM_GRACE_MAX )); do
  if ! alive "$pid"; then
    echo "[PASS] ${KIND}_STOPPED pid=$pid"
    rm -f "$PID_FILE"
    exit 0
  fi
  # Identity may have changed if the PID was reused while we waited; in that
  # case the new process is NOT ours and must not be escalated against.
  if ! identity_check; then
    echo "[WARN] pid=$pid identity changed during wait; refusing to escalate against a reused PID"
    rm -f "$PID_FILE"
    exit 0
  fi
  sleep "$TERM_GRACE_STEP"
  elapsed=$((elapsed + TERM_GRACE_STEP))
done

# Only escalate when the process is still the verified service.
if alive "$pid" && identity_check; then
  echo "[WARN] pid=$pid still alive after SIGTERM grace; escalating to SIGKILL"
  kill -9 -- "$pid" 2>/dev/null || true
else
  echo "[WARN] pid=$pid exited during TERM grace"
fi

elapsed=0
while (( elapsed < KILL_GRACE_MAX )); do
  if ! alive "$pid"; then
    echo "[PASS] ${KIND}_STOPPED pid=$pid"
    rm -f "$PID_FILE"
    exit 0
  fi
  sleep "$TERM_GRACE_STEP"
  elapsed=$((elapsed + TERM_GRACE_STEP))
done

if alive "$pid" && identity_check; then
  echo "[FAIL] $KIND pid=$pid is still alive after SIGKILL and still matches identity; keeping pid file for diagnosis"
  exit 1
fi

if alive "$pid"; then
  echo "[WARN] pid=$pid is alive but no longer matches the expected identity; removing stale pid file without killing"
  rm -f "$PID_FILE"
  exit 0
fi
echo "[PASS] ${KIND}_STOPPED pid=$pid"
rm -f "$PID_FILE"
exit 0
