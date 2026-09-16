#!/usr/bin/env bash
# Shared identity-safe graceful stop for Mage-Flow runtime processes.
#
# Usage:
#   stop_process.sh --pid-file FILE --kind t2i_worker|edit_worker|coordinator \
#     --project-root ROOT [--port N] [--device D] [--model-path MP]
#
# Behaviour:
#   * reads the PID file with the strict shared guard (process_identity.py);
#   * delegates the whole stop to process_identity.py `stop-process`, which
#     opens a pidfd for the PID, verifies identity while the pidfd is held,
#     sends SIGTERM via the pidfd, and only escalates to pidfd SIGKILL after a
#     bounded wait and a re-verification (never kills a reused PID);
#   * if the pidfd primitives are unavailable, fails closed with
#     `[HOLD] PIDFD_SIGNAL_AUTHORITY=UNAVAILABLE_FAIL_CLOSED` instead of
#     silently falling back to numeric-PID signaling;
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

if ! python scripts/process_identity.py pidfd-support >/dev/null 2>&1; then
  echo '[HOLD] PIDFD_SIGNAL_AUTHORITY=UNAVAILABLE_FAIL_CLOSED' >&2
  exit 1
fi

local_args=()
[[ -n "$PORT" ]] && local_args+=( --port "$PORT" )
[[ -n "$DEVICE" ]] && local_args+=( --device "$DEVICE" )
[[ -n "$MODEL_PATH" ]] && local_args+=( --model-path "$MODEL_PATH" )

python scripts/process_identity.py stop-process \
  --pid-file "$PID_FILE" \
  --kind "$KIND" \
  --project-root "$CWD_ROOT" \
  "${local_args[@]}" \
  --term-grace "$TERM_GRACE_MAX" \
  --kill-grace "$KILL_GRACE_MAX" \
  --step "$TERM_GRACE_STEP"
