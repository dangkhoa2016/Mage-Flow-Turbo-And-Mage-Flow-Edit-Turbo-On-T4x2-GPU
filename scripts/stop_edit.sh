#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f .runtime/edit.pid ]]; then
  pid="$(cat .runtime/edit.pid)"
  if kill -0 "$pid" 2>/dev/null; then
    if [[ -r "/proc/$pid/cmdline" ]] && grep -q "edit_runtime_server" "/proc/$pid/cmdline" 2>/dev/null; then
      kill "$pid"
      echo "[PASS] EDIT_WORKER_STOPPED pid=$pid"
    else
      echo "[WARN] pid=$pid does not match Edit runtime server; removing stale pid file without killing"
    fi
  else
    echo "[INFO] stale Edit pid file: $pid"
  fi
  rm -f .runtime/edit.pid
else
  echo '[PASS] EDIT_WORKER_ALREADY_STOPPED'
fi