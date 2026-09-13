#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f .runtime/t2i.pid ]]; then
  pid="$(cat .runtime/t2i.pid)"
  if kill -0 "$pid" 2>/dev/null; then
    if [[ -r "/proc/$pid/cmdline" ]] && grep -q "t2i_runtime_server" "/proc/$pid/cmdline" 2>/dev/null; then
      kill "$pid"
      echo "[PASS] T2I_WORKER_STOPPED pid=$pid"
    else
      echo "[WARN] pid=$pid does not match T2I runtime server; removing stale pid file without killing"
    fi
  else
    echo "[INFO] stale T2I pid file: $pid"
  fi
  rm -f .runtime/t2i.pid
else
  echo '[PASS] T2I_WORKER_ALREADY_STOPPED'
fi