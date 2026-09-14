#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f .runtime/server.pid ]]; then
  pid="$(cat .runtime/server.pid)"
  if kill -0 "$pid" 2>/dev/null; then
    if [[ -r "/proc/$pid/cmdline" ]] && grep -q "server.app:app" "/proc/$pid/cmdline" 2>/dev/null; then
      kill "$pid"
      echo "[PASS] REST_COORDINATOR_STOPPED pid=$pid"
    else
      echo "[WARN] pid=$pid does not match the REST coordinator; removing stale pid file without killing"
    fi
  else
    echo "[INFO] stale coordinator pid file: $pid"
  fi
  rm -f .runtime/server.pid
else
  echo '[PASS] REST_COORDINATOR_ALREADY_STOPPED'
fi

./scripts/stop_edit.sh

./scripts/stop_t2i.sh