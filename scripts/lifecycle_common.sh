#!/usr/bin/env bash
# Shared transactional-safety helpers for Mage-Flow lifecycle orchestrators.
#
# A start orchestrator must never leave processes resident that THIS invocation
# started when the overall start fails. These helpers track which runtime
# processes the invocation actually started (as opposed to healthy
# pre-existing/reused processes) and, on demand, stop exactly those in reverse
# dependency order: coordinator, Edit worker, then T2I worker.
#
# Source this file from a lifecycle script; it must only be sourced once per
# shell so the ``NEW_SERVICES`` array stays scoped to the orchestrator.

declare -a NEW_SERVICES=()

record_new_service() {
  local pid_file="$1" stop_script="$2"
  NEW_SERVICES+=("$pid_file|$stop_script")
}

cleanup_new_services() {
  local i entry pid_file stop_script n
  n="${#NEW_SERVICES[@]}"
  for (( i = n - 1; i >= 0; i-- )); do
    entry="${NEW_SERVICES[$i]}"
    [[ -n "$entry" ]] || continue
    pid_file="${entry%%|*}"
    stop_script="${entry##*|}"
    if [[ -f "$pid_file" ]]; then
      echo "[INFO] transactional cleanup: stopping newly started process in $pid_file"
      bash "$stop_script" || true
    fi
  done
}
