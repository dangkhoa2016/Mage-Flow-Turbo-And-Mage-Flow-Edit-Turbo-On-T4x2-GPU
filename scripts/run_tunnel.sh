#!/usr/bin/env bash
set -euo pipefail
if ! command -v cloudflared >/dev/null 2>&1; then
  echo '[FAIL] CLOUDFLARED_NOT_FOUND'
  echo 'Suggested action / Hướng xử lý: install cloudflared before enabling the optional tunnel.'
  exit 1
fi
echo '[INFO] Starting temporary Cloudflare Quick Tunnel for localhost:8090'
exec cloudflared tunnel --url http://127.0.0.1:8090
