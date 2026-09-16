from __future__ import annotations

import os
import sys
import time

import requests

URL = "http://127.0.0.1:8090/ready"
TIMEOUT = 180
token = os.environ.get("MAGE_FLOW_API_TOKEN", "")
if not token:
    print("[FAIL] MAGE_FLOW_API_TOKEN is not set", flush=True)
    sys.exit(1)
headers = {"Authorization": f"Bearer {token}"}
start = time.monotonic()
print("[INFO] Waiting for authenticated REST readiness...", flush=True)
while time.monotonic() - start < TIMEOUT:
    try:
        r = requests.get(URL, headers=headers, timeout=3)
        data = r.json()
        print(f"[HEARTBEAT] elapsed={int(time.monotonic() - start)}s status={data.get('status')}", flush=True)
        if r.ok and data.get("ready") is True:
            print("[PASS] REST_API_READY", flush=True)
            sys.exit(0)
    except Exception as exc:
        elapsed = int(time.monotonic() - start)
        print(
            f"[HEARTBEAT] elapsed={elapsed}s state=waiting detail={type(exc).__name__}",
            flush=True,
        )
    time.sleep(5)
print("[FAIL] REST_API_READY_TIMEOUT", flush=True)
sys.exit(1)
