from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def check(url: str, model: str, device: str, model_path: str | None = None) -> bool:
    """Return True only when the localhost worker identity matches exactly."""
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
            data = json.load(response)
    except Exception:
        return False
    if not (
        data.get("status") == "ready"
        and data.get("ready") is True
        and data.get("model") == model
        and data.get("device") == device
    ):
        return False
    if model_path is not None:
        import os

        if data.get("model_path") != os.path.realpath(model_path):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a localhost model worker health identity")
    parser.add_argument("url", help="internal base URL, e.g. http://127.0.0.1:8101")
    parser.add_argument("model", help="expected model name")
    parser.add_argument("device", help="expected device, e.g. cuda:0")
    parser.add_argument("model_path", help="realpath of the mounted model directory")
    args = parser.parse_args()
    if check(args.url, args.model, args.device, args.model_path):
        print("[PASS] HEALTHY")
        return 0
    print("[FAIL] UNHEALTHY")
    return 1


if __name__ == "__main__":
    sys.exit(main())