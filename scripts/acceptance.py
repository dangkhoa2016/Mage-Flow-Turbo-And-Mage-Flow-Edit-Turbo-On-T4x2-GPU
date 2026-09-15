"""Live public REST acceptance for the Mage-Flow T4 x2 demo.

Runs against the resident coordinator after both model workers are ready. It
performs the full dual-worker REST acceptance, saves small artifacts under
``.runtime/acceptance`` and writes a public-safe summary JSON. It never loads a
model itself and never restarts any service.

Usage:
    python scripts/acceptance.py [--base-url URL] [--token TOKEN]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

BAR = "━" * 46
DEFAULT_BASE_URL = "http://127.0.0.1:8090"
DEFAULT_EDIT_SOURCE = (
    "/kaggle/input/models/dangkhoa2016/mage-flow-community-mage-flow-edit-turbo/pytorch/default/1/assets/dog.jpg"
)


def log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def stage(name: str) -> None:
    print(BAR, flush=True)
    log("STAGE", name)
    print(BAR, flush=True)


@dataclass(frozen=True)
class AcceptanceResult:
    passed: bool
    summary: dict
    output_paths: list[str]


class LiveAcceptance:
    def __init__(self, base_url: str, token: str, output_dir: Path, edit_source: Path):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.output_dir = output_dir
        self.edit_source = edit_source
        self.headers = {"Authorization": f"Bearer {token}"}
        self.summary: dict = {"endpoints": {}}

    def _get(self, path: str, *, authenticated: bool):
        headers = self.headers if authenticated else {}
        response = requests.get(f"{self.base_url}{path}", headers=headers, timeout=10)
        return response.status_code, response.json()

    def health(self) -> None:
        stage("Coordinator /health")
        status, payload = self._get("/health", authenticated=False)
        assert status == 200, f"/health returned {status}"
        assert payload.get("status") == "ok", payload
        self.summary["health"] = {"status_code": status, "ok": True}
        log("PASS", f"/health -> {status} {payload.get('status')}")

    def wait_ready(self, timeout: float = 300.0) -> dict:
        stage("Coordinator /ready (fail-closed)")
        started = time.monotonic()
        last_report = 0.0
        while time.monotonic() - started < timeout:
            status, payload = self._get("/ready", authenticated=True)
            if status == 200 and payload.get("ready") is True:
                self.summary["ready"] = {"status_code": status, "payload": payload}
                log("PASS", f"/ready -> ready=true (t2i={payload.get('t2i_ready')} edit={payload.get('edit_ready')})")
                return payload
            now = time.monotonic() - started
            if now - last_report >= 15:
                log("HEARTBEAT", f"waiting /ready elapsed={int(now)}s status={payload.get('status')}")
                last_report = now
            time.sleep(3)
        raise AssertionError(f"/ready did not become true within {int(timeout)}s; last={payload}")

    def info(self) -> dict:
        stage("Coordinator /v1/info")
        status, payload = self._get("/v1/info", authenticated=True)
        assert status == 200, f"/v1/info returned {status}"
        t2i = payload.get("t2i", {})
        edit = payload.get("edit", {})
        assert t2i.get("device") == "cuda:0" and t2i.get("ready") is True, t2i
        assert edit.get("device") == "cuda:1" and edit.get("ready") is True, edit
        assert payload.get("cpu_fallback") is False, payload
        self.summary["info"] = payload
        log("PASS", f"/v1/info t2i={t2i.get('device')} edit={edit.get('device')} cpu_fallback=false")
        return payload

    def _save_data_url(self, data_url: str, name: str) -> bytes:
        match = re.match(r"^data:image/(png|jpeg|jpg);base64,([A-Za-z0-9+/=]+)$", data_url)
        assert match, f"unexpected output data URL prefix: {data_url[:40]}..."
        raw = base64.b64decode(match.group(2))
        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return raw

    def verify_png(self, raw: bytes) -> tuple[int, int]:
        from PIL import Image

        image = Image.open(io.BytesIO(raw))
        image.load()
        assert image.format == "PNG", image.format
        return image.size

    @staticmethod
    def sha256(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    def t2i_generation(self) -> dict:
        stage("T2I public generation")
        payload = {
            "prompt": "A tranquil mountain lake at sunrise, photorealistic landscape photography",
            "seed": 42,
            "steps": 4,
            "width": 1024,
            "height": 1024,
        }
        started = time.monotonic()
        response = requests.post(
            f"{self.base_url}/v1/images/generations",
            headers=self.headers,
            json=payload,
            timeout=float(os.environ.get("MAGE_FLOW_ACCEPTANCE_REQUEST_TIMEOUT_SECONDS","3600")),
        )
        wall = round(time.monotonic() - started, 3)
        assert response.status_code == 200, f"generation returned {response.status_code}: {response.text[:500]}"
        data = response.json()
        assert data.get("status") == "completed", data
        assert data.get("model") == "mage-flow-turbo", data.get("model")
        assert data.get("device") == "cuda:0", data.get("device")
        latency = data.get("elapsed_seconds", 0.0)
        assert latency > 0, data
        raw = self._save_data_url(data["output"], "t2i.png")
        size = self.verify_png(raw)
        assert size == (1024, 1024), size
        file_hash = self.sha256(raw)
        self.summary["t2i_generation"] = {
            "status_code": 200,
            "model": data.get("model"),
            "device": data.get("device"),
            "seed": data.get("seed"),
            "size": list(size),
            "latency_seconds": latency,
            "wall_seconds": wall,
            "sha256": file_hash,
        }
        log("PASS", f"T2I completed device=cuda:0 size={size} latency={latency}s sha256={file_hash[:16]}...")
        return self.summary["t2i_generation"]

    def edit_generation(self) -> dict:
        stage("Edit public generation")
        instruction = "Change the background to a sunny green meadow while keeping the main subject unchanged."
        assert self.edit_source.is_file(), f"edit source image missing: {self.edit_source}"
        source_bytes = self.edit_source.read_bytes()
        started = time.monotonic()
        response = requests.post(
            f"{self.base_url}/v1/images/edits",
            headers=self.headers,
            files={"image": (self.edit_source.name, source_bytes, "image/jpeg")},
            data={"prompt": instruction, "seed": "42"},
            timeout=float(os.environ.get("MAGE_FLOW_ACCEPTANCE_REQUEST_TIMEOUT_SECONDS","3600")),
        )
        wall = round(time.monotonic() - started, 3)
        assert response.status_code == 200, f"edit returned {response.status_code}: {response.text[:500]}"
        data = response.json()
        assert data.get("status") == "completed", data
        assert data.get("model") == "mage-flow-edit-turbo", data.get("model")
        assert data.get("device") == "cuda:1", data.get("device")
        latency = data.get("elapsed_seconds", 0.0)
        assert latency > 0, data
        output_raw = self._save_data_url(data["output"], "edit.png")
        self.verify_png(output_raw)
        source_path = self.output_dir / "edit-source.png"
        source_path.write_bytes(self._to_png(source_bytes))
        self.summary["edit_generation"] = {
            "status_code": 200,
            "model": data.get("model"),
            "device": data.get("device"),
            "seed": data.get("seed"),
            "source": self.edit_source.name,
            "latency_seconds": latency,
            "wall_seconds": wall,
            "sha256": self.sha256(output_raw),
            "output_size": list(self.verify_png(output_raw)),
        }
        log("PASS", f"Edit completed device=cuda:1 latency={latency}s sha256={self.sha256(output_raw)[:16]}...")
        return self.summary["edit_generation"]

    @staticmethod
    def _to_png(image_bytes: bytes) -> bytes:
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def run(self) -> AcceptanceResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.health()
        self.wait_ready()
        self.info()
        t2i_result = self.t2i_generation()
        edit_result = self.edit_generation()
        self.summary["passed"] = True
        self.summary["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        summary_path = self.output_dir / "acceptance-summary.json"
        summary_path.write_text(json.dumps(self.summary, indent=2) + "\n")
        log("PASS", f"acceptance summary written: {summary_path}")
        return AcceptanceResult(
            passed=True,
            summary=self.summary,
            output_paths=[str(p) for p in sorted(self.output_dir.iterdir())],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live public REST acceptance for Mage-Flow T4 x2 demo")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--token", default=None)
    parser.add_argument("--output-dir", default=".runtime/acceptance")
    parser.add_argument("--edit-source", default=DEFAULT_EDIT_SOURCE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = args.token or os.environ.get("MAGE_FLOW_API_TOKEN", "")
    if not token:
        log("FAIL", "MAGE_FLOW_API_TOKEN is required")
        return 1
    runner = LiveAcceptance(
        base_url=args.base_url,
        token=token,
        output_dir=Path(args.output_dir),
        edit_source=Path(args.edit_source),
    )
    try:
        result = runner.run()
    except Exception as exc:
        log("FAIL", f"{type(exc).__name__}: {exc}")
        return 1
    log("PASS", f"LIVE_REST_ACCEPTANCE passed={result.passed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
