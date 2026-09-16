"""Live public REST acceptance for the Mage-Flow T4 x2 demo.

Runs against the resident coordinator after both model workers are ready. It
performs the full dual-worker REST acceptance, saves small artifacts under
``.runtime/acceptance`` and writes a public-safe summary JSON. It never loads a
model itself and never restarts any service.

Safety-gate contract: no ``assert`` statements are used as operational checks
because ``python -O`` strips them. Every gate raises ``AssertionError``/other
exceptions explicitly so they cannot silently disappear.

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
from typing import Any

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

    def _get(self, path: str, *, authenticated: bool, headers: dict | None = None) -> tuple[int, dict]:
        request_headers = self.headers if authenticated else {}
        if headers:
            request_headers = {**request_headers, **headers}
        try:
            response = requests.get(f"{self.base_url}{path}", headers=request_headers, timeout=10)
        except requests.RequestException:
            return -1, {}
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        return response.status_code, payload

    def _post_json(self, path: str, payload: dict, *, headers: dict | None = None) -> int:
        request_headers = dict(self.headers)
        if headers:
            request_headers.update(headers)
        try:
            response = requests.post(
                f"{self.base_url}{path}",
                headers=request_headers,
                json=payload,
                timeout=10,
            )
            return response.status_code
        except requests.RequestException:
            return -1

    def _post_edit_upload(self, data: bytes, media_type: str, filename: str) -> int:
        try:
            response = requests.post(
                f"{self.base_url}/v1/images/edits",
                headers=self.headers,
                files={"image": (filename, data, media_type)},
                data={"prompt": "Change nothing; validation-only probe", "seed": "42"},
                timeout=10,
            )
            return response.status_code
        except requests.RequestException:
            return -1

    def health(self) -> None:
        stage("Coordinator /health")
        status, payload = self._get("/health", authenticated=False)
        if status != 200:
            raise AssertionError(f"/health returned {status}")
        if payload.get("status") != "ok":
            raise AssertionError(f"/health payload invalid: {payload}")
        self.summary["health"] = {"status_code": status, "ok": True}
        log("PASS", f"/health -> {status} {payload.get('status')}")

    def wait_ready(self, timeout: float = 300.0) -> dict:
        stage("Coordinator /ready (fail-closed, transient-tolerant)")
        started = time.monotonic()
        last_report = 0.0
        transient_errors = 0
        last_payload: dict = {}
        while time.monotonic() - started < timeout:
            status, payload = self._get("/ready", authenticated=True)
            if status != 200 or not payload.get("ready"):
                transient_errors += 1
            else:
                self.summary["ready"] = {
                    "status_code": status,
                    "payload": payload,
                    "transient_errors": transient_errors,
                }
                log(
                    "PASS",
                    f"/ready -> ready=true (t2i={payload.get('t2i_ready')} edit={payload.get('edit_ready')})",
                )
                return payload
            if payload:
                last_payload = payload
            now = time.monotonic() - started
            if now - last_report >= 15:
                log(
                    "HEARTBEAT",
                    f"waiting /ready elapsed={int(now)}s status={status} transient={transient_errors}",
                )
                last_report = now
            time.sleep(3)
        raise AssertionError(f"/ready did not become true within {int(timeout)}s; last={last_payload or 'no response'}")

    def _expect(self, actual: int, expected: int, label: str, checks: list[dict]) -> None:
        ok = actual == expected
        checks.append({"label": label, "expected": expected, "actual": actual, "ok": ok})
        if ok:
            log("PASS", f"{label} -> {actual}")
        else:
            log("FAIL", f"{label} -> got {actual}, expected {expected}")

    def security_cheap_checks(self) -> None:
        """Cheap no-model security acceptance, run before any GPU inference.

        These requests are rejected by FastAPI middleware, dependency or body
        validation before a worker is invoked, so they never load or run a model.
        """
        stage("Security acceptance (no model inference)")
        checks: list[dict] = []
        status, _ = self._get("/ready", authenticated=False)
        self._expect(status, 401, "reject unauthenticated /ready", checks)
        status, _ = self._get(
            "/ready",
            authenticated=True,
            headers={"Authorization": "Bearer invalid-token-000"},
        )
        self._expect(status, 401, "reject invalid bearer token on /ready", checks)
        status = self._post_json(
            "/v1/images/generations",
            {"prompt": "x", "bogus_fields_are_forbidden": True},
        )
        self._expect(status, 422, "reject unknown generation field (extra=forbid)", checks)
        status = self._post_edit_upload(b"not a real image payload", "image/jpeg", "broken.jpg")
        self._expect(status, 400, "reject malformed edit image bytes", checks)
        status = self._post_edit_upload(b"plain text", "text/plain", "note.txt")
        self._expect(status, 415, "reject unsupported edit media type", checks)
        failures = [c for c in checks if not c["ok"]]
        self.summary["security_checks"] = checks
        if failures:
            raise AssertionError(
                f"{len(failures)} security acceptance check(s) failed: " + "; ".join(c["label"] for c in failures)
            )

    def info(self) -> dict:
        stage("Coordinator /v1/info")
        status, payload = self._get("/v1/info", authenticated=True)
        if status != 200:
            raise AssertionError(f"/v1/info returned {status}")
        t2i = payload.get("t2i", {})
        edit = payload.get("edit", {})
        if t2i.get("device") != "cuda:0" or t2i.get("ready") is not True:
            raise AssertionError(f"t2i info invalid: {t2i}")
        if edit.get("device") != "cuda:1" or edit.get("ready") is not True:
            raise AssertionError(f"edit info invalid: {edit}")
        if payload.get("cpu_fallback") is not False:
            raise AssertionError(f"cpu_fallback must be false: {payload.get('cpu_fallback')}")
        self.summary["info"] = payload
        log("PASS", f"/v1/info t2i={t2i.get('device')} edit={edit.get('device')} cpu_fallback=false")
        return payload

    def _save_data_url(self, data_url: str, name: str) -> bytes:
        match = re.match(r"^data:image/(png|jpeg|jpg);base64,([A-Za-z0-9+/=]+)$", data_url)
        if not match:
            raise AssertionError(f"unexpected output data URL prefix: {data_url[:40]}...")
        raw = base64.b64decode(match.group(2))
        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return raw

    def verify_png(self, raw: bytes) -> tuple[int, int]:
        from PIL import Image

        image = Image.open(io.BytesIO(raw))
        image.load()
        if image.format != "PNG":
            raise AssertionError(f"expected PNG output, got {image.format}")
        return image.size

    @staticmethod
    def sha256(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    def t2i_generation(self) -> dict:
        stage("T2I public generation")
        payload: dict[str, Any] = {
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
            timeout=float(os.environ.get("MAGE_FLOW_ACCEPTANCE_REQUEST_TIMEOUT_SECONDS", "3600")),
        )
        wall = round(time.monotonic() - started, 3)
        if response.status_code != 200:
            raise AssertionError(f"generation returned {response.status_code}: {response.text[:500]}")
        data = response.json()
        if data.get("status") != "completed":
            raise AssertionError(f"generation not completed: {data}")
        if data.get("model") != "mage-flow-turbo":
            raise AssertionError(f"unexpected model: {data.get('model')}")
        if data.get("device") != "cuda:0":
            raise AssertionError(f"unexpected device: {data.get('device')}")
        latency = data.get("elapsed_seconds", 0.0)
        if not (latency > 0):
            raise AssertionError(f"invalid latency: {data}")
        raw = self._save_data_url(data["output"], "t2i.png")
        size = self.verify_png(raw)
        if size != (1024, 1024):
            raise AssertionError(f"unexpected T2I size: {size}")
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
        if not self.edit_source.is_file():
            raise AssertionError(f"edit source image missing: {self.edit_source}")
        source_bytes = self.edit_source.read_bytes()
        started = time.monotonic()
        response = requests.post(
            f"{self.base_url}/v1/images/edits",
            headers=self.headers,
            files={"image": (self.edit_source.name, source_bytes, "image/jpeg")},
            data={"prompt": instruction, "seed": "42"},
            timeout=float(os.environ.get("MAGE_FLOW_ACCEPTANCE_REQUEST_TIMEOUT_SECONDS", "3600")),
        )
        wall = round(time.monotonic() - started, 3)
        if response.status_code != 200:
            raise AssertionError(f"edit returned {response.status_code}: {response.text[:500]}")
        data = response.json()
        if data.get("status") != "completed":
            raise AssertionError(f"edit not completed: {data}")
        if data.get("model") != "mage-flow-edit-turbo":
            raise AssertionError(f"unexpected model: {data.get('model')}")
        if data.get("device") != "cuda:1":
            raise AssertionError(f"unexpected device: {data.get('device')}")
        latency = data.get("elapsed_seconds", 0.0)
        if not (latency > 0):
            raise AssertionError(f"invalid latency: {data}")
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
        self.security_cheap_checks()
        self.info()
        self.t2i_generation()
        self.edit_generation()
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
