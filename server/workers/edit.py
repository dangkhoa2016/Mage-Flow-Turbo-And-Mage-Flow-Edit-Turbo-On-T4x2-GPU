from __future__ import annotations

import base64
import json
import os
import subprocess
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from ..config import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    parse_timeout_seconds,
    validate_loopback_http_url,
)
from ..worker_contract import validate_edit_worker_response

DEFAULT_INTERNAL_URL = "http://127.0.0.1:8102"
# Public runtime-root abstraction. MAGE_FLOW_RUNTIME_ROOT overrides the default;
# the default below is the internal Kaggle path used by the already-qualified demo,
# NOT a new runtime qualification.
DEFAULT_RUNTIME_ROOT = "/kaggle/working/mage-flow-v5-t4x2-c1-concurrency-source-20260912"
DEFAULT_MODEL_PATH = "/kaggle/input/models/dangkhoa2016/mage-flow-community-mage-flow-edit-turbo/pytorch/default/1"
DEFAULT_DEVICE = "cuda:1"
# A 1024x1024 PNG output can exceed the error-body budget; success responses are
# bounded separately from compact error responses.
MAX_WORKER_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_WORKER_ERROR_RESPONSE_BYTES = 64 * 1024


@dataclass(frozen=True)
class EditWorkerConfig:
    internal_url: str = DEFAULT_INTERNAL_URL
    runtime_root: str = DEFAULT_RUNTIME_ROOT
    model_path: str = DEFAULT_MODEL_PATH
    device: str = DEFAULT_DEVICE
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "internal_url",
            validate_loopback_http_url(self.internal_url, name="MAGE_FLOW_EDIT_INTERNAL_URL"),
        )
        object.__setattr__(
            self,
            "request_timeout_seconds",
            parse_timeout_seconds(
                self.request_timeout_seconds,
                name="MAGE_FLOW_EDIT_REQUEST_TIMEOUT_SECONDS",
            ),
        )

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None) -> EditWorkerConfig:
        values = os.environ if env is None else env
        return cls(
            internal_url=values.get("MAGE_FLOW_EDIT_INTERNAL_URL", DEFAULT_INTERNAL_URL),
            runtime_root=values.get("MAGE_FLOW_RUNTIME_ROOT", DEFAULT_RUNTIME_ROOT),
            model_path=values.get("MAGE_FLOW_EDIT_MODEL_PATH", DEFAULT_MODEL_PATH),
            device=values.get("MAGE_FLOW_EDIT_DEVICE", DEFAULT_DEVICE),
            request_timeout_seconds=parse_timeout_seconds(
                values.get(
                    "MAGE_FLOW_EDIT_REQUEST_TIMEOUT_SECONDS",
                    DEFAULT_REQUEST_TIMEOUT_SECONDS,
                ),
                name="MAGE_FLOW_EDIT_REQUEST_TIMEOUT_SECONDS",
            ),
        )


class EditWorkerClient:
    """HTTP client for the localhost-only Mage-Flow Edit Turbo runtime worker.

    The coordinator intentionally does not import Mage/PyTorch. The heavy runtime
    lives in the validated restored virtual environment and exposes a tiny
    localhost-only internal API.
    """

    def __init__(self, config: EditWorkerConfig | None = None):
        self.config = config or EditWorkerConfig.from_environment()
        if self.config.device != DEFAULT_DEVICE:
            raise ValueError("Edit public routing contract requires device cuda:1")

    @property
    def device(self) -> str:
        return self.config.device

    @property
    def ready(self) -> bool:
        try:
            payload = self._request_json("GET", "/health", timeout=1.5)
        except (OSError, RuntimeError, ValueError):
            return False
        return (
            payload.get("status") == "ready"
            and payload.get("ready") is True
            and payload.get("device") == DEFAULT_DEVICE
            and payload.get("model") == "mage-flow-edit-turbo"
            and payload.get("model_path") == os.path.realpath(self.config.model_path)
        )

    def edit(self, *, image_bytes: bytes, prompt: str, seed: int) -> dict:
        if not isinstance(image_bytes, bytes) or not image_bytes:
            raise ValueError("image_bytes must be non-empty bytes")
        payload = {
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            "prompt": prompt,
            "seed": seed,
        }
        result = self._request_json(
            "POST",
            "/edit",
            payload=payload,
            timeout=self.config.request_timeout_seconds,
        )
        return validate_edit_worker_response(result)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict | None = None,
        timeout: float,
    ) -> dict:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            f"{self.config.internal_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read(MAX_WORKER_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            raw = exc.read(MAX_WORKER_ERROR_RESPONSE_BYTES + 1)
            body = raw.decode("utf-8", errors="replace")
            try:
                detail = json.loads(body).get("detail", body)
            except json.JSONDecodeError:
                detail = body
            raise RuntimeError(f"Edit worker HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise RuntimeError(f"Edit worker unavailable at {self.config.internal_url}: {exc}") from exc

        if len(raw) > MAX_WORKER_RESPONSE_BYTES:
            raise RuntimeError("Edit worker response exceeds the configured size bound")
        body = raw.decode("utf-8", errors="replace")

        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Edit worker returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise RuntimeError("Edit worker returned a non-object JSON payload")
        return result


def build_edit_worker_command(
    *,
    project_root: str | Path,
    config: EditWorkerConfig | None = None,
    host: str = "127.0.0.1",
    port: int = 8102,
) -> list[str]:
    """Build the exact child-process command without starting a model."""

    cfg = config or EditWorkerConfig.from_environment()
    if cfg.device != DEFAULT_DEVICE:
        raise ValueError("Edit public routing contract requires device cuda:1")

    runtime_python = Path(cfg.runtime_root) / ".venv" / "bin" / "python"
    runtime_server = Path(project_root) / "server" / "workers" / "edit_runtime_server.py"
    return [
        str(runtime_python),
        str(runtime_server),
        "--host",
        host,
        "--port",
        str(port),
        "--model-path",
        cfg.model_path,
        "--device",
        cfg.device,
    ]


def start_edit_worker_process(
    *,
    project_root: str | Path,
    config: EditWorkerConfig | None = None,
    stdout: IO[Any] | None = None,
    stderr: IO[Any] | None = None,
) -> subprocess.Popen:
    """Start the worker subprocess; readiness is checked separately by the caller."""

    cfg = config or EditWorkerConfig.from_environment()
    command = build_edit_worker_command(project_root=project_root, config=cfg)
    runtime_python = Path(command[0])
    runtime_server = Path(command[1])
    if not runtime_python.is_file():
        raise FileNotFoundError(f"validated runtime Python not found: {runtime_python}")
    if not runtime_server.is_file():
        raise FileNotFoundError(f"Edit runtime server not found: {runtime_server}")
    if not Path(cfg.model_path).is_dir():
        raise FileNotFoundError(f"Edit model directory not found: {cfg.model_path}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(command, cwd=str(project_root), env=env, stdout=stdout, stderr=stderr)
