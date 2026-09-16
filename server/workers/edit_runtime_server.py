from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# The standalone runtime is executed with ``python server/workers/...`` from the
# project root; expose the project package so the shared decoded-image contract
# in ``server.images`` is the single source of truth for both boundaries.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from server.images import ImageValidationError, decode_validated_worker_image  # noqa: E402
from server.safety.wiring import (  # noqa: E402
    clear_current_request_id,
    install_and_audit_safety,
    install_timing_probes,
    set_current_request_id,
    timing_report,
)

MODEL_NAME = "mage-flow-edit-turbo"
REQUIRED_DEVICE = "cuda:1"
MAX_BODY_BYTES = 16 * 1024 * 1024
EDIT_STEPS = 4
EDIT_CFG = 1.0

POLICY_EDIT_MAX_SIZE = os.environ.get("MAGE_FLOW_EDIT_MAX_SIZE", "1024")


def resolve_edit_max_size(value: str) -> int:
    """Resolve the public Edit max-size policy to the frozen value 1024."""
    if value != "1024":
        raise ValueError(f"MAGE_FLOW_EDIT_MAX_SIZE must resolve to the frozen public profile value 1024; got {value!r}")
    return int(value)


EDIT_MAX_SIZE = resolve_edit_max_size(POLICY_EDIT_MAX_SIZE)
PROMPT_TEMPLATE = "mage-flow-edit"
VL_COND_LONG_EDGE = 384


class Heartbeat:
    def __init__(self, label: str, interval_seconds: float = 30.0):
        self.label = label
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started = 0.0

    def __enter__(self):
        self.started = time.monotonic()
        self._thread = threading.Thread(target=self._run, name=f"heartbeat-{self.label}", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self):
        while not self._stop.wait(self.interval_seconds):
            elapsed = time.monotonic() - self.started
            print(f"[HEARTBEAT] stage={self.label} elapsed={elapsed:.0f}s", flush=True)


def log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def image_to_data_url(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def decode_image_bytes(image_bytes: bytes) -> Any:
    """Validate raw image bytes against the decoded-image resource contract."""
    if not image_bytes:
        raise ValueError("image payload must not be empty")
    try:
        return decode_validated_worker_image(image_bytes)
    except ImageValidationError as exc:
        raise ValueError(f"invalid image payload: {exc}") from exc


def validate_edit_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if len(prompt) > 4000:
        raise ValueError("prompt exceeds 4000 characters")

    seed = payload.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError(f"seed must be between 0 and {2**32 - 1}")

    encoded = payload.get("image_base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("image_base64 must be a non-empty base64 string")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("image_base64 is not valid base64") from exc
    if not image_bytes:
        raise ValueError("image_base64 decoded to empty bytes")

    return {"prompt": prompt, "seed": seed, "image_bytes": image_bytes}


def validate_gpu_contract(device: str) -> tuple[Any, list[str]]:
    if device != REQUIRED_DEVICE:
        raise RuntimeError(f"Edit worker must use {REQUIRED_DEVICE}; got {device}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Edit worker refuses CPU fallback")
    count = torch.cuda.device_count()
    if count < 2:
        raise RuntimeError(f"Kaggle T4 x2 contract requires at least 2 CUDA devices; found {count}")

    names = [torch.cuda.get_device_name(i) for i in range(count)]
    if "T4" not in names[0].upper() or "T4" not in names[1].upper():
        raise RuntimeError(f"Expected NVIDIA T4 x2 at cuda:0/cuda:1; detected {names[:2]}")

    torch.cuda.set_device(1)
    return torch, names


class RuntimeState:
    def __init__(self, model_path: str, device: str):
        self.model_path = os.path.realpath(model_path)
        self.device = device
        self.pipeline: Any = None
        self.ready = False
        self.gpu_names: list[str] = []
        self.edit_lock = threading.Lock()
        self.safety_adapter: Any = None

    def load(self) -> None:
        model_dir = Path(self.model_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Edit model path does not exist: {self.model_path}")
        required = [
            model_dir / "model_index.json",
            model_dir / "transformer" / "diffusion_pytorch_model.safetensors",
            model_dir / "transformer" / "config.json",
            model_dir / "scheduler" / "scheduler_config.json",
            model_dir / "vae" / "diffusion_pytorch_model.safetensors",
            model_dir / "text_encoder" / "config.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Edit model mount is incomplete; missing: {missing}")

        log("INFO", f"validating GPU contract device={self.device}")
        _, names = validate_gpu_contract(self.device)
        self.gpu_names = names
        log("PASS", f"GPU contract cuda:0={names[0]!r} cuda:1={names[1]!r}")

        from mage_flow import MageFlowPipeline

        log("INFO", f"loading {MODEL_NAME} from local path: {self.model_path}")
        with Heartbeat("edit_model_load"):
            self.pipeline = MageFlowPipeline.from_pretrained(self.model_path, device="cpu")

        log("INFO", f"moving DiT transformer + VAE to {self.device}; keeping text encoder on CPU (T4 14.56GiB split)")
        with Heartbeat("edit_gpu_split"):
            self.pipeline.model.transformer.to(self.device)
            if self.pipeline.model.vae is not None:
                self.pipeline.model.vae.to(self.device)
        self.pipeline.device = self.device

        from mage_flow.models.modules._attn_backend import set_attn_backend

        set_attn_backend("sdpa")
        log("INFO", "attention backend set to sdpa (flash-attn not installed)")

        self.safety_adapter = install_and_audit_safety(self.pipeline, request_type="EDIT")
        log("PASS", "EDIT_GGUF_SAFETY_INSTALLED backend=gguf")
        install_timing_probes("EDIT")

        self.ready = True
        log("PASS", f"EDIT_WORKER_READY model={MODEL_NAME} device={self.device}")

    def edit(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.ready or self.pipeline is None:
            raise RuntimeError("Edit runtime is not ready")
        request = validate_edit_payload(payload)
        image = decode_image_bytes(request["image_bytes"])

        started = time.monotonic()
        log(
            "INFO",
            "edit start "
            f"seed={request['seed']} steps={EDIT_STEPS} cfg={EDIT_CFG} "
            f"device={self.device} source_size={image.size}",
        )
        with Heartbeat("edit_generate"):
            set_current_request_id(f"edit_{request['seed']}")
            try:
                images = self.pipeline.edit(
                    [request["prompt"]],
                    [image],
                    neg_prompts=[" "],
                    seeds=[request["seed"]],
                    steps=EDIT_STEPS,
                    cfg=EDIT_CFG,
                    prompt_template=PROMPT_TEMPLATE,
                    vl_cond_long_edge=VL_COND_LONG_EDGE,
                    max_size=EDIT_MAX_SIZE,
                )
            finally:
                clear_current_request_id()
        if not isinstance(images, list) or len(images) != 1 or images[0] is None:
            raise RuntimeError("MageFlowPipeline.edit returned an unexpected result")

        result_image = images[0]
        width, height = result_image.size
        elapsed = time.monotonic() - started
        timing_report("EDIT")
        log("PASS", f"edit completed elapsed={elapsed:.3f}s actual_size={width}x{height}")
        return {
            "id": f"img_{uuid.uuid4().hex}",
            "status": "completed",
            "model": MODEL_NAME,
            "device": self.device,
            "seed": request["seed"],
            "width": width,
            "height": height,
            "elapsed_seconds": round(elapsed, 3),
            "output": image_to_data_url(result_image),
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "MageFlowEditRuntime/0.1"

    def log_message(self, format: str, *args: object) -> None:
        log("HTTP", format % args)

    @property
    def runtime(self) -> RuntimeState:
        return self.server.runtime_state  # type: ignore[attr-defined]

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._write_json(404, {"detail": "not found"})
            return
        self._write_json(
            200,
            {
                "status": "ready" if self.runtime.ready else "not_ready",
                "ready": self.runtime.ready,
                "model": MODEL_NAME,
                "device": self.runtime.device,
                "gpu_names": self.runtime.gpu_names[:2],
                "model_path": self.runtime.model_path,
                "pid": os.getpid(),
            },
        )

    def do_POST(self) -> None:
        if self.path != "/edit":
            self._write_json(404, {"detail": "not found"})
            return
        if not self.runtime.ready:
            self._write_json(503, {"detail": "Edit runtime is not ready"})
            return

        if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
            self._write_json(415, {"detail": "expected Content-Type application/json"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._write_json(400, {"detail": "invalid Content-Length"})
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._write_json(413 if length > MAX_BODY_BYTES else 400, {"detail": "invalid request body size"})
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._write_json(400, {"detail": str(exc)})
            return

        if not self.runtime.edit_lock.acquire(blocking=False):
            self._write_json(409, {"detail": "Edit generation is already running"})
            return
        try:
            result = self.runtime.edit(payload)
        except ValueError as exc:
            self._write_json(400, {"detail": str(exc)})
            return
        except Exception as exc:
            log("FAIL", f"edit error: {type(exc).__name__}: {exc}")
            self._write_json(500, {"detail": f"Edit generation failed: {type(exc).__name__}: {exc}"})
            return
        finally:
            self.runtime.edit_lock.release()

        self._write_json(200, result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Localhost-only Mage-Flow Edit Turbo runtime worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default=REQUIRED_DEVICE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.host != "127.0.0.1":
        raise SystemExit("[FAIL] internal Edit worker must bind to 127.0.0.1 only")

    print("=" * 60, flush=True)
    log("STAGE", "Mage-Flow Edit Turbo runtime worker")
    print("=" * 60, flush=True)
    log("INFO", f"bind={args.host}:{args.port}")
    log("INFO", f"device={args.device}")
    log("INFO", f"model_path={os.path.realpath(args.model_path)}")

    state = RuntimeState(args.model_path, args.device)
    try:
        state.load()
    except Exception as exc:
        log("FAIL", f"EDIT_WORKER_START {type(exc).__name__}: {exc}")
        raise

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.runtime_state = state  # type: ignore[attr-defined]
    log("PASS", f"EDIT_INTERNAL_HTTP_READY http://{args.host}:{args.port}")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        log("INFO", "Edit runtime worker stopped")


if __name__ == "__main__":
    main()
