from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


MODEL_NAME = "mage-flow-turbo"
REQUIRED_DEVICE = "cuda:0"
MAX_BODY_BYTES = 64 * 1024
# GPU-qualified frozen public acceptance profile. The internal runtime enforces
# the exact same values the public schema accepts, as defense in depth.
FROZEN_STEPS = 4
FROZEN_WIDTH = 1024
FROZEN_HEIGHT = 1024


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


def validate_generation_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if len(prompt) > 4000:
        raise ValueError("prompt exceeds 4000 characters")

    def integer(name: str, default: int, low: int, high: int) -> int:
        value = payload.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
        return value

    seed = integer("seed", 42, 0, 2**32 - 1)
    steps = integer("steps", FROZEN_STEPS, FROZEN_STEPS, FROZEN_STEPS)
    width = integer("width", FROZEN_WIDTH, FROZEN_WIDTH, FROZEN_WIDTH)
    height = integer("height", FROZEN_HEIGHT, FROZEN_HEIGHT, FROZEN_HEIGHT)

    return {
        "prompt": prompt,
        "seed": seed,
        "steps": steps,
        "width": width,
        "height": height,
    }


def image_to_data_url(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def validate_gpu_contract(device: str):
    if device != REQUIRED_DEVICE:
        raise RuntimeError(f"T2I worker must use {REQUIRED_DEVICE}; got {device}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; T2I worker refuses CPU fallback")
    count = torch.cuda.device_count()
    if count < 2:
        raise RuntimeError(f"Kaggle T4 x2 contract requires at least 2 CUDA devices; found {count}")

    names = [torch.cuda.get_device_name(i) for i in range(count)]
    if "T4" not in names[0].upper() or "T4" not in names[1].upper():
        raise RuntimeError(f"Expected NVIDIA T4 x2 at cuda:0/cuda:1; detected {names[:2]}")

    torch.cuda.set_device(0)
    return torch, names


class RuntimeState:
    def __init__(self, model_path: str, device: str):
        self.model_path = os.path.realpath(model_path)
        self.device = device
        self.pipeline = None
        self.ready = False
        self.gpu_names: list[str] = []
        self.generate_lock = threading.Lock()

    def load(self) -> None:
        model_dir = Path(self.model_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"T2I model path does not exist: {self.model_path}")
        required = [
            model_dir / "model_index.json",
            model_dir / "transformer" / "diffusion_pytorch_model.safetensors",
            model_dir / "transformer" / "config.json",
            model_dir / "scheduler" / "scheduler_config.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"T2I model mount is incomplete; missing: {missing}")

        log("INFO", f"validating GPU contract device={self.device}")
        _, names = validate_gpu_contract(self.device)
        self.gpu_names = names
        log("PASS", f"GPU contract cuda:0={names[0]!r} cuda:1={names[1]!r}")

        from mage_flow import MageFlowPipeline

        log("INFO", f"loading {MODEL_NAME} from local path: {self.model_path}")
        with Heartbeat("t2i_model_load"):
            self.pipeline = MageFlowPipeline.from_pretrained(self.model_path, device="cpu")

        log("INFO", f"moving DiT transformer + VAE to {self.device}; keeping text encoder on CPU (T4 14.56GiB split)")
        with Heartbeat("t2i_gpu_split"):
            self.pipeline.model.transformer.to(self.device)
            if self.pipeline.model.vae is not None:
                self.pipeline.model.vae.to(self.device)
        self.pipeline.device = self.device

        from mage_flow.models.modules._attn_backend import set_attn_backend
        set_attn_backend("sdpa")
        log("INFO", "attention backend set to sdpa (flash-attn not installed)")

        self.ready = True
        log("PASS", f"T2I_WORKER_READY model={MODEL_NAME} device={self.device}")

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.ready or self.pipeline is None:
            raise RuntimeError("T2I runtime is not ready")
        request = validate_generation_payload(payload)

        started = time.monotonic()
        log(
            "INFO",
            "generation start "
            f"seed={request['seed']} steps={request['steps']} "
            f"size={request['width']}x{request['height']} device={self.device}",
        )
        with Heartbeat("t2i_generate"):
                images = self.pipeline.generate(
                    [request["prompt"]],
                    neg_prompts=[" "],
                    seeds=[request["seed"]],
                    heights=[request["height"]],
                    widths=[request["width"]],
                    steps=request["steps"],
                    cfg=1.0,
                    prompt_template="mage-flow",
                )
        if not isinstance(images, list) or len(images) != 1 or images[0] is None:
            raise RuntimeError("MageFlowPipeline.generate returned an unexpected result")

        image = images[0]
        width, height = image.size
        elapsed = time.monotonic() - started
        log("PASS", f"generation completed elapsed={elapsed:.3f}s actual_size={width}x{height}")
        return {
            "id": f"img_{uuid.uuid4().hex}",
            "status": "completed",
            "model": MODEL_NAME,
            "device": self.device,
            "seed": request["seed"],
            "width": width,
            "height": height,
            "elapsed_seconds": round(elapsed, 3),
            "output": image_to_data_url(image),
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "MageFlowT2IRuntime/0.1"

    def log_message(self, format: str, *args) -> None:
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
        if self.path != "/generate":
            self._write_json(404, {"detail": "not found"})
            return
        if not self.runtime.ready:
            self._write_json(503, {"detail": "T2I runtime is not ready"})
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

        if not self.runtime.generate_lock.acquire(blocking=False):
            self._write_json(409, {"detail": "T2I generation is already running"})
            return
        try:
            result = self.runtime.generate(payload)
        except ValueError as exc:
            self._write_json(400, {"detail": str(exc)})
            return
        except Exception as exc:
            log("FAIL", f"generation error: {type(exc).__name__}: {exc}")
            self._write_json(500, {"detail": f"T2I generation failed: {type(exc).__name__}: {exc}"})
            return
        finally:
            self.runtime.generate_lock.release()

        self._write_json(200, result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Localhost-only Mage-Flow Turbo T2I runtime worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default=REQUIRED_DEVICE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("[FAIL] internal T2I worker must bind to localhost only")

    print("=" * 60, flush=True)
    log("STAGE", "Mage-Flow Turbo T2I runtime worker")
    print("=" * 60, flush=True)
    log("INFO", f"bind={args.host}:{args.port}")
    log("INFO", f"device={args.device}")
    log("INFO", f"model_path={os.path.realpath(args.model_path)}")

    state = RuntimeState(args.model_path, args.device)
    try:
        state.load()
    except Exception as exc:
        log("FAIL", f"T2I_WORKER_START {type(exc).__name__}: {exc}")
        raise

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.runtime_state = state  # type: ignore[attr-defined]
    log("PASS", f"T2I_INTERNAL_HTTP_READY http://{args.host}:{args.port}")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        log("INFO", "T2I runtime worker stopped")


if __name__ == "__main__":
    main()
