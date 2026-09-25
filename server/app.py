from __future__ import annotations

import functools
import logging
import os
import threading

import anyio
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from . import __version__
from .auth import require_bearer_token
from .health import ServiceState
from .images import (
    ImageValidationError,
    decode_validated_image,
    validate_declared_media_type,
)
from .middleware import (
    MAX_EDIT_REQUEST_BODY_BYTES,
    MAX_T2I_REQUEST_BODY_BYTES,
    RequestBodyLimitMiddleware,
)
from .schemas import EditResponse, GenerationRequest, GenerationResponse
from .worker_contract import (
    validate_edit_worker_response,
    validate_t2i_worker_response,
)
from .workers.edit import EditWorkerClient
from .workers.t2i import T2IWorkerClient

MAX_PUBLIC_UPLOAD_BYTES = 8 * 1024 * 1024

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mage-Flow-Turbo and Mage-Flow-Edit-Turbo on T4x2 GPU",
    version=__version__,
)
app.add_middleware(
    RequestBodyLimitMiddleware,
    routes={
        ("POST", "/v1/images/edits"): MAX_EDIT_REQUEST_BODY_BYTES,
        ("POST", "/v1/images/generations"): MAX_T2I_REQUEST_BODY_BYTES,
    },
)


@app.middleware("http")
async def add_response_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


app.state.service_state = ServiceState()
app.state.t2i_worker = T2IWorkerClient()
app.state.edit_worker = EditWorkerClient()
# One independent GPU admission gate per model lane: each lane allows a single
# active generation/edit, and T2I/Edit lanes are independent (separate GPUs).
app.state.t2i_gate = threading.Lock()
app.state.edit_gate = threading.Lock()
# full T2I conditioning temporarily uses cuda:1, where the Edit worker model is
# resident. When full is enabled, public T2I/Edit requests must therefore share
# one cross-lane admission gate; otherwise the original independent-lane
# behavior is preserved.
app.state.full_shared_gpu_gate = threading.Lock()


def _acquire_gate(gate: threading.Lock, detail: str) -> None:
    if not gate.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=detail,
            headers={"Retry-After": "1"},
        )


def _full_cross_lane_serialization_enabled() -> bool:
    return os.environ.get("MAGE_T2I_CONDITIONING_OFFLOAD", "off").strip().lower() == "full"


def _acquire_request_gate(lane: str) -> threading.Lock:
    if _full_cross_lane_serialization_enabled():
        gate = app.state.full_shared_gpu_gate
        _acquire_gate(gate, "GPU residency is busy (full cross-lane serialization)")
        return gate
    if lane == "t2i":
        gate = app.state.t2i_gate
        _acquire_gate(gate, "T2I worker is busy")
        return gate
    if lane == "edit":
        gate = app.state.edit_gate
        _acquire_gate(gate, "Edit worker is busy")
        return gate
    raise RuntimeError(f"unknown request lane: {lane}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "coordinator": "healthy"}


@app.get("/ready", dependencies=[Depends(require_bearer_token)])
def ready() -> dict:
    state: ServiceState = app.state.service_state
    t2i_worker = app.state.t2i_worker
    edit_worker = app.state.edit_worker
    state.t2i_ready = bool(t2i_worker is not None and getattr(t2i_worker, "ready", False))
    state.edit_ready = bool(edit_worker is not None and getattr(edit_worker, "ready", False))
    payload = state.as_dict()
    payload["status"] = "ready" if state.ready else "not_ready"
    return payload


@app.get("/v1/info", dependencies=[Depends(require_bearer_token)])
def info() -> dict:
    t2i_worker = app.state.t2i_worker
    edit_worker = app.state.edit_worker
    return {
        "project": "Mage-Flow-Turbo-And-Mage-Flow-Edit-Turbo-On-T4x2-GPU",
        "runtime_target": "Kaggle NVIDIA T4 x2",
        "t2i": {
            "model": "mage-flow-turbo",
            "device": "cuda:0",
            "ready": bool(t2i_worker is not None and getattr(t2i_worker, "ready", False)),
            "transport": "localhost-internal-http",
            "single_flight": True,
        },
        "edit": {
            "model": "mage-flow-edit-turbo",
            "device": "cuda:1",
            "ready": bool(edit_worker is not None and getattr(edit_worker, "ready", False)),
            "transport": "localhost-internal-http",
            "single_flight": True,
        },
        "cpu_fallback": False,
        "conditioning_offload": {
            "mode": os.environ.get("MAGE_T2I_CONDITIONING_OFFLOAD", "off").strip().lower(),
            "cross_lane_serialized": _full_cross_lane_serialization_enabled(),
        },
        "production_claim": "public Kaggle workflow; not SLA-backed, HA, or multi-tenant",
    }


def _worker_request_failed(worker_name: str, exc: Exception) -> HTTPException:
    logger.warning("%s worker request failed: %s", worker_name, exc)
    return HTTPException(status_code=502, detail=f"{worker_name} worker request failed")


@app.post("/v1/images/generations", response_model=GenerationResponse, dependencies=[Depends(require_bearer_token)])
def generate(request: GenerationRequest) -> GenerationResponse:
    worker = app.state.t2i_worker
    if worker is None or not getattr(worker, "ready", False):
        raise HTTPException(status_code=503, detail="T2I worker is not ready")
    request_gate = _acquire_request_gate("t2i")
    try:
        result = worker.generate(**request.model_dump())
        validate_t2i_worker_response(result)
        return GenerationResponse(**result)
    except RuntimeError as exc:
        raise _worker_request_failed("T2I", exc) from exc
    finally:
        request_gate.release()


@app.post("/v1/images/edits", response_model=EditResponse, dependencies=[Depends(require_bearer_token)])
async def edit(
    image: UploadFile = File(...),
    prompt: str = Form(..., min_length=1, max_length=4000),
    seed: int = Form(42, ge=0, le=2**32 - 1),
) -> EditResponse:
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must be a non-empty string")
    worker = app.state.edit_worker
    if worker is None or not getattr(worker, "ready", False):
        raise HTTPException(status_code=503, detail="Edit worker is not ready")
    try:
        validate_declared_media_type(image.content_type)
    except ImageValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    image_bytes = await image.read(MAX_PUBLIC_UPLOAD_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="image file is empty")
    if len(image_bytes) > MAX_PUBLIC_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="image upload exceeds the public byte limit",
        )
    try:
        decode_validated_image(image_bytes, image.content_type)
    except ImageValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    request_gate = _acquire_request_gate("edit")
    try:
        result = await anyio.to_thread.run_sync(
            functools.partial(
                worker.edit,
                image_bytes=image_bytes,
                prompt=prompt,
                seed=seed,
            )
        )
        validate_edit_worker_response(result)
        return EditResponse(**result)
    except RuntimeError as exc:
        raise _worker_request_failed("Edit", exc) from exc
    finally:
        request_gate.release()
