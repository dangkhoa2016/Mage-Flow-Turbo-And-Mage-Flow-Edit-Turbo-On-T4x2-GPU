from __future__ import annotations

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
from .workers.edit import EditWorkerClient
from .workers.t2i import T2IWorkerClient

MAX_PUBLIC_UPLOAD_BYTES = 8 * 1024 * 1024

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
app.state.service_state = ServiceState()
app.state.t2i_worker = T2IWorkerClient()
app.state.edit_worker = EditWorkerClient()


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
        "project": "mage-flow-turbo-and-mage-flow-edit-turbo-on-t4x2-gpu",
        "runtime_target": "Kaggle NVIDIA T4 x2",
        "t2i": {
            "model": "mage-flow-turbo",
            "device": "cuda:0",
            "ready": bool(t2i_worker is not None and getattr(t2i_worker, "ready", False)),
            "transport": "localhost-internal-http",
        },
        "edit": {
            "model": "mage-flow-edit-turbo",
            "device": "cuda:1",
            "ready": bool(edit_worker is not None and getattr(edit_worker, "ready", False)),
            "transport": "localhost-internal-http",
        },
        "cpu_fallback": False,
        "production_claim": "production-style demo; not SLA-backed, HA, or multi-tenant",
    }


@app.post("/v1/images/generations", response_model=GenerationResponse, dependencies=[Depends(require_bearer_token)])
def generate(request: GenerationRequest) -> GenerationResponse:
    worker = app.state.t2i_worker
    if worker is None or not getattr(worker, "ready", False):
        raise HTTPException(status_code=503, detail="T2I worker is not ready")
    try:
        result = worker.generate(**request.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"T2I worker request failed: {exc}") from exc
    return GenerationResponse(**result)


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
    try:
        result = worker.edit(image_bytes=image_bytes, prompt=prompt, seed=seed)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"Edit worker request failed: {exc}") from exc
    return EditResponse(**result)
