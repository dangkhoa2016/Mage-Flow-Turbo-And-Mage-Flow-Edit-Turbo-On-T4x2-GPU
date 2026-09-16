"""Full coordinator-facing worker response contract validation.

Every candidate response from an internal worker is validated strictly before
it reaches the public surface, so internal contract drift becomes a controlled
502 gateway failure instead of an uncaught 500. The public 502 detail is always
generic; diagnostics are logged server-side only.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Any

from PIL import Image

MODEL_T2I = "mage-flow-turbo"
MODEL_EDIT = "mage-flow-edit-turbo"
DEVICE_T2I = "cuda:0"
DEVICE_EDIT = "cuda:1"
T2I_WIDTH = 1024
T2I_HEIGHT = 1024
OUTPUT_DATA_URL_PREFIX = "data:image/png;base64,"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SEED_MAX = 2**32 - 1


class WorkerResponseViolation(RuntimeError):
    """Raised when an internal worker violates the coordinator-facing contract."""


def _require(condition: bool, worker: str, field: str, value: Any) -> None:
    if not condition:
        raise WorkerResponseViolation(f"{worker} worker violated the response contract: {field}={value!r}")


def _check_common_fields(value: dict, worker: str, model: str, device: str) -> None:
    _require(isinstance(value, dict), worker, "response type", value)
    _require(value.get("status") == "completed", worker, "status", value.get("status"))
    _require(value.get("model") == model, worker, "model", value.get("model"))
    _require(value.get("device") == device, worker, "device", value.get("device"))

    request_id = value.get("id")
    _require(
        bool(isinstance(request_id, str) and request_id),
        worker,
        "id",
        request_id,
    )
    seed = value.get("seed")
    _require(
        isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed <= SEED_MAX,
        worker,
        "seed",
        seed,
    )
    elapsed = value.get("elapsed_seconds")
    _require(
        isinstance(elapsed, (int, float))
        and not isinstance(elapsed, bool)
        and elapsed >= 0
        and not math.isnan(elapsed)
        and not math.isinf(elapsed),
        worker,
        "elapsed_seconds",
        elapsed,
    )
    output = value.get("output")
    _require(
        isinstance(output, str)
        and output.startswith(OUTPUT_DATA_URL_PREFIX)
        and len(output) > len(OUTPUT_DATA_URL_PREFIX),
        worker,
        "output",
        "expected a PNG data URL" if isinstance(output, str) else type(output).__name__,
    )
    assert isinstance(output, str)
    _check_output_png(output, worker)


def _check_output_png(data_url: str, worker: str) -> None:
    """Prove the data-URL payload is strict Base64 and valid PNG content.

    The error diagnostics never embed the decoded bytes or the payload, so no
    worker output can leak into the public error surface.
    """
    payload = data_url[len(OUTPUT_DATA_URL_PREFIX) :]
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (ValueError, TypeError):
        raise WorkerResponseViolation(
            f"{worker} worker violated the response contract: output payload is not strict Base64"
        ) from None
    if not decoded:
        raise WorkerResponseViolation(f"{worker} worker violated the response contract: output payload is empty")
    if not decoded.startswith(PNG_SIGNATURE):
        raise WorkerResponseViolation(f"{worker} worker violated the response contract: output is not a PNG image")
    try:
        with Image.open(io.BytesIO(decoded)) as image:
            image.verify()
    except Exception:
        raise WorkerResponseViolation(
            f"{worker} worker violated the response contract: output is not a decodable PNG image"
        ) from None


def validate_t2i_worker_response(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkerResponseViolation(
            f"{MODEL_T2I} worker violated the response contract: response type={type(value).__name__}"
        )
    _check_common_fields(value, MODEL_T2I, MODEL_T2I, DEVICE_T2I)
    _require(value.get("width") == T2I_WIDTH, MODEL_T2I, "width", value.get("width"))
    _require(value.get("height") == T2I_HEIGHT, MODEL_T2I, "height", value.get("height"))
    return value


def validate_edit_worker_response(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkerResponseViolation(
            f"{MODEL_EDIT} worker violated the response contract: response type={type(value).__name__}"
        )
    _check_common_fields(value, MODEL_EDIT, MODEL_EDIT, DEVICE_EDIT)
    return value
