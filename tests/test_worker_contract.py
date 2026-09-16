from __future__ import annotations

import base64
import io

import pytest
from PIL import Image
from server.worker_contract import (
    OUTPUT_DATA_URL_PREFIX,
    WorkerResponseViolation,
    validate_edit_worker_response,
    validate_t2i_worker_response,
)


def _png_data_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"{OUTPUT_DATA_URL_PREFIX}{payload}"


def _non_png_data_url() -> str:
    payload = base64.b64encode(b"plain bytes that are not a PNG").decode("ascii")
    return f"{OUTPUT_DATA_URL_PREFIX}{payload}"


def _valid_t2i_response(**overrides) -> dict:
    payload = {
        "id": "img_1",
        "status": "completed",
        "model": "mage-flow-turbo",
        "device": "cuda:0",
        "seed": 42,
        "width": 1024,
        "height": 1024,
        "elapsed_seconds": 1.2,
        "output": _png_data_url(),
    }
    payload.update(overrides)
    return payload


def _valid_edit_response(**overrides) -> dict:
    payload = {
        "id": "img_2",
        "status": "completed",
        "model": "mage-flow-edit-turbo",
        "device": "cuda:1",
        "seed": 7,
        "elapsed_seconds": 1.2,
        "output": _png_data_url(),
    }
    payload.update(overrides)
    return payload


def test_contract_accepts_valid_responses():
    assert validate_t2i_worker_response(_valid_t2i_response())["model"] == "mage-flow-turbo"
    assert validate_edit_worker_response(_valid_edit_response())["model"] == "mage-flow-edit-turbo"


@pytest.mark.parametrize(
    "override",
    [
        {"status": "running"},
        {"model": "unexpected-model"},
        {"device": "cuda:1"},
        {"id": ""},
        {"seed": -1},
        {"seed": True},
        {"seed": "42"},
        {"width": 512},
        {"height": 2048},
        {"elapsed_seconds": -0.5},
        {"elapsed_seconds": float("nan")},
        {"elapsed_seconds": "1.2"},
        {"output": "not-a-data-url"},
        {"output": f"{OUTPUT_DATA_URL_PREFIX}"},
        {"output": f"{OUTPUT_DATA_URL_PREFIX}@@@not-base64@@@"},
        {"output": _non_png_data_url()},
        {"output": "data:image/jpeg;base64," + _png_data_url().split(",", 1)[1]},
        {"output": 123},
        {"output": None},
        {"output": b"data:image/png;base64,AAAA"},
    ],
)
def test_t2i_contract_rejects_invalid_field(override):
    with pytest.raises(WorkerResponseViolation):
        validate_t2i_worker_response(_valid_t2i_response(**override))


@pytest.mark.parametrize(
    "override",
    [
        {"status": "failed"},
        {"model": "mage-flow-turbo"},
        {"device": "cuda:0"},
        {"id": ""},
        {"seed": -1},
        {"seed": True},
        {"seed": "42"},
        {"elapsed_seconds": -0.5},
        {"elapsed_seconds": float("inf")},
        {"elapsed_seconds": "1.2"},
        {"output": "not-a-data-url"},
        {"output": ""},
        {"output": f"{OUTPUT_DATA_URL_PREFIX}"},
        {"output": f"{OUTPUT_DATA_URL_PREFIX}___not_base64___"},
        {"output": _non_png_data_url()},
        {"output": "data:image/webp;base64," + _png_data_url().split(",", 1)[1]},
        {"output": 42},
        {"output": ["data:image/png;base64,AAAA"]},
    ],
)
def test_edit_contract_rejects_invalid_field(override):
    with pytest.raises(WorkerResponseViolation):
        validate_edit_worker_response(_valid_edit_response(**override))


@pytest.mark.parametrize("bad", [[], "json", 42, None, ("tuple",)])
def test_contract_rejects_non_object_responses(bad):
    with pytest.raises(WorkerResponseViolation):
        validate_t2i_worker_response(bad)
    with pytest.raises(WorkerResponseViolation):
        validate_edit_worker_response(bad)


def test_contract_missing_output_rejected():
    payload = _valid_t2i_response()
    del payload["output"]
    with pytest.raises(WorkerResponseViolation):
        validate_t2i_worker_response(payload)


def test_png_violation_does_not_leak_output_bytes():
    secret = base64.b64encode(b"x" * 400).decode("ascii")
    with pytest.raises(WorkerResponseViolation) as excinfo:
        validate_t2i_worker_response(_valid_t2i_response(output=f"{OUTPUT_DATA_URL_PREFIX}{secret}"))
    message = str(excinfo.value)
    assert secret not in message
    assert "x" * 400 not in message
