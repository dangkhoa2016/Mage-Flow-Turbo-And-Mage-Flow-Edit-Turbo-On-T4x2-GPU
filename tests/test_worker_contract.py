from __future__ import annotations

import pytest

from server.worker_contract import (
    OUTPUT_DATA_URL_PREFIX,
    WorkerResponseViolation,
    validate_edit_worker_response,
    validate_t2i_worker_response,
)


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
        "output": f"{OUTPUT_DATA_URL_PREFIX}AAAA",
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
        "output": f"{OUTPUT_DATA_URL_PREFIX}BBBB",
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