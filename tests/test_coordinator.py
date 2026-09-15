from __future__ import annotations

import io
import threading
import time

import pytest
from PIL import Image

from server.worker_contract import (
    OUTPUT_DATA_URL_PREFIX,
    WorkerResponseViolation,
    validate_edit_worker_response,
    validate_t2i_worker_response,
)

AUTH_TOKEN = "kaggle-demo-test-token-0123456789abcdef0123456789abcdef"


def _tiny_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 200, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


# --- bearer-token semantics ---


def test_auth_missing_credentials_is_401_with_challenge(monkeypatch):
    from fastapi.testclient import TestClient

    from server.app import app

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", AUTH_TOKEN)
    response = TestClient(app).get("/ready")
    assert response.status_code == 401
    assert response.headers.get("www-authenticate", "").lower().startswith("bearer")


@pytest.mark.parametrize(
    "header",
    [
        "Basic dXNlcjpwYXNz",
        "Bearer",
        "Bearer ",
        "Token abc",
        "Bearer {}".format("x" * 60),
    ],
)
def test_auth_malformed_credentials_are_401(monkeypatch, header):
    from fastapi.testclient import TestClient

    from server.app import app

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", AUTH_TOKEN)
    response = TestClient(app).get("/v1/info", headers={"Authorization": header})
    assert response.status_code == 401


def test_auth_invalid_token_is_401_not_403(monkeypatch):
    from fastapi.testclient import TestClient

    from server.app import app

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", AUTH_TOKEN)
    response = TestClient(app).get(
        "/v1/info",
        headers={"Authorization": f"Bearer {'wrong' * 12}"},
    )
    assert response.status_code == 401
    assert response.headers.get("www-authenticate", "").lower().startswith("bearer")


def test_auth_unconfigured_token_is_503(monkeypatch):
    from fastapi.testclient import TestClient

    from server.app import app

    monkeypatch.delenv("MAGE_FLOW_API_TOKEN", raising=False)
    response = TestClient(app).get(
        "/v1/info",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
    )
    assert response.status_code == 503


def test_auth_weak_configured_token_is_503_fail_closed(monkeypatch):
    from fastapi.testclient import TestClient
    from server.auth import MIN_PUBLIC_TOKEN_LENGTH

    from server.app import app

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", "short-token")
    client = TestClient(app)
    response = client.get("/v1/info", headers={"Authorization": f"Bearer {'t' * MIN_PUBLIC_TOKEN_LENGTH}"})
    assert response.status_code == 503
    assert "token contract" in response.json()["detail"]


def test_openapi_documents_bearer_security():
    import fastapi

    from server.app import app

    schema = fastapi.FastAPI.openapi(app)
    schemes = schema["components"]["securitySchemes"]
    bearer = next((s for s in schemes.values() if s.get("scheme") == "bearer"), None)
    assert bearer is not None
    assert bearer["type"] == "http"
    for path, operations in schema["paths"].items():
        if path.startswith("/v1/"):
            for operation in operations.values():
                if isinstance(operation, dict):
                    security = operation.get("security")
                    assert security, f"{path} {operation} must require auth"


# --- security headers ---


def test_api_responses_carry_no_store_and_nosniff(monkeypatch):
    from fastapi.testclient import TestClient

    from server.app import app

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", AUTH_TOKEN)
    client = TestClient(app)
    for path, kwargs in [
        ("/health", {}),
        ("/ready", {"headers": {"Authorization": f"Bearer {AUTH_TOKEN}"}}),
        ("/v1/info", {"headers": {"Authorization": f"Bearer {AUTH_TOKEN}"}}),
    ]:
        response = client.get(path, **kwargs)
        assert response.headers.get("cache-control") == "no-store"
        assert response.headers.get("x-content-type-options") == "nosniff"


# --- strict T2I request model ---


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": "x", "unknown_field": 1},
        {"prompt": "x", "seed": True},
        {"prompt": "x", "seed": "42"},
        {"prompt": "x", "steps": True},
        {"prompt": "x", "width": "1024"},
        {"prompt": "x", "steps": 5},
    ],
)
def test_strict_generation_request_rejects_invalid_bodies(monkeypatch, body):
    from fastapi.testclient import TestClient

    from server.app import app

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", AUTH_TOKEN)
    response = TestClient(app).post(
        "/v1/images/generations",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        json=body,
    )
    assert response.status_code == 422


def test_qualified_generation_request_unchanged(monkeypatch):
    from fastapi.testclient import TestClient

    from server.app import app

    class _ReadyT2IWorker:
        ready = True
        device = "cuda:0"
        captured = {}

        def generate(self, **kwargs):
            _ReadyT2IWorker.captured = kwargs
            return {
                "id": "img_test",
                "status": "completed",
                "model": "mage-flow-turbo",
                "device": "cuda:0",
                "seed": kwargs["seed"],
                "width": 1024,
                "height": 1024,
                "elapsed_seconds": 1.0,
                "output": f"{OUTPUT_DATA_URL_PREFIX}abc",
            }

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", AUTH_TOKEN)
    old = app.state.t2i_worker
    app.state.t2i_worker = _ReadyT2IWorker()
    try:
        response = TestClient(app).post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={"prompt": "cat", "seed": 42, "steps": 4, "width": 1024, "height": 1024},
        )
        assert response.status_code == 200
        assert _ReadyT2IWorker.captured == {
            "prompt": "cat",
            "seed": 42,
            "steps": 4,
            "width": 1024,
            "height": 1024,
        }
    finally:
        app.state.t2i_worker = old


# --- response model constraints ---


@pytest.mark.parametrize(
    "field,value",
    [("seed", -1), ("seed", 2**32), ("elapsed_seconds", -0.1), ("output", ""), ("id", "")],
)
def test_generation_response_model_constraints(field, value):
    from pydantic import ValidationError

    from server.schemas import GenerationResponse

    payload = {
        "id": "img_test",
        "status": "completed",
        "model": "mage-flow-turbo",
        "device": "cuda:0",
        "seed": 42,
        "width": 1024,
        "height": 1024,
        "elapsed_seconds": 1.0,
        "output": f"{OUTPUT_DATA_URL_PREFIX}abc",
    }
    payload[field] = value
    with pytest.raises(ValidationError):
        GenerationResponse(**payload)


@pytest.mark.parametrize("field,value", [("seed", -1), ("elapsed_seconds", -0.1), ("output", "")])
def test_edit_response_model_constraints(field, value):
    from pydantic import ValidationError

    from server.schemas import EditResponse

    payload = {
        "id": "img_test",
        "status": "completed",
        "model": "mage-flow-edit-turbo",
        "device": "cuda:1",
        "seed": 42,
        "elapsed_seconds": 1.0,
        "output": f"{OUTPUT_DATA_URL_PREFIX}abc",
    }
    payload[field] = value
    with pytest.raises(ValidationError):
        EditResponse(**payload)


# --- thread offload ---


def test_health_remains_responsive_while_edit_worker_blocked(monkeypatch):
    from fastapi.testclient import TestClient

    from server.app import app

    entered = threading.Event()
    release = threading.Event()

    class _BlockingEditWorker:
        ready = True
        device = "cuda:1"

        def edit(self, **kwargs):
            entered.set()
            release.wait(timeout=10)
            return {
                "id": "img_test",
                "status": "completed",
                "model": "mage-flow-edit-turbo",
                "device": "cuda:1",
                "seed": kwargs["seed"],
                "elapsed_seconds": 1.0,
                "output": f"{OUTPUT_DATA_URL_PREFIX}abc",
            }

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", AUTH_TOKEN)
    old = app.state.edit_worker
    app.state.edit_worker = _BlockingEditWorker()
    box = {}
    try:

        def _do_edit():
            box["r"] = TestClient(app).post(
                "/v1/images/edits",
                headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
                files={"image": ("source.png", _tiny_png_bytes(), "image/png")},
                data={"prompt": "make it sunny", "seed": "42"},
            )

        thread = threading.Thread(target=_do_edit)
        thread.start()
        assert entered.wait(timeout=5)
        started = time.monotonic()
        health = TestClient(app).get("/health")
        elapsed = time.monotonic() - started
        assert health.status_code == 200
        assert elapsed < 2.0
        release.set()
        thread.join(timeout=10)
        assert box["r"].status_code == 200
    finally:
        app.state.edit_worker = old
        release.set()


# --- single-flight request gates ---


def test_second_t2i_request_while_busy_gets_429(monkeypatch):
    from fastapi.testclient import TestClient

    from server.app import app

    entered = threading.Event()
    release = threading.Event()

    class _BlockingT2IWorker:
        ready = True
        device = "cuda:0"

        def generate(self, **kwargs):
            entered.set()
            release.wait(timeout=10)
            return {
                "id": "img_test",
                "status": "completed",
                "model": "mage-flow-turbo",
                "device": "cuda:0",
                "seed": kwargs["seed"],
                "width": 1024,
                "height": 1024,
                "elapsed_seconds": 1.0,
                "output": f"{OUTPUT_DATA_URL_PREFIX}abc",
            }

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", AUTH_TOKEN)
    old = app.state.t2i_worker
    app.state.t2i_worker = _BlockingT2IWorker()
    box = {}
    try:

        def _first():
            box["r"] = TestClient(app).post(
                "/v1/images/generations",
                headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
                json={"prompt": "first"},
            )

        thread = threading.Thread(target=_first)
        thread.start()
        assert entered.wait(timeout=5)
        second = TestClient(app).post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={"prompt": "second"},
        )
        assert second.status_code == 429
        assert second.headers.get("retry-after") == "1"
        assert "busy" in second.json()["detail"]
        release.set()
        thread.join(timeout=10)
        assert box["r"].status_code == 200
    finally:
        app.state.t2i_worker = old
        release.set()


def test_second_edit_request_while_busy_gets_429(monkeypatch):
    from fastapi.testclient import TestClient

    from server.app import app

    entered = threading.Event()
    release = threading.Event()

    class _BlockingEditWorker:
        ready = True
        device = "cuda:1"

        def edit(self, **kwargs):
            entered.set()
            release.wait(timeout=10)
            return {
                "id": "img_test",
                "status": "completed",
                "model": "mage-flow-edit-turbo",
                "device": "cuda:1",
                "seed": kwargs["seed"],
                "elapsed_seconds": 1.0,
                "output": f"{OUTPUT_DATA_URL_PREFIX}abc",
            }

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", AUTH_TOKEN)
    old = app.state.edit_worker
    app.state.edit_worker = _BlockingEditWorker()
    box = {}
    try:

        def _first():
            box["r"] = TestClient(app).post(
                "/v1/images/edits",
                headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
                files={"image": ("source.png", _tiny_png_bytes(), "image/png")},
                data={"prompt": "make it sunny", "seed": "42"},
            )

        thread = threading.Thread(target=_first)
        thread.start()
        assert entered.wait(timeout=5)
        second = TestClient(app).post(
            "/v1/images/edits",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            files={"image": ("source.png", _tiny_png_bytes(), "image/png")},
            data={"prompt": "make it rainy", "seed": "43"},
        )
        assert second.status_code == 429
        assert second.headers.get("retry-after") == "1"
        assert "busy" in second.json()["detail"]
        release.set()
        thread.join(timeout=10)
        assert box["r"].status_code == 200
    finally:
        app.state.edit_worker = old
        release.set()


def test_t2i_and_edit_lanes_run_concurrently(monkeypatch):
    from fastapi.testclient import TestClient

    from server.app import app

    entered = threading.Event()
    release = threading.Event()

    class _BlockingT2IWorker:
        ready = True
        device = "cuda:0"

        def generate(self, **kwargs):
            entered.set()
            release.wait(timeout=10)
            return {
                "id": "img_test",
                "status": "completed",
                "model": "mage-flow-turbo",
                "device": "cuda:0",
                "seed": kwargs["seed"],
                "width": 1024,
                "height": 1024,
                "elapsed_seconds": 1.0,
                "output": f"{OUTPUT_DATA_URL_PREFIX}abc",
            }

    class _ReadyEditWorker:
        ready = True
        device = "cuda:1"

        def edit(self, **kwargs):
            return {
                "id": "img_test",
                "status": "completed",
                "model": "mage-flow-edit-turbo",
                "device": "cuda:1",
                "seed": kwargs["seed"],
                "elapsed_seconds": 1.0,
                "output": f"{OUTPUT_DATA_URL_PREFIX}abc",
            }

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", AUTH_TOKEN)
    old_t2i = app.state.t2i_worker
    old_edit = app.state.edit_worker
    app.state.t2i_worker = _BlockingT2IWorker()
    app.state.edit_worker = _ReadyEditWorker()
    box = {}
    try:

        def _t2i():
            box["t2i"] = TestClient(app).post(
                "/v1/images/generations",
                headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
                json={"prompt": "cat"},
            )

        thread = threading.Thread(target=_t2i)
        thread.start()
        assert entered.wait(timeout=5)
        edit = TestClient(app).post(
            "/v1/images/edits",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            files={"image": ("source.png", _tiny_png_bytes(), "image/png")},
            data={"prompt": "make it sunny", "seed": "42"},
        )
        assert edit.status_code == 200
        release.set()
        thread.join(timeout=10)
        assert box["t2i"].status_code == 200
    finally:
        app.state.t2i_worker = old_t2i
        app.state.edit_worker = old_edit
        release.set()


def test_gate_releases_on_worker_exception(monkeypatch):
    from fastapi.testclient import TestClient

    from server.app import app

    class _FlakyT2IWorker:
        ready = True
        device = "cuda:0"
        calls = 0

        def generate(self, **kwargs):
            _FlakyT2IWorker.calls += 1
            if _FlakyT2IWorker.calls == 1:
                raise RuntimeError("internal burst")
            return {
                "id": "img_test",
                "status": "completed",
                "model": "mage-flow-turbo",
                "device": "cuda:0",
                "seed": kwargs["seed"],
                "width": 1024,
                "height": 1024,
                "elapsed_seconds": 1.0,
                "output": f"{OUTPUT_DATA_URL_PREFIX}abc",
            }

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", AUTH_TOKEN)
    old = app.state.t2i_worker
    app.state.t2i_worker = _FlakyT2IWorker()
    try:
        client = TestClient(app)
        first = client.post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={"prompt": "cat"},
        )
        assert first.status_code == 502
        second = client.post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={"prompt": "cat"},
        )
        assert second.status_code == 200
    finally:
        app.state.t2i_worker = old


def test_invalid_request_rejected_before_gate_admission(monkeypatch):
    from fastapi.testclient import TestClient

    from server.app import app

    class _TrackingT2IWorker:
        ready = True
        device = "cuda:0"
        calls = 0

        def generate(self, **kwargs):
            _TrackingT2IWorker.calls += 1
            return {
                "id": "img_test",
                "status": "completed",
                "model": "mage-flow-turbo",
                "device": "cuda:0",
                "seed": kwargs["seed"],
                "width": 1024,
                "height": 1024,
                "elapsed_seconds": 1.0,
                "output": f"{OUTPUT_DATA_URL_PREFIX}abc",
            }

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", AUTH_TOKEN)
    old = app.state.t2i_worker
    app.state.t2i_worker = _TrackingT2IWorker()
    try:
        response = TestClient(app).post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={"prompt": "cat", "unknown_field": 1},
        )
        assert response.status_code == 422
        assert _TrackingT2IWorker.calls == 0
        follow = TestClient(app).post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={"prompt": "cat"},
        )
        assert follow.status_code == 200
    finally:
        app.state.t2i_worker = old


# --- worker contract violation 502 behavior ---


def test_worker_contract_violation_maps_to_sanitized_502(monkeypatch):
    from fastapi.testclient import TestClient

    from server.app import app

    class _BadT2IWorker:
        ready = True
        device = "cuda:0"

        def generate(self, **kwargs):
            return {"id": "img_test", "status": "completed", "model": "wrong-model"}

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", AUTH_TOKEN)
    old = app.state.t2i_worker
    app.state.t2i_worker = _BadT2IWorker()
    try:
        response = TestClient(app).post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={"prompt": "cat"},
        )
        assert response.status_code == 502
        assert response.json()["detail"] == "T2I worker request failed"
    finally:
        app.state.t2i_worker = old


def test_502_detail_does_not_leak_internal_diagnostics(monkeypatch):
    from fastapi.testclient import TestClient

    from server.app import app

    class _LeakingT2IWorker:
        ready = True
        device = "cuda:0"

        def generate(self, **kwargs):
            raise RuntimeError(
                "internal request failed at http://127.0.0.1:8101 "
                "model /kaggle/input/secret/path token abc123"
            )

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", AUTH_TOKEN)
    old = app.state.t2i_worker
    app.state.t2i_worker = _LeakingT2IWorker()
    try:
        response = TestClient(app).post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={"prompt": "cat"},
        )
        assert response.status_code == 502
        detail = response.json()["detail"]
        assert detail == "T2I worker request failed"
        assert "8101" not in detail
        assert "kaggle" not in detail
        assert "secret" not in detail
    finally:
        app.state.t2i_worker = old


def test_edit_contract_violation_maps_to_sanitized_502(monkeypatch):
    from fastapi.testclient import TestClient

    from server.app import app

    class _BadEditWorker:
        ready = True
        device = "cuda:1"

        def edit(self, **kwargs):
            return {"id": "img_test", "status": "completed", "model": "wrong-model"}

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", AUTH_TOKEN)
    old = app.state.edit_worker
    app.state.edit_worker = _BadEditWorker()
    try:
        response = TestClient(app).post(
            "/v1/images/edits",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            files={"image": ("source.png", _tiny_png_bytes(), "image/png")},
            data={"prompt": "make it sunny", "seed": "42"},
        )
        assert response.status_code == 502
        assert response.json()["detail"] == "Edit worker request failed"
    finally:
        app.state.edit_worker = old