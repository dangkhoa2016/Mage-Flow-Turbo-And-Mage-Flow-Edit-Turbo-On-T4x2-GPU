import json
from unittest.mock import patch

import pytest
from server.workers.t2i import (
    T2IWorkerClient,
    T2IWorkerConfig,
    build_t2i_worker_command,
)
from server.workers.t2i_runtime_server import validate_generation_payload

AUTH_TOKEN = "kaggle-demo-test-token-0123456789abcdef0123456789abcdef"


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, amt: int = -1):
        data = json.dumps(self.payload).encode()
        if amt is None or amt < 0:
            return data
        return data[:amt]


def test_build_worker_command_uses_validated_runtime_and_cuda0(tmp_path):
    cfg = T2IWorkerConfig(
        runtime_root="/runtime",
        model_path="/models/t2i",
        device="cuda:0",
    )
    command = build_t2i_worker_command(project_root=tmp_path, config=cfg)
    assert command[0] == "/runtime/.venv/bin/python"
    assert command[1] == str(tmp_path / "server/workers/t2i_runtime_server.py")
    assert command[-2:] == ["--device", "cuda:0"]
    assert "/models/t2i" in command


def test_build_worker_command_rejects_wrong_device(tmp_path):
    cfg = T2IWorkerConfig(device="cuda:1")
    with pytest.raises(ValueError, match="cuda:0"):
        build_t2i_worker_command(project_root=tmp_path, config=cfg)


def test_client_ready_requires_exact_worker_identity():
    client = T2IWorkerClient(T2IWorkerConfig())
    model_path = client.config.model_path
    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(
            {"status": "ready", "ready": True, "device": "cuda:0", "model": "mage-flow-turbo", "model_path": model_path}
        ),
    ):
        assert client.ready is True

    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(
            {"status": "ready", "ready": True, "device": "cuda:1", "model": "mage-flow-turbo", "model_path": model_path}
        ),
    ):
        assert client.ready is False


def test_client_ready_rejects_stale_model_path():
    client = T2IWorkerClient(T2IWorkerConfig())
    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(
            {
                "status": "ready",
                "ready": True,
                "device": "cuda:0",
                "model": "mage-flow-turbo",
                "model_path": "/different/model/path",
            }
        ),
    ):
        assert client.ready is False


def test_client_generate_forwards_public_contract():
    result = {
        "id": "img_test",
        "status": "completed",
        "model": "mage-flow-turbo",
        "device": "cuda:0",
        "seed": 42,
        "width": 1024,
        "height": 1024,
        "elapsed_seconds": 1.2,
        "output": "data:image/png;base64,abc",
    }
    client = T2IWorkerClient(T2IWorkerConfig())
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return FakeResponse(result)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        response = client.generate(prompt="cat", seed=42, steps=4, width=1024, height=1024)

    assert response == result
    assert captured["url"].endswith("/generate")
    assert captured["body"] == {"prompt": "cat", "seed": 42, "steps": 4, "width": 1024, "height": 1024}


def test_runtime_payload_validation_defaults_and_alignment():
    payload = validate_generation_payload({"prompt": "hello"})
    assert payload["seed"] == 42
    assert payload["steps"] == 4
    assert payload["width"] == 1024
    assert payload["height"] == 1024


@pytest.mark.parametrize(
    "overrides",
    [
        {"steps": 5},
        {"steps": 1},
        {"steps": 100},
        {"width": 512, "height": 512},
        {"width": 1024, "height": 512},
        {"width": 2048, "height": 2048},
        {"width": 256, "height": 256},
    ],
)
def test_runtime_rejects_unfrozen_t2i_profile(overrides):
    with pytest.raises(ValueError):
        validate_generation_payload({"prompt": "hello", **overrides})


def test_runtime_rejects_bool_as_int_profile_params():
    with pytest.raises(ValueError):
        validate_generation_payload({"prompt": "hello", "steps": True})
    with pytest.raises(ValueError):
        validate_generation_payload({"prompt": "hello", "width": True})
    with pytest.raises(ValueError):
        validate_generation_payload({"prompt": "hello", "height": True})


def test_runtime_payload_rejects_empty_prompt_and_bool_seed():
    with pytest.raises(ValueError, match="non-empty"):
        validate_generation_payload({"prompt": ""})
    with pytest.raises(ValueError, match="seed must be an integer"):
        validate_generation_payload({"prompt": "hello", "seed": True})


def test_client_rejects_oversized_worker_response():
    from server.workers.t2i import MAX_WORKER_RESPONSE_BYTES

    class BigResponse(FakeResponse):
        def read(self, amt: int = -1):
            return b"x" * (MAX_WORKER_RESPONSE_BYTES + 1)

    client = T2IWorkerClient(T2IWorkerConfig())
    with (
        patch("urllib.request.urlopen", return_value=BigResponse({"status": "completed"})),
        pytest.raises(RuntimeError, match="size bound"),
    ):
        client.generate(prompt="cat", seed=42, steps=4, width=1024, height=1024)


def test_client_transport_timeout_normalized_to_runtime_error():

    client = T2IWorkerClient(T2IWorkerConfig())

    def timeout(*args, **kwargs):
        raise TimeoutError("timed out")

    with patch("urllib.request.urlopen", side_effect=timeout), pytest.raises(RuntimeError, match="unavailable"):
        client.generate(prompt="cat", seed=42, steps=4, width=1024, height=1024)


def test_client_socket_oserror_normalized_to_runtime_error():
    client = T2IWorkerClient(T2IWorkerConfig())

    def refused(*args, **kwargs):
        raise OSError("connection refused")

    with patch("urllib.request.urlopen", side_effect=refused), pytest.raises(RuntimeError, match="unavailable"):
        client.generate(prompt="cat", seed=42, steps=4, width=1024, height=1024)


def test_client_ready_false_on_transient_transport_error():

    client = T2IWorkerClient(T2IWorkerConfig())

    def timeout(*args, **kwargs):
        raise TimeoutError("timed out")

    with patch("urllib.request.urlopen", side_effect=timeout):
        assert client.ready is False


class _ReadyT2IWorker:
    ready = True
    device = "cuda:0"
    generate_call_count = 0

    def generate(self, *, prompt, seed, steps, width, height):
        _ReadyT2IWorker.generate_call_count += 1
        return {
            "id": "img_test",
            "status": "completed",
            "model": "mage-flow-turbo",
            "device": "cuda:0",
            "seed": seed,
            "width": width,
            "height": height,
            "elapsed_seconds": 1.0,
            "output": "data:image/png;base64,abc",
        }


def test_coordinator_rejects_empty_t2i_prompt(monkeypatch):
    from fastapi.testclient import TestClient
    from server.app import app

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", AUTH_TOKEN)
    old = app.state.t2i_worker
    app.state.t2i_worker = _ReadyT2IWorker()
    _ReadyT2IWorker.generate_call_count = 0
    try:
        client = TestClient(app)
        response = client.post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={"prompt": ""},
        )
        assert response.status_code == 422
        assert _ReadyT2IWorker.generate_call_count == 0
    finally:
        app.state.t2i_worker = old


def test_coordinator_rejects_whitespace_t2i_prompt(monkeypatch):
    from fastapi.testclient import TestClient
    from server.app import app

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", AUTH_TOKEN)
    old = app.state.t2i_worker
    app.state.t2i_worker = _ReadyT2IWorker()
    _ReadyT2IWorker.generate_call_count = 0
    try:
        client = TestClient(app)
        response = client.post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={"prompt": "   "},
        )
        assert response.status_code == 422
        assert _ReadyT2IWorker.generate_call_count == 0
    finally:
        app.state.t2i_worker = old


def test_coordinator_rejects_tab_newline_t2i_prompt(monkeypatch):
    from fastapi.testclient import TestClient
    from server.app import app

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", AUTH_TOKEN)
    old = app.state.t2i_worker
    app.state.t2i_worker = _ReadyT2IWorker()
    _ReadyT2IWorker.generate_call_count = 0
    try:
        client = TestClient(app)
        response = client.post(
            "/v1/images/generations",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={"prompt": "\t\n"},
        )
        assert response.status_code == 422
        assert _ReadyT2IWorker.generate_call_count == 0
    finally:
        app.state.t2i_worker = old
