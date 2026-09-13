import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from server.workers.t2i import (
    T2IWorkerClient,
    T2IWorkerConfig,
    build_t2i_worker_command,
)
from server.workers.t2i_runtime_server import validate_generation_payload


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


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

    with pytest.raises(ValueError, match="multiples of 16"):
        validate_generation_payload({"prompt": "hello", "width": 1001})


def test_runtime_payload_rejects_empty_prompt_and_bool_seed():
    with pytest.raises(ValueError, match="non-empty"):
        validate_generation_payload({"prompt": ""})
    with pytest.raises(ValueError, match="seed must be an integer"):
        validate_generation_payload({"prompt": "hello", "seed": True})
