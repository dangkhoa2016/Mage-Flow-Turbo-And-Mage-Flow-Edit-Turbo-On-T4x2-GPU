import base64
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from server.workers.edit import (
    EditWorkerClient,
    EditWorkerConfig,
    build_edit_worker_command,
    start_edit_worker_process,
)
from server.workers.edit_runtime_server import (
    decode_image_bytes,
    main,
    validate_edit_payload,
    validate_gpu_contract,
)
from server.workers.t2i import build_t2i_worker_command, T2IWorkerConfig


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def _tiny_png_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (120, 40, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_edit_config_defaults_to_cuda1():
    cfg = EditWorkerConfig()
    assert cfg.device == "cuda:1"
    assert cfg.internal_url == "http://127.0.0.1:8102"
    assert cfg.model_path == (
        "/kaggle/input/models/dangkhoa2016/mage-flow-community-mage-flow-edit-turbo/pytorch/default/1"
    )
    client = EditWorkerClient(cfg)
    assert client.ready is False


def test_edit_client_rejects_non_cuda1_device():
    with pytest.raises(ValueError, match="cuda:1"):
        EditWorkerClient(EditWorkerConfig(device="cuda:0"))


def test_build_edit_worker_command_uses_validated_runtime_and_cuda1(tmp_path):
    cfg = EditWorkerConfig(
        runtime_root="/runtime",
        model_path="/models/edit",
        device="cuda:1",
    )
    command = build_edit_worker_command(project_root=tmp_path, config=cfg)
    assert command[0] == "/runtime/.venv/bin/python"
    assert command[1] == str(tmp_path / "server/workers/edit_runtime_server.py")
    assert "--host" in command and command[command.index("--host") + 1] == "127.0.0.1"
    assert "--port" in command and command[command.index("--port") + 1] == "8102"
    assert command[-2:] == ["--device", "cuda:1"]
    assert "/models/edit" in command


def test_build_edit_worker_command_rejects_wrong_device(tmp_path):
    cfg = EditWorkerConfig(device="cuda:0")
    with pytest.raises(ValueError, match="cuda:1"):
        build_edit_worker_command(project_root=tmp_path, config=cfg)


def test_start_edit_process_fails_closed_on_missing_runtime(tmp_path):
    cfg = EditWorkerConfig(runtime_root="/missing-runtime", model_path="/models/edit", device="cuda:1")
    with pytest.raises(FileNotFoundError, match="runtime Python"):
        start_edit_worker_process(project_root=tmp_path, config=cfg)


def test_start_edit_process_fails_closed_on_missing_model(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / ".venv").mkdir()
    python = runtime / ".venv" / "bin"
    python.mkdir(parents=True)
    (python / "python").write_text("#!/bin/sh\n")
    server_file = tmp_path / "server" / "workers" / "edit_runtime_server.py"
    server_file.parent.mkdir(parents=True)
    server_file.write_text("")
    cfg = EditWorkerConfig(runtime_root=str(runtime), model_path="/missing-model", device="cuda:1")
    with pytest.raises(FileNotFoundError, match="model directory"):
        start_edit_worker_process(project_root=tmp_path, config=cfg)


def test_client_ready_requires_exact_edit_identity():
    client = EditWorkerClient(EditWorkerConfig())
    model_path = client.config.model_path
    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(
            {
                "status": "ready",
                "ready": True,
                "device": "cuda:1",
                "model": "mage-flow-edit-turbo",
                "model_path": model_path,
            }
        ),
    ):
        assert client.ready is True

    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(
            {
                "status": "ready",
                "ready": True,
                "device": "cuda:0",
                "model": "mage-flow-edit-turbo",
                "model_path": model_path,
            }
        ),
    ):
        assert client.ready is False


def test_client_ready_rejects_stale_edit_model_path():
    client = EditWorkerClient(EditWorkerConfig())
    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(
            {
                "status": "ready",
                "ready": True,
                "device": "cuda:1",
                "model": "mage-flow-edit-turbo",
                "model_path": "/different/edit/model/path",
            }
        ),
    ):
        assert client.ready is False


def test_client_edit_forwards_base64_image_and_public_contract():
    image_bytes = _tiny_png_bytes()
    result = {
        "id": "img_test",
        "status": "completed",
        "model": "mage-flow-edit-turbo",
        "device": "cuda:1",
        "seed": 42,
        "width": 16,
        "height": 16,
        "elapsed_seconds": 1.2,
        "output": "data:image/png;base64,abc",
    }
    client = EditWorkerClient(EditWorkerConfig())
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return FakeResponse(result)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        response = client.edit(image_bytes=image_bytes, prompt="make it sunny", seed=42)

    assert response == result
    assert captured["url"].endswith("/edit")
    assert captured["body"]["prompt"] == "make it sunny"
    assert captured["body"]["seed"] == 42
    assert base64.b64decode(captured["body"]["image_base64"]) == image_bytes


def test_client_edit_rejects_empty_image_bytes():
    client = EditWorkerClient(EditWorkerConfig())
    with pytest.raises(ValueError, match="non-empty"):
        client.edit(image_bytes=b"", prompt="make it sunny", seed=42)


def test_runtime_payload_validation_defaults_and_roundtrip():
    image_bytes = _tiny_png_bytes()
    payload = validate_edit_payload(
        {"prompt": "make it sunny", "seed": 42, "image_base64": base64.b64encode(image_bytes).decode("ascii")}
    )
    assert payload["seed"] == 42
    assert payload["image_bytes"] == image_bytes

    decoded = decode_image_bytes(payload["image_bytes"])
    assert decoded.mode == "RGB"
    assert decoded.size == (16, 16)


def test_runtime_payload_rejects_empty_prompt_and_bool_seed():
    image_bytes = _tiny_png_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    with pytest.raises(ValueError, match="non-empty"):
        validate_edit_payload({"prompt": "", "image_base64": encoded})
    with pytest.raises(ValueError, match="non-empty"):
        validate_edit_payload({"prompt": "   ", "image_base64": encoded})
    with pytest.raises(ValueError, match="seed must be an integer"):
        validate_edit_payload({"prompt": "x", "seed": True, "image_base64": encoded})
    with pytest.raises(ValueError, match="body must be a JSON object"):
        validate_edit_payload(["not", "a", "dict"])


def test_runtime_payload_rejects_malformed_base64_image():
    with pytest.raises(ValueError, match="base64"):
        validate_edit_payload({"prompt": "x", "image_base64": "!!!not-base64!!!"})
    with pytest.raises(ValueError, match="non-empty"):
        validate_edit_payload({"prompt": "x", "image_base64": ""})
    with pytest.raises(ValueError, match="base64"):
        validate_edit_payload({"prompt": "x"})
    with pytest.raises(ValueError, match="non-empty base64"):
        validate_edit_payload({"prompt": "x", "image_base64": None})


def test_decode_image_bytes_rejects_garbage_bytes():
    with pytest.raises(ValueError, match="invalid image payload"):
        decode_image_bytes(b"this is not an image at all" * 100)


def test_gpu_contract_requires_cuda1_only():
    with pytest.raises(RuntimeError, match="cuda:1"):
        validate_gpu_contract("cuda:0")


def test_no_cpu_fallback_when_cuda_unavailable(monkeypatch):
    import sys
    import types

    fake_cuda = types.ModuleType("torch.cuda")
    fake_cuda.is_available = lambda: False
    fake_cuda.device_count = lambda: 1
    fake_cuda.set_device = lambda *_args: None
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = fake_cuda
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.cuda", fake_cuda)
    with pytest.raises(RuntimeError, match="refuses CPU fallback"):
        validate_gpu_contract("cuda:1")


def test_worker_rejects_non_loopback_bind(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["edit_runtime_server", "--host", "0.0.0.0", "--port", "8102", "--model-path", "/models/edit", "--device", "cuda:1"],
    )
    with pytest.raises(SystemExit, match="localhost"):
        main()


def test_edit_and_t2i_ports_are_disjoint(tmp_path):
    edit_cmd = build_edit_worker_command(
        project_root=tmp_path,
        config=EditWorkerConfig(runtime_root="/r", model_path="/m", device="cuda:1"),
    )
    t2i_cmd = build_t2i_worker_command(
        project_root=tmp_path,
        config=T2IWorkerConfig(runtime_root="/r", model_path="/m", device="cuda:0"),
    )
    edit_port = edit_cmd[edit_cmd.index("--port") + 1]
    t2i_port = t2i_cmd[t2i_cmd.index("--port") + 1]
    assert edit_port == "8102"
    assert t2i_port == "8101"
    assert "8101" not in edit_cmd[edit_cmd.index("--port"):]
    assert "8102" not in t2i_cmd[t2i_cmd.index("--port"):]