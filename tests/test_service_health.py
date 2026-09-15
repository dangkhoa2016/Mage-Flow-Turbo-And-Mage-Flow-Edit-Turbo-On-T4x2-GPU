import http.server
import json
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

from scripts.service_health import check

ROOT = Path(__file__).resolve().parent.parent


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_health_identity_match():
    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(
            {"status": "ready", "ready": True, "model": "mage-flow-turbo", "device": "cuda:0", "model_path": "/models/t2i"}
        ),
    ):
        assert check("http://127.0.0.1:8101", "mage-flow-turbo", "cuda:0", "/models/t2i") is True


def test_health_identity_mismatch_device():
    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(
            {"status": "ready", "ready": True, "model": "mage-flow-turbo", "device": "cuda:1", "model_path": "/models/t2i"}
        ),
    ):
        assert check("http://127.0.0.1:8101", "mage-flow-turbo", "cuda:0", "/models/t2i") is False


def test_health_identity_mismatch_model():
    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(
            {
                "status": "ready",
                "ready": True,
                "model": "mage-flow-edit-turbo",
                "device": "cuda:1",
                "model_path": "/models/edit",
            }
        ),
    ):
        assert check("http://127.0.0.1:8102", "mage-flow-turbo", "cuda:1", "/models/edit") is False


def test_health_identity_mismatch_model_path():
    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(
            {
                "status": "ready",
                "ready": True,
                "model": "mage-flow-turbo",
                "device": "cuda:0",
                "model_path": "/stale/model/path",
            }
        ),
    ):
        assert check("http://127.0.0.1:8101", "mage-flow-turbo", "cuda:0", "/expected/model/path") is False


def test_health_not_ready_yet_is_unhealthy():
    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(
            {"status": "not_ready", "ready": False, "model": "mage-flow-turbo", "device": "cuda:0"}
        ),
    ):
        assert check("http://127.0.0.1:8101", "mage-flow-turbo", "cuda:0") is False


def test_health_connection_error_is_unhealthy():
    def boom(*args, **kwargs):
        raise OSError("connection refused")

    with patch("urllib.request.urlopen", side_effect=boom):
        assert check("http://127.0.0.1:8101", "mage-flow-turbo", "cuda:0") is False


class _HealthServer:
    def __init__(self, payload):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/health":
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._server.shutdown()
        self._server.server_close()


def _run_cli(url, model, device, model_path):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "service_health.py"),
            url,
            model,
            device,
            model_path,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cli_requires_and_accepts_model_path():
    payload = {
        "status": "ready",
        "ready": True,
        "model": "mage-flow-turbo",
        "device": "cuda:0",
        "model_path": "/models/t2i",
    }
    with _HealthServer(payload) as server:
        url = f"http://127.0.0.1:{server.port}"
        good = _run_cli(url, "mage-flow-turbo", "cuda:0", "/models/t2i")
        assert good.returncode == 0, good.stdout + good.stderr
        assert "[PASS] HEALTHY" in good.stdout

        wrong_path = _run_cli(url, "mage-flow-turbo", "cuda:0", "/stale/model/path")
        assert wrong_path.returncode != 0

        wrong_model = _run_cli(url, "mage-flow-edit-turbo", "cuda:0", "/models/t2i")
        assert wrong_model.returncode != 0

        wrong_device = _run_cli(url, "mage-flow-turbo", "cuda:1", "/models/t2i")
        assert wrong_device.returncode != 0


def test_cli_rejects_unreachable_endpoint():
    result = _run_cli("http://127.0.0.1:1", "mage-flow-turbo", "cuda:0", "/models/t2i")
    assert result.returncode != 0
    assert "[FAIL] UNHEALTHY" in result.stdout


def test_cli_fails_fast_on_missing_model_path_argument():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "service_health.py"),
            "http://127.0.0.1:8101",
            "mage-flow-turbo",
            "cuda:0",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "model_path" in result.stderr
