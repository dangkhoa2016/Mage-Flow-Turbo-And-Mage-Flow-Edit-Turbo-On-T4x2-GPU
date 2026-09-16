import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scripts.acceptance import LiveAcceptance

VALID_TOKEN = "kaggle-demo-test-token-0123456789abcdef0123456789abcdef"


class _State:
    ready_calls = 0
    fail_generation = False


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        if isinstance(payload, dict):
            body = json.dumps(payload).encode("utf-8")
            content_type = "application/json"
        else:
            body = payload
            content_type = "text/html; charset=utf-8"
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _route(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        auth = self.headers.get("Authorization", "")

        if self.path == "/health":
            self._send(200, {"status": "ok"})
            return
        if self.path == "/ready":
            if auth != f"Bearer {VALID_TOKEN}":
                self._send(401, {"detail": "unauthorized"})
                return
            _State.ready_calls += 1
            if _State.ready_calls <= 2:
                self._send(503, b"<html>Service Unavailable</html>")
                return
            self._send(
                200,
                {"ready": True, "status": "ok", "t2i_ready": True, "edit_ready": True},
            )
            return
        if self.path == "/v1/info":
            if auth != f"Bearer {VALID_TOKEN}":
                self._send(401, {"detail": "unauthorized"})
                return
            self._send(
                200,
                {
                    "cpu_fallback": False,
                    "t2i": {"device": "cuda:0", "ready": True},
                    "edit": {"device": "cuda:1", "ready": True},
                },
            )
            return
        if self.path == "/v1/images/generations":
            if _State.fail_generation:
                self._send(400, {"detail": "unexpectedly accepted"})
            else:
                self._send(422, {"detail": "validation error"})
            return
        if self.path == "/v1/images/edits":
            if b"text/plain" in body:
                self._send(415, {"detail": "unsupported media"})
                return
            if b"not a real image payload" in body:
                self._send(400, {"detail": "invalid image"})
                return
            self._send(200, {"status": "completed"})
            return
        self._send(404, {"detail": "not found"})

    def do_GET(self):
        self._route()

    def do_POST(self):
        self._route()


@pytest.fixture()
def server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=5)


@pytest.fixture()
def runner(server, tmp_path):
    return LiveAcceptance(
        base_url=f"http://127.0.0.1:{server.server_address[1]}",
        token=VALID_TOKEN,
        output_dir=tmp_path / ".runtime" / "acceptance",
        edit_source=tmp_path / "dog.jpg",
    )


def test_wait_ready_retries_transient_failures(server, runner):
    _State.ready_calls = 0
    payload = runner.wait_ready(timeout=10)
    assert payload.get("ready") is True
    assert runner.summary["ready"]["transient_errors"] == 2


def test_security_cheap_checks_record_all_gates(server, runner):
    _State.ready_calls = 100
    runner.health()
    runner.security_cheap_checks()
    checks = runner.summary["security_checks"]
    assert len(checks) == 5
    assert all(c["ok"] for c in checks)
    assert "reject unauthenticated /ready" in [c["label"] for c in checks]
    assert len([c for c in checks if "invalid bearer token" in c["label"]]) == 1
    assert len([c for c in checks if "unknown generation field" in c["label"]]) == 1
    assert len([c for c in checks if "malformed edit image bytes" in c["label"]]) == 1
    assert len([c for c in checks if "unsupported edit media type" in c["label"]]) == 1


def test_security_cheap_checks_fail_closed(server, runner):
    _State.ready_calls = 100
    _State.fail_generation = True
    runner.health()
    with pytest.raises(AssertionError, match="security acceptance check"):
        runner.security_cheap_checks()


def test_health_explicitly_raises_instead_of_asserting(server, runner):
    _State.ready_calls = 100
    runner.health()
    assert runner.summary["health"]["ok"] is True