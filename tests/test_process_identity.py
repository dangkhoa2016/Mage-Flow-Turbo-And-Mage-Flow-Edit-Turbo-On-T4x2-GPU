import http.server
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from scripts.process_identity import (
    PIDFD_SUPPORTED,
    ProcessIdentityError,
    _fdinfo_pid,
    argv_matches_signature,
    check_process,
    open_pidfd,
    parse_strict_pid,
    pidfd_wait_exit,
    read_pid_file,
    verify_coordinator_authenticated,
    verify_worker_health,
)

ROOT = Path(__file__).resolve().parent.parent

WORKER_ARGV = [
    "server/workers/t2i_runtime_server.py",
    "--host",
    "127.0.0.1",
    "--port",
    "8101",
    "--model-path",
    "/models/t2i",
    "--device",
    "cuda:0",
]


# ---------------- strict PID parser ----------------
def test_parse_valid_pid():
    assert parse_strict_pid("123") == 123
    assert parse_strict_pid(" 123 ") == 123


@pytest.mark.parametrize(
    "value",
    ["", "   ", "-1", "0", "1", "abc", "12a", "12.0", "1 2", "999999999999999999999999999999"],
)
def test_parse_invalid_pid_rejected(value):
    with pytest.raises(ProcessIdentityError):
        parse_strict_pid(value)


# ---------------- argv signature matcher ----------------
def test_argv_matches_full_signature():
    assert argv_matches_signature(WORKER_ARGV, "t2i_worker", port=8101, device="cuda:0", model_path="/models/t2i") == []


def test_unrelated_process_without_expected_token_rejected():
    unrelated = ["/usr/bin/sleep", "60"]
    assert argv_matches_signature(unrelated, "t2i_worker") != []


def test_same_substring_unrelated_command_rejected():
    sibling = ["server/workers/t2i_runtime_server_backup.py", "--port", "8101"]
    assert argv_matches_signature(sibling, "t2i_worker", port=8101) != []


def test_wrong_port_rejected():
    assert argv_matches_signature(WORKER_ARGV, "t2i_worker", port=8102) != []


def test_wrong_device_rejected():
    assert argv_matches_signature(WORKER_ARGV, "t2i_worker", device="cuda:1") != []


def test_wrong_model_path_rejected():
    assert argv_matches_signature(WORKER_ARGV, "t2i_worker", model_path="/stale/models") != []


def test_missing_model_path_rejected():
    partial = [
        "server/workers/t2i_runtime_server.py",
        "--host",
        "127.0.0.1",
        "--port",
        "8101",
        "--device",
        "cuda:0",
    ]
    assert argv_matches_signature(partial, "t2i_worker", model_path="/models/t2i") != []


def test_unknown_kind_rejected():
    assert argv_matches_signature(WORKER_ARGV, "mystery") != []


# ---------------- live /proc identity ----------------
def _spawn_fixture(argv0: str, args: list[str], cwd: str):
    cmd = ["bash", "-c", f"exec -a {argv0} {sys.executable} -c 'import time; time.sleep(60)' {' '.join(args)}", "_"]
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        proc.wait(timeout=3)
        raise AssertionError("fixture exited early")
    except subprocess.TimeoutExpired:
        return proc


def test_live_matching_worker_identity_passes():
    proc = _spawn_fixture(
        "server/workers/t2i_runtime_server.py",
        ["--host", "127.0.0.1", "--port", "8101", "--device", "cuda:0", "--model-path", "/models/t2i"],
        str(ROOT),
    )
    try:
        check_process(
            proc.pid,
            "t2i_worker",
            project_root=str(ROOT),
            port=8101,
            device="cuda:0",
            model_path="/models/t2i",
        )
    finally:
        proc.kill()


def test_live_wrong_cwd_rejected(tmp_path):
    proc = _spawn_fixture(
        "server/workers/edit_runtime_server.py",
        ["--host", "127.0.0.1", "--port", "8102", "--device", "cuda:1", "--model-path", "/models/edit"],
        str(tmp_path),
    )
    try:
        with pytest.raises(ProcessIdentityError):
            check_process(
                proc.pid,
                "edit_worker",
                project_root=str(ROOT),
                port=8102,
                device="cuda:1",
                model_path="/models/edit",
            )
    finally:
        proc.kill()


def test_stale_dead_pid_rejected():
    with pytest.raises(ProcessIdentityError):
        check_process(99999999, "t2i_worker")


def test_read_pid_requires_decimal(tmp_path):
    pid_file = tmp_path / "x.pid"
    pid_file.write_text("-5")
    with pytest.raises(ProcessIdentityError):
        read_pid_file(pid_file)


# ---------------- health-payload identity (PID reuse simulation) ----------------
class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_health_pid_mismatch_is_pid_reuse():
    from unittest.mock import patch

    payload = {
        "status": "ready",
        "ready": True,
        "model": "mage-flow-turbo",
        "device": "cuda:0",
        "model_path": "/models/t2i",
        "pid": 424242,
    }
    with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)), pytest.raises(ProcessIdentityError):
        verify_worker_health(
            "http://127.0.0.1:8101",
            1234,
            model="mage-flow-turbo",
            device="cuda:0",
            model_path="/models/t2i",
        )


def test_health_identity_match():
    from unittest.mock import patch

    payload = {
        "status": "ready",
        "ready": True,
        "model": "mage-flow-turbo",
        "device": "cuda:0",
        "model_path": "/models/t2i",
        "pid": 1234,
    }
    with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)):
        verify_worker_health(
            "http://127.0.0.1:8101",
            1234,
            model="mage-flow-turbo",
            device="cuda:0",
            model_path="/models/t2i",
        )


def test_health_not_ready_yet_rejected():
    from unittest.mock import patch

    payload = {"status": "not_ready", "ready": False, "pid": 1234}
    with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)), pytest.raises(ProcessIdentityError):
        verify_worker_health(
            "http://127.0.0.1:8101",
            1234,
            model="mage-flow-turbo",
            device="cuda:0",
            model_path="/models/t2i",
        )


# ---------------- authenticated coordinator reuse (P0-13) ----------------
class _CoordinatorServer:
    def __init__(self, payload, auth_token):
        self.auth_token = auth_token
        self.payload = payload

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/v1/info":
                    self.send_response(404)
                    self.end_headers()
                    return
                auth = self.headers.get("Authorization", "")
                if auth != f"Bearer {self.server.auth_token}":
                    self.send_response(401)
                    self.end_headers()
                    return
                body = json.dumps(self.server.payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.auth_token = auth_token
        self.server.payload = payload
        self.token = auth_token
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.server.shutdown()
        self.server.server_close()


def _coordinator_payload():
    return {
        "project": "mage-flow-t4x2-production-rest-api-demo",
        "runtime_target": "Kaggle NVIDIA T4 x2",
        "t2i": {"model": "mage-flow-turbo", "device": "cuda:0", "ready": True, "single_flight": True},
        "edit": {"model": "mage-flow-edit-turbo", "device": "cuda:1", "ready": True, "single_flight": True},
        "cpu_fallback": False,
    }


def test_coordinator_reuse_passes_with_current_token(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("current-token-0123456789abcdef0123456789abcdef")
    with _CoordinatorServer(_coordinator_payload(), "current-token-0123456789abcdef0123456789abcdef") as server:
        verify_coordinator_authenticated(
            f"http://127.0.0.1:{server.port}",
            token_file=str(token_file),
            token_env=None,
        )


def test_coordinator_reuse_rejected_with_old_token(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("old-token-0123456789abcdef0123456789abcdef")
    with (
        _CoordinatorServer(_coordinator_payload(), "current-token-0123456789abcdef0123456789abcdef") as server,
        pytest.raises(ProcessIdentityError),
    ):
        verify_coordinator_authenticated(
            f"http://127.0.0.1:{server.port}",
            token_file=str(token_file),
            token_env=None,
        )


def test_coordinator_reuse_rejected_when_project_mismatches(tmp_path):
    token = "current-token-0123456789abcdef0123456789abcdef"
    token_file = tmp_path / "token"
    token_file.write_text(token)
    payload = _coordinator_payload()
    payload["project"] = "some-other-project"
    with _CoordinatorServer(payload, token) as server, pytest.raises(ProcessIdentityError):
        verify_coordinator_authenticated(
            f"http://127.0.0.1:{server.port}",
            token_file=str(token_file),
            token_env=None,
        )


# ---------------- CLI behaviour ----------------
def test_cli_read_pid_prints_valid_pid(tmp_path):
    pid_file = tmp_path / "p.pid"
    pid_file.write_text("  42  ")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "process_identity.py"), "read-pid", "--pid-file", str(pid_file)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "42"


def test_cli_rejects_non_numeric_pid(tmp_path):
    pid_file = tmp_path / "p.pid"
    pid_file.write_text("abc")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "process_identity.py"), "read-pid", "--pid-file", str(pid_file)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "[FAIL]" in result.stdout


# ---------------- pidfd race-safe signal authority (P0-06) ----------------
def _spawn_sigterm_ignoring(argv0: str, args: list[str], cwd: str):
    code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)"
    cmd = ["bash", "-c", f"exec -a {argv0} {sys.executable} -c '{code}' {' '.join(args)}", "_"]
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        proc.wait(timeout=3)
        raise AssertionError("fixture exited early")
    except subprocess.TimeoutExpired:
        return proc


def _run_stop_process(pid_file: Path, *, term_grace: int = 20, kill_grace: int = 10, model_path: str | None = None):
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "process_identity.py"),
        "stop-process",
        "--pid-file",
        str(pid_file),
        "--kind",
        "t2i_worker",
        "--project-root",
        str(ROOT),
        "--port",
        "8101",
        "--device",
        "cuda:0",
        "--term-grace",
        str(term_grace),
        "--kill-grace",
        str(kill_grace),
        "--step",
        "1",
    ]
    if model_path:
        cmd += ["--model-path", model_path]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)


def test_pidfd_support_available_on_linux():
    assert PIDFD_SUPPORTED is True


def test_pidfd_fdinfo_references_expected_pid():
    proc = _spawn_fixture(
        "server/workers/t2i_runtime_server.py",
        ["--host", "127.0.0.1", "--port", "8101", "--device", "cuda:0", "--model-path", "/models/t2i"],
        str(ROOT),
    )
    try:
        fd = open_pidfd(proc.pid)
        try:
            assert _fdinfo_pid(fd) == proc.pid
        finally:
            os.close(fd)
    finally:
        proc.kill()


def test_pidfd_reused_pid_cannot_be_signalled():
    # Once the tracked task has exited AND been reaped, the pidfd no longer
    # resolves to any task (ESRCH), so a subsequent pidfd signal can never reach
    # a different process that happened to reuse the numeric PID.
    proc = _spawn_fixture(
        "server/workers/t2i_runtime_server.py",
        ["--host", "127.0.0.1", "--port", "8101", "--device", "cuda:0", "--model-path", "/models/t2i"],
        str(ROOT),
    )
    fd = open_pidfd(proc.pid)
    try:
        assert _fdinfo_pid(fd) == proc.pid
        proc.kill()
        proc.wait(timeout=15)
        with pytest.raises(ProcessLookupError):
            signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)


def test_pidfd_zombie_observed_as_exited():
    # A process that was killed but not reaped is a zombie: it cannot be
    # signalled/escalated against, and the pidfd wait must not report a
    # lingering "timeout" that could suggest escalation.
    proc = _spawn_fixture(
        "server/workers/t2i_runtime_server.py",
        ["--host", "127.0.0.1", "--port", "8101", "--device", "cuda:0", "--model-path", "/models/t2i"],
        str(ROOT),
    )
    try:
        proc.kill()
        fd = open_pidfd(proc.pid)
        try:
            assert pidfd_wait_exit(fd, proc.pid, 5, 1) != "timeout"
        finally:
            os.close(fd)
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_stop_process_escalates_to_sigkill_when_term_ignored(tmp_path):
    proc = _spawn_sigterm_ignoring(
        "server/workers/t2i_runtime_server.py",
        ["--host", "127.0.0.1", "--port", "8101", "--device", "cuda:0", "--model-path", "/models/t2i"],
        str(ROOT),
    )
    pid_file = tmp_path / "t2i.pid"
    pid_file.write_text(str(proc.pid))
    try:
        result = _run_stop_process(pid_file, term_grace=1, kill_grace=5, model_path="/models/t2i")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "[PASS] t2i_worker_STOPPED" in result.stdout
        assert "escalating to pidfd SIGKILL" in result.stdout
        assert proc.wait(timeout=15) == -signal.SIGKILL
        assert not pid_file.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
