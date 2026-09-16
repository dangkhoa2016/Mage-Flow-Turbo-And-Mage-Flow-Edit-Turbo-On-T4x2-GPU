import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STOP_WRAPPER = ROOT / "scripts" / "stop_process.sh"


def _spawn_sleeper(argv0: str, args: list[str], cwd: str):
    cmd = ["bash", "-c", f"exec -a {argv0} {sys.executable} -c 'import time; time.sleep(300)' {' '.join(args)}", "_"]
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        proc.wait(timeout=3)
        raise AssertionError("fixture exited early")
    except subprocess.TimeoutExpired:
        return proc


def _run_stop(kind: str, pid_file: Path, *, port: str | None = None, device: str | None = None):
    cmd = [
        "bash",
        str(STOP_WRAPPER),
        "--pid-file",
        str(pid_file),
        "--kind",
        kind,
        "--project-root",
        str(ROOT),
    ]
    if port:
        cmd += ["--port", port]
    if device:
        cmd += ["--device", device]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)


def test_stop_worker_terms_and_removes_pid_file(tmp_path):
    proc = _spawn_sleeper(
        "server/workers/t2i_runtime_server.py",
        ["--host", "127.0.0.1", "--port", "8101", "--device", "cuda:0", "--model-path", "/models/t2i"],
        str(ROOT),
    )
    pid_file = tmp_path / "t2i.pid"
    pid_file.write_text(str(proc.pid))
    try:
        result = _run_stop("t2i_worker", pid_file, port="8101", device="cuda:0")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "[PASS] t2i_worker_STOPPED" in result.stdout
        assert not pid_file.exists()
        assert proc.wait(timeout=15) == -signal.SIGTERM
    finally:
        if _is_alive(proc.pid):
            proc.kill()


def test_stop_refuses_unrelated_process(tmp_path):
    proc = _spawn_sleeper(
        "some/unrelated/tool.py",
        ["--host", "127.0.0.1", "--port", "8101", "--device", "cuda:0", "--model-path", "/models/t2i"],
        str(ROOT),
    )
    pid_file = tmp_path / "t2i.pid"
    pid_file.write_text(str(proc.pid))
    try:
        result = _run_stop("t2i_worker", pid_file, port="8101", device="cuda:0")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "without killing" in result.stdout
        assert not pid_file.exists()
        assert _is_alive(proc.pid)
    finally:
        if _is_alive(proc.pid):
            proc.kill()


def test_stop_cleans_non_numeric_pid_file(tmp_path):
    pid_file = tmp_path / "t2i.pid"
    pid_file.write_text("not-a-pid")
    result = _run_stop("t2i_worker", pid_file)
    assert result.returncode == 0
    assert "without killing" in result.stdout
    assert not pid_file.exists()


def test_stop_handles_stale_dead_pid(tmp_path):
    pid_file = tmp_path / "t2i.pid"
    pid_file.write_text("314159265358")
    result = _run_stop("t2i_worker", pid_file)
    assert result.returncode == 0
    assert "without killing" in result.stdout or "STOPPED" in result.stdout
    assert not pid_file.exists()


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False