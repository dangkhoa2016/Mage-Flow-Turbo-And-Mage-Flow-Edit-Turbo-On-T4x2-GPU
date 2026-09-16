from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

STOP_SCRIPT_COMMON = (
    "stop scripts must delegate to the shared identity guard so a stale pid "
    "file cannot silently stop a running worker/coordinator"
)


def test_stop_t2i_delegates_to_shared_guard():
    text = (SCRIPTS / "stop_t2i.sh").read_text(encoding="utf-8")
    assert "bash scripts/stop_process.sh" in text
    assert "--kind t2i_worker" in text
    assert "--device cuda:0" in text


def test_stop_edit_delegates_to_shared_guard():
    text = (SCRIPTS / "stop_edit.sh").read_text(encoding="utf-8")
    assert "bash scripts/stop_process.sh" in text
    assert "--kind edit_worker" in text
    assert "--device cuda:1" in text


def test_stop_coordinator_delegates_to_shared_guard():
    text = (SCRIPTS / "stop.sh").read_text(encoding="utf-8")
    assert "bash scripts/stop_process.sh" in text
    assert "--kind coordinator" in text


def test_shared_guard_never_falls_back_to_blind_kill(monkeypatch):
    guard = (SCRIPTS / "process_identity.py").read_text(encoding="utf-8")
    assert "read_pid_file" in guard
    assert "/proc/" in guard
    assert "os.kill(" not in guard
    assert "subprocess" not in guard
    assert "signal.SIGKILL" not in guard


def test_shared_stop_wrapper_is_present():
    wrapper = (SCRIPTS / "stop_process.sh").read_text(encoding="utf-8")
    assert "kill --" in wrapper
    assert "kill -9 --" in wrapper
    assert "identity" in wrapper.lower()
