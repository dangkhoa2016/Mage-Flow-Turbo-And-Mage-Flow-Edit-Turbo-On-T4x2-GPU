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
    assert "--model-path" in text
    assert "resolve-model-path --kind t2i" in text


def test_stop_edit_delegates_to_shared_guard():
    text = (SCRIPTS / "stop_edit.sh").read_text(encoding="utf-8")
    assert "bash scripts/stop_process.sh" in text
    assert "--kind edit_worker" in text
    assert "--device cuda:1" in text
    assert "--model-path" in text
    assert "resolve-model-path --kind edit" in text


def test_stop_coordinator_delegates_to_shared_guard():
    text = (SCRIPTS / "stop.sh").read_text(encoding="utf-8")
    assert "bash scripts/stop_process.sh" in text
    assert "--kind coordinator" in text


def test_shared_guard_signal_path_is_pidfd_only(monkeypatch):
    guard = (SCRIPTS / "process_identity.py").read_text(encoding="utf-8")
    assert "read_pid_file" in guard
    assert "/proc/" in guard
    assert "os.kill(" not in guard
    assert "subprocess" not in guard
    assert "pidfd_open" in guard
    assert "pidfd_send_signal" in guard
    assert "stop-process" in guard
    assert "fail closed" in guard


def test_shared_stop_wrapper_requires_pidfd_authority():
    wrapper = (SCRIPTS / "stop_process.sh").read_text(encoding="utf-8")
    assert "pidfd-support" in wrapper
    assert "stop-process" in wrapper
    assert "UNAVAILABLE_FAIL_CLOSED" in wrapper
    assert "identity" in wrapper.lower()


def test_shared_stop_wrapper_has_no_numeric_fallback_loop():
    wrapper = (SCRIPTS / "stop_process.sh").read_text(encoding="utf-8")
    assert "kill --" not in wrapper
    assert "kill -9 --" not in wrapper
