from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

STOP_SCRIPT_COMMON = (
    "must refuse to kill a pid whose /proc cmdline does not match its own server identity, "
    "so a stale pid file cannot silently stop a running worker or coordinator"
)


def test_stop_t2i_checks_process_identity_before_kill():
    text = (SCRIPTS / "stop_t2i.sh").read_text(encoding="utf-8")
    assert 'grep -q "t2i_runtime_server" "/proc/$pid/cmdline"' in text, STOP_SCRIPT_COMMON
    assert "removing stale pid file without killing" in text


def test_stop_edit_checks_process_identity_before_kill():
    text = (SCRIPTS / "stop_edit.sh").read_text(encoding="utf-8")
    assert 'grep -q "edit_runtime_server" "/proc/$pid/cmdline"' in text, STOP_SCRIPT_COMMON
    assert "removing stale pid file without killing" in text


def test_stop_coordinator_checks_process_identity_before_kill():
    text = (SCRIPTS / "stop.sh").read_text(encoding="utf-8")
    assert 'grep -q "server.app:app" "/proc/$pid/cmdline"' in text, STOP_SCRIPT_COMMON
    assert "removing stale pid file without killing" in text