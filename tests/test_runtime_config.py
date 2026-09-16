"""Subprocess tests against the real shell configuration authority.

These tests exercise ``scripts/runtime_config.py`` exactly the way the
production lifecycle scripts invoke it, so the shell start/stop paths are
proven to consume the shared ``server.config`` parsers (P0-02, P0-03).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNTIME_CONFIG = ROOT / "scripts" / "runtime_config.py"


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNTIME_CONFIG), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


# ---------------- port triple validation (P0-02) ----------------
def test_ports_defaults_pass():
    result = _run("validate-ports", "--rest", "8090", "--t2i", "8101", "--edit", "8102")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[PASS]" in result.stdout


def test_ports_custom_non_colliding_pass():
    result = _run("validate-ports", "--rest", "9000", "--t2i", "9101", "--edit", "9202")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("rest", "t2i", "edit"),
    [
        ("1", "1", "2"),
        ("1", "2", "1"),
        ("1", "2", "2"),
    ],
)
def test_ports_collision_rejected(rest, t2i, edit):
    result = _run("validate-ports", "--rest", rest, "--t2i", t2i, "--edit", edit)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "collide" in result.stdout or "collide" in result.stderr


@pytest.mark.parametrize("bad", ["0", "65536", "-1", "abc", "1.5", "", "   ", "1e3", "0x50"])
@pytest.mark.parametrize("which", ["--rest", "--t2i", "--edit"])
def test_port_position_rejects_invalid_value(which, bad):
    args = {"--rest": "8090", "--t2i": "8101", "--edit": "8102"}
    args[which] = bad
    result = _run("validate-ports", "--rest", args["--rest"], "--t2i", args["--t2i"], "--edit", args["--edit"])
    assert result.returncode == 1, result.stdout + result.stderr
    assert "[FAIL]" in result.stdout


def test_single_port_validation_pass():
    result = _run("validate-port", "--name", "MAGE_FLOW_T2I_INTERNAL_PORT", "--value", "8101")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bad", ["0", "65536", "-1", "abc", "1.5", "  "])
def test_single_port_validation_rejects(bad):
    result = _run("validate-port", "--name", "MAGE_FLOW_T2I_INTERNAL_PORT", "--value", bad)
    assert result.returncode == 1, result.stdout + result.stderr


# ---------------- lifecycle timeout validation (P0-03) ----------------
def test_timeout_default_passes():
    assert (
        _run(
            "validate-timeout",
            "--name",
            "MAGE_FLOW_T2I_START_TIMEOUT_SECONDS",
            "--value",
            "900",
            "--default",
            "900",
            "--upper-bound",
            "7200",
        ).returncode
        == 0
    )
    assert (
        _run(
            "validate-timeout",
            "--name",
            "MAGE_FLOW_T2I_START_TIMEOUT_SECONDS",
            "--value",
            "",
            "--default",
            "900",
            "--upper-bound",
            "7200",
        ).returncode
        == 0
    )


def test_timeout_valid_override_passes():
    result = _run(
        "validate-timeout",
        "--name",
        "MAGE_FLOW_STOP_TERM_GRACE_SECONDS",
        "--value",
        "20",
        "--default",
        "20",
        "--upper-bound",
        "7200",
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bad", ["0", "-1", "-20", "nan", "inf", "-inf", "abc", "1.5", "7201", "99999999"])
def test_timeout_rejects_invalid_values(bad):
    result = _run(
        "validate-timeout",
        "--name",
        "MAGE_FLOW_T2I_START_TIMEOUT_SECONDS",
        f"--value={bad}",
        "--default",
        "900",
        "--upper-bound",
        "7200",
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "[FAIL]" in result.stdout


# ---------------- token validation (P0-01) ----------------
def test_token_valid_passes():
    result = _run("validate-token", "--value", "a" * 32)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bad", ["a" * 31, "", " " * 32, "a" * 31 + "\n", "a" * 31 + "\r"])
def test_token_invalid_rejected(bad):
    result = _run("validate-token", f"--value={bad}")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "[FAIL]" in result.stdout


def test_token_with_nul_rejected_via_python_authority():
    # A NUL cannot cross the subprocess argv boundary, so the fail-closed
    # rejection is proven at the Python authority level (and for persisted files
    # in tests/test_token_store.py).
    from server.auth import validate_public_token
    from server.token_contract import TokenContractError

    with pytest.raises(ValueError):
        validate_public_token("a" * 31 + "\x00")
    with pytest.raises(TokenContractError):
        from server.token_contract import validate_public_token_value

        validate_public_token_value("a" * 31 + "\x00")


# ---------------- resolve-model-path (P0-04 single authority) ----------------
def test_resolve_model_path_t2i_default():
    env = dict(os.environ)
    env.pop("MAGE_FLOW_T2I_MODEL_PATH", None)
    env.pop("MAGE_FLOW_EDIT_MODEL_PATH", None)
    result = _run("resolve-model-path", "--kind", "t2i", env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == (
        "/kaggle/input/models/dangkhoa2016/mage-flow-community-mage-flow-turbo/pytorch/default/1"
    )


def test_resolve_model_path_edit_default():
    env = dict(os.environ)
    env.pop("MAGE_FLOW_T2I_MODEL_PATH", None)
    env.pop("MAGE_FLOW_EDIT_MODEL_PATH", None)
    result = _run("resolve-model-path", "--kind", "edit", env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == (
        "/kaggle/input/models/dangkhoa2016/mage-flow-community-mage-flow-edit-turbo/pytorch/default/1"
    )


def test_resolve_model_path_env_override():
    env = dict(os.environ)
    env["MAGE_FLOW_EDIT_MODEL_PATH"] = "/custom/models/edit"
    env.pop("MAGE_FLOW_T2I_MODEL_PATH", None)
    result = _run("resolve-model-path", "--kind", "edit", env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "/custom/models/edit"


def test_resolve_model_path_unknown_kind_rejected():
    result = _run("resolve-model-path", "--kind", "nope")
    assert result.returncode != 0


# ---------------- production shell wiring ----------------
def test_start_sh_validates_token_and_ports():
    text = (ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")
    assert "validate-token" in text and "MAGE_FLOW_API_TOKEN" in text
    assert "validate-ports" in text and "--rest" in text and "--t2i" in text and "--edit" in text


def test_start_scripts_validate_when_resolved_from_memory():
    t2i = (ROOT / "scripts" / "start_t2i.sh").read_text(encoding="utf-8")
    edit = (ROOT / "scripts" / "start_edit.sh").read_text(encoding="utf-8")
    assert "validate-port" in t2i and "validate-timeout" in t2i
    assert "validate-port" in edit and "validate-timeout" in edit


def test_stop_scripts_validate_port_before_identity():
    for name in ("stop_t2i.sh", "stop_edit.sh"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "validate-port" in text
        assert "stop_process.sh" in text


def test_stop_process_validates_timeouts_and_port():
    text = (ROOT / "scripts" / "stop_process.sh").read_text(encoding="utf-8")
    assert "validate-timeout" in text
    assert "validate-port" in text


def test_run_public_candidate_validates_ports_and_token():
    text = (ROOT / "scripts" / "run_public_candidate.sh").read_text(encoding="utf-8")
    assert "validate-ports" in text
    assert "validate-token" in text


def test_start_sh_short_token_fails_before_coordinator_launch(tmp_path):
    env = dict(os.environ)
    env["MAGE_FLOW_API_TOKEN"] = "abc"
    result = subprocess.run(
        ["bash", "scripts/start.sh"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "[FAIL]" in combined
    assert "Starting authenticated REST coordinator" not in combined
