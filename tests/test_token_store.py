import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.token_store import (
    TokenStoreError,
    enforce_directory,
    enforce_token_file,
    ensure_token,
    verify_token,
)

ROOT = Path(__file__).resolve().parent.parent


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_ensure_token_creates_dir_and_token_securely(tmp_path):
    token_file = tmp_path / ".runtime" / "api_token"
    token = ensure_token(token_file)
    assert len(token) == 43
    assert token_file.is_file()
    assert _mode(token_file) == 0o600
    assert _mode(token_file.parent) == 0o700
    assert token_file.read_text() == token


def test_ensure_token_returns_existing_value(tmp_path):
    token_file = tmp_path / ".runtime" / "api_token"
    first = ensure_token(token_file)
    second = ensure_token(token_file)
    assert second == first
    assert token_file.read_text() == first


def test_ensure_token_fixes_overly_permissive_existing_file(tmp_path):
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    token_file = runtime / "api_token"
    existing = "existing-secret-0123456789abcdef0123456789abcdef"
    token_file.write_text(existing)
    os.chmod(token_file, 0o644)
    os.chmod(runtime, 0o755)
    assert ensure_token(token_file) == existing
    assert _mode(token_file) == 0o600
    assert _mode(runtime) == 0o700


def test_verify_token_rejects_symlink(tmp_path):
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    real = runtime / "real-token"
    real.write_text("secret")
    link = runtime / "api_token"
    os.symlink(real.name, link)
    with pytest.raises(TokenStoreError):
        verify_token(link)


def test_enforce_directory_rejects_symlinked_dir(tmp_path):
    target = tmp_path / "real-runtime"
    target.mkdir()
    link = tmp_path / ".runtime"
    os.symlink(target.name, link)
    with pytest.raises(TokenStoreError):
        enforce_directory(link)


def test_verify_token_rejects_empty_file(tmp_path):
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    token_file = runtime / "api_token"
    token_file.write_text("")
    os.chmod(token_file, 0o600)
    with pytest.raises(TokenStoreError):
        verify_token(tmp_path / ".runtime" / "api_token")


def test_enforce_token_rejects_extra_hardlink(tmp_path):
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    token_file = runtime / "api_token"
    token_file.write_text("secret")
    os.link(token_file, runtime / "second-link")
    with pytest.raises(TokenStoreError):
        enforce_token_file(token_file)


def test_cli_ensure_token_prints_only_token(tmp_path):
    token_file = tmp_path / "runtime" / "api_token"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "token_store.py"), "ensure-token", str(token_file)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == token_file.read_text().strip()
    assert "secret" not in result.stderr.lower()


def test_cli_prepare_dir_enforces_mode(tmp_path):
    directory = tmp_path / "runtime"
    directory.mkdir()
    os.chmod(directory, 0o755)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "token_store.py"), "prepare-dir", str(directory)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert _mode(directory) == 0o700


# ---------------- shared token-value contract ----------------
def _write_persisted_token(tmp_path, value: str) -> Path:
    runtime = tmp_path / ".runtime"
    runtime.mkdir(exist_ok=True)
    token_file = runtime / "api_token"
    token_file.write_text(value, encoding="utf-8")
    os.chmod(token_file, 0o600)
    return token_file


@pytest.mark.parametrize(
    "value",
    [
        "a" * 31,
        "",
        " " * 32,
        "\t" * 32,
        "a" * 31 + "\n",
        "a" * 31 + "\r",
        "a" * 31 + "\x00",
        " " + "a" * 32,
        "a" * 32 + " ",
        "a" * 32 + "\n",
    ],
)
def test_persisted_token_value_contract_fails_closed(tmp_path, value):
    token_file = _write_persisted_token(tmp_path, value)
    with pytest.raises(TokenStoreError):
        verify_token(token_file)
    with pytest.raises(TokenStoreError):
        ensure_token(token_file)


@pytest.mark.parametrize("value", ["a" * 32, "a" * 64, "abc-123-" * 4])
def test_persisted_valid_token_value_accepted(tmp_path, value):
    token_file = _write_persisted_token(tmp_path, value)
    verify_token(token_file)
    assert ensure_token(token_file) == value


def test_new_generated_token_meets_public_contract(tmp_path):
    token = ensure_token(tmp_path / ".runtime" / "api_token")
    assert len(token) >= 32
    from server.token_contract import validate_public_token_value

    assert validate_public_token_value(token) == token


def test_server_auth_and_token_store_agree_on_valid_value():
    from server.auth import validate_public_token
    from server.token_contract import validate_public_token_value

    good = "a" * 32
    assert validate_public_token(good) == good
    assert validate_public_token_value(good) == good


def test_server_auth_and_token_store_agree_on_invalid_values():
    from server.auth import validate_public_token
    from server.token_contract import TokenContractError, validate_public_token_value

    for bad in ("a" * 31, "", " " * 32, "a" * 32 + "\n", "a" * 32 + "\r", "a" * 32 + "\x00"):
        with pytest.raises(ValueError):
            validate_public_token(bad)
        with pytest.raises(TokenContractError):
            validate_public_token_value(bad)
