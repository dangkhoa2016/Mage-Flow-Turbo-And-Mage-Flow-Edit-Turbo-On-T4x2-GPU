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
    token_file.write_text("existing-secret")
    os.chmod(token_file, 0o644)
    os.chmod(runtime, 0o755)
    assert ensure_token(token_file) == "existing-secret"
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