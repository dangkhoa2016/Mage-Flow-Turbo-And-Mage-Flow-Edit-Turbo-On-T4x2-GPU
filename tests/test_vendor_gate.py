import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = ROOT / "scripts" / "validate_vendor_source.py"
CANONICAL_MAGE_COMMIT = "76bec2bb3818863f470de7e867c2dc7f1d0bfd83"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_vendor_source", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_cli(source: Path, env_extra: dict[str, str] | None = None):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(source)],
        capture_output=True,
        text=True,
        env=env,
    )


def _init_repo(tmp_path, commit: bool = True) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    if commit:
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test User"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "--allow-empty", "-q", "-m", "seed"], check=True)
    return repo


def test_canonical_commit_constant_is_immutable_and_single_source():
    validator = _load_validator()
    assert validator.MAGE_CANONICAL_COMMIT == CANONICAL_MAGE_COMMIT
    orchestrator = (ROOT / "scripts" / "run_public_acceptance.sh").read_text(encoding="utf-8")
    assert "EXPECTED_MAGE_COMMIT" not in orchestrator
    assert "ACTUAL_COMMIT=" not in orchestrator
    assert CANONICAL_MAGE_COMMIT not in orchestrator


def test_validator_never_reads_environment():
    source = VALIDATOR.read_text(encoding="utf-8")
    assert "os.environ" not in source
    assert "getenv" not in source
    assert "from os import" not in source
    assert "import os" not in source


def test_matching_expected_commit_passes(tmp_path):
    repo = _init_repo(tmp_path)
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"]).decode().strip()
    validator = _load_validator()
    ok, message = validator.check_vendor_commit(repo, expected_commit=actual)
    assert ok is True
    assert message == actual


def test_cli_default_enforces_immutable_canonical_commit(tmp_path):
    repo = _init_repo(tmp_path)
    result = _run_cli(repo)
    assert result.returncode != 0
    assert "[FAIL]" in result.stdout
    assert "commit mismatch" in result.stdout
    assert CANONICAL_MAGE_COMMIT in result.stdout


def test_wrong_commit_fails(tmp_path):
    repo = _init_repo(tmp_path)
    result = _run_cli(repo)
    assert result.returncode != 0
    assert "[FAIL]" in result.stdout


def test_environment_cannot_redefine_expected_commit(tmp_path):
    repo = _init_repo(tmp_path)
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"]).decode().strip()
    env_extra = {"EXPECTED_MAGE_COMMIT": actual, "MAGE_VENDOR_COMMIT": actual}
    result = _run_cli(repo, env_extra=env_extra)
    assert result.returncode != 0
    assert "commit mismatch" in result.stdout
    assert CANONICAL_MAGE_COMMIT in result.stdout


def test_missing_source_fails(tmp_path):
    missing = tmp_path / "does-not-exist"
    result = _run_cli(missing)
    assert result.returncode != 0
    assert "FAIL" in result.stdout
    assert "vendor source missing" in result.stdout


def test_non_git_source_fails(tmp_path):
    source = tmp_path / "vendor"
    source.mkdir()
    result = _run_cli(source)
    assert result.returncode != 0
    assert "FAIL" in result.stdout
    assert "no resolvable Git HEAD" in result.stdout


def test_unborn_repo_fails(tmp_path):
    repo = _init_repo(tmp_path, commit=False)
    result = _run_cli(repo)
    assert result.returncode != 0
    assert "FAIL" in result.stdout
    assert "no resolvable Git HEAD" in result.stdout


def test_orchestrator_delegates_to_reusable_validator():
    script = (ROOT / "scripts" / "run_public_acceptance.sh").read_text(encoding="utf-8")
    start_script = (ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")
    assert 'python scripts/validate_vendor_source.py "$MAGE_SOURCE"' in script
    assert 'if [[ "$ACTUAL_COMMIT" != "$EXPECTED_MAGE_COMMIT" ]]; then' not in script
    assert "found ${ACTUAL_COMMIT:-<unresolved>}" not in script
    assert 'validate-token --value="$MAGE_FLOW_API_TOKEN"' in script
    assert 'validate-token --value "$MAGE_FLOW_API_TOKEN"' not in script
    assert 'validate-token --value="$MAGE_FLOW_API_TOKEN"' in start_script
    assert 'validate-token --value "$MAGE_FLOW_API_TOKEN"' not in start_script


def test_orchestrator_passes_model_path_to_service_health_cli():
    script = (ROOT / "scripts" / "run_public_acceptance.sh").read_text(encoding="utf-8")
    assert 'python scripts/service_health.py "$url" "$model" "$device" "$model_path"' in script
    assert 'is_endpoint_healthy "$url" "$model" "$device" "$model_path"' in script


def _extract_bash_function(script: str, name: str) -> list[str]:
    lines = script.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"{name}()"):
            start = i
            break
    assert start is not None, f"function {name} not found"
    body = []
    for line in lines[start + 1 :]:
        if line.strip() == "}":
            break
        body.append(line)
    return body


def _run_token_function(tmp_path, prepopulate: str | None):
    script = (ROOT / "scripts" / "run_public_acceptance.sh").read_text(encoding="utf-8")
    body = _extract_bash_function(script, "prepare_api_token")
    token_file = tmp_path / ".runtime" / "api_token"
    if prepopulate is not None:
        token_file.parent.mkdir(parents=True)
        token_file.write_text(prepopulate)
        os.chmod(token_file, 0o644)
        os.chmod(token_file.parent, 0o755)
    bash_src = (
        "set -euo pipefail\n"
        "info() { printf '[INFO] %s\\n' \"$*\"; }\n"
        "prepare_api_token() {\n"
        + "\n".join(body)
        + "\n}\n"
        + f'prepare_api_token "{token_file}"\n'
        + 'printf "%s" "$MAGE_FLOW_API_TOKEN"\n'
    )
    result = subprocess.run(["bash", "-c", bash_src], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return lines[-1], token_file


def _mode(path: Path) -> str:
    return subprocess.check_output(["stat", "-c", "%a", str(path)]).decode().strip()


def test_new_shell_token_gets_0600_and_private_dir(tmp_path):
    value, token_file = _run_token_function(tmp_path, prepopulate=None)
    assert len(value) == 43  # secrets.token_urlsafe(32)
    assert token_file.is_file()
    assert _mode(token_file) == "600"
    assert _mode(token_file.parent) == "700"


def test_reused_shell_token_permission_is_corrected_to_0600(tmp_path):
    token = "-existing-secret-0123456789abcdef0123456789abcdef"
    value, token_file = _run_token_function(tmp_path, prepopulate=token)
    assert value == token
    assert token_file.read_text() == token
    assert _mode(token_file) == "600"
