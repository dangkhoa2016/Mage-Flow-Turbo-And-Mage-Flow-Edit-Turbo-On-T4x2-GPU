import os
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The vendor gate must be fail-closed: an unresolvable or non-Git source must
# fail instead of being accepted by default.
GATE_SNIPPET = r"""
EXPECTED_MAGE_COMMIT='abc123def456'
check_gate() {
  local mage_source="$1"
  ACTUAL_COMMIT="$(git -C "$mage_source" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$ACTUAL_COMMIT" != "$EXPECTED_MAGE_COMMIT" ]]; then
    echo "FAIL: expected $EXPECTED_MAGE_COMMIT, found ${ACTUAL_COMMIT:-<unresolved>}"
    return 1
  fi
  echo "PASS"
  return 0
}
"""


def _run_gate(mage_source: str) -> tuple[int, str]:
    script = textwrap.dedent(GATE_SNIPPET) + f"check_gate '{mage_source}'\n"
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    return result.returncode, result.stdout.strip() + result.stderr.strip()


def test_matching_commit_passes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test User"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "--allow-empty", "-q", "-m", "seed"], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"]).decode().strip()
    script = textwrap.dedent(GATE_SNIPPET) + f"EXPECTED_MAGE_COMMIT='{commit}'\ncheck_gate '{repo}'\n"
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip() == "PASS"


def test_wrong_commit_fails(tmp_path):
    expected = "abc123def456"
    script = textwrap.dedent(GATE_SNIPPET) + f"EXPECTED_MAGE_COMMIT='{expected}'\ncheck_gate '{tmp_path}'\n"
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode != 0
    assert "FAIL" in result.stdout + result.stderr


def test_non_git_path_fails(tmp_path):
    (tmp_path / "vendor" / "Mage").mkdir(parents=True)
    script = textwrap.dedent(GATE_SNIPPET) + f"check_gate '{tmp_path / 'vendor' / 'Mage'}'\n"
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode != 0
    assert "FAIL" in result.stdout + result.stderr
    assert "<unresolved>" in result.stdout + result.stderr


def test_unresolvable_head_fails(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    script = textwrap.dedent(GATE_SNIPPET) + f"check_gate '{repo}'\n"
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode != 0
    assert "FAIL" in result.stdout + result.stderr


def test_missing_source_fails(tmp_path):
    missing = tmp_path / "does-not-exist"
    script = textwrap.dedent(GATE_SNIPPET) + f"check_gate '{missing}'\n"
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode != 0
    assert "FAIL" in result.stdout + result.stderr
    assert "<unresolved>" in result.stdout + result.stderr


def test_orchestrator_uses_fail_closed_gate():
    script = (ROOT / "scripts" / "run_public_candidate.sh").read_text(encoding="utf-8")
    assert 'if [[ "$ACTUAL_COMMIT" != "$EXPECTED_MAGE_COMMIT" ]]; then' in script
    assert "found ${ACTUAL_COMMIT:-<unresolved>}" in script
    assert '[[ -n "$ACTUAL_COMMIT" &&' not in script


def test_expected_mage_commit_is_immutable_project_constant():
    script = (ROOT / "scripts" / "run_public_candidate.sh").read_text(encoding="utf-8")
    assert 'EXPECTED_MAGE_COMMIT="76bec2bb3818863f470de7e867c2dc7f1d0bfd83"' in script
    assert 'EXPECTED_MAGE_COMMIT="${EXPECTED_MAGE_COMMIT:-' not in script
    assert "EXPECTED_MAGE_COMMIT:-" not in script


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
    script = (ROOT / "scripts" / "run_public_candidate.sh").read_text(encoding="utf-8")
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
    value, token_file = _run_token_function(tmp_path, prepopulate="existing-secret")
    assert value == "existing-secret"
    assert token_file.read_text() == "existing-secret"
    assert _mode(token_file) == "600"


def test_orchestrator_passes_model_path_to_service_health_cli():
    script = (ROOT / "scripts" / "run_public_candidate.sh").read_text(encoding="utf-8")
    assert 'python scripts/service_health.py "$url" "$model" "$device" "$model_path"' in script
    assert 'is_endpoint_healthy "$url" "$model" "$device" "$model_path"' in script
