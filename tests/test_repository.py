import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = ROOT / "scripts" / "validate_repository.py"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(VALIDATOR), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_repository_validator_passes_on_repository():
    result = _run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[PASS] REPOSITORY_HYGIENE_VALID" in result.stdout


def test_repository_validator_rejects_tracked_generated_paths(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("x\n", encoding="utf-8")
    (root / ".runtime").mkdir()
    (root / ".runtime" / "api_token").write_text("secret\n", encoding="utf-8")
    result = _run("--root", str(root))
    assert result.returncode != 0
    assert "tracked" in result.stdout


def test_repository_validator_rejects_missing_final_newline(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("no trailing newline", encoding="utf-8")
    result = _run("--root", str(root))
    assert result.returncode != 0
    assert "final newline" in result.stdout


def test_repository_validator_rejects_project_state_drift(tmp_path):
    root = tmp_path / "repo"
    for directory in (
        "notebooks",
        "server",
        "scripts",
        "tests",
        ".github/workflows",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for file in (
        "LICENSE",
        "README.md",
        "README.vi.md",
        "pyproject.toml",
        ".gitignore",
    ):
        (root / file).write_text("", encoding="utf-8")
    (root / "notebooks/mage-flow-turbo-and-mage-flow-edit-turbo-on-t4x2-gpu.ipynb").write_text("{}\n")
    (root / "server/app.py").write_text("")
    (root / "scripts/run_public_candidate.sh").write_text("")
    (root / "scripts/run_public_candidate.sh").chmod(0o755)
    (root / "tests/test_acceptance.py").write_text("")
    (root / ".github/workflows/ci.yml").write_text("")
    (root / ".github/SECURITY.md").write_text("")
    project_state = {
        "runtime": {
            "accelerator": "Kaggle NVIDIA T4 x2",
            "t2i_device": "cuda:0",
            "edit_device": "cuda:1",
            "cpu_fallback": False,
        },
        "t2i_integration": {
            "device": "cuda:0",
            "internal_url": "http://127.0.0.1:8101",
        },
        "edit_integration": {
            "device": "cuda:1",
            "internal_url": "http://127.0.0.1:8102",
        },
        "license_status": "MIT",
        "public_notebook": "notebooks/mage-flow-turbo-and-mage-flow-edit-turbo-on-t4x2-gpu.ipynb",
        "github_ci_configured": True,
        "github_community_metadata_configured": True,
    }
    state_file = root / "project_state.json"
    state_file.write_text(__import__("json").dumps(project_state) + "\n", encoding="utf-8")
    good = _run("--root", str(root))
    assert good.returncode == 0, good.stdout + good.stderr
    project_state["runtime"]["cpu_fallback"] = True
    state_file.write_text(__import__("json").dumps(project_state) + "\n", encoding="utf-8")
    bad = _run("--root", str(root))
    assert bad.returncode != 0
    assert "cpu_fallback" in bad.stdout
