import json
import subprocess
import sys
from pathlib import Path

import pytest

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


def _valid_project_state() -> dict:
    return {
        "runtime": {
            "accelerator": "Kaggle NVIDIA T4 x2",
            "t2i_device": "cuda:0",
            "edit_device": "cuda:1",
            "cpu_fallback": False,
        },
        "t2i_integration": {
            "device": "cuda:0",
            "internal_url": "http://127.0.0.1:8101",
            "output_format": "data:image/png;base64",
            "acceptance_width": 1024,
            "acceptance_height": 1024,
            "request_timeout_seconds": 3600,
        },
        "edit_integration": {
            "device": "cuda:1",
            "internal_url": "http://127.0.0.1:8102",
            "output_format": "data:image/png;base64",
            "max_size": 1024,
        },
        "license_status": "MIT",
        "public_notebook": "notebooks/mage-flow-turbo-and-mage-flow-edit-turbo-on-t4x2-gpu.ipynb",
        "github_ci_configured": True,
        "github_community_metadata_configured": True,
        "public_notebook_validation": {
            "structural_validation": True,
            "live_run_all": False,
            "saved_version_verified": False,
        },
    }


def _build_repo(tmp_path):
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
    (root / "scripts/run_public_acceptance.sh").write_text("")
    (root / "scripts/run_public_acceptance.sh").chmod(0o755)
    (root / "tests/test_acceptance.py").write_text("")
    (root / ".github/workflows/ci.yml").write_text("")
    (root / ".github/SECURITY.md").write_text("")
    state_file = root / "project_state.json"
    state_file.write_text(json.dumps(_valid_project_state()) + "\n", encoding="utf-8")
    return root


def test_repository_validator_passes_on_repository():
    result = _run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[PASS] REPOSITORY_HYGIENE_VALID" in result.stdout
    assert "[PASS] PROJECT_STATE_MACHINE_AUTHORITY" in result.stdout
    assert "[PASS] PROJECT_STATE_PUBLICATION_FLAGS_TRUTHFUL" in result.stdout


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


def test_repository_validator_rejects_project_state_drift_default_root(tmp_path):
    root = _build_repo(tmp_path)
    good = _run("--root", str(root))
    assert good.returncode == 0, good.stdout + good.stderr
    state = json.loads((root / "project_state.json").read_text(encoding="utf-8"))
    state["runtime"]["cpu_fallback"] = True
    (root / "project_state.json").write_text(json.dumps(state) + "\n", encoding="utf-8")
    bad = _run("--root", str(root))
    assert bad.returncode != 0
    assert "cpu_fallback" in bad.stdout


@pytest.mark.parametrize(
    "path,value,fragment",
    [
        (("t2i_integration", "acceptance_width"), 512, "acceptance_width"),
        (("t2i_integration", "acceptance_height"), 768, "acceptance_height"),
        (("t2i_integration", "request_timeout_seconds"), 60, "request_timeout_seconds"),
        (("edit_integration", "max_size"), 512, "max_size"),
        (("t2i_integration", "output_format"), "image/jpeg", "output_format"),
        (("edit_integration", "output_format"), "image/webp", "output_format"),
        (("runtime", "t2i_device"), "cuda:1", "t2i_device"),
        (("runtime", "edit_device"), "cuda:0", "edit_device"),
        (("runtime", "cpu_fallback"), True, "cpu_fallback"),
        (("t2i_integration", "internal_url"), "http://127.0.0.1:9999", "internal_url"),
        (("edit_integration", "internal_url"), "http://127.0.0.1:9999", "internal_url"),
        (("public_notebook_validation", "structural_validation"), False, "structural_validation"),
        (("public_notebook_validation", "live_run_all"), True, "live_run_all"),
        (("public_notebook_validation", "saved_version_verified"), True, "saved_version_verified"),
        (("runtime", "accelerator"), "NVIDIA A100", "accelerator"),
    ],
)
def test_repository_validator_rejects_project_state_machine_drift(tmp_path, path, value, fragment):
    root = _build_repo(tmp_path)
    assert _run("--root", str(root)).returncode == 0
    state = json.loads((root / "project_state.json").read_text(encoding="utf-8"))
    node = state
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    (root / "project_state.json").write_text(json.dumps(state) + "\n", encoding="utf-8")
    result = _run("--root", str(root))
    assert result.returncode != 0
    assert fragment in result.stdout
