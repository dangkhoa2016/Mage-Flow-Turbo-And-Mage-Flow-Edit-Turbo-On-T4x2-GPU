#!/usr/bin/env python
"""Validate repository hygiene and project_state.json consistency.

Offline checks:
  1. No generated/temporary/secret files are tracked by git.
  2. Core repository files exist (LICENSE, bilingual README, pyproject.toml,
     .gitignore, project_state.json, source notebook, server/scripts/tests).
  3. project_state.json fields are type-consistent and do not drift from the
     repository reality (devices, internal worker URLs, CPU fallback disabled,
     frozen acceptance profile, output-format contract, truthful publication
     flags, MIT license, public notebook path, CI/community metadata files).

Usage:
    python scripts/validate_repository.py [--root DIR]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HARMFUL_PREFIXES = (
    ".runtime/",
    "outputs/",
    "artifacts/",
    "dist/",
    "build/",
    ".egg-info/",
    ".pytest_cache/",
    ".venv",
    ".venv-authority",
    "venv/",
    "__pycache__/",
    ".ipynb_checkpoints/",
    ".mypy_cache/",
    ".ruff_cache/",
)
HARMFUL_SUFFIXES = (".log", ".pid", ".pyc", ".tmp", ".bak", ".orig", ".swp")

MANDATORY_PATHS = (
    "LICENSE",
    "README.md",
    "README.vi.md",
    "pyproject.toml",
    ".gitignore",
    "project_state.json",
    "notebooks/mage-flow-t4x2-production-rest-api-demo.ipynb",
    "server/app.py",
    "scripts/run_public_acceptance.sh",
    "tests/test_acceptance.py",
    ".github/workflows/ci.yml",
)

# project_state.json machine authority: the frozen worker/coordinator profile.
T2I_DEVICE = "cuda:0"
EDIT_DEVICE = "cuda:1"
T2I_INTERNAL_URL = "http://127.0.0.1:8101"
EDIT_INTERNAL_URL = "http://127.0.0.1:8102"
T2I_ACCEPTANCE_WIDTH = 1024
T2I_ACCEPTANCE_HEIGHT = 1024
T2I_REQUEST_TIMEOUT_SECONDS = 3600
EDIT_MAX_SIZE = 1024
OUTPUT_FORMAT = "data:image/png;base64"
ACCELERATOR = "Kaggle NVIDIA T4 x2"


def tracked_files(root: Path) -> list[str]:
    if not (root / ".git").is_dir():
        return sorted(
            str(p.relative_to(root)).replace("\\", "/")
            for p in root.rglob("*")
            if p.is_file() and ".git" not in p.parts
        )
    out = subprocess.check_output(["git", "-C", str(root), "ls-files"], text=True)
    return sorted(line for line in out.splitlines() if line.strip())


def is_secret_like_path(path: str) -> bool:
    parts = path.split("/")
    return any(
        part in {"token", "secret", "api_token", "secrets.json", ".env"} or part.startswith("api_token")
        for part in parts
    )


def check_hygiene(root: Path) -> list[str]:
    errors: list[str] = []
    for path in tracked_files(root):
        lowered = path.lower()
        if any(lowered.startswith(prefix) or prefix in lowered for prefix in HARMFUL_PREFIXES):
            errors.append(f"tracked generated/temporary path: {path}")
        if lowered.endswith(HARMFUL_SUFFIXES):
            errors.append(f"tracked file with harmful suffix: {path}")
        if is_secret_like_path(path):
            errors.append(f"tracked secret/token-like path: {path}")
    for path in MANDATORY_PATHS:
        if not (root / path).is_file():
            errors.append(f"mandatory file missing: {path}")
    return errors


def check_text_files(root: Path) -> list[str]:
    errors: list[str] = []
    for rel in tracked_files(root):
        path = root / rel
        try:
            data = path.read_bytes()
        except OSError as exc:
            errors.append(f"cannot read tracked file {rel}: {exc}")
            continue
        if b"\x00" in data[:8192]:
            continue  # binary; not a normal text file
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"tracked text file is not valid UTF-8: {rel}")
            continue
        if data and not data.endswith(b"\n"):
            errors.append(f"tracked text file missing final newline: {rel}")
        if rel.endswith(".sh"):
            if b"\r\n" in data:
                errors.append(f"tracked shell script contains CRLF: {rel}")
            try:
                if not (path.stat().st_mode & 0o111):
                    errors.append(f"tracked shell script not executable: {rel}")
            except OSError as exc:
                errors.append(f"cannot stat {rel}: {exc}")
    return errors


def check_project_state(root: Path) -> list[str]:
    errors: list[str] = []
    state_path = root / "project_state.json"
    if not state_path.is_file():
        return [f"project_state.json missing: {state_path}"]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    runtime = state.get("runtime", {})
    if runtime.get("accelerator") != ACCELERATOR:
        errors.append("project_state runtime.accelerator does not match README accelerator")
    if runtime.get("cpu_fallback") is not False:
        errors.append("project_state runtime.cpu_fallback must be false")
    if runtime.get("t2i_device") != T2I_DEVICE:
        errors.append("project_state runtime.t2i_device must be cuda:0")
    if runtime.get("edit_device") != EDIT_DEVICE:
        errors.append("project_state runtime.edit_device must be cuda:1")
    t2i = state.get("t2i_integration", {})
    edit = state.get("edit_integration", {})
    if t2i.get("device") != T2I_DEVICE:
        errors.append("project_state t2i_integration.device must be cuda:0")
    if edit.get("device") != EDIT_DEVICE:
        errors.append("project_state edit_integration.device must be cuda:1")
    if t2i.get("internal_url") != T2I_INTERNAL_URL:
        errors.append("project_state t2i_integration.internal_url does not match worker default")
    if edit.get("internal_url") != EDIT_INTERNAL_URL:
        errors.append("project_state edit_integration.internal_url does not match worker default")
    if t2i.get("output_format") != OUTPUT_FORMAT:
        errors.append("project_state t2i_integration.output_format must be the PNG data URL contract")
    if edit.get("output_format") != OUTPUT_FORMAT:
        errors.append("project_state edit_integration.output_format must be the PNG data URL contract")
    if t2i.get("acceptance_width") != T2I_ACCEPTANCE_WIDTH:
        errors.append("project_state t2i_integration.acceptance_width must be 1024")
    if t2i.get("acceptance_height") != T2I_ACCEPTANCE_HEIGHT:
        errors.append("project_state t2i_integration.acceptance_height must be 1024")
    if t2i.get("request_timeout_seconds") != T2I_REQUEST_TIMEOUT_SECONDS:
        errors.append("project_state t2i_integration.request_timeout_seconds must be 3600")
    if edit.get("max_size") != EDIT_MAX_SIZE:
        errors.append("project_state edit_integration.max_size must be 1024")
    notebook = state.get("public_notebook_validation", {})
    if notebook.get("structural_validation") is not True:
        errors.append("project_state public_notebook_validation.structural_validation must be true")
    if notebook.get("live_run_all") is not False:
        errors.append(
            "project_state public_notebook_validation.live_run_all must remain false until fresh Kaggle evidence"
        )
    if notebook.get("saved_version_verified") is not False:
        errors.append(
            "project_state public_notebook_validation.saved_version_verified must remain false until verified"
        )
    if state.get("license_status") != "MIT":
        errors.append("project_state license_status must be MIT")
    public_notebook = state.get("public_notebook")
    if not public_notebook or not (root / public_notebook).is_file():
        errors.append(f"project_state public_notebook does not exist: {public_notebook}")
    if state.get("github_ci_configured") is not True:
        errors.append("project_state github_ci_configured must be true")
    if not (root / ".github/workflows/ci.yml").is_file():
        errors.append("project_state github_ci_configured=true but .github/workflows/ci.yml missing")
    if state.get("github_community_metadata_configured") is not True:
        errors.append("project_state github_community_metadata_configured must be true")
    if not (root / ".github/SECURITY.md").is_file():
        errors.append("project_state github_community_metadata_configured=true but community metadata missing")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    errors = sorted(set(check_hygiene(root) + check_text_files(root) + check_project_state(root)))
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print(f"[INFO] tracked files checked (hygiene): {len(tracked_files(root))}")
    print("[INFO] mandatory repository files present; project_state.json consistent")
    print("[PASS] REPOSITORY_HYGIENE_VALID")
    print("[PASS] PROJECT_STATE_MACHINE_AUTHORITY")
    print("[PASS] PROJECT_STATE_PUBLICATION_FLAGS_TRUTHFUL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
