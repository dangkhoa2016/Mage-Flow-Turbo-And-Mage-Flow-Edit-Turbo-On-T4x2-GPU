from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_mit_license_identifies_copyright_holder() -> None:
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License\n")
    assert "Copyright (c) 2026 Đăng Khoa <i.am@dangkhoa.dev>" in text
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in text


def test_github_ci_runs_cpu_safe_validation() -> None:
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    required = [
        "permissions:\n  contents: read",
        "fetch-depth: 0",
        "persist-credentials: false",
        "python -m pytest -q",
        "python scripts/validate_bilingual_docs.py",
        "python scripts/validate_notebook.py",
        "python -m pip check",
    ]
    for item in required:
        assert item in text


def test_github_community_metadata_is_present() -> None:
    required_paths = [
        ".github/CONTRIBUTING.md",
        ".github/CONTRIBUTING.vi.md",
        ".github/CODE_OF_CONDUCT.md",
        ".github/CODE_OF_CONDUCT.vi.md",
        ".github/SECURITY.md",
        ".github/SECURITY.vi.md",
        ".github/SUPPORT.md",
        ".github/SUPPORT.vi.md",
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/PULL_REQUEST_TEMPLATE.vi.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/ISSUE_TEMPLATE/question.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
        ".github/dependabot.yml",
        ".github/workflows/ci.yml",
    ]
    missing = [path for path in required_paths if not (ROOT / path).is_file()]
    assert not missing, f"missing GitHub repository metadata: {missing}"
