import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = ROOT / "scripts" / "validate_docs_links.py"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(VALIDATOR), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_docs_links_validator_passes_offline():
    result = _run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[PASS] DOCS_LINKS_VALID" in result.stdout


def test_docs_links_validator_does_not_fail_on_remote_urls():
    root = ROOT
    remote_docs = [
        p
        for p in root.rglob("*.md")
        if ".git" not in p.parts and "https://" in p.read_text(encoding="utf-8")
    ]
    assert remote_docs, "expected at least one doc with a remote link"
    assert _run().returncode == 0


def test_docs_links_validator_rejects_broken_relative_target(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text(
        "Readme\n\n[Broken link](missing-file.md)\n",
        encoding="utf-8",
    )
    result = _run("--root", str(root))
    assert result.returncode != 0
    assert "broken relative link" in result.stdout


def test_docs_links_validator_reports_missing_counterpart_link(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "guide.md").write_text(
        "en doc, no language switcher banner\n",
        encoding="utf-8",
    )
    (root / "guide.vi.md").write_text(
        "vi doc\n\n[English](guide.md)\n",
        encoding="utf-8",
    )
    result = _run("--root", str(root))
    assert result.returncode != 0
    assert "missing bilingual counterpart link" in result.stdout