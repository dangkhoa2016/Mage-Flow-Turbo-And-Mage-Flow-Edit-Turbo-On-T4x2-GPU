#!/usr/bin/env python
"""Validate bilingual Markdown pairing in the working tree and Git history."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def counterpart(path: str) -> str:
    p = Path(path)
    name = p.name
    if name.endswith(".vi.md"):
        paired_name = name[:-6] + ".md"
    elif name.endswith(".md"):
        paired_name = name[:-3] + ".vi.md"
    else:
        raise ValueError(f"not a Markdown path: {path}")
    return str(p.with_name(paired_name)).replace("\\", "/")


def tracked_markdown() -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files", "--", "*.md"], text=True
        )
        return sorted(line for line in out.splitlines() if line.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return sorted(
            str(p.relative_to(ROOT)).replace("\\", "/")
            for p in ROOT.rglob("*.md")
            if ".git" not in p.parts
        )


def expected_banner(path: str) -> str:
    p = Path(path)
    pair = Path(counterpart(path)).name
    if p.name.endswith(".vi.md"):
        return f"> 🌐 Language / Ngôn ngữ: [English]({pair}) | **Tiếng Việt**"
    return f"> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt]({pair})"


def validate_tree() -> list[str]:
    errors: list[str] = []
    docs = set(tracked_markdown())
    for path in sorted(docs):
        pair = counterpart(path)
        if pair not in docs:
            errors.append(f"missing bilingual counterpart: {path} -> {pair}")
            continue
        text = (ROOT / path).read_text(encoding="utf-8")
        banner = expected_banner(path)
        first_lines = "\n".join(text.splitlines()[:8])
        if banner not in first_lines:
            errors.append(f"missing/incorrect language switcher in {path}: expected {banner!r}")
    return errors


def changed_markdown(commit: str) -> set[str]:
    out = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "diff-tree",
            "--root",
            "--no-commit-id",
            "--name-only",
            "-r",
            commit,
            "--",
            "*.md",
        ],
        text=True,
    )
    return {line for line in out.splitlines() if line.strip()}


def validate_history() -> list[str]:
    if not (ROOT / ".git").exists():
        return []
    errors: list[str] = []
    commits = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-list", "--reverse", "HEAD"], text=True
    ).splitlines()
    for commit in commits:
        changed = changed_markdown(commit)
        for path in sorted(changed):
            pair = counterpart(path)
            if pair not in changed:
                errors.append(
                    f"commit {commit[:12]} changes {path} without paired {pair}"
                )
    return errors


def main() -> int:
    errors = validate_tree() + validate_history()
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print(f"[INFO] bilingual Markdown files: {len(tracked_markdown())}")
    print("[PASS] BILINGUAL_DOCUMENTATION_PAIRING_VALID")
    return 0


if __name__ == "__main__":
    sys.exit(main())
