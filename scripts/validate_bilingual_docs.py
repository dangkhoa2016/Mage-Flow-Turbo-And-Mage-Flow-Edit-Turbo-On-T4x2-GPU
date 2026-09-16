#!/usr/bin/env python
"""Validate bilingual Markdown pairing in the working tree and Git history."""

from __future__ import annotations

import re
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
        out = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "--", "*.md"], text=True)
        return sorted(line for line in out.splitlines() if line.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in ROOT.rglob("*.md") if ".git" not in p.parts)


# require structural parity (P1-20): heading-level sequence, fenced code
# blocks (count + language tags), bullet-list count and numbered-list count
# must match between English and Vietnamese copies so a section/example added
# in one language but omitted in the other is flagged. Semantic translation
# scoring is intentionally not attempted.
_HEADING_RE = re.compile(r"^(#+)\s+")
_FENCE_RE = re.compile(r"^(?P<f>`{3,}|~{3,})(?P<tag>\S*)\s*$")
_BULLET_RE = re.compile(r"^\s*[-*]\s+")
_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+")


def structural_signature(text: str) -> tuple:
    lines = text.splitlines()
    headings = tuple(_HEADING_RE.findall(text))
    fences = tuple((len(m.group("f")), m.group("tag")) for line in lines if (m := _FENCE_RE.match(line)))
    bullets = len(_BULLET_RE.findall(text))
    numbered = len(_NUMBERED_RE.findall(text))
    return headings, fences, bullets, numbered


def validate_parity(path: str, text: str, pair_path: str) -> list[str]:
    errors: list[str] = []
    signature = structural_signature(text)
    pair_signature = structural_signature((ROOT / pair_path).read_text(encoding="utf-8"))
    labels = (
        ("heading-level sequence", signature[0], pair_signature[0]),
        ("fenced code blocks (count + language tags)", signature[1], pair_signature[1]),
        ("bullet-list item count", signature[2], pair_signature[2]),
        ("numbered-list item count", signature[3], pair_signature[3]),
    )
    for label, actual, expected in labels:
        if actual != expected:
            errors.append(f"structural parity mismatch EN/VI in {path}: {label} differs")
    return errors


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
        errors.extend(validate_parity(path, text, pair))
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
    commits = subprocess.check_output(["git", "-C", str(ROOT), "rev-list", "--reverse", "HEAD"], text=True).splitlines()
    for commit in commits:
        changed = changed_markdown(commit)
        for path in sorted(changed):
            pair = counterpart(path)
            if pair not in changed:
                errors.append(f"commit {commit[:12]} changes {path} without paired {pair}")
    return errors


def main() -> int:
    errors = validate_tree() + validate_history()
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print(f"[INFO] bilingual Markdown files: {len(tracked_markdown())}")
    print("[INFO] structural EN/VI parity enforced (headings, code fences, bullet/numbered counts)")
    print("[PASS] BILINGUAL_DOCUMENTATION_PAIRING_VALID")
    return 0


if __name__ == "__main__":
    sys.exit(main())
