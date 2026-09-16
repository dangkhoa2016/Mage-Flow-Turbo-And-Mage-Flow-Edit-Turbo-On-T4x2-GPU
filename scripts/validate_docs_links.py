#!/usr/bin/env python
"""Validate repository-relative Markdown links, fully offline (offline link validation).

Checks, with no network access:
  1. Every repository-relative link target exists (files or directories).
  2. Paired-language links exist (each EN/VI copy references its counterpart).
  3. README documentation links resolve.
  4. Community-document links (.github/*) resolve.

Remote URLs (http/https/ftp), anchors, mailto:, data: and protocol-relative
links are intentionally not checked: external network availability is
nondeterministic and must not be a required CI gate.

Usage:
    python scripts/validate_docs_links.py [--root DIR]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_INLINE_LINK_RE = re.compile(r"\]\(([^)\s]+)")
_REF_DEF_RE = re.compile(r"^\[[^\]]+\]:\s*(\S+)", flags=re.M)
_REMOTE_SCHEMES = ("http://", "https://", "ftp://", "mailto:", "tel:", "data:")
_TEMPLATE_PREFIXES = ("{{", "{%")


def markdown_files(root: Path) -> list[str]:
    if (root / ".git").is_dir():
        try:
            git_md = subprocess.check_output(
                ["git", "-C", str(root), "ls-files", "--", "*.md"],
                text=True,
            )
            return sorted(line for line in git_md.splitlines() if line.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*.md") if ".git" not in p.parts)


def counterpart(path: str) -> str:
    p = Path(path)
    name = p.name
    if name.endswith(".vi.md"):
        paired = name[:-6] + ".md"
    elif name.endswith(".md"):
        paired = name[:-3] + ".vi.md"
    else:
        return ""
    return str(p.with_name(paired)).replace("\\", "/")


def is_external(target: str) -> bool:
    if target.startswith(_REMOTE_SCHEMES):
        return True
    if target.startswith("//"):
        return True
    if target.startswith("#"):
        return True
    return bool(target.startswith(_TEMPLATE_PREFIXES))


def resolve_target(root: Path, doc: str, target: str) -> Path | None:
    path_text = target.split("#", 1)[0].strip().strip("<>")
    if not path_text:
        return None
    decoded = urllib.parse.unquote(path_text)
    path = Path(decoded)
    if path.is_absolute():
        return root / str(path).lstrip("/")
    return (root / Path(doc).parent / path).resolve()


def validate_tree(root: Path) -> list[str]:
    errors: list[str] = []
    docs = set(markdown_files(root))
    for doc in sorted(docs):
        text = (root / doc).read_text(encoding="utf-8")
        targets = set(_INLINE_LINK_RE.findall(text))
        targets.update(_REF_DEF_RE.findall(text))
        for target in sorted(targets):
            if is_external(target):
                continue
            resolved = resolve_target(root, doc, target)
            if resolved is None:
                continue
            if not resolved.exists():
                errors.append(f"{doc}: broken relative link -> {target}")
        pair = counterpart(doc)
        if pair and pair in docs:
            pair_resolved = (root / pair).resolve()
            linked = False
            for target in sorted(targets):
                if is_external(target):
                    continue
                resolved = resolve_target(root, doc, target)
                if resolved is not None and resolved.resolve() == pair_resolved:
                    linked = True
                    break
            if not linked:
                errors.append(f"{doc}: missing bilingual counterpart link -> {pair}")
    if not docs:
        errors.append("no Markdown files found")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    errors = sorted(set(validate_tree(root)))
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    files = markdown_files(root)
    print(f"[INFO] Markdown files checked: {len(files)}")
    print("[INFO] relative links resolve locally; remote URLs intentionally unchecked")
    print("[PASS] DOCS_LINKS_VALID")
    return 0


if __name__ == "__main__":
    sys.exit(main())
