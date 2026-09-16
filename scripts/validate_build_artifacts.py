#!/usr/bin/env python
"""CPU-only validation of ``python -m build`` artifacts (wheel and sdist validation).

Checks (no network, no model load):
  1. Exactly one wheel and one sdist are produced in the dist directory.
  2. The distribution version equals the canonical ``server/_version.py`` value.
  3. LICENSE is present in the wheel metadata and at the sdist root.
  4. No local-review residue is packaged: .runtime, caches, PID/token/log
     artifacts, notebook checkpoints, dist/build/egg-info or local outputs.

Usage:
    python scripts/validate_build_artifacts.py [--dist-dir DIR] [--version-file PATH]
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from pathlib import Path

FORBIDDEN_COMPONENTS = {
    ".runtime",
    ".ipynb_checkpoints",
    "__pycache__",
    "outputs",
    "artifacts",
    "build",
    "dist",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
}
FORBIDDEN_SUFFIXES = (".pyc", ".log", ".pid", ".tmp", ".bak", ".orig", ".swp", ".ipynb")
VERSION_RE = re.compile(r"__version__\s*=\s*\"([^\"]+)\"")
REQUIRED_SUBMODULES = (
    "server/workers/conditioning_offload.py",
    "server/workers/edit_runtime_server.py",
    "server/workers/t2i_runtime_server.py",
    "server/safety/__init__.py",
    "server/safety/gguf_safety.py",
    "server/safety/wiring.py",
)


def read_version(version_file: Path) -> str:
    match = VERSION_RE.search(version_file.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"cannot read __version__ from {version_file}")
    return match.group(1)


def missing_members(names: list[str]) -> list[str]:
    """Return required modules absent from a wheel or sdist member listing.

    Wheels store members relative to the package root (``server/...``) while
    sdists prefix members with the project root (``<name>-<version>/server/...``),
    so a member matches when the required path appears at either form.
    """
    normalized = {name.strip("/") for name in names}
    return [
        member
        for member in REQUIRED_SUBMODULES
        if member not in normalized and not any(name.endswith(f"/{member}") for name in normalized)
    ]


def forbidden_members(names: list[str]) -> list[str]:
    bad: list[str] = []
    for name in names:
        components = [part for part in name.strip("/").split("/") if part]
        if not components:
            continue
        base = components[-1]
        if base == "token" or base.endswith(".pid") or base.endswith(".token"):
            bad.append(name)
        for part in components:
            if part in FORBIDDEN_COMPONENTS:
                bad.append(name)
                break
        if base.lower().endswith(FORBIDDEN_SUFFIXES):
            bad.append(name)
    return sorted(set(bad))


def validate_wheel(path: Path, version: str) -> list[str]:
    problems: list[str] = []
    stem = path.name.removesuffix(".whl")
    parts = stem.split("-")
    if len(parts) < 2:
        problems.append(f"wheel name does not follow PEP 427: {path.name}")
    else:
        wheel_version = parts[1]
        if wheel_version != version:
            problems.append(f"wheel version {wheel_version!r} != server version {version!r}")
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        license_ok = any("/licenses/LICENSE" in name or name.endswith("/LICENSE") for name in names)
        if not license_ok:
            problems.append("wheel is missing LICENSE in dist-info/licenses")
        bad = forbidden_members(names)
        if bad:
            problems.append(f"wheel contains forbidden members: {', '.join(bad)}")
        absent = missing_members(names)
        if absent:
            problems.append(f"wheel is missing required modules: {', '.join(absent)}")
    return problems


def validate_sdist(path: Path, version: str) -> list[str]:
    problems: list[str] = []
    name = path.name
    match = re.match(r".+-([0-9][^-]*)\.tar\.gz$", name)
    if not match:
        problems.append(f"sdist name does not embed a version: {name}")
    elif match.group(1) != version:
        problems.append(f"sdist version {match.group(1)!r} != server version {version!r}")
    with tarfile.open(path, "r:gz") as tf:
        names = tf.getnames()
        if not any(n.endswith("/LICENSE") for n in names):
            problems.append("sdist is missing LICENSE")
        bad = forbidden_members(names)
        if bad:
            problems.append(f"sdist contains forbidden members: {', '.join(bad)}")
        absent = missing_members(names)
        if absent:
            problems.append(f"sdist is missing required modules: {', '.join(absent)}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", default=Path(__file__).resolve().parent.parent / "dist", type=Path)
    parser.add_argument("--version-file", default=None, type=Path)
    args = parser.parse_args(argv)

    dist_dir = Path(args.dist_dir)
    version_file = Path(args.version_file) if args.version_file else dist_dir.parent / "server" / "_version.py"
    version = read_version(version_file)

    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    problems: list[str] = []
    if len(wheels) != 1:
        problems.append(f"expected exactly one wheel, found {len(wheels)} in {dist_dir}")
    if len(sdists) != 1:
        problems.append(f"expected exactly one sdist, found {len(sdists)} in {dist_dir}")
    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}", flush=True)
        return 1

    problems += validate_wheel(wheels[0], version)
    problems += validate_sdist(sdists[0], version)
    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}", flush=True)
        return 1

    print(f"[INFO] dist has one wheel + one sdist (version {version})", flush=True)
    print("[PASS] BUILD_ARTIFACTS_VALID", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
