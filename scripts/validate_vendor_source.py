#!/usr/bin/env python
"""Single-source authority for the canonical Mage vendor commit gate.

The production orchestrator (``scripts/run_public_candidate.sh``) invokes this
helper so the "expected Mage source commit" is defined and validated in exactly
one implementation with no copied Bash gate. The canonical value is immutable:
this script never reads environment variables, so no ``EXPECTED_MAGE_COMMIT`` /
``MAGE_*`` override can redefine it from the environment.

The gate is fail-closed: a missing source, non-Git source, unborn repository, or
a commit that differs from the expected value all exit non-zero.

Exit code ``0`` means PASS and ``1`` means FAIL; nothing is silently clamped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MAGE_CANONICAL_COMMIT = "76bec2bb3818863f470de7e867c2dc7f1d0bfd83"


def check_vendor_commit(source: Path, expected_commit: str | None = None) -> tuple[bool, str]:
    """Validate that a vendor checkout resolves to the expected Mage commit.

    ``expected_commit`` defaults to the immutable canonical constant. It exists
    only so tests can exercise the pass path without depending on a real vendor
    checkout whose HEAD equals the canonical commit; the production CLI never
    overrides it.
    """
    expected = expected_commit or MAGE_CANONICAL_COMMIT
    if not source.is_dir():
        return False, f"vendor source missing: {source}"
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return False, (f"vendor source has no resolvable Git HEAD: {detail or source}")
    actual = (result.stdout or "").strip()
    if actual != expected:
        return False, f"vendor source commit mismatch: expected {expected}, found {actual or '<unresolved>'}"
    return True, actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="path to the Mage vendor checkout")
    args = parser.parse_args(argv)
    ok, message = check_vendor_commit(args.source)
    if not ok:
        print(f"[FAIL] {message}", flush=True)
        return 1
    print(f"[PASS] VENDOR_SOURCE_COMMIT={message}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
