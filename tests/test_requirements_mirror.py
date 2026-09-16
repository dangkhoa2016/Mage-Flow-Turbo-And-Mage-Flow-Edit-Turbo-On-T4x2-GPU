from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROJECT_DEPS_RE = re.compile(r"dependencies\s*=\s*\[(.*?)\n\]", re.DOTALL)
TEST_DEPS_RE = re.compile(r"^\s*test\s*=\s*\[([^\]]*)\]", re.MULTILINE)
QUOTED = re.compile(r"\"([^\"]+)\"")

# quality extras intentionally live only in pyproject.toml; the convenience
# mirror covers the runtime dependencies plus the test extra.


def _pyproject_specs() -> list[str]:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    specs = []
    deps_match = PROJECT_DEPS_RE.search(text)
    if not deps_match:
        raise AssertionError("pyproject.toml dependencies block not found")
    specs += QUOTED.findall(deps_match.group(1))
    test_match = TEST_DEPS_RE.search(text)
    if not test_match:
        raise AssertionError("pyproject.toml test extra not found")
    specs += QUOTED.findall(test_match.group(1))
    return specs


def _requirement_specs() -> list[str]:
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    specs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        specs.append(line.split(";", 1)[0].strip())
    return specs


def test_requirements_mirror_matches_pyproject_authority() -> None:
    pyproject_specs = _pyproject_specs()
    requirement_specs = _requirement_specs()
    assert pyproject_specs, "no dependency specs parsed from pyproject.toml"
    missing_in_requirements = sorted(set(pyproject_specs) - set(requirement_specs))
    assert not missing_in_requirements, f"missing in requirements.txt: {missing_in_requirements}"


def test_requirements_mirror_has_no_extra_members() -> None:
    pyproject_specs = _pyproject_specs()
    requirement_specs = _requirement_specs()
    unexpected = sorted(set(requirement_specs) - set(pyproject_specs))
    assert not unexpected, f"extra spec not declared in pyproject.toml: {unexpected}"
