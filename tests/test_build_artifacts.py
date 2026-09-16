from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import pytest
from scripts.validate_build_artifacts import main, read_version

VERSION = "1.2.3"

_REQUIRED_MODULES = {
    "server/workers/conditioning_offload.py",
    "server/workers/edit_runtime_server.py",
    "server/workers/t2i_runtime_server.py",
    "server/safety/__init__.py",
    "server/safety/gguf_safety.py",
    "server/safety/wiring.py",
}


@pytest.fixture
def dist(tmp_path: Path) -> tuple[Path, Path]:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    version_file = tmp_path / "_version.py"
    version_file.write_text(f'__version__ = "{VERSION}"\n', encoding="utf-8")
    return dist_dir, version_file


def _make_wheel(path: Path, *, version: str, license_present: bool = True) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        files = {
            "magedemo/__init__.py": "__version__ = 'x'\n",
            f"magedemo-{version}.dist-info/METADATA": "",
            **{member: "" for member in _REQUIRED_MODULES},
        }
        if license_present:
            files[f"magedemo-{version}.dist-info/licenses/LICENSE"] = "MIT"
        for name, content in files.items():
            zf.writestr(name, content)


def _make_sdist(path: Path, *, version: str, license_present: bool = True) -> None:
    root = f"magedemo-{version}"
    with tarfile.open(path, "w:gz") as tf:
        members = [
            (f"{root}/magedemo/__init__.py", "x"),
            (f"{root}/PKG-INFO", ""),
            *((f"{root}/{member}", "") for member in _REQUIRED_MODULES),
        ]
        if license_present:
            members.append((f"{root}/LICENSE", "MIT"))
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tf.addfile(info, __import__("io").BytesIO(content.encode()))


def _run_main(dist_dir: Path, version_file: Path) -> int:
    return main(["--dist-dir", str(dist_dir), "--version-file", str(version_file)])


def test_valid_dist_artifacts(dist: tuple[Path, Path]) -> None:
    dist_dir, version_file = dist
    _make_wheel(dist_dir / f"magedemo-{VERSION}-py3-none-any.whl", version=VERSION)
    _make_sdist(dist_dir / f"magedemo-{VERSION}.tar.gz", version=VERSION)
    assert _run_main(dist_dir, version_file) == 0


def test_version_mismatch_rejected(dist: tuple[Path, Path]) -> None:
    dist_dir, version_file = dist
    _make_wheel(dist_dir / "magedemo-9.9.9-py3-none-any.whl", version="9.9.9")
    _make_sdist(dist_dir / "magedemo-9.9.9.tar.gz", version="9.9.9")
    assert _run_main(dist_dir, version_file) == 1


def test_missing_license_rejected(dist: tuple[Path, Path]) -> None:
    dist_dir, version_file = dist
    _make_sdist(dist_dir / "magedemo-9.9.9.tar.gz", version="9.9.9", license_present=False)
    _make_wheel(dist_dir / f"magedemo-{VERSION}-py3-none-any.whl", version=VERSION, license_present=False)
    assert _run_main(dist_dir, version_file) == 1


def test_forbidden_member_rejected(dist: tuple[Path, Path]) -> None:
    dist_dir, version_file = dist
    _make_wheel(dist_dir / f"magedemo-{VERSION}-py3-none-any.whl", version=VERSION)
    path = dist_dir / f"magedemo-{VERSION}.tar.gz"
    root = f"magedemo-{VERSION}"
    with tarfile.open(path, "w:gz") as tf:
        for name, content in ((f"{root}/LICENSE", "MIT"), (f"{root}/.runtime/token", "secret")):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tf.addfile(info, __import__("io").BytesIO(content.encode()))
    assert _run_main(dist_dir, version_file) == 1


def test_missing_distribution_rejected(dist: tuple[Path, Path]) -> None:
    dist_dir, version_file = dist
    _make_wheel(dist_dir / f"magedemo-{VERSION}-py3-none-any.whl", version=VERSION)
    assert _run_main(dist_dir, version_file) == 1


def test_read_version_rejects_missing_assignment(tmp_path: Path) -> None:
    version_file = tmp_path / "_version.py"
    version_file.write_text("# no version here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_version(version_file)
