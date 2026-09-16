"""Fail-closed runtime secret/directory handling for the Mage-Flow demo.

This module is the single authority for the ``.runtime`` directory and persisted
API-token contracts used by the shell orchestrators and the notebook:

- the runtime directory must be a real directory (not a symlink) owned by the
  current user with mode ``0700``;
- the API token file must be a regular file (not a symlink) owned by the current
  user with mode ``0600``, a link count of exactly one, and content satisfying
  the shared token-value contract in ``server.token_contract`` (same authority
  the coordinator auth middleware enforces, so the store can never report valid
  a token the server would reject);
- new tokens are created with restrictive mode from the first write (``O_EXCL``
  with ``0o600``), never written wide and chmodded afterwards.

Every check fails closed: if a contract cannot be enforced the process aborts
instead of reporting normal progress.
"""

from __future__ import annotations

import argparse
import os
import secrets
import stat
import sys
from pathlib import Path

# Allow direct CLI execution (``python scripts/token_store.py``) to import the
# shared authorities from the project package.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from server.token_contract import TokenContractError, validate_public_token_value  # noqa: E402

DEFAULT_DIR_MODE = 0o700
DEFAULT_TOKEN_MODE = 0o600


class TokenStoreError(RuntimeError):
    """Raised when the runtime directory/token contract cannot be enforced."""


def _real_owner_ok(path: str) -> bool:
    st = os.stat(path)
    if st.st_uid != os.geteuid():
        raise TokenStoreError(f"{path} is owned by uid {st.st_uid}, not the current user")
    return True


def enforce_directory(path: str | os.PathLike[str]) -> str:
    """Create/enforce the runtime directory contract and return its real path.

    The directory must not itself be a symlink, must be owned by the current
    user, and must have mode ``0700``. Permission fixes are applied here; a
    failure to apply them aborts (fail closed).
    """
    directory = os.path.abspath(os.fspath(path))
    if os.path.islink(directory):
        raise TokenStoreError(f"runtime directory must not be a symlink: {directory}")
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        raise TokenStoreError(f"cannot create runtime directory {directory}: {exc}") from exc
    st = os.stat(directory)
    if not stat.S_ISDIR(st.st_mode):
        raise TokenStoreError(f"{directory} is not a directory")
    _real_owner_ok(directory)
    if st.st_nlink == 0:
        raise TokenStoreError(f"{directory} has an empty link count")
    try:
        os.chmod(directory, DEFAULT_DIR_MODE)
    except OSError as exc:
        raise TokenStoreError(f"cannot chmod runtime directory {directory}: {exc}") from exc
    if (os.stat(directory).st_mode & 0o777) != DEFAULT_DIR_MODE:
        raise TokenStoreError(f"runtime directory {directory} is not mode {DEFAULT_DIR_MODE:o}")
    return directory


def enforce_token_file(path: str | os.PathLike[str]) -> None:
    """Enforce the token file contract (regular, owned, single link, mode 0600)."""
    token_path = str(path)
    if os.path.islink(token_path):
        raise TokenStoreError(f"token file must not be a symlink: {token_path}")
    try:
        st = os.stat(token_path)
    except FileNotFoundError as exc:
        raise TokenStoreError(f"token file does not exist: {token_path}") from exc
    except OSError as exc:
        raise TokenStoreError(f"cannot stat token file {token_path}: {exc}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise TokenStoreError(f"token file {token_path} is not a regular file (mode {stat.S_IFMT(st.st_mode):o})")
    _real_owner_ok(token_path)
    if st.st_nlink != 1:
        raise TokenStoreError(f"token file {token_path} has link count {st.st_nlink} (surprising extra hardlinks)")
    try:
        os.chmod(token_path, DEFAULT_TOKEN_MODE)
    except OSError as exc:
        raise TokenStoreError(f"cannot chmod token file {token_path}: {exc}") from exc
    if (os.stat(token_path).st_mode & 0o777) != DEFAULT_TOKEN_MODE:
        raise TokenStoreError(f"token file {token_path} is not mode {DEFAULT_TOKEN_MODE:o}")


def _read_token(path: str) -> str:
    """Read and validate a persisted token value against the shared authority.

    The raw file content is validated without stripping: a persisted value that
    carries leading/trailing whitespace or an embedded newline is ambiguous with
    the exact value the server compares, so it fails closed instead of being
    silently normalized.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise TokenStoreError(f"cannot read token file {path}: {exc}") from exc
    try:
        return validate_public_token_value(raw)
    except TokenContractError as exc:
        raise TokenStoreError(f"token file {path} contains an invalid persisted token value: {exc}") from exc


def _create_token_secure(path: str) -> str:
    """Create a new token with restrictive mode from the very first write."""
    token = secrets.token_urlsafe(32)
    validate_public_token_value(token)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path, flags, DEFAULT_TOKEN_MODE)
    except FileExistsError:
        return _read_token(path)
    except OSError as exc:
        raise TokenStoreError(f"cannot create token file {path}: {exc}") from exc
    try:
        os.write(fd, token.encode("utf-8"))
        os.fsync(fd)
    except OSError as exc:
        os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass
        raise TokenStoreError(f"cannot write token file {path}: {exc}") from exc
    os.close(fd)
    enforce_token_file(path)
    return token


def ensure_token(path: str | os.PathLike[str]) -> str:
    """Return a valid persisted token, creating it securely if absent."""
    token_path = os.path.abspath(os.fspath(path))
    directory = enforce_directory(os.path.dirname(token_path))
    token_path = os.path.join(directory, os.path.basename(token_path))
    if os.path.exists(token_path):
        enforce_token_file(token_path)
        token = _read_token(token_path)
        if not token:
            raise TokenStoreError(f"token file {token_path} is empty")
        return token
    return _create_token_secure(token_path)


def verify_token(path: str | os.PathLike[str]) -> None:
    """Validate the token contract for an existing token file."""
    token_path = os.path.abspath(os.fspath(path))
    enforce_directory(os.path.dirname(token_path))
    enforce_token_file(token_path)
    _read_token(token_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed runtime secret handling for the Mage-Flow demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dirs_parser = subparsers.add_parser("prepare-dir", help="create/enforce runtime directories")
    dirs_parser.add_argument("directories", nargs="+", help="runtime directories to enforce")

    ensure_parser = subparsers.add_parser("ensure-token", help="return a valid persisted token (creating if absent)")
    ensure_parser.add_argument("path", help="token file path")

    verify_parser = subparsers.add_parser("verify-token", help="validate an existing token file; prints nothing")
    verify_parser.add_argument("path", help="token file path")

    args = parser.parse_args(argv)

    try:
        if args.command == "prepare-dir":
            for directory in args.directories:
                enforce_directory(directory)
            return 0
        if args.command == "ensure-token":
            print(ensure_token(args.path), flush=True)
            return 0
        if args.command == "verify-token":
            verify_token(args.path)
            return 0
    except TokenStoreError as exc:
        print(f"[FAIL] {exc}", flush=True)
        return 1

    print(f"[FAIL] unknown command: {args.command!r}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
