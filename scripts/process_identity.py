"""Shared process-identity and lifecycle guard for the Mage-Flow services.

This module is the single authority used by the start, stop, and publication
orchestrator logic before any PID file value may be trusted or signalled.

Invariants enforced:

- a PID file contains decimal digits only, is greater than 1, and is not an
  absurd huge integer;
- ``/proc/<pid>`` exists;
- ``/proc/<pid>/cmdline`` exactly matches the expected runtime script/module
  and the key arguments (host, port, device, model path);
- ``/proc/<pid>/cwd`` resolves to the expected project root;
- when a health endpoint is available the reported fields - including the
  worker's own ``os.getpid()`` - must match the PID file value and identity;
- for the coordinator, reuse additionally requires an authenticated endpoint
  call with the current ``MAGE_FLOW_API_TOKEN`` so a coordinator from another
  session cannot be reused by mistake.

Together with ``scripts/stop_process.sh`` this guarantees that ``kill`` is only
ever signalled after strict positive-integer and process-identity validation,
and that termination is graceful, bounded, and re-verified before escalation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

KIND_T2I = "t2i_worker"
KIND_EDIT = "edit_worker"
KIND_COORD = "coordinator"

LOCALHOST_ARGS = {"127.0.0.1", "localhost", "::1"}

EXPECTED_SCRIPT_TOKENS = {
    KIND_T2I: "server/workers/t2i_runtime_server.py",
    KIND_EDIT: "server/workers/edit_runtime_server.py",
    KIND_COORD: "server.app:app",
}

PROJECT_NAME = "mage-flow-t4x2-production-rest-api-demo"

MAX_PID_DIGITS = 10


class ProcessIdentityError(RuntimeError):
    """Raised when a process identity check fails fail-closed."""


def parse_strict_pid(text: str) -> int:
    """Parse and validate a PID file value in a fail-closed way."""
    value = text.strip()
    if not value:
        raise ProcessIdentityError("PID file is empty")
    if not value.isdecimal():
        raise ProcessIdentityError(f"PID is not decimal digits only: {value!r}")
    if len(value) > MAX_PID_DIGITS:
        raise ProcessIdentityError(f"PID is unrealistically large: {value!r}")
    pid = int(value)
    if pid <= 1:
        raise ProcessIdentityError(f"PID must be greater than 1, got {pid}")
    return pid


def read_pid_file(path: str | os.PathLike[str]) -> int:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ProcessIdentityError(f"cannot read PID file {path}: {exc}") from exc
    return parse_strict_pid(text)


def read_cmdline(pid: int) -> list[str]:
    """Return the NUL-separated argv of a process, dropping empty segments."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise ProcessIdentityError(f"/proc/{pid}/cmdline unreadable: {exc}") from exc
    tokens = raw.split(b"\x00")
    argv = [token.decode("utf-8", "replace") for token in tokens if token]
    if not argv:
        raise ProcessIdentityError(f"/proc/{pid}/cmdline is empty (kernel thread or zombie)")
    return argv


def read_cwd(pid: int) -> str:
    try:
        return os.path.realpath(f"/proc/{pid}/cwd")
    except OSError as exc:
        raise ProcessIdentityError(f"/proc/{pid}/cwd unreadable: {exc}") from exc


def scan_argv(argv: list[str]) -> dict[str, str]:
    """Convert ``--key value`` argument pairs into an option map (last wins)."""
    options: dict[str, str] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token.startswith("--") and "=" in token:
            key, _, value = token.partition("=")
            options[key] = value
        elif token.startswith("--") and index + 1 < len(argv):
            options[token] = argv[index + 1]
            index += 1
        index += 1
    return options


def argv_matches_signature(
    argv: list[str],
    kind: str,
    *,
    port: int | None = None,
    device: str | None = None,
    model_path: str | None = None,
) -> list[str]:
    """Return a list of signature violations, empty when the process matches.

    The script/module token must appear verbatim and every supplied argument is
    compared by full value equality so an unrelated command that merely shares a
    substring is always rejected.
    """
    violations: list[str] = []
    expected_token = EXPECTED_SCRIPT_TOKENS.get(kind)
    if expected_token is None:
        violations.append(f"unknown process kind: {kind!r}")
        return violations
    if expected_token not in argv:
        violations.append(f"missing expected script/module token {expected_token!r}")
        return violations

    options = scan_argv(argv)

    host = options.get("--host")
    if host not in LOCALHOST_ARGS:
        violations.append(f"process is not bound to localhost (--host={host!r})")

    if port is not None and options.get("--port") != str(port):
        violations.append(f"--port is {options.get('--port')!r}, expected {port}")

    if device is not None and options.get("--device") != device:
        violations.append(f"--device is {options.get('--device')!r}, expected {device!r}")

    if model_path is not None:
        actual = options.get("--model-path")
        if actual is None:
            violations.append("missing --model-path argument")
        else:
            try:
                resolved = os.path.realpath(actual)
            except (TypeError, ValueError):
                resolved = ""
            if resolved != os.path.realpath(model_path):
                violations.append(f"--model-path is {resolved!r}, expected {os.path.realpath(model_path)!r}")
    return violations


def check_process(
    pid: int,
    kind: str,
    *,
    project_root: str | None = None,
    port: int | None = None,
    device: str | None = None,
    model_path: str | None = None,
) -> None:
    """Validate the local ``/proc`` identity of a live process, raising on failure."""
    if not os.path.isdir(f"/proc/{pid}"):
        raise ProcessIdentityError(f"PID {pid} is not running (/proc/{pid} missing)")
    argv = read_cmdline(pid)
    violations = argv_matches_signature(
        argv, kind, port=port, device=device, model_path=model_path
    )
    if violations:
        raise ProcessIdentityError(f"PID {pid} identity mismatch: {'; '.join(violations)}")
    if project_root is not None:
        expected_cwd = os.path.realpath(project_root)
        actual_cwd = read_cwd(pid)
        if actual_cwd != expected_cwd:
            raise ProcessIdentityError(
                f"PID {pid} cwd is {actual_cwd!r}, expected {expected_cwd!r}"
            )


def _urlopen_json(url: str, *, headers: dict[str, str] | None = None, timeout: float = 2.0) -> dict:
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise ProcessIdentityError(f"cannot reach {url}: {exc}") from exc


def verify_worker_health(
    url: str,
    pid: int,
    *,
    model: str,
    device: str,
    model_path: str,
) -> None:
    """Require the localhost worker's own health payload to match exactly."""
    data = _urlopen_json(f"{url}/health")
    if data.get("status") != "ready" or data.get("ready") is not True:
        raise ProcessIdentityError(
            f"worker on {url} is not ready: status={data.get('status')!r} ready={data.get('ready')!r}"
        )
    if data.get("pid") != pid:
        raise ProcessIdentityError(
            f"worker on {url} reports pid={data.get('pid')!r}, expected {pid} (PID reuse or wrong process)"
        )
    if data.get("model") != model:
        raise ProcessIdentityError(
            f"worker on {url} reports model={data.get('model')!r}, expected {model!r}"
        )
    if data.get("device") != device:
        raise ProcessIdentityError(
            f"worker on {url} reports device={data.get('device')!r}, expected {device!r}"
        )
    reported_path = data.get("model_path")
    if reported_path is None or os.path.realpath(reported_path) != os.path.realpath(model_path):
        raise ProcessIdentityError(
            f"worker on {url} reports model_path={reported_path!r}, expected {os.path.realpath(model_path)!r}"
        )


def _load_token(token_file: str | None, token_env: str | None) -> str:
    if token_file is not None:
        try:
            token = Path(token_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ProcessIdentityError(f"cannot read token file {token_file}: {exc}") from exc
        if not token:
            raise ProcessIdentityError(f"token file {token_file} is empty")
        return token
    if token_env is not None:
        token = os.environ.get(token_env, "").strip()
        if not token:
            raise ProcessIdentityError(f"environment token {token_env} is not set")
        return token
    raise ProcessIdentityError("coordinator authentication requires --token-file or --token-env")


def verify_coordinator_authenticated(url: str, *, token_file: str | None, token_env: str | None) -> None:
    """Prove current-token coordinator identity before reuse."""
    token = _load_token(token_file, token_env)
    data = _urlopen_json(
        f"{url}/v1/info",
        headers={"Authorization": f"Bearer {token}"},
        timeout=3.0,
    )
    if data.get("project") != PROJECT_NAME:
        raise ProcessIdentityError(
            f"coordinator on {url} reports project={data.get('project')!r}, expected {PROJECT_NAME!r}"
        )
    if data.get("cpu_fallback") is not False:
        raise ProcessIdentityError(f"coordinator on {url} is not the GPU-only identity (cpu_fallback={data.get('cpu_fallback')!r})")
    t2i = data.get("t2i", {})
    edit = data.get("edit", {})
    if t2i.get("device") != "cuda:0" or edit.get("device") != "cuda:1":
        raise ProcessIdentityError(
            f"coordinator on {url} reports incompatible GPU routing (t2i={t2i.get('device')!r} edit={edit.get('device')!r})"
        )


def _add_check_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pid-file", required=True, help="path to the runtime PID file")
    parser.add_argument(
        "--kind",
        required=True,
        choices=[KIND_T2I, KIND_EDIT, KIND_COORD],
        help="expected process kind",
    )
    parser.add_argument("--project-root", default=None, help="expected project root for cwd checks")
    parser.add_argument("--port", type=int, default=None, help="expected --port argument")
    parser.add_argument("--device", default=None, help="expected --device argument")
    parser.add_argument("--model", default=None, help="expected model name (health identity)")
    parser.add_argument("--model-path", default=None, help="expected --model-path argument")
    parser.add_argument("--health-url", default=None, help="worker /health base URL")
    parser.add_argument("--token-file", default=None, help="token file for authenticated coordinator reuse")
    parser.add_argument("--token-env", default=None, help="environment variable holding the coordinator token")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Process identity and lifecycle guard for Mage-Flow services")
    subparsers = parser.add_subparsers(dest="command", required=True)

    read_parser = subparsers.add_parser("read-pid", help="read and validate a PID file value")
    read_parser.add_argument("--pid-file", required=True)

    check_parser = subparsers.add_parser("check", help="validate process identity before reuse or signal")
    _add_check_args(check_parser)

    args = parser.parse_args(argv)

    if args.command == "read-pid":
        try:
            pid = read_pid_file(args.pid_file)
        except ProcessIdentityError as exc:
            print(f"[FAIL] {exc}", flush=True)
            return 1
        print(pid, flush=True)
        return 0

    if args.command == "check":
        try:
            pid = read_pid_file(args.pid_file)
            if args.kind == KIND_COORD:
                check_process(
                    pid,
                    KIND_COORD,
                    project_root=args.project_root,
                    port=args.port,
                )
                if args.health_url is not None or args.token_file is not None or args.token_env is not None:
                    if args.health_url is not None:
                        _urlopen_json(f"{args.health_url}/health")
                    verify_coordinator_authenticated(
                        args.health_url or f"http://127.0.0.1:{args.port}",
                        token_file=args.token_file,
                        token_env=args.token_env,
                    )
            else:
                check_process(
                    pid,
                    args.kind,
                    project_root=args.project_root,
                    port=args.port,
                    device=args.device,
                    model_path=args.model_path,
                )
                if args.health_url is not None:
                    if args.model is None or args.model_path is None or args.device is None:
                        raise ProcessIdentityError(
                            "--model, --device and --model-path are required for worker health identity"
                        )
                    verify_worker_health(
                        args.health_url,
                        pid,
                        model=args.model,
                        device=args.device,
                        model_path=args.model_path,
                    )
        except ProcessIdentityError as exc:
            print(f"[FAIL] {exc}", flush=True)
            return 1
        print(f"[PASS] {args.kind} identity confirmed (pid={pid})", flush=True)
        return 0

    print(f"[FAIL] unknown command: {args.command!r}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())