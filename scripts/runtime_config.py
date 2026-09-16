#!/usr/bin/env python
"""Executable configuration authority for the shell lifecycle.

The shell orchestrators must never re-implement numeric validation in Bash.
This helper is the single CLI authority they invoke so the production start/stop
paths consume exactly the same parsers the coordinator and workers use:

- ``validate-ports`` enforces the global port invariants
  (each port in 1..65535 and mutually distinct);
- ``validate-port`` validates one named TCP port value;
- ``validate-token`` validates an API token value with the shared token contract;
- ``validate-timeout`` validates a lifecycle timeout (positive, whole-number
  seconds, bounded) for start-readiness waits and stop grace intervals.

Exit code ``0`` means PASS and ``1`` means FAIL; nothing is silently clamped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow direct CLI execution (``python scripts/runtime_config.py``) to import the
# shared authorities from the project package.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from server.config import ConfigError, parse_lifecycle_timeout_seconds, parse_tcp_port  # noqa: E402
from server.token_contract import TokenContractError, validate_public_token_value  # noqa: E402


def validate_ports(rest: str, t2i: str, edit: str) -> tuple[int, int, int]:
    """Validate the reserved port triple and reject any collision."""
    rest_port = parse_tcp_port(rest, name="MAGE_FLOW_REST_PORT")
    t2i_port = parse_tcp_port(t2i, name="MAGE_FLOW_T2I_INTERNAL_PORT")
    edit_port = parse_tcp_port(edit, name="MAGE_FLOW_EDIT_INTERNAL_PORT")
    if rest_port == t2i_port:
        raise ConfigError("MAGE_FLOW_REST_PORT and MAGE_FLOW_T2I_INTERNAL_PORT must not collide")
    if rest_port == edit_port:
        raise ConfigError("MAGE_FLOW_REST_PORT and MAGE_FLOW_EDIT_INTERNAL_PORT must not collide")
    if t2i_port == edit_port:
        raise ConfigError("MAGE_FLOW_T2I_INTERNAL_PORT and MAGE_FLOW_EDIT_INTERNAL_PORT must not collide")
    return rest_port, t2i_port, edit_port


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Executable configuration authority for shell lifecycle scripts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ports_parser = subparsers.add_parser("validate-ports", help="validate the reserved port triple")
    ports_parser.add_argument("--rest", required=True, help="public REST coordinator port value")
    ports_parser.add_argument("--t2i", required=True, help="internal T2I worker port value")
    ports_parser.add_argument("--edit", required=True, help="internal Edit worker port value")

    port_parser = subparsers.add_parser("validate-port", help="validate a single named TCP port value")
    port_parser.add_argument("--name", required=True, help="source environment variable name")
    port_parser.add_argument("--value", required=True, help="raw port value from the environment")

    timeout_parser = subparsers.add_parser("validate-timeout", help="validate a lifecycle timeout value")
    timeout_parser.add_argument("--name", required=True, help="source environment variable name")
    timeout_parser.add_argument("--value", default="", help="raw timeout value (empty means use the default)")
    timeout_parser.add_argument("--default", required=True, help="documented default seconds")
    timeout_parser.add_argument("--upper-bound", default="7200", help="documented upper bound in seconds")

    token_parser = subparsers.add_parser("validate-token", help="validate an API token value")
    token_parser.add_argument("--value", required=True, help="raw API token value")

    args = parser.parse_args(argv)

    try:
        if args.command == "validate-ports":
            _, _, _ = validate_ports(args.rest, args.t2i, args.edit)
            print("[PASS] PORT_CONFIGURATION_VALID", flush=True)
            return 0
        if args.command == "validate-port":
            parse_tcp_port(args.value, name=args.name)
            print(f"[PASS] {args.name}_VALID", flush=True)
            return 0
        if args.command == "validate-timeout":
            parse_lifecycle_timeout_seconds(
                None if args.value == "" else args.value,
                name=args.name,
                default=float(args.default),
                upper_bound=float(args.upper_bound),
            )
            print(f"[PASS] {args.name}_VALID", flush=True)
            return 0
        if args.command == "validate-token":
            validate_public_token_value(args.value)
            print("[PASS] TOKEN_VALUE_VALID", flush=True)
            return 0
    except (ConfigError, TokenContractError, ValueError) as exc:
        print(f"[FAIL] {exc}", flush=True)
        return 1
    except (TypeError, AttributeError):
        print("[FAIL] invalid command invocation", flush=True)
        return 1

    print(f"[FAIL] unknown command: {args.command!r}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
