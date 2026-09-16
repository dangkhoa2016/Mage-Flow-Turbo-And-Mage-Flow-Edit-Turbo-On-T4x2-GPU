"""Shared configuration validation helpers for the public coordinator and workers.

These helpers encode project invariants as machine-checkable contracts:

- internal worker URLs must be localhost-only HTTP (no credentials, path, query,
  or fragment), so the "workers are localhost-only" claim is fail-closed;
- request/lifecycle timeouts must be finite, positive, and bounded so bad
  environment values cannot loop forever or behave like infinity;
- TCP ports must be decimal integers within 1..65535 and cannot collide.
"""

from __future__ import annotations

import math
import re
from typing import Any
from urllib.parse import urlsplit

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
DEFAULT_REQUEST_TIMEOUT_SECONDS = 3600.0
MAX_REQUEST_TIMEOUT_SECONDS = 7200.0


class ConfigError(ValueError):
    """Raised when a configuration value violates a project invariant."""


def validate_loopback_http_url(value: Any, *, name: str) -> str:
    """Validate and canonicalize a localhost-only HTTP base URL.

    Returns the URL with a trailing slash removed. Raises ``ConfigError`` for
    every non-loopback or malformed variant so misconfiguration is fail-closed.
    """
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty HTTP loopback URL")
    url = value.strip().rstrip("/")
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise ConfigError(f"{name} is not a valid URL: {value!r}") from exc

    if parts.scheme != "http":
        raise ConfigError(f"{name} must use the http scheme, got {parts.scheme!r}")
    if parts.username is not None or parts.password is not None:
        raise ConfigError(f"{name} must not contain embedded credentials")
    if parts.query:
        raise ConfigError(f"{name} must not contain a query string")
    if parts.fragment:
        raise ConfigError(f"{name} must not contain a fragment")
    if parts.path not in ("", "/"):
        raise ConfigError(f"{name} must not contain a path, got {parts.path!r}")

    if parts.hostname is None:
        raise ConfigError(f"{name} must include a hostname")
    host = parts.hostname.lower()
    if host not in LOOPBACK_HOSTS:
        raise ConfigError(f"{name} host must be a loopback host, got {host!r}")

    try:
        port = parts.port
    except ValueError as exc:
        raise ConfigError(f"{name} has an invalid TCP port: {value!r}") from exc
    if port is None:
        raise ConfigError(f"{name} must include an explicit TCP port")
    if not 1 <= port <= 65535:
        raise ConfigError(f"{name} port must be in 1..65535, got {port}")

    display_host = f"[{host}]" if host == "::1" else host
    return f"http://{display_host}:{port}"


def parse_timeout_seconds(
    value: Any,
    *,
    name: str,
    default: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> float:
    """Parse a finite, positive, bounded timeout in seconds.

    Accepts a numeric string or a number. ``None`` uses ``default``. Values that
    are zero, negative, NaN, infinite, non-numeric, or above the project maximum
    raise ``ConfigError`` instead of being silently clamped.
    """
    if value is None:
        value = default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be a finite number of seconds") from None
    if math.isnan(parsed) or math.isinf(parsed):
        raise ConfigError(f"{name} must be finite, got {value!r}")
    if parsed <= 0:
        raise ConfigError(f"{name} must be positive, got {parsed!r}")
    if parsed > MAX_REQUEST_TIMEOUT_SECONDS:
        raise ConfigError(f"{name} must not exceed {MAX_REQUEST_TIMEOUT_SECONDS} seconds, got {parsed!r}")
    return parsed


_PORT_DECIMAL = re.compile(r"^\d+$")


def parse_tcp_port(value: Any, *, name: str) -> int:
    """Parse a TCP port as a decimal integer within 1..65535."""
    if not isinstance(value, str) or not _PORT_DECIMAL.fullmatch(value.strip()):
        raise ConfigError(f"{name} must be a decimal TCP port, got {value!r}")
    port = int(value.strip())
    if not 1 <= port <= 65535:
        raise ConfigError(f"{name} must be in 1..65535, got {port}")
    return port
