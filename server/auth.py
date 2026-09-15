from __future__ import annotations

import hmac
import os

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

TOKEN_ENV = "MAGE_FLOW_API_TOKEN"
MIN_PUBLIC_TOKEN_LENGTH = 32

bearer_scheme = HTTPBearer(auto_error=False)


def validate_public_token(token: str) -> str:
    """Enforce the public start contract for configured tokens (fail closed)."""
    if len(token) < MIN_PUBLIC_TOKEN_LENGTH:
        raise ValueError(
            f"API token must be at least {MIN_PUBLIC_TOKEN_LENGTH} characters"
        )
    if not token.strip():
        raise ValueError("API token must not be whitespace-only")
    if "\n" in token or "\x00" in token:
        raise ValueError("API token must not contain newlines or NUL bytes")
    return token


def require_bearer_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    expected = os.environ.get(TOKEN_ENV, "")
    if not expected:
        raise HTTPException(status_code=503, detail="API authentication token is not configured")
    try:
        validate_public_token(expected)
    except ValueError:
        raise HTTPException(
            status_code=503,
            detail="API authentication token does not meet the public token contract",
        ) from None

    if credentials is None or str(credentials.scheme).lower() != "bearer" or not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not hmac.compare_digest(
        credentials.credentials.encode("utf-8", errors="surrogateescape"),
        expected.encode("utf-8", errors="surrogateescape"),
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )