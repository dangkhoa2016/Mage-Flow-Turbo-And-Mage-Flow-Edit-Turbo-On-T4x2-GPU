from __future__ import annotations

import hmac
import os
from fastapi import Header, HTTPException

TOKEN_ENV = "MAGE_FLOW_API_TOKEN"


def require_bearer_token(authorization: str | None = Header(default=None)) -> None:
    expected = os.environ.get(TOKEN_ENV, "")
    if not expected:
        raise HTTPException(status_code=503, detail="API authentication token is not configured")
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    supplied = authorization[len(prefix):]
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=403, detail="Invalid bearer token")
