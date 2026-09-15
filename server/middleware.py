"""Project-owned ASGI middleware that enforces whole-request HTTP body limits.

Multipart parsing happens before the endpoint's bounded ``UploadFile.read``, so
the compressed-image byte limit alone does not bound the total raw request body.
This middleware counts actual bytes received at the ASGI layer, rejects declared
``Content-Length`` values that exceed the route limit before reading, and stops
reading chunked/understated/missing-length bodies once the limit is exceeded.

The over-limit response is ``413 Payload Too Large`` and the oversized body is
never fully read or stored.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

MAX_EDIT_REQUEST_BODY_BYTES = 9 * 1024 * 1024  # 9 MiB keeps safe multipart overhead
MAX_T2I_REQUEST_BODY_BYTES = 64 * 1024  # 64 KiB JSON limits
DEFAULT_REQUEST_BODY_LIMIT_BYTES = 1 * 1024  # small explicit limit for other routes

logger = logging.getLogger(__name__)


class RequestBodyLimitMiddleware:
    """Pure ASGI middleware that bounds total bytes per request body."""

    def __init__(
        self,
        app: Callable[..., Any],
        routes: dict[tuple[str, str], int] | None = None,
        default_limit: int = DEFAULT_REQUEST_BODY_LIMIT_BYTES,
    ) -> None:
        self.app = app
        self.routes = routes or {}
        self.default_limit = default_limit

    def _limit_for(self, scope: dict[str, Any]) -> int | None:
        if scope.get("type") != "http":
            return None
        method = scope.get("method", "")
        path = scope.get("path", "")
        return self.routes.get((method, path), self.default_limit)

    @staticmethod
    def _declared_content_length(scope: dict[str, Any]) -> int | None:
        headers = dict(scope.get("headers") or [])
        raw = headers.get(b"content-length")
        if raw is None:
            return None
        try:
            length = int(raw)
        except (TypeError, ValueError):
            return None
        return length if length >= 0 else None

    @staticmethod
    async def _send_error(send: Callable[..., Any], payload: bytes) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Any],
        send: Callable[..., Any],
    ) -> None:
        limit = self._limit_for(scope)
        if limit is None:
            await self.app(scope, receive, send)
            return

        declared = self._declared_content_length(scope)
        if declared is not None and declared > limit:
            logger.warning("rejecting request %s %s: declared body %d > %d", scope.get("method"), scope.get("path"), declared, limit)
            await self._send_error(send, b'{"detail":"request body exceeds the configured limit"}')
            return

        received = 0
        rejected = False

        async def bounded_receive() -> dict[str, Any]:
            nonlocal received, rejected
            message = await receive()
            if rejected:
                return {"type": "http.request", "body": b"", "more_body": False}
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    rejected = True
                    logger.warning(
                        "rejecting request %s %s after %d body bytes (limit %d)",
                        scope.get("method"),
                        scope.get("path"),
                        received,
                        limit,
                    )
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def guarded_send(message: dict[str, Any]) -> None:
            if rejected:
                return
            await send(message)

        try:
            await self.app(scope, bounded_receive, guarded_send)
        except Exception:
            if not rejected:
                raise
            return
        finally:
            if rejected:
                await self._send_error(
                    send, b'{"detail":"request body exceeds the configured limit"}'
                )