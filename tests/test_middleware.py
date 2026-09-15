from __future__ import annotations

import pytest

from server.middleware import (
    DEFAULT_REQUEST_BODY_LIMIT_BYTES,
    MAX_EDIT_REQUEST_BODY_BYTES,
    MAX_T2I_REQUEST_BODY_BYTES,
    RequestBodyLimitMiddleware,
)

AUTH_TOKEN = "kaggle-demo-test-token-0123456789abcdef0123456789abcdef"

ROUTES = {
    ("POST", "/v1/images/edits"): MAX_EDIT_REQUEST_BODY_BYTES,
    ("POST", "/v1/images/generations"): MAX_T2I_REQUEST_BODY_BYTES,
}


def _http_scope(method="POST", path="/v1/images/edits", content_length: int | None = None) -> dict:
    headers = []
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode("ascii")))
    return {"type": "http", "method": method, "path": path, "headers": headers}


def _run(app, scope, chunks) -> tuple[bytes | None, list[dict]]:
    """Drive the middleware manually and return (downstream body, asgi messages)."""
    messages: list[dict] = []

    async def receive():
        if chunks:
            return {"type": "http.request", "body": chunks.pop(0), "more_body": bool(chunks)}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    async def driver():
        await RequestBodyLimitMiddleware(app, routes=ROUTES)(scope, receive, send)

    import asyncio

    asyncio.run(driver())
    return messages


def test_limits_are_documented_constants():
    assert DEFAULT_REQUEST_BODY_LIMIT_BYTES == 1024
    assert MAX_T2I_REQUEST_BODY_BYTES == 64 * 1024
    assert MAX_EDIT_REQUEST_BODY_BYTES == 9 * 1024 * 1024


def test_oversized_declared_content_length_rejects_before_reading():
    async def downstream(scope, receive, send):
        raise AssertionError("declared-length rejection must not invoke the application")

    messages = _run(
        downstream,
        _http_scope(content_length=MAX_EDIT_REQUEST_BODY_BYTES + 1),
        [b"x" * 16],
    )
    statuses = [m["status"] for m in messages if m["type"] == "http.response.start"]
    assert statuses == [413]


def test_body_at_limit_reaches_the_application_unchanged():
    body = b"a" * MAX_EDIT_REQUEST_BODY_BYTES
    captured = {}

    async def downstream(scope, receive, send):
        received = b""
        while True:
            message = await receive()
            received += message.get("body", b"")
            if not message.get("more_body"):
                break
        captured["body"] = received
        await send(
            {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]}
        )
        await send({"type": "http.response.body", "body": b"ok"})

    messages = _run(downstream, _http_scope(content_length=len(body)), [body])
    assert captured["body"] == body
    assert [m["status"] for m in messages if m["type"] == "http.response.start"] == [200]


def test_chunked_body_over_default_limit_is_truncated_and_413():

    async def downstream(scope, receive, send):
        received = b""
        while True:
            message = await receive()
            received += message.get("body", b"")
            if not message.get("more_body"):
                break
        captured["body"] = received
        await send(
            {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]}
        )
        await send({"type": "http.response.body", "body": b"ok"})

    captured = {}
    chunks = [b"a" * 4096] * 8  # 32 KiB, declared length unknown
    messages = _run(
        downstream,
        _http_scope(method="POST", path="/v1/other", content_length=None),
        list(chunks),
    )
    assert len(captured["body"]) <= DEFAULT_REQUEST_BODY_LIMIT_BYTES
    statuses = [m["status"] for m in messages if m["type"] == "http.response.start"]
    assert statuses == [413]


def test_routes_without_route_limit_use_small_default():
    async def downstream(scope, receive, send):
        while True:
            message = await receive()
            if not message.get("more_body"):
                break
        await send(
            {"type": "http.response.start", "status": 204, "headers": [(b"content-type", b"text/plain")]}
        )
        await send({"type": "http.response.body", "body": b""})

    scope = _http_scope(method="POST", path="/v1/some-other-route", content_length=2048)
    messages = _run(downstream, scope, [b"y" * 2048])
    assert [m["status"] for m in messages if m["type"] == "http.response.start"] == [413]


def test_non_http_scopes_are_passthrough():
    async def downstream(scope, receive, send):
        await send({"type": "lifespan.startup.complete"})

    messages: list[dict] = []
    import asyncio

    async def send(message):
        messages.append(message)

    async def receive():
        return {"type": "http.disconnect"}

    async def driver():
        middleware = RequestBodyLimitMiddleware(downstream, routes=ROUTES)
        await middleware({"type": "lifespan"}, receive, send)

    asyncio.run(driver())
    assert messages == [{"type": "lifespan.startup.complete"}]


def test_app_generation_route_rejects_body_over_64kib(monkeypatch):
    from fastapi.testclient import TestClient

    from server.app import app

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", AUTH_TOKEN)
    client = TestClient(app)
    response = client.post(
        "/v1/images/generations",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        content=b'{"prompt":"' + b"a" * (64 * 1024) + b'"}',
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "request body exceeds the configured limit"


def test_app_body_limiter_leaves_public_health_untouched(monkeypatch):
    from fastapi.testclient import TestClient

    from server.app import app

    monkeypatch.setenv("MAGE_FLOW_API_TOKEN", AUTH_TOKEN)
    client = TestClient(app)
    assert client.get("/health").status_code == 200