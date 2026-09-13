import os
import asyncio
import io

from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from server.app import MAX_PUBLIC_UPLOAD_BYTES, app
from server.schemas import EditResponse


def test_health_is_public_minimal_liveness():
    client = TestClient(app)
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_ready_requires_authentication(monkeypatch):
    monkeypatch.setenv('MAGE_FLOW_API_TOKEN', 'test-token')
    client = TestClient(app)
    assert client.get('/ready').status_code == 401


def test_ready_is_fail_closed_before_workers_load(monkeypatch):
    monkeypatch.setenv('MAGE_FLOW_API_TOKEN', 'test-token')
    client = TestClient(app)
    response = client.get('/ready', headers={'Authorization': 'Bearer test-token'})
    assert response.status_code == 200
    data = response.json()
    assert data['ready'] is False
    assert data['status'] == 'not_ready'


def test_info_requires_authentication(monkeypatch):
    monkeypatch.setenv('MAGE_FLOW_API_TOKEN', 'test-token')
    client = TestClient(app)
    assert client.get('/v1/info').status_code == 401
    response = client.get('/v1/info', headers={'Authorization': 'Bearer test-token'})
    assert response.status_code == 200
    assert response.json()['t2i']['device'] == 'cuda:0'
    assert response.json()['edit']['device'] == 'cuda:1'


def test_generation_is_fail_closed_until_worker_ready(monkeypatch):
    monkeypatch.setenv('MAGE_FLOW_API_TOKEN', 'test-token')
    client = TestClient(app)
    response = client.post(
        '/v1/images/generations',
        headers={'Authorization': 'Bearer test-token'},
        json={'prompt': 'test'},
    )
    assert response.status_code == 503
    assert response.json()['detail'] == 'T2I worker is not ready'


class _ReadyT2IWorker:
    ready = True
    device = 'cuda:0'

    def generate(self, **kwargs):
        return {
            'id': 'img_test',
            'status': 'completed',
            'model': 'mage-flow-turbo',
            'device': 'cuda:0',
            'seed': kwargs['seed'],
            'width': kwargs['width'],
            'height': kwargs['height'],
            'elapsed_seconds': 1.0,
            'output': 'data:image/png;base64,abc',
        }


def test_generation_success_with_ready_worker(monkeypatch):
    monkeypatch.setenv('MAGE_FLOW_API_TOKEN', 'test-token')
    old = app.state.t2i_worker
    app.state.t2i_worker = _ReadyT2IWorker()
    try:
        client = TestClient(app)
        response = client.post(
            '/v1/images/generations',
            headers={'Authorization': 'Bearer test-token'},
            json={'prompt': 'test', 'seed': 42, 'steps': 4, 'width': 1024, 'height': 1024},
        )
        assert response.status_code == 200
        data = response.json()
        assert data['device'] == 'cuda:0'
        assert data['output'].startswith('data:image/png;base64,')
    finally:
        app.state.t2i_worker = old


class _FailingT2IWorker:
    ready = True
    device = 'cuda:0'

    def generate(self, **kwargs):
        raise RuntimeError('boom')


def test_generation_maps_worker_runtime_error_to_502(monkeypatch):
    monkeypatch.setenv('MAGE_FLOW_API_TOKEN', 'test-token')
    old = app.state.t2i_worker
    app.state.t2i_worker = _FailingT2IWorker()
    try:
        client = TestClient(app)
        response = client.post(
            '/v1/images/generations',
            headers={'Authorization': 'Bearer test-token'},
            json={'prompt': 'test'},
        )
        assert response.status_code == 502
        assert 'T2I worker request failed' in response.json()['detail']
    finally:
        app.state.t2i_worker = old


def _tiny_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 200, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


class _CountingUpload:
    """UploadFile stand-in that records every read() size argument."""

    def __init__(self, data: bytes, content_type: str = "image/png"):
        self.data = data
        self.content_type = content_type
        self.read_sizes = []

    async def read(self, size: int = -1):
        self.read_sizes.append(size)
        if size is None or size < 0:
            return self.data
        return self.data[:size]


class _NeverCalledEditWorker:
    ready = True
    device = "cuda:1"

    def edit(self, *, image_bytes, prompt, seed):
        raise AssertionError("Edit worker must not be called for rejected uploads")


def _call_edit(upload: _CountingUpload, worker) -> tuple[object, list[int]]:
    old = app.state.edit_worker
    app.state.edit_worker = worker
    try:
        from server.app import edit

        return asyncio.run(edit(image=upload, prompt="make it sunny", seed=42)), upload.read_sizes
    except HTTPException as exc:
        return exc, upload.read_sizes
    finally:
        app.state.edit_worker = old


def test_edit_empty_upload_is_400_and_worker_not_called():
    exc, read_sizes = _call_edit(_CountingUpload(b""), _NeverCalledEditWorker())
    assert isinstance(exc, HTTPException) and exc.status_code == 400
    assert "empty" in exc.detail
    assert read_sizes == [MAX_PUBLIC_UPLOAD_BYTES + 1]


def test_edit_bounded_read_rejects_oversized_before_worker():
    payload = b"\x89PNG\r\n\x1a\n" + b"A" * (MAX_PUBLIC_UPLOAD_BYTES + 1)
    exc, read_sizes = _call_edit(_CountingUpload(payload), _NeverCalledEditWorker())
    assert isinstance(exc, HTTPException) and exc.status_code == 413
    assert "exceeds the public byte limit" in exc.detail
    assert read_sizes == [MAX_PUBLIC_UPLOAD_BYTES + 1]


def test_edit_malformed_png_bytes_is_400_and_worker_not_called():
    exc, read_sizes = _call_edit(
        _CountingUpload(b"\x89PNG\r\n\x1a\n" + b"not-an-image" * 32),
        _NeverCalledEditWorker(),
    )
    assert isinstance(exc, HTTPException) and exc.status_code == 400
    assert "decodable" in exc.detail
    assert read_sizes == [MAX_PUBLIC_UPLOAD_BYTES + 1]


def test_edit_invalid_jpeg_bytes_is_400_and_worker_not_called():
    exc, read_sizes = _call_edit(
        _CountingUpload(b"\xff\xd8\xff\xe0garbagegarbagegarbage", content_type="image/jpeg"),
        _NeverCalledEditWorker(),
    )
    assert isinstance(exc, HTTPException) and exc.status_code == 400
    assert read_sizes == [MAX_PUBLIC_UPLOAD_BYTES + 1]


def test_edit_unsupported_media_type_is_415():
    exc, read_sizes = _call_edit(
        _CountingUpload(_tiny_png_bytes(), content_type="text/plain"),
        _NeverCalledEditWorker(),
    )
    assert isinstance(exc, HTTPException) and exc.status_code == 415
    assert read_sizes == []  # rejected before any read


def test_edit_valid_within_limit_reaches_worker():
    class _GoodEditWorker:
        ready = True
        device = "cuda:1"
        called_with = {}

        def edit(self, *, image_bytes, prompt, seed):
            self.called_with = {"image_bytes": image_bytes, "prompt": prompt, "seed": seed}
            return {
                "id": "img_test",
                "status": "completed",
                "model": "mage-flow-edit-turbo",
                "device": "cuda:1",
                "seed": seed,
                "elapsed_seconds": 0.5,
                "output": "data:image/png;base64,abc",
            }

    worker = _GoodEditWorker()
    raw = _tiny_png_bytes()
    result, read_sizes = _call_edit(_CountingUpload(raw), worker)
    assert isinstance(result, EditResponse)
    assert result.model == "mage-flow-edit-turbo"
    assert result.device == "cuda:1"
    assert worker.called_with == {"image_bytes": raw, "prompt": "make it sunny", "seed": 42}
    assert read_sizes == [MAX_PUBLIC_UPLOAD_BYTES + 1]
