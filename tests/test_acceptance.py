import base64
import hashlib
from pathlib import Path

import pytest

from scripts.acceptance import LiveAcceptance


def _tiny_png_bytes() -> bytes:
    from PIL import Image

    import io

    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 200, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


def _tiny_jpeg_bytes() -> bytes:
    from PIL import Image

    import io

    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (200, 20, 90)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _runner(tmp_path: Path) -> LiveAcceptance:
    return LiveAcceptance(
        base_url="http://127.0.0.1:8090",
        token="test-token",
        output_dir=tmp_path / ".runtime" / "acceptance",
        edit_source=tmp_path / "dog.jpg",
    )


def test_save_data_url_roundtrip(tmp_path):
    runner = _runner(tmp_path)
    raw = _tiny_png_bytes()
    data_url = f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"
    saved = runner._save_data_url(data_url, "t2i.png")
    assert saved == raw
    assert (runner.output_dir / "t2i.png").read_bytes() == raw


@pytest.mark.parametrize(
    "data_url",
    [
        "data:image/png;base64,{bad}",
        "http://example.com/not-a-data-url",
        "data:image/gif;base64,AAAA",
    ],
)
def test_save_data_url_rejects_unexpected_prefix(tmp_path, data_url):
    runner = _runner(tmp_path)
    with pytest.raises(AssertionError):
        runner._save_data_url(data_url, "t2i.png")


def test_verify_png_and_sha256(tmp_path):
    runner = _runner(tmp_path)
    raw = _tiny_png_bytes()
    assert runner.verify_png(raw) == (16, 16)
    assert runner.sha256(raw) == hashlib.sha256(raw).hexdigest()


def test_to_png_converts_jpeg(tmp_path):
    runner = _runner(tmp_path)
    png = runner._to_png(_tiny_jpeg_bytes())
    assert runner.verify_png(png) == (32, 32)