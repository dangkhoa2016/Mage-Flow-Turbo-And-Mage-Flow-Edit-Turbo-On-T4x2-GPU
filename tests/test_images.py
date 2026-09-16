from __future__ import annotations

import io

import pytest
from PIL import Image
from server.images import (
    ALLOWED_IMAGE_FORMATS,
    ALLOWED_IMAGE_MEDIA_TYPES,
    MAX_PUBLIC_IMAGE_EDGE,
    MAX_PUBLIC_IMAGE_PIXELS,
    MEDIA_TYPE_TO_FORMAT,
    ImageValidationError,
    decode_validated_image,
    decode_validated_worker_image,
)
from server.workers.edit_runtime_server import decode_image_bytes


def _image_bytes(size=(8, 8), fmt="PNG", **save_kwargs) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 200, 90)).save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


def test_declared_media_type_allowlist_and_format_map_agree():
    assert ALLOWED_IMAGE_FORMATS == ("PNG", "JPEG", "WEBP")
    assert {"image/png", "image/jpeg", "image/jpg", "image/webp"} == ALLOWED_IMAGE_MEDIA_TYPES
    assert set(MEDIA_TYPE_TO_FORMAT) == ALLOWED_IMAGE_MEDIA_TYPES
    assert set(MEDIA_TYPE_TO_FORMAT.values()) == set(ALLOWED_IMAGE_FORMATS)
    assert MEDIA_TYPE_TO_FORMAT["image/jpeg"] == MEDIA_TYPE_TO_FORMAT["image/jpg"] == "JPEG"


@pytest.mark.parametrize(
    ("media_type", "fmt"),
    [("image/png", "PNG"), ("image/jpeg", "JPEG"), ("image/jpg", "JPEG"), ("image/webp", "WEBP")],
)
def test_valid_images_across_allowed_formats(media_type, fmt):
    decoded = decode_validated_image(_image_bytes(fmt=fmt), media_type)
    assert decoded.mode == "RGB"
    assert decoded.size == (8, 8)


def test_worker_surrogate_accepts_image_without_declared_media_type():
    decoded = decode_validated_worker_image(_image_bytes())
    assert decoded.mode == "RGB"
    assert decoded.size == (8, 8)


def test_worker_runtime_decode_image_bytes_matches_contract():
    decoded = decode_image_bytes(_image_bytes())
    assert decoded.mode == "RGB"


def test_worker_runtime_rejects_garbage_with_value_error():
    with pytest.raises(ValueError, match="invalid image payload"):
        decode_image_bytes(b"\x89PNG\r\n\x1a\n" + b"not-an-image" * 32)


def test_worker_runtime_rejects_empty_payload():
    with pytest.raises(ValueError, match="image payload must not be empty"):
        decode_image_bytes(b"")


def test_empty_bytes_rejected():
    with pytest.raises(ImageValidationError) as excinfo:
        decode_validated_image(b"", "image/png")
    assert excinfo.value.status_code == 400


@pytest.mark.parametrize("media_type", ["text/plain", "image/bmp", "image/tiff", ""])
def test_missing_or_unsupported_declared_media_type_is_415(media_type):
    with pytest.raises(ImageValidationError) as excinfo:
        decode_validated_image(_image_bytes(), media_type)
    assert excinfo.value.status_code == 415


def test_declared_media_type_mismatch_is_400():
    with pytest.raises(ImageValidationError) as excinfo:
        decode_validated_image(_image_bytes(fmt="PNG"), "image/jpeg")
    assert excinfo.value.status_code == 400
    assert "does not match" in str(excinfo.value)


def test_garbage_bytes_are_400():
    with pytest.raises(ImageValidationError) as excinfo:
        decode_validated_image(b"\x89PNG\r\n\x1a\n" + b"not-an-image" * 32, "image/png")
    assert excinfo.value.status_code == 400


def test_disallowed_real_format_is_400_even_if_pillow_can_parse_it():
    bmp_buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(bmp_buffer, format="BMP")
    with pytest.raises(ImageValidationError) as excinfo:
        decode_validated_image(bmp_buffer.getvalue(), "image/png")
    assert excinfo.value.status_code == 400


def test_excessive_pixel_count_is_400():
    oversized_buffer = io.BytesIO()
    wide, tall = 5000, 4000
    Image.new("RGB", (wide, tall), "gray").save(oversized_buffer, format="PNG")
    assert wide * tall > MAX_PUBLIC_IMAGE_PIXELS
    with pytest.raises(ImageValidationError) as excinfo:
        decode_validated_image(oversized_buffer.getvalue(), "image/png")
    assert excinfo.value.status_code == 400
    assert "pixel" in str(excinfo.value)


def test_excessive_edge_dimension_is_400():
    edge_buffer = io.BytesIO()
    Image.new("RGB", (MAX_PUBLIC_IMAGE_EDGE + 1, 8), "gray").save(edge_buffer, format="PNG")
    with pytest.raises(ImageValidationError) as excinfo:
        decode_validated_image(edge_buffer.getvalue(), "image/png")
    assert excinfo.value.status_code == 400
    assert "edge" in str(excinfo.value)


def test_animated_webp_is_400_single_frame_only():
    buffer = io.BytesIO()
    frame_a = Image.new("RGB", (8, 8), (255, 0, 0))
    frame_b = Image.new("RGB", (8, 8), (0, 0, 255))
    frame_a.save(buffer, format="WEBP", save_all=True, append_images=[frame_b], duration=100)
    raw = buffer.getvalue()
    frame_count = Image.open(io.BytesIO(raw), formats=["WEBP"]).n_frames
    assert frame_count > 1
    with pytest.raises(ImageValidationError) as excinfo:
        decode_validated_image(raw, "image/webp")
    assert excinfo.value.status_code == 400
    assert "multi-frame" in str(excinfo.value)


def test_decompression_bomb_warning_is_treated_as_error(monkeypatch):
    monkeypatch.setattr("PIL.Image.MAX_IMAGE_PIXELS", 8 * 8 - 1)  # below our tiny fixture
    with pytest.raises(ImageValidationError) as excinfo:
        decode_validated_image(_image_bytes(), "image/png")
    assert excinfo.value.status_code == 400


def test_exif_orientation_is_normalized():
    exif_buffer = io.BytesIO()
    image = Image.new("RGB", (32, 16), (10, 200, 90))
    exif = Image.Exif()
    exif[0x0112] = 6  # EXIF orientation 6: 90 degrees clockwise rotation
    image.save(exif_buffer, format="JPEG", exif=exif)
    decoded = decode_validated_image(exif_buffer.getvalue(), "image/jpeg")
    assert decoded.size == (16, 32)
