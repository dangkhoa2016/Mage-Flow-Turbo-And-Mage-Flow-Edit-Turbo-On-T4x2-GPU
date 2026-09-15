"""Public Edit image resource and format contract validation.

The worker/application intentionally never trusts the client-declared MIME type
alone. A payload is accepted only when:

- the declared media type is in the allowlist,
- the payload size is within the compressed byte budget (checked by the caller),
- Pillow opens the payload as exactly one of the approved formats,
- the Pillow-identified format matches the declared media type,
- dimensions are positive and bounded (edge and total-pixel limits),
- Pillow decompression warnings/errors fail closed,
- the image is single-frame,
- and the image decodes fully before being forwarded.

``decode_validated_image`` normalizes EXIF orientation and returns an RGB image.
"""

from __future__ import annotations

import io
import warnings
from typing import Any

from PIL import Image, ImageOps

MAX_PUBLIC_IMAGE_PIXELS = 4096 * 4096  # 16,777,216 pixels
MAX_PUBLIC_IMAGE_EDGE = 8192
ALLOWED_IMAGE_FORMATS = ("PNG", "JPEG", "WEBP")
ALLOWED_IMAGE_MEDIA_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
MEDIA_TYPE_TO_FORMAT = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",  # intentionally retained for backward compatibility
    "image/webp": "WEBP",
}


class ImageValidationError(ValueError):
    """Raised when an image payload violates the public Edit image contract."""

    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def validate_declared_media_type(media_type: str | None) -> str:
    """Reject unsupported declared media types (mapped to HTTP 415)."""
    if media_type not in ALLOWED_IMAGE_MEDIA_TYPES:
        raise ImageValidationError("Unsupported media type", status_code=415)
    return media_type


def open_image_restricted(image_bytes: bytes) -> Image.Image:
    """Open with a format allowlist and decompression bombs treated as errors."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            return Image.open(io.BytesIO(image_bytes), formats=list(ALLOWED_IMAGE_FORMATS))
    except Image.DecompressionBombError as exc:
        raise ImageValidationError("image exceeds the decoded-pixel safety limit") from exc
    except Image.DecompressionBombWarning as exc:
        raise ImageValidationError("image triggers the decoded-pixel safety warning") from exc
    except Exception as exc:
        raise ImageValidationError(
            "image data is not a decodable supported image"
        ) from exc


def decode_validated_image(
    image_bytes: bytes,
    declared_media_type: str | None,
) -> Image.Image:
    """Validate raw bytes against the decoded-image resource contract.

    Validates the full pipeline (MIME match, format allowlist, dimensions,
    pixel limit, single frame, decompression safety) before and during decode,
    then normalizes EXIF orientation and returns an RGB image.
    """
    if not image_bytes:
        raise ImageValidationError("image data is empty")

    if declared_media_type is not None:
        declared_media_type = validate_declared_media_type(declared_media_type)

    image = open_image_restricted(image_bytes)

    if not isinstance(image.format, str) or image.format not in ALLOWED_IMAGE_FORMATS:
        raise ImageValidationError("image uses a format outside the supported allowlist")
    if declared_media_type is not None:
        expected_format = MEDIA_TYPE_TO_FORMAT[declared_media_type]
        if image.format != expected_format:
            raise ImageValidationError(
                "declared media type does not match the decoded image format"
            )

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ImageValidationError("image dimensions must be positive")
    if width > MAX_PUBLIC_IMAGE_EDGE or height > MAX_PUBLIC_IMAGE_EDGE:
        raise ImageValidationError("image edge dimension exceeds the public limit")
    if width * height > MAX_PUBLIC_IMAGE_PIXELS:
        raise ImageValidationError("image pixel count exceeds the public limit")

    if getattr(image, "n_frames", 1) > 1:
        raise ImageValidationError("animated or multi-frame images are not supported")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image.load()
    except Image.DecompressionBombError as exc:
        raise ImageValidationError("image exceeds the decoded-pixel safety limit") from exc
    except Image.DecompressionBombWarning as exc:
        raise ImageValidationError("image triggers the decoded-pixel safety warning") from exc
    except Exception as exc:
        raise ImageValidationError("image data failed to decode fully") from exc

    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def decode_validated_worker_image(image_bytes: bytes) -> Image.Image:
    """Worker-side defense-in-depth decoder with the same resource contract."""
    return decode_validated_image(image_bytes, declared_media_type=None)


def image_format_name(image: Image.Image) -> str:
    """Best-effort canonical format name for diagnostics."""
    value: Any = getattr(image, "format", None)
    return str(value) if value else "unknown"