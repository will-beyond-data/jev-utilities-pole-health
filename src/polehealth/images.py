"""Image loading and resizing for the model and for the demo site."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageOps

MODEL_MAX_EDGE = 1024
MODEL_JPEG_QUALITY = 85
DEMO_MAX_EDGE = 960
DEMO_JPEG_QUALITY = 80


def load_image(path: str | Path) -> Image.Image:
    """Open an image, apply EXIF orientation (phone photos), and convert to RGB."""
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw)
        return image.convert("RGB")


def resize_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
    """Shrink so the longer edge is at most max_edge. Never upscales."""
    if max(image.size) <= max_edge:
        return image
    scale = max_edge / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def data_url(path: str | Path, max_edge: int = MODEL_MAX_EDGE, quality: int = MODEL_JPEG_QUALITY) -> str:
    """The request-ready `data:image/jpeg;base64,...` string the model server accepts."""
    payload = jpeg_bytes(resize_max_edge(load_image(path), max_edge), quality)
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")


def save_demo_photo(src: str | Path, dst: str | Path) -> tuple[int, int]:
    """Write the resized photo the static site shows. Returns the output (width, height)."""
    image = resize_max_edge(load_image(src), DEMO_MAX_EDGE)
    destination = Path(dst)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(jpeg_bytes(image, DEMO_JPEG_QUALITY))
    return image.size


def normalise_data_url(url: str, max_edge: int = MODEL_MAX_EDGE, quality: int = MODEL_JPEG_QUALITY) -> str:
    """Re-encode an uploaded data URL exactly like a file photo (EXIF-aware, max edge, JPEG)."""
    _, _, encoded = url.partition(",")
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    payload = jpeg_bytes(resize_max_edge(image, max_edge), quality)
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")
