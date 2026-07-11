"""Raw RGB image payload helpers."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_raw_rgb(path: str | Path) -> tuple[bytes, int, int]:
    path = Path(path).expanduser().resolve()
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        payload = image.tobytes()
    return payload, int(width), int(height)


def save_raw_rgb(payload: bytes, width: int, height: int, path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    expected_size = width * height * 3
    pixels = bytes(payload[:expected_size]).ljust(expected_size, b"\x00")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("RGB", (width, height), pixels).save(path, format="PNG")
    return path
