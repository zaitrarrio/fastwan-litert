"""Frames out: uncompressed `.y4m` (ffmpeg reads it) and `.png`, dependency-free."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def rgb_to_yuv444(rgb: np.ndarray) -> np.ndarray:
    """BT.601 limited range, `[H, W, 3]` uint8 → `[3, H, W]` uint8."""
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    y = 16 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    u = 128 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255.0
    v = 128 + (112.0 * r - 93.786 * g - 18.214 * b) / 255.0
    return np.clip(np.stack([y, u, v]) + 0.5, 0, 255).astype(np.uint8)


def write_y4m(path: Path, frames: np.ndarray, fps: int = 16) -> None:
    f, h, w, _ = frames.shape
    with open(path, "wb") as out:
        out.write(f"YUV4MPEG2 W{w} H{h} F{fps}:1 Ip A1:1 C444\n".encode())
        for i in range(f):
            out.write(b"FRAME\n")
            out.write(rgb_to_yuv444(frames[i]).tobytes())


def write_png(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    Path(path).write_bytes(png)
