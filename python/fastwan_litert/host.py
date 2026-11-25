"""Host-side tensor plumbing shared by the pipeline and the tests: patchify /
unpatchify in Diffusers' orders, the DMD step, and chunk padding. Pure numpy;
every other runtime ports these verbatim."""

from __future__ import annotations

import numpy as np


def patchify(latent: np.ndarray, p: int) -> np.ndarray:
    """`[C, T, H, W]` → `[T·(H/p)·(W/p), C·p²]`, feature `c·p² + kh·p + kw`
    (the Conv3d `(1, p, p)` weight flattened)."""
    c, t, h, w = latent.shape
    x = latent.reshape(c, t, h // p, p, w // p, p)
    x = x.transpose(1, 2, 4, 0, 3, 5)  # t, h', w', c, kh, kw
    return np.ascontiguousarray(x.reshape(t * (h // p) * (w // p), c * p * p))


def unpatchify(rows: np.ndarray, c: int, t: int, h: int, w: int, p: int) -> np.ndarray:
    """`[T·(H/p)·(W/p), p²·C]` (feature `(kh·p + kw)·c + c'`, Diffusers
    unpatchify) → `[C, T, H, W]`."""
    x = rows.reshape(t, h // p, w // p, p, p, c)
    x = x.transpose(5, 0, 1, 3, 2, 4)  # c, t, h', kh, w', kw
    return np.ascontiguousarray(x.reshape(c, t, h, w))


def dmd_step(x: np.ndarray, v: np.ndarray, sigma_t: float, sigma_next: float | None, noise: np.ndarray | None) -> np.ndarray:
    """`x0 = x − σ_t·v`; then `(1 − σ_n)·x0 + σ_n·noise` when `sigma_next` is given."""
    x0 = x - np.float32(sigma_t) * v
    if sigma_next is None:
        return x0.astype(np.float32)
    assert noise is not None
    s = np.float32(sigma_next)
    return ((np.float32(1.0) - s) * x0 + s * noise).astype(np.float32)


def pad_rows(x: np.ndarray, chunk: int) -> tuple[np.ndarray, int]:
    """Zero-pad the first axis to a multiple of `chunk`; returns `(padded, real rows)`."""
    n = x.shape[0]
    full = (n + chunk - 1) // chunk * chunk
    if full == n:
        return x, n
    pad = np.zeros((full - n,) + x.shape[1:], dtype=x.dtype)
    return np.concatenate([x, pad], axis=0), n


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def to_rgb8(frames: np.ndarray) -> np.ndarray:
    """`[F, 3, H, W]` in `[0, 1]` → `[F, H, W, 3]` uint8 with fastwan-rs's
    rounding (`(clamp(v)·255 + 0.5) as u8`)."""
    q = np.clip(frames, 0.0, 1.0) * 255.0 + 0.5
    return np.ascontiguousarray(q.astype(np.uint8).transpose(0, 2, 3, 1))
