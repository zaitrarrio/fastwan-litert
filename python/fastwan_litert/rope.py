"""3-axis RoPE tables (Diffusers `WanRotaryPosEmbed`; fastwan-core `rope.rs`).

Each head of width `d` is split `(t, h, w)` (44/42/42 at d = 128) and each
axis rotates interleaved pairs `(x[2i], x[2i+1])` by `pos · θ^(−2i/axis)`.
Both members of a pair share one angle, so the table stores `d/2` angles per
token. Angles are formed in f64 and their cos/sin cast to f32, as Diffusers.
"""

from __future__ import annotations

import numpy as np

from .config import WanConfig


def _axis_freqs(dim: int, theta: float) -> np.ndarray:
    i = np.arange(dim // 2, dtype=np.float64)
    return 1.0 / theta ** (2.0 * i / dim)


def rope_table(cfg: WanConfig, grid: tuple[int, int, int], start_frame: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """`(cos, sin)` of shape `[tokens, head_dim/2]`, tokens in `(t, h, w)` order."""
    dt, dh, dw = cfg.rope_split()
    gt, gh, gw = grid
    ft, fh, fw = (_axis_freqs(dt, cfg.rope_theta), _axis_freqs(dh, cfg.rope_theta), _axis_freqs(dw, cfg.rope_theta))
    t = np.arange(start_frame, start_frame + gt, dtype=np.float64)
    h = np.arange(gh, dtype=np.float64)
    w = np.arange(gw, dtype=np.float64)
    # angles per axis: [n_axis, freqs]
    at = np.outer(t, ft)  # [gt, dt/2]
    ah = np.outer(h, fh)
    aw = np.outer(w, fw)
    angles = np.concatenate(
        [
            np.broadcast_to(at[:, None, None, :], (gt, gh, gw, at.shape[1])),
            np.broadcast_to(ah[None, :, None, :], (gt, gh, gw, ah.shape[1])),
            np.broadcast_to(aw[None, None, :, :], (gt, gh, gw, aw.shape[1])),
        ],
        axis=-1,
    ).reshape(gt * gh * gw, cfg.head_dim // 2)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
