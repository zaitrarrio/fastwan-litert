"""The uncut PyTorch pipeline: the same host loop as `pipeline.py` with the
whole-model modules in place of the graphs. It produces the tiny fixture's
`expected.safetensors` and is what the graph pipeline is tested against."""

from __future__ import annotations

import numpy as np
import torch

from .geometry import Geometry
from .host import dmd_step, to_rgb8
from .modules import TaeHv, WanDit
from .rng import Rng
from .rope import rope_table
from .schedule import DmdSchedule


@torch.no_grad()
def denoise(dit: WanDit, sched: DmdSchedule, embeds: np.ndarray, g: Geometry, seed: int) -> np.ndarray:
    c = dit.cfg
    g.validate(c)
    cos, sin = rope_table(c, g.grid(c), 0)
    cos_t, sin_t = torch.from_numpy(cos), torch.from_numpy(sin)
    emb = torch.from_numpy(np.ascontiguousarray(embeds, dtype=np.float32))
    lt, lh, lw = g.latent()
    rng = Rng(seed)
    x = rng.normal(c.in_channels * lt * lh * lw).reshape(c.in_channels, lt, lh, lw)
    for i, t in enumerate(sched.timesteps):
        v = dit(torch.from_numpy(x), emb, torch.tensor([float(t)]), cos_t, sin_t).numpy()
        sigma, nxt = sched.coeffs(i)
        noise = rng.normal(x.size).reshape(x.shape) if nxt is not None else None
        x = dmd_step(x, v, sigma, nxt, noise)
    return x


@torch.no_grad()
def decode(tae: TaeHv, latent: np.ndarray, chunk: int) -> np.ndarray:
    """`[4T − 3, H, W, 3]` uint8."""
    frames = tae.decode_all(torch.from_numpy(np.ascontiguousarray(latent)), chunk).numpy()
    return to_rgb8(frames)
