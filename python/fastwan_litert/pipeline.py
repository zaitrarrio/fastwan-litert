"""Prompt embeddings → DMD denoise → TAEHV decode → RGB frames, driving the
graphs exactly as `docs/graphs.md`'s host loop says. This is the reference
loop the web, Android and iOS runtimes mirror."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .config import WanConfig
from .geometry import Geometry
from .host import dmd_step, pad_rows, patchify, to_rgb8, unpatchify
from .rng import Rng
from .rope import rope_table
from .runtime import GraphStore, Manifest

Event = dict
OnEvent = Callable[[Event], None]


@dataclass
class Timings:
    prepare_s: float = 0.0
    steps_s: list[float] = field(default_factory=list)
    decode_s: float = 0.0


@dataclass
class Video:
    frames: int
    width: int
    height: int
    rgb: np.ndarray  # [frames, H, W, 3] uint8


class Prompt:
    """Per-prompt cross-attention K/V of every block."""

    def __init__(self, kv: list[tuple[np.ndarray, np.ndarray]]):
        self.kv = kv


class Pipeline:
    def __init__(self, store: GraphStore):
        self.store = store
        self.m: Manifest = store.m
        self.cfg: WanConfig = self.m.cfg

    # ---- prompt -----------------------------------------------------------------
    def prepare_prompt(self, embeds: np.ndarray) -> Prompt:
        c = self.cfg
        if embeds.shape != (c.text_len, c.text_dim):
            raise ValueError(f"prompt embeds are {embeds.shape}, expected {(c.text_len, c.text_dim)}")
        (ctx,) = self.store.fixed(self.m.graphs["text_embed"]).run(embeds=embeds)
        kv = []
        for b in self.m.graphs["blocks"]:
            k2, v2 = self.store.block(b["kv"]).run(ctx=ctx)
            kv.append((k2, v2))
        return Prompt(kv)

    # ---- one DiT forward --------------------------------------------------------
    def _rows(self, graph, chunk: int, n_real: int, **arrays: np.ndarray) -> list[np.ndarray]:
        """Run a per-row graph over row chunks of the arrays (padded); returns the
        outputs with padding dropped, concatenated along rows."""
        padded = {k: pad_rows(v, chunk)[0] for k, v in arrays.items()}
        n = next(iter(padded.values())).shape[0]
        outs: list[list[np.ndarray]] = []
        for r in range(0, n, chunk):
            o = graph.run(**{k: v[r : r + chunk] for k, v in padded.items()}, **self._fixed_inputs)
            outs.append(o)
        return [np.concatenate([o[i] for o in outs], axis=0)[:n_real] for i in range(len(outs[0]))]

    def forward(self, x: np.ndarray, prompt: Prompt, t: float, cos: np.ndarray, sin: np.ndarray, attn: dict) -> np.ndarray:
        c = self.cfg
        C = self.m.chunk
        ch, tt, hh, ww = x.shape
        s = tt * (hh // c.patch) * (ww // c.patch)
        block_mods, head_mods = self.store.fixed(self.m.graphs["time_mods"]).run(t=np.array([t], np.float32))
        patches = patchify(x, c.patch)
        self._fixed_inputs = {}
        (h,) = self._rows(self.store.fixed(self.m.graphs["patch_embed"]), C, s, patches=patches)
        attn_graph = self.store.fixed(attn["file"])
        cq = int(attn["q_chunk"])
        for l, b in enumerate(self.m.graphs["blocks"]):
            mods = block_mods[l]
            self._fixed_inputs = {"mods": mods}
            q, k, v = self._rows(self.store.block(b["qkv"]), C, s, h=h, cos=cos, sin=sin)
            self._fixed_inputs = {"k": k, "v": v}
            (o,) = self._rows(attn_graph, cq, s, q=q)
            k2, v2 = prompt.kv[l]
            self._fixed_inputs = {"mods": mods, "k2": k2, "v2": v2}
            (h,) = self._rows(self.store.block(b["post"]), C, s, h=h, o=o)
        self._fixed_inputs = {"mods": head_mods}
        (rows,) = self._rows(self.store.fixed(self.m.graphs["head"]), C, s, h=h)
        self._fixed_inputs = {}
        return unpatchify(rows, c.out_channels, tt, hh, ww, c.patch)

    # ---- denoise ------------------------------------------------------------------
    def denoise(self, embeds: np.ndarray, g: Geometry, seed: int, on: OnEvent | None = None, timings: Timings | None = None) -> np.ndarray:
        c = self.cfg
        g.validate(c)
        tm = timings or Timings()
        t0 = time.perf_counter()
        attn = self.m.attention_for(g.tokens(c))
        prompt = self.prepare_prompt(embeds)
        cos, sin = rope_table(c, g.grid(c), 0)
        lt, lh, lw = g.latent()
        rng = Rng(seed)
        x = rng.normal(c.in_channels * lt * lh * lw).reshape(c.in_channels, lt, lh, lw)
        tm.prepare_s = time.perf_counter() - t0
        if on:
            on({"type": "prepared", "secs": tm.prepare_s, "tokens": g.tokens(c)})
        sched = self.m.schedule
        for i, t in enumerate(sched.timesteps):
            ts = time.perf_counter()
            v = self.forward(x, prompt, float(t), cos, sin, attn)
            sigma, nxt = sched.coeffs(i)
            noise = rng.normal(x.size).reshape(x.shape) if nxt is not None else None
            x = dmd_step(x, v, sigma, nxt, noise)
            tm.steps_s.append(time.perf_counter() - ts)
            if on:
                on({"type": "step", "index": i, "total": len(sched), "secs": tm.steps_s[-1]})
        return x

    # ---- decode ---------------------------------------------------------------------
    def decode(self, latent: np.ndarray, on: OnEvent | None = None, timings: Timings | None = None) -> Video:
        tcfg = self.m.tae
        tm = timings or Timings()
        t0 = time.perf_counter()
        c, t, h, w = latent.shape
        entry = self.m.tae_for(h, w)
        graph = self.store.fixed(entry["file"])
        f = int(entry["frames"])
        nmem = 3 * tcfg.mem_blocks
        mems = [np.zeros((1, tcfg.widths[s], h << s, w << s), np.float32) for s in range(3) for _ in range(tcfg.mem_blocks)]
        up = tcfg.t_upscale()
        to_trim = tcfg.frames_to_trim()
        frames: list[np.ndarray] = []
        total = 0
        for f0 in range(0, t, f):
            n = min(f, t - f0)
            z = np.zeros((f, c, h, w), np.float32)
            z[:n] = latent[:, f0 : f0 + n].transpose(1, 0, 2, 3)
            out = graph.run(z=z, **{f"mem_{i}": mems[i] for i in range(nmem)})
            rgb = out[0][: n * up]
            mems = list(out[1:])
            skip = min(to_trim, rgb.shape[0])
            to_trim -= skip
            rgb8 = to_rgb8(rgb[skip:])
            if rgb8.shape[0]:
                frames.append(rgb8)
                if on:
                    on({"type": "frames", "first": total, "count": rgb8.shape[0], "width": rgb8.shape[2], "height": rgb8.shape[1], "rgb": rgb8})
                total += rgb8.shape[0]
        tm.decode_s = time.perf_counter() - t0
        all_frames = np.concatenate(frames, axis=0)
        return Video(all_frames.shape[0], all_frames.shape[2], all_frames.shape[1], all_frames)

    def generate(self, embeds: np.ndarray, g: Geometry, seed: int, on: OnEvent | None = None) -> tuple[Video, Timings]:
        tm = Timings()
        latent = self.denoise(embeds, g, seed, on, tm)
        video = self.decode(latent, on, tm)
        return video, tm
