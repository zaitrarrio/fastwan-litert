"""The model as PyTorch modules, cut exactly as `docs/graphs.md` describes.

Each class below is one `.tflite` graph; `WanDit` and `TaeHv` compose them
into the uncut forward that the parity tests compare the graphs against
(and that `tests/test_parity.py` compares to Diffusers' model). Weights are
plain parameters in the Diffusers layout, so a checkpoint's tensors load by
name (`weights.py`). Nothing here depends on `litert-torch`: this module
imports only torch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TaeConfig, WanConfig, WAN21_LATENT_MEAN, WAN21_LATENT_STD

SHIFT_MSA, SCALE_MSA, GATE_MSA, SHIFT_FFN, SCALE_FFN, GATE_FFN = range(6)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm across the last axis (all heads: Diffusers `rms_norm_across_heads`)."""
    var = (x * x).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * w


def layer_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), eps=eps)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """Interleaved-pair rotation of `[rows, heads·head_dim]` by `[rows, head_dim/2]`
    angles (the same angle for every head); returns `[rows, heads, head_dim]`."""
    rows = x.shape[0]
    x = x.reshape(rows, heads, head_dim // 2, 2)
    c = cos.reshape(rows, 1, head_dim // 2)
    s = sin.reshape(rows, 1, head_dim // 2)
    x0, x1 = x[..., 0], x[..., 1]
    y = torch.stack([x0 * c - x1 * s, x1 * c + x0 * s], dim=-1)
    return y.reshape(rows, heads, head_dim)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Softmax attention over `[rows, H, hd]` tensors; returns `[rows_q, H·hd]`."""
    hd = q.shape[-1]
    qh, kh, vh = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
    scores = torch.matmul(qh, kh.transpose(1, 2)) * (1.0 / math.sqrt(hd))
    p = torch.softmax(scores, dim=-1)
    o = torch.matmul(p, vh)  # [H, rows_q, hd]
    return o.permute(1, 0, 2).reshape(q.shape[0], -1)


class TimeMods(nn.Module):
    """`time_mods`: `t[1]` → `block_mods[L, 6, d]`, `head_mods[2, d]`."""

    def __init__(self, cfg: WanConfig):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        half = cfg.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)
        self.linear_1 = nn.Linear(cfg.freq_dim, d)
        self.linear_2 = nn.Linear(d, d)
        self.time_proj = nn.Linear(d, 6 * d)
        self.block_tables = nn.Parameter(torch.zeros(cfg.layers, 6, d))
        self.head_table = nn.Parameter(torch.zeros(2, d))

    def forward(self, t: torch.Tensor):
        a = t.reshape(1, 1) * self.freqs.reshape(1, -1)
        emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)  # [1, freq_dim]
        temb = self.linear_2(F.silu(self.linear_1(emb)))  # [1, d]
        proj = self.time_proj(F.silu(temb)).reshape(1, 6, -1)
        block_mods = self.block_tables + proj
        head_mods = self.head_table + temb
        return block_mods, head_mods


class TextEmbed(nn.Module):
    def __init__(self, cfg: WanConfig):
        super().__init__()
        self.linear_1 = nn.Linear(cfg.text_dim, cfg.dim)
        self.linear_2 = nn.Linear(cfg.dim, cfg.dim)

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.gelu(self.linear_1(embeds), approximate="tanh"))


class BlockKV(nn.Module):
    """`block_N_kv`: `ctx[L, d]` → `k2[H, L, hd]`, `v2[H, L, hd]`."""

    def __init__(self, cfg: WanConfig):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.to_k = nn.Linear(d, d)
        self.to_v = nn.Linear(d, d)
        self.norm_k = nn.Parameter(torch.ones(d))

    def forward(self, ctx: torch.Tensor):
        c = self.cfg
        rows = ctx.shape[0]
        k = rms_norm(self.to_k(ctx), self.norm_k, c.eps).reshape(rows, c.heads, c.head_dim).permute(1, 0, 2)
        v = self.to_v(ctx).reshape(rows, c.heads, c.head_dim).permute(1, 0, 2)
        return k.contiguous(), v.contiguous()


class PatchEmbed(nn.Module):
    def __init__(self, cfg: WanConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.patch_features_in(), cfg.dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.proj(patches)


class BlockQkv(nn.Module):
    """`block_N_qkv`: `h, mods, cos, sin` → `q, k, v [C, H, hd]`."""

    def __init__(self, cfg: WanConfig):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.to_q = nn.Linear(d, d)
        self.to_k = nn.Linear(d, d)
        self.to_v = nn.Linear(d, d)
        self.norm_q = nn.Parameter(torch.ones(d))
        self.norm_k = nn.Parameter(torch.ones(d))

    def forward(self, h: torch.Tensor, mods: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        c = self.cfg
        n = layer_norm(h, c.eps) * (1 + mods[SCALE_MSA]) + mods[SHIFT_MSA]
        q = rope(rms_norm(self.to_q(n), self.norm_q, c.eps), cos, sin, c.heads, c.head_dim)
        k = rope(rms_norm(self.to_k(n), self.norm_k, c.eps), cos, sin, c.heads, c.head_dim)
        v = self.to_v(n).reshape(h.shape[0], c.heads, c.head_dim)
        return q, k, v


class Attention(nn.Module):
    """`attn_s*_q*_h*`: dense attention, weight-free."""

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return attention(q, k, v)


class BlockPost(nn.Module):
    """`block_N_post`: attention output → the block's new hidden state."""

    def __init__(self, cfg: WanConfig):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.to_out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d, eps=cfg.eps, elementwise_affine=True)
        self.to_q2 = nn.Linear(d, d)
        self.norm_q2 = nn.Parameter(torch.ones(d))
        self.to_out2 = nn.Linear(d, d)
        self.ffn_in = nn.Linear(d, cfg.ffn_dim)
        self.ffn_out = nn.Linear(cfg.ffn_dim, d)

    def forward(self, h: torch.Tensor, o: torch.Tensor, mods: torch.Tensor, k2: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        rows = h.shape[0]
        h = h + self.to_out(o) * mods[GATE_MSA]
        n = self.norm2(h)
        q2 = rms_norm(self.to_q2(n), self.norm_q2, c.eps).reshape(rows, c.heads, c.head_dim)
        cross = attention(q2, k2.permute(1, 0, 2), v2.permute(1, 0, 2))
        h = h + self.to_out2(cross)
        n = layer_norm(h, c.eps) * (1 + mods[SCALE_FFN]) + mods[SHIFT_FFN]
        ff = self.ffn_out(F.gelu(self.ffn_in(n), approximate="tanh"))
        return h + ff * mods[GATE_FFN]


class Head(nn.Module):
    def __init__(self, cfg: WanConfig):
        super().__init__()
        self.cfg = cfg
        self.proj_out = nn.Linear(cfg.dim, cfg.patch_features_out())

    def forward(self, h: torch.Tensor, mods: torch.Tensor) -> torch.Tensor:
        n = layer_norm(h, self.cfg.eps) * (1 + mods[1]) + mods[0]
        return self.proj_out(n)


class WanDit(nn.Module):
    """The whole DiT from the graph modules: the uncut reference forward.
    Input latent `[C, T, H, W]`, prompt `[L, text_dim]`, timestep scalar;
    output velocity `[C, T, H, W]`."""

    def __init__(self, cfg: WanConfig, with_blocks: bool = True):
        """`with_blocks=False` allocates everything but the blocks (the exporter builds those one at a time)."""
        super().__init__()
        self.cfg = cfg
        self.time = TimeMods(cfg)
        self.text = TextEmbed(cfg)
        self.patch = PatchEmbed(cfg)
        n = cfg.layers if with_blocks else 0
        self.kv = nn.ModuleList(BlockKV(cfg) for _ in range(n))
        self.qkv = nn.ModuleList(BlockQkv(cfg) for _ in range(n))
        self.post = nn.ModuleList(BlockPost(cfg) for _ in range(n))
        self.head = Head(cfg)
        self.attn = Attention()

    def forward(self, latent: torch.Tensor, embeds: torch.Tensor, t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        p = c.patch
        ch, tt, hh, ww = latent.shape
        x = latent.reshape(ch, tt, hh // p, p, ww // p, p).permute(1, 2, 4, 0, 3, 5).reshape(-1, ch * p * p)
        block_mods, head_mods = self.time(t)
        ctx = self.text(embeds)
        h = self.patch(x)
        for l in range(c.layers):
            k2, v2 = self.kv[l](ctx)
            q, k, v = self.qkv[l](h, block_mods[l], cos, sin)
            o = self.attn(q, k, v)
            h = self.post[l](h, o, block_mods[l], k2, v2)
        rows = self.head(h, head_mods)
        out = rows.reshape(tt, hh // p, ww // p, p, p, c.out_channels).permute(5, 0, 1, 3, 2, 4)
        return out.reshape(c.out_channels, tt, hh, ww)


# ---- TAEHV -----------------------------------------------------------------


def conv3(cin: int, cout: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 3, padding=1, bias=bias)


class MemBlock(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.conv0 = conv3(2 * n, n)
        self.conv2 = conv3(n, n)
        self.conv4 = conv3(n, n)

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv0(torch.cat([x, past], dim=1)))
        y = F.relu(self.conv2(y))
        return F.relu(self.conv4(y) + x)


class TaeStage(nn.Module):
    def __init__(self, cfg: TaeConfig, s: int):
        super().__init__()
        n, n_next, stride = cfg.widths[s], cfg.widths[s + 1], cfg.tgrow[s]
        self.n, self.stride, self.last = n, stride, s == 2
        self.mems = nn.ModuleList(MemBlock(n) for _ in range(cfg.mem_blocks))
        self.grow = nn.Conv2d(n, n * stride, 1, bias=False)
        self.conv = conv3(n, n_next, bias=False)

    def forward(self, x: torch.Tensor, mems: list[torch.Tensor]):
        """`x [f, n, h, w]`, `mems` one `[1, n, h, w]` per MemBlock →
        `(x' [f·stride, n_next, 2h, 2w], new mems)`."""
        new = []
        for m, mem in zip(self.mems, mems):
            past = torch.cat([mem, x[:-1]], dim=0)
            new.append(x[-1:])
            x = m(x, past)
        f, n, h, w = x.shape
        g = self.grow(x).reshape(f * self.stride, n, h, w)  # TGrow before the upsample (exact)
        g = F.interpolate(g, scale_factor=2.0, mode="nearest")
        y = self.conv(g)
        if self.last:
            y = F.relu(y)
        return y, new


class TaeHv(nn.Module):
    """TAEHV decoder as one chunk graph: `z[F, C, h, w]` and 9 memories →
    `rgb[4F, 3, 8ph, 8pw]` in `[0, 1]` and the 9 new memories."""

    def __init__(self, cfg: TaeConfig):
        super().__init__()
        self.cfg = cfg
        self.first = conv3(cfg.latent_channels, cfg.widths[0])
        self.stages = nn.ModuleList(TaeStage(cfg, s) for s in range(3))
        self.last = conv3(cfg.widths[3], cfg.out_channels())
        if cfg.denormalize:
            self.register_buffer("lat_std", torch.tensor(WAN21_LATENT_STD).reshape(1, -1, 1, 1), persistent=False)
            self.register_buffer("lat_mean", torch.tensor(WAN21_LATENT_MEAN).reshape(1, -1, 1, 1), persistent=False)

    def memory_shapes(self, h: int, w: int) -> list[tuple[int, ...]]:
        return [(1, self.cfg.widths[s], h << s, w << s) for s in range(3) for _ in range(self.cfg.mem_blocks)]

    def zero_memory(self, h: int, w: int) -> list[torch.Tensor]:
        return [torch.zeros(s) for s in self.memory_shapes(h, w)]

    def forward(self, z: torch.Tensor, *mems: torch.Tensor):
        cfg = self.cfg
        if cfg.denormalize:
            z = z * self.lat_std + self.lat_mean
        x = torch.tanh(z / 3.0) * 3.0
        x = F.relu(self.first(x))
        out_mems: list[torch.Tensor] = []
        for s, stage in enumerate(self.stages):
            k = cfg.mem_blocks
            x, new = stage(x, list(mems[s * k : (s + 1) * k]))
            out_mems.extend(new)
        rgb = self.last(x)
        if cfg.patch > 1:
            rgb = F.pixel_shuffle(rgb, cfg.patch)
        rgb = torch.clamp(rgb, 0.0, 1.0)
        return (rgb, *out_mems)

    def decode_all(self, latent: torch.Tensor, chunk: int) -> torch.Tensor:
        """Reference decode of a `[C, T, h, w]` latent in chunks of `chunk`
        frames (no padding), priming frames dropped: `[4T − 3, 3, H, W]`."""
        c, t, h, w = latent.shape
        mems = self.zero_memory(h, w)
        frames = []
        f0 = 0
        while f0 < t:
            n = min(chunk, t - f0)
            z = latent[:, f0 : f0 + n].permute(1, 0, 2, 3)
            out = self(z, *mems)
            frames.append(out[0])
            mems = list(out[1:])
            f0 += n
        return torch.cat(frames, dim=0)[self.cfg.frames_to_trim() :]
