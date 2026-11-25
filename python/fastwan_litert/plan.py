"""Memory plan (fastwan-core `plan.rs`, for the graph cut): what a generation
holds at once, before anything is allocated. Hosts call it from `plan` and
before `generate` to refuse what will not fit."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import TaeConfig, WanConfig
from .geometry import Geometry

BYTES_PER_WEIGHT = {"f32": 4.0, "f16": 2.0, "int8": 1.0, "int4": 0.5}


@dataclass
class Plan:
    tokens: int
    chunk: int
    q_chunk: int
    weights_total: int
    weights_resident: int
    cross_kv: int
    activations: int
    attention_scratch: int
    decode_peak: int
    peak: int
    budget: int
    fits: bool
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        mib = lambda b: b / (1 << 20)  # noqa: E731
        lines = [
            f"tokens            {self.tokens}",
            f"weights (all)     {mib(self.weights_total):9.1f} MiB",
            f"weights resident  {mib(self.weights_resident):9.1f} MiB",
            f"cross-attn K/V    {mib(self.cross_kv):9.1f} MiB",
            f"DiT activations   {mib(self.activations):9.1f} MiB (row chunk {self.chunk})",
            f"attention scratch {mib(self.attention_scratch):9.1f} MiB (query chunk {self.q_chunk})",
            f"decode peak       {mib(self.decode_peak):9.1f} MiB",
            f"peak              {mib(self.peak):9.1f} MiB of {mib(self.budget):.0f} MiB  fits: {self.fits}",
        ]
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)


def plan(
    cfg: WanConfig,
    tae: TaeConfig,
    g: Geometry,
    weights: str,
    chunk: int,
    q_chunk: int,
    tae_frames: int,
    budget_bytes: int,
    resident_blocks: int = 2,
) -> Plan:
    s = g.tokens(cfg)
    d = cfg.dim
    a = 4  # f32 activations
    wb = BYTES_PER_WEIGHT[weights]
    block_params = (8 * (d * d + d) + 2 * (d * cfg.ffn_dim) + cfg.ffn_dim + d + 6 * d)
    other_params = cfg.params() - cfg.layers * block_params
    weights_total = int(cfg.layers * block_params * wb + other_params * 2)
    weights_resident = int(resident_blocks * block_params * wb + other_params * 2)
    cross_kv = cfg.layers * 2 * cfg.text_len * d * a
    # h, q, k, v, o: [S, d] each; the latent, velocity and noise; rope tables.
    activations = (5 * s * d + 3 * g.latent_len(cfg) + s * cfg.head_dim) * a
    attention_scratch = cfg.heads * q_chunk * s * a
    notes = []
    # TAEHV peak: the last stage's 64-channel map at 8x resolution for the chunk's 4F frames,
    # plus input/output of the MemBlock convs (x, past, two temporaries, out).
    t, lh, lw = g.latent()
    f = min(tae_frames, t)
    nf = tae.widths
    px = lh * lw
    s0 = f * nf[0] * px
    s1 = f * tae.tgrow[0] * nf[1] * 4 * px
    s2 = f * tae.tgrow[0] * tae.tgrow[1] * nf[2] * 16 * px
    out = f * tae.t_upscale() * nf[3] * 64 * px
    decode_peak = max(5 * s0, 5 * s1, 5 * s2, s2 + 2 * out) * a
    peak = max(weights_resident + cross_kv + activations + attention_scratch, weights_resident + decode_peak)
    if resident_blocks < cfg.layers:
        notes.append(f"{resident_blocks} of {cfg.layers} block graphs resident; the rest stream from disk / the network")
    if s % chunk:
        notes.append(f"{s} rows run as {(s + chunk - 1) // chunk} chunks of {chunk} ({chunk - s % chunk} padded rows)")
    fits = peak <= budget_bytes
    return Plan(s, chunk, q_chunk, weights_total, weights_resident, cross_kv, activations, attention_scratch, decode_peak, peak, budget_bytes, fits, notes)
