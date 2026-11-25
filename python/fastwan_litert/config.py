"""Architecture constants, as fastwan-rs `fastwan-core/src/config.rs` records
them from each checkpoint's Diffusers `config.json`."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WanConfig:
    """The Wan 2.1 DiT. Batch is always 1: the DMD-distilled FastWan models
    run without classifier-free guidance."""

    patch: int
    heads: int
    head_dim: int
    in_channels: int
    out_channels: int
    text_dim: int
    text_len: int
    freq_dim: int
    ffn_dim: int
    layers: int
    eps: float
    rope_max_seq_len: int
    rope_theta: float

    @property
    def dim(self) -> int:
        return self.heads * self.head_dim

    def patch_features_in(self) -> int:
        return self.patch * self.patch * self.in_channels

    def patch_features_out(self) -> int:
        return self.patch * self.patch * self.out_channels

    def rope_split(self) -> tuple[int, int, int]:
        """`(t, h, w)` widths of one head (Diffusers `WanRotaryPosEmbed`:
        `h = w = 2·(d/6)`, `t = d − h − w`; 44/42/42 at d = 128)."""
        hw = 2 * (self.head_dim // 6)
        return (self.head_dim - 2 * hw, hw, hw)

    def params(self) -> int:
        """Dense parameter count of the DiT (VSA gates excluded)."""
        d = self.dim

        def lin(i: int, o: int) -> int:
            return i * o + o

        block = (
            8 * lin(d, d)
            + 4 * d
            + 2 * d
            + lin(d, self.ffn_dim)
            + lin(self.ffn_dim, d)
            + 6 * d
        )
        return (
            self.layers * block
            + lin(self.patch_features_in(), d)
            + lin(self.freq_dim, d)
            + lin(d, d)
            + lin(d, 6 * d)
            + lin(self.text_dim, d)
            + lin(d, d)
            + lin(d, self.patch_features_out())
            + 2 * d
        )

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, v: dict) -> "WanConfig":
        return cls(**{k: v[k] for k in cls.__dataclass_fields__})

    @classmethod
    def fastwan_1_3b(cls) -> "WanConfig":
        """`FastVideo/FastWan2.1-T2V-1.3B-Diffusers` (and Wan 2.1 T2V 1.3B)."""
        return cls(
            patch=2,
            heads=12,
            head_dim=128,
            in_channels=16,
            out_channels=16,
            text_dim=4096,
            text_len=512,
            freq_dim=256,
            ffn_dim=8960,
            layers=30,
            eps=1e-6,
            rope_max_seq_len=1024,
            rope_theta=10000.0,
        )

    @classmethod
    def tiny(cls) -> "WanConfig":
        """A graph small enough for the CPU in a unit test (fastwan-rs's
        `WanConfig::tiny`): head_dim a multiple of 6 so the 3-axis RoPE split
        is non-trivial."""
        return cls(
            patch=2,
            heads=2,
            head_dim=12,
            in_channels=4,
            out_channels=4,
            text_dim=16,
            text_len=8,
            freq_dim=16,
            ffn_dim=40,
            layers=2,
            eps=1e-6,
            rope_max_seq_len=64,
            rope_theta=10000.0,
        )

    @classmethod
    def from_name(cls, name: str) -> "WanConfig":
        if name in ("fastwan-1.3b", "fastwan21-1.3b", "fastwan", "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"):
            return cls.fastwan_1_3b()
        if name == "tiny":
            return cls.tiny()
        raise ValueError(f"unknown model `{name}` (fastwan-1.3b or tiny)")


WAN21_LATENT_MEAN = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]
WAN21_LATENT_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]


@dataclass(frozen=True)
class TaeConfig:
    """TAEHV (madebyollin/taehv) decoder for Wan 2.1 latents."""

    latent_channels: int
    widths: tuple[int, int, int, int]
    tgrow: tuple[int, int, int]
    mem_blocks: int
    patch: int
    denormalize: bool

    def t_upscale(self) -> int:
        n = 1
        for g in self.tgrow:
            n *= g
        return n

    def frames_to_trim(self) -> int:
        """Priming frames emitted first and dropped: `4T − 3` frames for Wan."""
        return self.t_upscale() - 1

    def spatial_scale(self) -> int:
        return 8 * self.patch

    def out_channels(self) -> int:
        return 3 * self.patch * self.patch

    def to_json(self) -> dict:
        d = asdict(self)
        d["widths"] = list(self.widths)
        d["tgrow"] = list(self.tgrow)
        return d

    @classmethod
    def from_json(cls, v: dict) -> "TaeConfig":
        return cls(
            latent_channels=v["latent_channels"],
            widths=tuple(v["widths"]),
            tgrow=tuple(v["tgrow"]),
            mem_blocks=v["mem_blocks"],
            patch=v["patch"],
            denormalize=v["denormalize"],
        )

    @classmethod
    def taew2_1(cls) -> "TaeConfig":
        return cls(16, (256, 128, 64, 64), (1, 2, 2), 3, 1, False)

    @classmethod
    def lighttaew2_1(cls) -> "TaeConfig":
        """LightX2V's LightTAE: TAEHV's network on raw (denormalised) latents."""
        return cls(16, (256, 128, 64, 64), (1, 2, 2), 3, 1, True)

    @classmethod
    def taew2_2(cls) -> "TaeConfig":
        return cls(48, (256, 128, 64, 64), (1, 2, 2), 3, 2, False)

    @classmethod
    def tiny(cls) -> "TaeConfig":
        return cls(4, (8, 6, 4, 4), (1, 2, 2), 3, 1, False)

    @classmethod
    def from_file_name(cls, name: str) -> "TaeConfig":
        n = name.lower()
        if "lighttaew2_1" in n:
            return cls.lighttaew2_1()
        if "lighttae" in n:
            raise ValueError("LightTAE for Wan 2.2 is not supported (no Wan 2.2 DiT here)")
        if "taew2_1" in n:
            return cls.taew2_1()
        if "taew2_2" in n:
            return cls.taew2_2()
        raise ValueError(f"cannot tell the TAEHV variant from `{name}`")
