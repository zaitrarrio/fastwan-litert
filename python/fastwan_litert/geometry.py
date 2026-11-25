"""Pixel ↔ latent ↔ token shapes for a Wan 2.1 clip (fastwan-core `geometry.rs`)."""

from __future__ import annotations

from dataclasses import dataclass

from .config import WanConfig

VAE_SPATIAL = 8
VAE_TEMPORAL = 4


@dataclass(frozen=True)
class Geometry:
    width: int
    height: int
    frames: int

    def validate(self, cfg: WanConfig) -> None:
        cell = VAE_SPATIAL * cfg.patch
        if self.width <= 0 or self.height <= 0 or self.width % cell or self.height % cell:
            raise ValueError(f"{self.width}x{self.height} is not a positive multiple of {cell}")
        if self.frames <= 0 or (self.frames - 1) % VAE_TEMPORAL:
            raise ValueError(f"{self.frames} frames is not 4n+1")

    def latent(self) -> tuple[int, int, int]:
        """Latent `(T, H, W)`."""
        return ((self.frames - 1) // VAE_TEMPORAL + 1, self.height // VAE_SPATIAL, self.width // VAE_SPATIAL)

    def grid(self, cfg: WanConfig) -> tuple[int, int, int]:
        t, h, w = self.latent()
        return (t, h // cfg.patch, w // cfg.patch)

    def tokens(self, cfg: WanConfig) -> int:
        t, h, w = self.grid(cfg)
        return t * h * w

    def latent_len(self, cfg: WanConfig) -> int:
        t, h, w = self.latent()
        return cfg.in_channels * t * h * w

    def to_json(self) -> dict:
        return {"width": self.width, "height": self.height, "frames": self.frames}

    @classmethod
    def parse(cls, s: str) -> "Geometry":
        """`WxHxF`, e.g. `832x480x81`."""
        try:
            w, h, f = (int(v) for v in s.lower().split("x"))
        except ValueError as e:
            raise ValueError(f"geometry `{s}` is not WIDTHxHEIGHTxFRAMES") from e
        return cls(w, h, f)

    def __str__(self) -> str:
        return f"{self.width}x{self.height}x{self.frames}"


FASTWAN_480P = Geometry(832, 480, 81)
"""FastWan 2.1 1.3B's served default: 5 s at 16 fps."""
MOBILE_480x272 = Geometry(480, 272, 33)
"""fastwan-rs's phone tier."""
