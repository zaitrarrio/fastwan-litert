"""The host math against the values fastwan-rs's own tests pin down. The
TypeScript, Kotlin and Swift ports carry these same cases."""

import numpy as np

from fastwan_litert.config import TaeConfig, WanConfig
from fastwan_litert.geometry import FASTWAN_480P, Geometry
from fastwan_litert.host import patchify, unpatchify
from fastwan_litert.rng import Rng
from fastwan_litert.rope import rope_table
from fastwan_litert.schedule import DmdSchedule


def test_fastwan_sigmas():
    s = DmdSchedule.fastwan_1_3b()
    assert s.sigmas[0] == 1.0
    assert abs(s.sigmas[1] - 0.757) < 1e-3
    assert abs(s.sigmas[2] - 0.522) < 1e-3
    assert s.sigmas[3] == 0.0
    assert s.coeffs(2)[1] is None


def test_480p_geometry():
    cfg = WanConfig.fastwan_1_3b()
    assert FASTWAN_480P.latent() == (21, 60, 104)
    assert FASTWAN_480P.grid(cfg) == (21, 30, 52)
    assert FASTWAN_480P.tokens(cfg) == 32760
    for bad in (Geometry(830, 480, 81), Geometry(832, 480, 80)):
        try:
            bad.validate(cfg)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    Geometry(832, 480, 1).validate(cfg)


def test_params_is_1_4b():
    p = WanConfig.fastwan_1_3b().params()
    assert 1_400_000_000 <= p <= 1_440_000_000


def test_rope_split_and_origin():
    cfg = WanConfig.fastwan_1_3b()
    assert cfg.rope_split() == (44, 42, 42)
    assert WanConfig.tiny().rope_split() == (4, 4, 4)
    cos, sin = rope_table(cfg, (2, 3, 4), 0)
    assert cos.shape == (24, 64)
    assert np.all(cos[0] == 1.0) and np.all(sin[0] == 0.0)
    row = 12  # token (t=1, 0, 0): the first temporal pair rotates by 1 rad
    assert abs(cos[row, 0] - np.cos(1.0)) < 1e-7
    assert cos[row, 22] == 1.0  # the spatial pairs do not


def test_taehv_frames():
    t = TaeConfig.taew2_1()
    assert t.t_upscale() == 4
    assert 21 * t.t_upscale() - t.frames_to_trim() == 81


def test_rng_is_splitmix64_box_muller():
    # fastwan-rs seeds the state as `seed ^ GOLDEN`, so the canonical
    # SplitMix64 stream (state 0: first output 0xE220A8397B1DCDAF) appears at seed GOLDEN.
    r = Rng(0x9E3779B97F4A7C15)
    assert r.next_u64() == 0xE220A8397B1DCDAF
    assert r.next_u64() == 0x6E789E6AA1B965F4
    v = Rng(7).normal(200_000).astype(np.float64)
    assert abs(v.mean()) < 0.01 and abs(v.var() - 1.0) < 0.02
    # The vectorised stream equals the scalar definition.
    a = Rng(3)
    scalar = []
    for _ in range(4):
        u1, u2 = a._unit(), a._unit()
        rr = np.sqrt(-2.0 * np.log(u1))
        th = 2.0 * np.pi * u2
        scalar += [rr * np.cos(th), rr * np.sin(th)]
    assert np.allclose(Rng(3).normal(8), np.array(scalar, np.float32), atol=1e-6)


def test_patchify_round_trip():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 3, 8, 12)).astype(np.float32)
    rows = patchify(x, 2)
    assert rows.shape == (3 * 4 * 6, 16)
    # feature index c·p² + kh·p + kw
    assert rows[0, 1 * 4 + 1 * 2 + 0] == x[1, 0, 1, 0]
    # Diffusers unpatchify reads (kh·p + kw)·c + c'
    back = unpatchify(rows.reshape(72, 4, 2, 2).transpose(0, 2, 3, 1).reshape(72, 16), 4, 3, 8, 12, 2)
    assert np.array_equal(back, x)
