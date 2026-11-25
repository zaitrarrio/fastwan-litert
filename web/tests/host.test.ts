/**
 * The host math against the values fastwan-rs's own tests pin down: every case
 * of `python/tests/test_host.py` (and `test_plan.py`).
 */

import { describe, expect, it } from "vitest";

import { TaeConfig, WanConfig } from "../src/config.js";
import { FASTWAN_480P, Geometry, MOBILE_480x272 } from "../src/geometry.js";
import { dmdStep, padRows, patchify, toRgba8, unpatchify } from "../src/host.js";
import { parseManifest } from "../src/manifest.js";
import { plan } from "../src/plan.js";
import { Rng, mul64 } from "../src/rng.js";
import { ropeTable } from "../src/rope.js";
import { DmdSchedule } from "../src/schedule.js";

describe("schedule", () => {
  it("fastwan sigmas are 1.0 / 0.757 / 0.522 / 0", () => {
    const s = DmdSchedule.fastwan13b();
    expect(s.sigmas[0]).toBe(1.0);
    expect(Math.abs(s.sigmas[1] - 0.757)).toBeLessThan(1e-3);
    expect(Math.abs(s.sigmas[2] - 0.522)).toBeLessThan(1e-3);
    expect(s.sigmas[3]).toBe(0.0);
    expect(s.coeffs(2)[1]).toBeNull();
    // the float32 table values the exporter writes into the manifest
    expect(s.sigmas[1]).toBe(0.7567567229270935);
    expect(s.sigmas[2]).toBe(0.52173912525177);
  });
});

describe("geometry", () => {
  it("480p is (21, 60, 104) latent and 32760 tokens", () => {
    const cfg = WanConfig.fastwan13b();
    expect(FASTWAN_480P.latent()).toEqual([21, 60, 104]);
    expect(FASTWAN_480P.grid(cfg)).toEqual([21, 30, 52]);
    expect(FASTWAN_480P.tokens(cfg)).toBe(32760);
    expect(() => new Geometry(830, 480, 81).validate(cfg)).toThrow(/multiple of 16/);
    expect(() => new Geometry(832, 480, 80).validate(cfg)).toThrow(/4n\+1/);
    new Geometry(832, 480, 1).validate(cfg);
    expect(Geometry.parse("480x272x33").equals(MOBILE_480x272)).toBe(true);
    expect(() => Geometry.parse("480x272")).toThrow(/WIDTHxHEIGHTxFRAMES/);
  });

  it("the 1.3B config has 1.4B parameters", () => {
    const p = WanConfig.fastwan13b().params();
    expect(p).toBeGreaterThanOrEqual(1_400_000_000);
    expect(p).toBeLessThanOrEqual(1_440_000_000);
  });
});

describe("rope", () => {
  it("splits (44, 42, 42) and is the identity at the origin", () => {
    const cfg = WanConfig.fastwan13b();
    expect(cfg.ropeSplit()).toEqual([44, 42, 42]);
    expect(WanConfig.tiny().ropeSplit()).toEqual([4, 4, 4]);
    const { cos, sin, tokens, width } = ropeTable(cfg, [2, 3, 4], 0);
    expect([tokens, width]).toEqual([24, 64]);
    for (let i = 0; i < 64; i++) {
      expect(cos[i]).toBe(1.0);
      expect(sin[i]).toBe(0.0);
    }
    const row = 12; // token (t=1, 0, 0): the first temporal pair rotates by 1 rad
    expect(Math.abs(cos[row * 64] - Math.cos(1.0))).toBeLessThan(1e-7);
    expect(cos[row * 64 + 22]).toBe(1.0); // the spatial pairs do not
  });
});

describe("taehv", () => {
  it("decodes 21 latent frames to 81", () => {
    const t = TaeConfig.taew21();
    expect(t.tUpscale()).toBe(4);
    expect(21 * t.tUpscale() - t.framesToTrim()).toBe(81);
  });
});

describe("rng", () => {
  it("mul64 wraps like u64", () => {
    expect(mul64(0, 0xffffffff, 0, 0xffffffff)).toEqual([0xfffffffe, 0x00000001]);
    expect(mul64(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff)).toEqual([0, 1]);
    const p = BigInt.asUintN(64, 0x9e3779b97f4a7c15n * 0xbf58476d1ce4e5b9n);
    expect(mul64(0x9e3779b9, 0x7f4a7c15, 0xbf58476d, 0x1ce4e5b9)).toEqual([Number(p >> 32n), Number(p & 0xffffffffn)]);
  });

  it("is SplitMix64 → Box–Muller", () => {
    // fastwan-rs seeds the state as `seed ^ GOLDEN`, so the canonical
    // SplitMix64 stream (state 0: first output 0xE220A8397B1DCDAF) appears at seed GOLDEN.
    const r = new Rng(0x9e3779b97f4a7c15n);
    expect(r.nextU64()).toBe(0xe220a8397b1dcdafn);
    expect(r.nextU64()).toBe(0x6e789e6aa1b965f4n);
    const v = new Rng(7).normal(200_000);
    let sum = 0;
    let sq = 0;
    for (const x of v) {
      sum += x;
      sq += x * x;
    }
    const mean = sum / v.length;
    const variance = sq / v.length - mean * mean;
    expect(Math.abs(mean)).toBeLessThan(0.01);
    expect(Math.abs(variance - 1.0)).toBeLessThan(0.02);
    // The stream equals the scalar definition.
    const a = new Rng(3);
    const scalar: number[] = [];
    for (let i = 0; i < 4; i++) {
      const u1 = a.unit();
      const u2 = a.unit();
      const rr = Math.sqrt(-2.0 * Math.log(u1));
      const th = 2.0 * Math.PI * u2;
      scalar.push(rr * Math.cos(th), rr * Math.sin(th));
    }
    const got = new Rng(3).normal(8);
    for (let i = 0; i < 8; i++) expect(Math.abs(got[i] - scalar[i])).toBeLessThan(1e-6);
    // odd counts drop the second half of the last pair
    expect(new Rng(3).normal(7)).toEqual(got.subarray(0, 7));
  });

  it("matches the Python stream for seed 7 (first normals)", () => {
    // python: Rng(7).normal(4) → computed from the same SplitMix64 outputs
    const r = new Rng(7);
    const u = [r.unit(), r.unit()];
    const rr = Math.sqrt(-2.0 * Math.log(u[0]));
    const th = 2.0 * Math.PI * u[1];
    const first = new Rng(7).normal(2);
    expect(first[0]).toBe(Math.fround(rr * Math.cos(th)));
    expect(first[1]).toBe(Math.fround(rr * Math.sin(th)));
    // unit() is ((x >> 11) + 1) / 2^53, in (0, 1]
    const x = new Rng(7).nextU64();
    expect(new Rng(7).unit()).toBe(Number((x >> 11n) + 1n) / 2 ** 53);
  });
});

describe("host", () => {
  it("patchify / unpatchify use Diffusers' index conventions", () => {
    const [c, t, h, w, p] = [4, 3, 8, 12, 2];
    const x = new Rng(0).normal(c * t * h * w);
    const rows = patchify(x, c, t, h, w, p);
    const feat = c * p * p;
    expect(rows.length).toBe(3 * 4 * 6 * feat);
    // feature index c·p² + kh·p + kw
    expect(rows[1 * 4 + 1 * 2 + 0]).toBe(x[((1 * t + 0) * h + 1) * w + 0]);
    // Diffusers unpatchify reads (kh·p + kw)·c + c': permute the features and go back
    const permuted = new Float32Array(rows.length);
    const nRows = rows.length / feat;
    for (let r = 0; r < nRows; r++) {
      for (let cc = 0; cc < c; cc++) {
        for (let kh = 0; kh < p; kh++) {
          for (let kw = 0; kw < p; kw++) {
            permuted[r * feat + (kh * p + kw) * c + cc] = rows[r * feat + cc * p * p + kh * p + kw];
          }
        }
      }
    }
    const back = unpatchify(permuted, c, t, h, w, p);
    expect(back).toEqual(x);
  });

  it("dmd_step is float32 `x0 = x − σv; (1 − σ')x0 + σ'ε`", () => {
    const x = Float32Array.of(1, -2, 0.5);
    const v = Float32Array.of(0.25, 0.5, -1);
    const e = Float32Array.of(0.1, 0.2, 0.3);
    const f = Math.fround;
    const last = dmdStep(x, v, 0.52173912525177, null, null);
    for (let i = 0; i < 3; i++) expect(last[i]).toBe(f(x[i] - f(f(0.52173912525177) * v[i])));
    const mid = dmdStep(x, v, 1.0, 0.7567567229270935, e);
    const s = f(0.7567567229270935);
    for (let i = 0; i < 3; i++) {
      const x0 = f(x[i] - f(f(1.0) * v[i]));
      expect(mid[i]).toBe(f(f(f(f(1) - s) * x0) + f(s * e[i])));
    }
    expect(() => dmdStep(x, v, 1.0, 0.5, null)).toThrow();
  });

  it("pad_rows zero-pads to a multiple of the chunk", () => {
    const x = Float32Array.of(1, 2, 3, 4, 5, 6); // 3 rows of 2
    const [p, n] = padRows(x, 3, 2, 4);
    expect(n).toBe(3);
    expect(Array.from(p)).toEqual([1, 2, 3, 4, 5, 6, 0, 0]);
    const [same] = padRows(x, 3, 2, 3);
    expect(same).toBe(x);
  });

  it("to_rgb8 rounds like fastwan-rs and writes opaque RGBA", () => {
    // [1, 3, 1, 2]: two pixels
    const frames = Float32Array.of(0.0, 1.0, 0.5, -0.2, 0.998, 1.7);
    const [px] = toRgba8(frames, 1, 1, 2);
    expect(Array.from(px)).toEqual([0, 128, 254, 255, 255, 0, 255, 255]);
  });
});

describe("plan", () => {
  it("480p f16 fits 16 GiB with two resident blocks", () => {
    const p = plan(WanConfig.fastwan13b(), TaeConfig.taew21(), FASTWAN_480P, "f16", 1024, 256, 3, 16 * 2 ** 30);
    expect(p.fits).toBe(true);
    expect(p.tokens).toBe(32760);
    expect(p.attentionScratch).toBe(12 * 256 * 32760 * 4);
  });
  it("the phone tier in int4 fits 2 GiB and 480p does not", () => {
    const ok = plan(WanConfig.fastwan13b(), TaeConfig.taew21(), MOBILE_480x272, "int4", 1024, 256, 3, 2 * 2 ** 30);
    expect(ok.fits).toBe(true);
    expect(ok.weightsTotal).toBeLessThan(2 ** 30);
    const no = plan(WanConfig.fastwan13b(), TaeConfig.taew21(), FASTWAN_480P, "int4", 1024, 256, 3, 2 * 2 ** 30);
    expect(no.fits).toBe(false);
  });
});

describe("manifest", () => {
  const tiny = {
    format: "fastwan-litert/1",
    model: "tiny",
    config: WanConfig.tiny().toJson(),
    tae: TaeConfig.tiny().toJson(),
    schedule: DmdSchedule.fastwan13b().toJson(),
    weights: "f16",
    chunk: 16,
    graphs: {
      time_mods: "time_mods.tflite",
      text_embed: "text_embed.tflite",
      patch_embed: "patch_embed.tflite",
      head: "head.tflite",
      blocks: [
        { kv: "block_00_kv.tflite", qkv: "block_00_qkv.tflite", post: "block_00_post.tflite", bytes: 1 },
        { kv: "block_01_kv.tflite", qkv: "block_01_qkv.tflite", post: "block_01_post.tflite", bytes: 1 },
      ],
    },
    attention: [{ tokens: 18, q_chunk: 8, heads: 2, file: "attn_s18_q8_h2.tflite" }],
    tae_chunks: [{ frames: 2, latent_h: 6, latent_w: 4, file: "tae_f2_6x4.tflite" }],
    sizes: [{ width: 32, height: 48, frames: 9 }],
    files: { "head.tflite": { bytes: 6824, sha256: "x" } },
  };

  it("parses the tiny manifest and picks graphs by exact geometry", () => {
    const m = parseManifest(tiny);
    expect(m.cfg.dim).toBe(24);
    expect(m.attentionFor(18).file).toBe("attn_s18_q8_h2.tflite");
    expect(m.taeFor(6, 4).frames).toBe(2);
    expect(() => m.attentionFor(32760)).toThrow(/no attention graph for 32760 tokens \(it has \[18\]\)/);
    expect(() => m.taeFor(60, 104)).toThrow(/no TAEHV graph for latent 60x104 \(it has \[6x4\]\)/);
    expect(m.bytesOf("head.tflite")).toBe(6824);
    expect(m.allFiles()).toHaveLength(4 + 6 + 1 + 1);
  });

  it("refuses another format or a malformed field", () => {
    expect(() => parseManifest({ ...tiny, format: "fastwan-litert/2" })).toThrow(/format "fastwan-litert\/2", expected "fastwan-litert\/1"/);
    expect(() => parseManifest({ ...tiny, chunk: "16" })).toThrow(/chunk is not a number/);
    expect(() => parseManifest({ ...tiny, weights: "int2" })).toThrow(/weights/);
    expect(() => parseManifest({ ...tiny, graphs: { ...tiny.graphs, blocks: tiny.graphs.blocks.slice(0, 1) } })).toThrow(
      /lists 1 blocks but config.layers is 2/,
    );
  });
});
