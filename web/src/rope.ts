/**
 * 3-axis RoPE tables (Diffusers `WanRotaryPosEmbed`; fastwan-core `rope.rs`).
 *
 * Each head of width `d` is split `(t, h, w)` (44/42/42 at d = 128) and each
 * axis rotates interleaved pairs `(x[2i], x[2i+1])` by `pos · θ^(−2i/axis)`.
 * Both members of a pair share one angle, so the table stores `d/2` angles per
 * token. Angles are formed in f64 and their cos/sin cast to f32, as Diffusers.
 */

import type { WanConfig } from "./config.js";

function axisFreqs(dim: number, theta: number): Float64Array {
  const n = Math.floor(dim / 2);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = 1.0 / Math.pow(theta, (2.0 * i) / dim);
  return out;
}

export interface RopeTable {
  /** `[tokens, head_dim/2]` row-major. */
  cos: Float32Array;
  sin: Float32Array;
  tokens: number;
  width: number;
}

/** `(cos, sin)` of shape `[tokens, head_dim/2]`, tokens in `(t, h, w)` order. */
export function ropeTable(cfg: WanConfig, grid: readonly [number, number, number], startFrame = 0): RopeTable {
  const [dt, dh, dw] = cfg.ropeSplit();
  const [gt, gh, gw] = grid;
  const ft = axisFreqs(dt, cfg.ropeTheta);
  const fh = axisFreqs(dh, cfg.ropeTheta);
  const fw = axisFreqs(dw, cfg.ropeTheta);
  const width = Math.floor(cfg.headDim / 2);
  if (ft.length + fh.length + fw.length !== width) {
    throw new Error(`rope split ${dt}/${dh}/${dw} does not fill head_dim ${cfg.headDim}`);
  }
  const tokens = gt * gh * gw;
  const cos = new Float32Array(tokens * width);
  const sin = new Float32Array(tokens * width);
  let row = 0;
  for (let t = 0; t < gt; t++) {
    const pt = startFrame + t;
    for (let h = 0; h < gh; h++) {
      for (let w = 0; w < gw; w++) {
        let k = row * width;
        for (let i = 0; i < ft.length; i++, k++) {
          const a = pt * ft[i];
          cos[k] = Math.cos(a);
          sin[k] = Math.sin(a);
        }
        for (let i = 0; i < fh.length; i++, k++) {
          const a = h * fh[i];
          cos[k] = Math.cos(a);
          sin[k] = Math.sin(a);
        }
        for (let i = 0; i < fw.length; i++, k++) {
          const a = w * fw[i];
          cos[k] = Math.cos(a);
          sin[k] = Math.sin(a);
        }
        row++;
      }
    }
  }
  return { cos, sin, tokens, width };
}
