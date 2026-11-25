/**
 * Host-side tensor plumbing shared by the pipeline and the tests: patchify /
 * unpatchify in Diffusers' orders, the DMD step, chunk padding and RGB
 * quantisation (port of `python/fastwan_litert/host.py`). Row-major
 * `Float32Array`s with explicit shapes; float32 arithmetic is reproduced with
 * `Math.fround`.
 */

const f = Math.fround;

/** `[C, T, H, W]` → `[T·(H/p)·(W/p), C·p²]`, feature `c·p² + kh·p + kw` (the Conv3d `(1, p, p)` weight flattened). */
export function patchify(latent: Float32Array, c: number, t: number, h: number, w: number, p: number): Float32Array {
  if (latent.length !== c * t * h * w) throw new Error(`patchify: ${latent.length} values for [${c}, ${t}, ${h}, ${w}]`);
  if (h % p !== 0 || w % p !== 0) throw new Error(`patchify: ${h}x${w} is not a multiple of patch ${p}`);
  const hp = h / p;
  const wp = w / p;
  const feat = c * p * p;
  const out = new Float32Array(t * hp * wp * feat);
  let row = 0;
  for (let tt = 0; tt < t; tt++) {
    for (let hh = 0; hh < hp; hh++) {
      for (let ww = 0; ww < wp; ww++, row++) {
        const base = row * feat;
        for (let cc = 0; cc < c; cc++) {
          for (let kh = 0; kh < p; kh++) {
            const src = ((cc * t + tt) * h + hh * p + kh) * w + ww * p;
            const dst = base + cc * p * p + kh * p;
            for (let kw = 0; kw < p; kw++) out[dst + kw] = latent[src + kw];
          }
        }
      }
    }
  }
  return out;
}

/** `[T·(H/p)·(W/p), p²·C]` (feature `(kh·p + kw)·c + c'`, Diffusers unpatchify) → `[C, T, H, W]`. */
export function unpatchify(rows: Float32Array, c: number, t: number, h: number, w: number, p: number): Float32Array {
  const hp = h / p;
  const wp = w / p;
  const feat = p * p * c;
  if (rows.length !== t * hp * wp * feat) {
    throw new Error(`unpatchify: ${rows.length} values for ${t * hp * wp} rows of ${feat}`);
  }
  const out = new Float32Array(c * t * h * w);
  let row = 0;
  for (let tt = 0; tt < t; tt++) {
    for (let hh = 0; hh < hp; hh++) {
      for (let ww = 0; ww < wp; ww++, row++) {
        const base = row * feat;
        for (let kh = 0; kh < p; kh++) {
          for (let kw = 0; kw < p; kw++) {
            const src = base + (kh * p + kw) * c;
            const y = hh * p + kh;
            const x = ww * p + kw;
            for (let cc = 0; cc < c; cc++) out[((cc * t + tt) * h + y) * w + x] = rows[src + cc];
          }
        }
      }
    }
  }
  return out;
}

/** `x0 = x − σ_t·v`; then `(1 − σ_n)·x0 + σ_n·noise` when `sigmaNext` is given. Float32 throughout. */
export function dmdStep(
  x: Float32Array,
  v: Float32Array,
  sigmaT: number,
  sigmaNext: number | null,
  noise: Float32Array | null,
): Float32Array {
  if (v.length !== x.length) throw new Error(`dmd_step: v has ${v.length} values, x has ${x.length}`);
  const out = new Float32Array(x.length);
  const s = f(sigmaT);
  if (sigmaNext === null) {
    for (let i = 0; i < x.length; i++) out[i] = f(x[i] - f(s * v[i]));
    return out;
  }
  if (noise === null || noise.length !== x.length) throw new Error("dmd_step: re-noise needs noise of the latent's size");
  const sn = f(sigmaNext);
  const one = f(f(1.0) - sn);
  for (let i = 0; i < x.length; i++) {
    const x0 = f(x[i] - f(s * v[i]));
    out[i] = f(f(one * x0) + f(sn * noise[i]));
  }
  return out;
}

/** Zero-pad the row axis of `[rows, rowSize]` to a multiple of `chunk`; returns `[padded, real rows]`. */
export function padRows(x: Float32Array, rows: number, rowSize: number, chunk: number): [Float32Array, number] {
  if (x.length !== rows * rowSize) throw new Error(`pad_rows: ${x.length} values for ${rows} rows of ${rowSize}`);
  const full = Math.ceil(rows / chunk) * chunk;
  if (full === rows) return [x, rows];
  const out = new Float32Array(full * rowSize);
  out.set(x);
  return [out, rows];
}

export function geluTanh(x: number): number {
  return 0.5 * x * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
}

/**
 * `[F, 3, H, W]` in `[0, 1]` → `F` RGBA frames (`Uint8ClampedArray` of `H·W·4`,
 * alpha 255, ready for `ImageData`) with fastwan-rs's rounding
 * (`(clamp(v)·255 + 0.5) as u8`, float32 arithmetic, truncation).
 */
export function toRgba8(frames: Float32Array, n: number, h: number, w: number): Uint8ClampedArray[] {
  if (frames.length !== n * 3 * h * w) throw new Error(`to_rgb8: ${frames.length} values for [${n}, 3, ${h}, ${w}]`);
  const out: Uint8ClampedArray[] = [];
  const plane = h * w;
  for (let i = 0; i < n; i++) {
    const px = new Uint8ClampedArray(plane * 4);
    const base = i * 3 * plane;
    for (let j = 0; j < plane; j++) {
      px[j * 4] = quantize(frames[base + j]);
      px[j * 4 + 1] = quantize(frames[base + plane + j]);
      px[j * 4 + 2] = quantize(frames[base + 2 * plane + j]);
      px[j * 4 + 3] = 255;
    }
    out.push(px);
  }
  return out;
}

function quantize(v: number): number {
  const c = v < 0 ? 0 : v > 1 ? 1 : v;
  return Math.floor(f(f(c * 255.0) + 0.5));
}
