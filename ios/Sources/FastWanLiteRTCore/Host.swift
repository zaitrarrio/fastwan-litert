// Host-side tensor plumbing shared by the pipeline and the tests (python
// host.py): patchify / unpatchify in Diffusers' orders, the DMD step, chunk
// padding, and the 8-bit frame conversion. Ported verbatim.

import Foundation

public enum Host {
    /// `[C, T, H, W]` → `[T·(H/p)·(W/p), C·p²]`, feature `c·p² + kh·p + kw`
    /// (the Conv3d `(1, p, p)` weight flattened).
    public static func patchify(_ latent: Tensor, patch p: Int) -> Tensor {
        precondition(latent.rank == 4, "patchify wants [C, T, H, W], got \(latent.shape)")
        let c = latent.shape[0], t = latent.shape[1], h = latent.shape[2], w = latent.shape[3]
        precondition(h % p == 0 && w % p == 0, "latent \(h)x\(w) is not a multiple of patch \(p)")
        let hp = h / p, wp = w / p
        let rows = t * hp * wp
        let feats = c * p * p
        var out = [Float](repeating: 0, count: rows * feats)
        let src = latent.data
        for ti in 0..<t {
            for hi in 0..<hp {
                for wi in 0..<wp {
                    let row = (ti * hp + hi) * wp + wi
                    let base = row * feats
                    for ci in 0..<c {
                        for kh in 0..<p {
                            let y = hi * p + kh
                            let srcRow = ((ci * t + ti) * h + y) * w + wi * p
                            for kw in 0..<p {
                                out[base + ci * p * p + kh * p + kw] = src[srcRow + kw]
                            }
                        }
                    }
                }
            }
        }
        return Tensor(shape: [rows, feats], data: out)
    }

    /// `[T·(H/p)·(W/p), p²·C]` (feature `(kh·p + kw)·c + c'`, Diffusers unpatchify) → `[C, T, H, W]`.
    public static func unpatchify(_ rows: Tensor, channels c: Int, t: Int, h: Int, w: Int, patch p: Int) -> Tensor {
        let hp = h / p, wp = w / p
        let feats = p * p * c
        precondition(rows.shape == [t * hp * wp, feats], "unpatchify rows \(rows.shape) for c=\(c) t=\(t) h=\(h) w=\(w) p=\(p)")
        var out = [Float](repeating: 0, count: c * t * h * w)
        let src = rows.data
        for ti in 0..<t {
            for hi in 0..<hp {
                for wi in 0..<wp {
                    let base = ((ti * hp + hi) * wp + wi) * feats
                    for kh in 0..<p {
                        for kw in 0..<p {
                            let f = (kh * p + kw) * c
                            let y = hi * p + kh
                            let x = wi * p + kw
                            for ci in 0..<c {
                                out[((ci * t + ti) * h + y) * w + x] = src[base + f + ci]
                            }
                        }
                    }
                }
            }
        }
        return Tensor(shape: [c, t, h, w], data: out)
    }

    /// `x0 = x − σ_t·v`; then `(1 − σ_n)·x0 + σ_n·noise` when `sigmaNext` is given. Float32 arithmetic, as numpy.
    public static func dmdStep(x: Tensor, v: Tensor, sigma: Double, sigmaNext: Double?, noise: Tensor?) -> Tensor {
        precondition(x.shape == v.shape, "dmd step x \(x.shape) vs v \(v.shape)")
        let st = Float(sigma)
        var out = [Float](repeating: 0, count: x.count)
        let xd = x.data, vd = v.data
        if let sn = sigmaNext {
            guard let nz = noise, nz.shape == x.shape else {
                preconditionFailure("dmd step needs noise of shape \(x.shape)")
            }
            let s = Float(sn)
            let oneMinus = Float(1.0) - s
            let nd = nz.data
            for i in 0..<out.count {
                let x0 = xd[i] - st * vd[i]
                out[i] = oneMinus * x0 + s * nd[i]
            }
        } else {
            for i in 0..<out.count {
                out[i] = xd[i] - st * vd[i]
            }
        }
        return Tensor(shape: x.shape, data: out)
    }

    /// Zero-pad the first axis to a multiple of `chunk`; returns `(padded, real rows)`.
    public static func padRows(_ x: Tensor, chunk: Int) -> (padded: Tensor, real: Int) {
        let n = x.rows
        let full = (n + chunk - 1) / chunk * chunk
        if full == n { return (x, n) }
        var data = x.data
        data.append(contentsOf: [Float](repeating: 0, count: (full - n) * x.rowStride))
        return (Tensor(shape: [full] + Array(x.shape.dropFirst()), data: data), n)
    }

    public static func geluTanh(_ x: Float) -> Float {
        let c = Float((2.0 / Double.pi).squareRoot())
        return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)))
    }

    /// `[F, 3, H, W]` in `[0, 1]` → `[F, H, W, 3]` uint8 with fastwan-rs's
    /// rounding (`(clamp(v)·255 + 0.5) as u8`).
    public static func toRgb8(_ frames: Tensor) -> (rgb: [UInt8], frames: Int, height: Int, width: Int) {
        precondition(frames.rank == 4 && frames.shape[1] == 3, "toRgb8 wants [F, 3, H, W], got \(frames.shape)")
        let f = frames.shape[0], h = frames.shape[2], w = frames.shape[3]
        let plane = h * w
        var out = [UInt8](repeating: 0, count: f * plane * 3)
        let src = frames.data
        for fi in 0..<f {
            let fBase = fi * 3 * plane
            let oBase = fi * plane * 3
            for p in 0..<plane {
                for c in 0..<3 {
                    out[oBase + p * 3 + c] = quantize(src[fBase + c * plane + p])
                }
            }
        }
        return (out, f, h, w)
    }

    /// `[F, 3, H, W]` in `[0, 1]` → `[F, H, W, 4]` RGBA (alpha 255), what `CGImage` wants.
    public static func toRgba8(_ frames: Tensor) -> (rgba: [UInt8], frames: Int, height: Int, width: Int) {
        precondition(frames.rank == 4 && frames.shape[1] == 3, "toRgba8 wants [F, 3, H, W], got \(frames.shape)")
        let f = frames.shape[0], h = frames.shape[2], w = frames.shape[3]
        let plane = h * w
        var out = [UInt8](repeating: 255, count: f * plane * 4)
        let src = frames.data
        for fi in 0..<f {
            let fBase = fi * 3 * plane
            let oBase = fi * plane * 4
            for p in 0..<plane {
                for c in 0..<3 {
                    out[oBase + p * 4 + c] = quantize(src[fBase + c * plane + p])
                }
            }
        }
        return (out, f, h, w)
    }

    /// `[F, H, W, 4]` RGBA → `[F, H, W, 3]` RGB (for comparisons against `expected.safetensors`).
    public static func rgbaToRgb(_ rgba: [UInt8]) -> [UInt8] {
        let pixels = rgba.count / 4
        var out = [UInt8](repeating: 0, count: pixels * 3)
        for p in 0..<pixels {
            out[p * 3] = rgba[p * 4]
            out[p * 3 + 1] = rgba[p * 4 + 1]
            out[p * 3 + 2] = rgba[p * 4 + 2]
        }
        return out
    }

    @inline(__always)
    static func quantize(_ v: Float) -> UInt8 {
        // numpy: clip to [0, 1] (NaN stays NaN and casts to 0 on most platforms; we map it to 0), ·255 + 0.5, truncate.
        if v.isNaN { return 0 }
        let c = min(max(v, 0.0), 1.0)
        let q = c * 255.0 + 0.5
        return UInt8(q)  // q is in [0.5, 255.5]; truncation never exceeds 255
    }
}
