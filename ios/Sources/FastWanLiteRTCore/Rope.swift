// 3-axis RoPE tables (Diffusers `WanRotaryPosEmbed`; python rope.py; fastwan-core `rope.rs`).
//
// Each head of width `d` is split `(t, h, w)` (44/42/42 at d = 128) and each
// axis rotates interleaved pairs `(x[2i], x[2i+1])` by `pos · θ^(−2i/axis)`.
// Both members of a pair share one angle, so the table stores `d/2` angles per
// token. Angles are formed in f64 and their cos/sin cast to f32, as Diffusers.

import Foundation

public enum Rope {
    static func axisFreqs(dim: Int, theta: Double) -> [Double] {
        (0..<(dim / 2)).map { i in 1.0 / pow(theta, 2.0 * Double(i) / Double(dim)) }
    }

    /// `(cos, sin)` of shape `[tokens, head_dim/2]`, tokens in `(t, h, w)` order.
    public static func table(cfg: WanConfig, grid: (t: Int, h: Int, w: Int), startFrame: Int = 0) -> (cos: Tensor, sin: Tensor) {
        let split = cfg.ropeSplit
        let ft = axisFreqs(dim: split.t, theta: cfg.ropeTheta)
        let fh = axisFreqs(dim: split.h, theta: cfg.ropeTheta)
        let fw = axisFreqs(dim: split.w, theta: cfg.ropeTheta)
        let half = cfg.headDim / 2
        precondition(ft.count + fh.count + fw.count == half, "rope split does not cover head_dim/2")
        let tokens = grid.t * grid.h * grid.w
        var cosv = [Float](repeating: 0, count: tokens * half)
        var sinv = [Float](repeating: 0, count: tokens * half)
        var row = 0
        for t in 0..<grid.t {
            let pt = Double(startFrame + t)
            for h in 0..<grid.h {
                let ph = Double(h)
                for w in 0..<grid.w {
                    let pw = Double(w)
                    var j = row * half
                    for f in ft {
                        let a = pt * f
                        cosv[j] = Float(cos(a))
                        sinv[j] = Float(sin(a))
                        j += 1
                    }
                    for f in fh {
                        let a = ph * f
                        cosv[j] = Float(cos(a))
                        sinv[j] = Float(sin(a))
                        j += 1
                    }
                    for f in fw {
                        let a = pw * f
                        cosv[j] = Float(cos(a))
                        sinv[j] = Float(sin(a))
                        j += 1
                    }
                    row += 1
                }
            }
        }
        return (Tensor(shape: [tokens, half], data: cosv), Tensor(shape: [tokens, half], data: sinv))
    }
}
