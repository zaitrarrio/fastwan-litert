package dev.fastwan.litert.core

import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

/** `cos` / `sin` of shape `[tokens, head_dim/2]`, tokens in `(t, h, w)` order. */
public class RopeTable(public val cos: Tensor, public val sin: Tensor) {
    public val tokens: Int get() = cos.shape[0]
    public val half: Int get() = cos.shape[1]
}

/**
 * 3-axis RoPE tables (Diffusers `WanRotaryPosEmbed`; `rope.py`; fastwan-core `rope.rs`).
 *
 * Each head of width `d` is split `(t, h, w)` (44/42/42 at d = 128) and each axis rotates
 * interleaved pairs `(x[2i], x[2i+1])` by `pos · θ^(−2i/axis)`. Both members of a pair share one
 * angle, so the table stores `d/2` angles per token. Angles are formed in f64 and their cos/sin
 * cast to f32, as Diffusers.
 */
public object Rope {
    internal fun axisFreqs(dim: Int, theta: Double): DoubleArray =
        DoubleArray(dim / 2) { i -> 1.0 / theta.pow(2.0 * i / dim) }

    public fun table(cfg: WanConfig, grid: Dims3, startFrame: Int = 0): RopeTable {
        val (dt, dh, dw) = cfg.ropeSplit()
        val ft = axisFreqs(dt, cfg.ropeTheta)
        val fh = axisFreqs(dh, cfg.ropeTheta)
        val fw = axisFreqs(dw, cfg.ropeTheta)
        val half = cfg.headDim / 2
        require(ft.size + fh.size + fw.size == half) { "rope split ${cfg.ropeSplit()} does not fill head_dim ${cfg.headDim}" }
        val n = grid.count
        val cosA = FloatArray(n * half)
        val sinA = FloatArray(n * half)
        var row = 0
        for (t in 0 until grid.t) {
            val pt = (startFrame + t).toDouble()
            for (h in 0 until grid.h) {
                val ph = h.toDouble()
                for (w in 0 until grid.w) {
                    val pw = w.toDouble()
                    var c = row * half
                    for (f in ft) { val a = pt * f; cosA[c] = cos(a).toFloat(); sinA[c] = sin(a).toFloat(); c++ }
                    for (f in fh) { val a = ph * f; cosA[c] = cos(a).toFloat(); sinA[c] = sin(a).toFloat(); c++ }
                    for (f in fw) { val a = pw * f; cosA[c] = cos(a).toFloat(); sinA[c] = sin(a).toFloat(); c++ }
                    row++
                }
            }
        }
        val shape = intArrayOf(n, half)
        return RopeTable(Tensor(cosA, shape), Tensor(sinA, shape.copyOf()))
    }
}
