package dev.fastwan.litert.core

/**
 * Host-side tensor plumbing (`host.py`): patchify / unpatchify in Diffusers' orders, the DMD step,
 * chunk padding and the RGB quantisation. Ported verbatim; every arithmetic step is single
 * precision in the same order as the numpy reference.
 */
public object Host {
    /**
     * `[C, T, H, W]` → `[T·(H/p)·(W/p), C·p²]`, feature `c·p² + kh·p + kw` (the Conv3d `(1, p, p)`
     * weight flattened).
     */
    public fun patchify(latent: Tensor, p: Int): Tensor {
        require(latent.rank == 4) { "patchify wants [C, T, H, W], got ${latent.shapeString()}" }
        val (c, t, h, w) = latent.shape.let { listOf(it[0], it[1], it[2], it[3]) }
        require(h % p == 0 && w % p == 0) { "latent ${h}x$w is not a multiple of patch $p" }
        val hp = h / p
        val wp = w / p
        val rows = t * hp * wp
        val feat = c * p * p
        val out = FloatArray(rows * feat)
        val src = latent.data
        for (tt in 0 until t) for (hh in 0 until hp) for (ww in 0 until wp) {
            val row = (tt * hp + hh) * wp + ww
            val base = row * feat
            for (cc in 0 until c) for (kh in 0 until p) for (kw in 0 until p) {
                out[base + cc * p * p + kh * p + kw] = src[((cc * t + tt) * h + hh * p + kh) * w + ww * p + kw]
            }
        }
        return Tensor(out, intArrayOf(rows, feat))
    }

    /**
     * `[T·(H/p)·(W/p), p²·C]` (feature `(kh·p + kw)·c + c'`, Diffusers unpatchify) → `[C, T, H, W]`.
     */
    public fun unpatchify(rows: Tensor, c: Int, t: Int, h: Int, w: Int, p: Int): Tensor {
        val hp = h / p
        val wp = w / p
        val feat = p * p * c
        require(rows.rank == 2 && rows.shape[0] == t * hp * wp && rows.shape[1] == feat) {
            "unpatchify wants [${t * hp * wp}, $feat], got ${rows.shapeString()}"
        }
        val out = FloatArray(c * t * h * w)
        val src = rows.data
        for (tt in 0 until t) for (hh in 0 until hp) for (ww in 0 until wp) {
            val base = ((tt * hp + hh) * wp + ww) * feat
            for (kh in 0 until p) for (kw in 0 until p) for (cc in 0 until c) {
                out[((cc * t + tt) * h + hh * p + kh) * w + ww * p + kw] = src[base + (kh * p + kw) * c + cc]
            }
        }
        return Tensor(out, intArrayOf(c, t, h, w))
    }

    /** `x0 = x − σ_t·v`; then `(1 − σ_n)·x0 + σ_n·noise` when `sigmaNext` is given. */
    public fun dmdStep(x: Tensor, v: Tensor, sigmaT: Float, sigmaNext: Float?, noise: Tensor?): Tensor {
        require(x.size == v.size) { "dmdStep: x ${x.shapeString()} vs v ${v.shapeString()}" }
        val out = FloatArray(x.size)
        val xd = x.data
        val vd = v.data
        if (sigmaNext == null) {
            for (i in out.indices) out[i] = xd[i] - sigmaT * vd[i]
        } else {
            requireNotNull(noise) { "dmdStep needs noise when sigmaNext is given" }
            require(noise.size == x.size) { "dmdStep: noise ${noise.shapeString()} vs x ${x.shapeString()}" }
            val nd = noise.data
            val s = sigmaNext
            val oneMinus = 1.0f - s
            for (i in out.indices) {
                val x0 = xd[i] - sigmaT * vd[i]
                out[i] = oneMinus * x0 + s * nd[i]
            }
        }
        return Tensor(out, x.shape.copyOf())
    }

    public class Padded(public val tensor: Tensor, public val realRows: Int)

    /** Zero-pad the first axis to a multiple of `chunk`; returns `(padded, real rows)`. */
    public fun padRows(x: Tensor, chunk: Int): Padded {
        require(chunk > 0) { "chunk must be positive" }
        val n = x.rows
        val full = (n + chunk - 1) / chunk * chunk
        if (full == n) return Padded(x, n)
        val rs = x.rowSize
        val data = FloatArray(full * rs)
        System.arraycopy(x.data, 0, data, 0, x.data.size)
        return Padded(Tensor(data, intArrayOf(full) + x.shape.copyOfRange(1, x.shape.size)), n)
    }

    private fun q8(v: Float): Byte {
        val c = if (v < 0.0f) 0.0f else if (v > 1.0f) 1.0f else v
        return (c * 255.0f + 0.5f).toInt().toByte()
    }

    /**
     * `[F, 3, H, W]` in `[0, 1]` → `[F, H, W, 3]` uint8 with fastwan-rs's rounding
     * (`(clamp(v)·255 + 0.5) as u8`).
     */
    public fun toRgb8(frames: Tensor): ByteArray {
        require(frames.rank == 4 && frames.shape[1] == 3) { "toRgb8 wants [F, 3, H, W], got ${frames.shapeString()}" }
        val f = frames.shape[0]
        val h = frames.shape[2]
        val w = frames.shape[3]
        val out = ByteArray(f * h * w * 3)
        val src = frames.data
        val plane = h * w
        for (i in 0 until f) for (y in 0 until h) for (x in 0 until w) {
            val o = ((i * h + y) * w + x) * 3
            val s = i * 3 * plane + y * w + x
            out[o] = q8(src[s])
            out[o + 1] = q8(src[s + plane])
            out[o + 2] = q8(src[s + 2 * plane])
        }
        return out
    }

    /** As [toRgb8] but `[F, H, W, 4]` with alpha 255: the byte order of an `ARGB_8888` `Bitmap`. */
    public fun toRgba8(frames: Tensor): ByteArray {
        require(frames.rank == 4 && frames.shape[1] == 3) { "toRgba8 wants [F, 3, H, W], got ${frames.shapeString()}" }
        val f = frames.shape[0]
        val h = frames.shape[2]
        val w = frames.shape[3]
        val out = ByteArray(f * h * w * 4)
        val src = frames.data
        val plane = h * w
        for (i in 0 until f) for (y in 0 until h) for (x in 0 until w) {
            val o = ((i * h + y) * w + x) * 4
            val s = i * 3 * plane + y * w + x
            out[o] = q8(src[s])
            out[o + 1] = q8(src[s + plane])
            out[o + 2] = q8(src[s + 2 * plane])
            out[o + 3] = -1 // 0xFF
        }
        return out
    }

    /** `[F, H, W, 3]` uint8 → `[F, H, W, 4]` uint8, alpha 255. */
    public fun rgbToRgba(rgb: ByteArray): ByteArray {
        require(rgb.size % 3 == 0) { "rgb byte count ${rgb.size} is not a multiple of 3" }
        val px = rgb.size / 3
        val out = ByteArray(px * 4)
        for (i in 0 until px) {
            out[i * 4] = rgb[i * 3]
            out[i * 4 + 1] = rgb[i * 3 + 1]
            out[i * 4 + 2] = rgb[i * 3 + 2]
            out[i * 4 + 3] = -1
        }
        return out
    }
}
