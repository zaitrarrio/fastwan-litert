package dev.fastwan.litert.core

import kotlinx.serialization.Serializable

public const val VAE_SPATIAL: Int = 8
public const val VAE_TEMPORAL: Int = 4

/** `(t, h, w)` of a latent or of the token grid. */
public data class Dims3(val t: Int, val h: Int, val w: Int) {
    val count: Int get() = t * h * w
}

/** Pixel ↔ latent ↔ token shapes for a Wan 2.1 clip (`geometry.py`; fastwan-core `geometry.rs`). */
@Serializable
public data class Geometry(val width: Int, val height: Int, val frames: Int) {
    public fun validate(cfg: WanConfig) {
        val cell = VAE_SPATIAL * cfg.patch
        if (width <= 0 || height <= 0 || width % cell != 0 || height % cell != 0) {
            throw IllegalArgumentException("${width}x$height is not a positive multiple of $cell")
        }
        if (frames <= 0 || (frames - 1) % VAE_TEMPORAL != 0) {
            throw IllegalArgumentException("$frames frames is not 4n+1")
        }
    }

    /** Latent `(T, H, W)`. */
    public fun latent(): Dims3 = Dims3((frames - 1) / VAE_TEMPORAL + 1, height / VAE_SPATIAL, width / VAE_SPATIAL)

    public fun grid(cfg: WanConfig): Dims3 {
        val (t, h, w) = latent()
        return Dims3(t, h / cfg.patch, w / cfg.patch)
    }

    public fun tokens(cfg: WanConfig): Int = grid(cfg).count

    public fun latentLen(cfg: WanConfig): Int = cfg.inChannels * latent().count

    override fun toString(): String = "${width}x${height}x$frames"

    public companion object {
        /** `WxHxF`, e.g. `832x480x81`. */
        public fun parse(s: String): Geometry {
            val parts = s.lowercase().split("x")
            val v = parts.map { it.toIntOrNull() }
            if (v.size != 3 || v.any { it == null }) {
                throw IllegalArgumentException("geometry `$s` is not WIDTHxHEIGHTxFRAMES")
            }
            return Geometry(v[0]!!, v[1]!!, v[2]!!)
        }

        /** FastWan 2.1 1.3B's served default: 5 s at 16 fps. */
        public val FASTWAN_480P: Geometry = Geometry(832, 480, 81)

        /** fastwan-rs's phone tier. */
        public val MOBILE_480x272: Geometry = Geometry(480, 272, 33)
    }
}
