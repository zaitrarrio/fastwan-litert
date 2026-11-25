package dev.fastwan.litert.core

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Architecture constants of the Wan 2.1 DiT (`python/fastwan_litert/config.py`; fastwan-rs
 * `config.rs`). Batch is always 1: the DMD-distilled FastWan models run without CFG.
 */
@Serializable
public data class WanConfig(
    val patch: Int,
    val heads: Int,
    @SerialName("head_dim") val headDim: Int,
    @SerialName("in_channels") val inChannels: Int,
    @SerialName("out_channels") val outChannels: Int,
    @SerialName("text_dim") val textDim: Int,
    @SerialName("text_len") val textLen: Int,
    @SerialName("freq_dim") val freqDim: Int,
    @SerialName("ffn_dim") val ffnDim: Int,
    val layers: Int,
    val eps: Double,
    // `docs/graphs.md`'s manifest example omits this field (the Python reference requires it);
    // default to the 1.3B value so both manifests parse.
    @SerialName("rope_max_seq_len") val ropeMaxSeqLen: Int = 1024,
    @SerialName("rope_theta") val ropeTheta: Double,
) {
    /** `d = heads · head_dim`. */
    val dim: Int get() = heads * headDim

    public fun patchFeaturesIn(): Int = patch * patch * inChannels

    public fun patchFeaturesOut(): Int = patch * patch * outChannels

    /**
     * `(t, h, w)` widths of one head (Diffusers `WanRotaryPosEmbed`: `h = w = 2·(d/6)`,
     * `t = d − h − w`; 44/42/42 at d = 128).
     */
    public fun ropeSplit(): Triple<Int, Int, Int> {
        val hw = 2 * (headDim / 6)
        return Triple(headDim - 2 * hw, hw, hw)
    }

    /** Dense parameter count of the DiT (VSA gates excluded). */
    public fun params(): Long {
        val d = dim.toLong()
        fun lin(i: Long, o: Long): Long = i * o + o
        val block = 8 * lin(d, d) + 4 * d + 2 * d + lin(d, ffnDim.toLong()) + lin(ffnDim.toLong(), d) + 6 * d
        return layers * block +
            lin(patchFeaturesIn().toLong(), d) +
            lin(freqDim.toLong(), d) +
            lin(d, d) +
            lin(d, 6 * d) +
            lin(textDim.toLong(), d) +
            lin(d, d) +
            lin(d, patchFeaturesOut().toLong()) +
            2 * d
    }

    public companion object {
        /** `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` (and Wan 2.1 T2V 1.3B). */
        public fun fastwan13b(): WanConfig = WanConfig(
            patch = 2, heads = 12, headDim = 128, inChannels = 16, outChannels = 16,
            textDim = 4096, textLen = 512, freqDim = 256, ffnDim = 8960, layers = 30,
            eps = 1e-6, ropeMaxSeqLen = 1024, ropeTheta = 10000.0,
        )

        /** The unit-test model (fastwan-rs `WanConfig::tiny`): head_dim a multiple of 6. */
        public fun tiny(): WanConfig = WanConfig(
            patch = 2, heads = 2, headDim = 12, inChannels = 4, outChannels = 4,
            textDim = 16, textLen = 8, freqDim = 16, ffnDim = 40, layers = 2,
            eps = 1e-6, ropeMaxSeqLen = 64, ropeTheta = 10000.0,
        )

        public fun fromName(name: String): WanConfig = when (name) {
            "fastwan-1.3b", "fastwan21-1.3b", "fastwan", "FastVideo/FastWan2.1-T2V-1.3B-Diffusers" -> fastwan13b()
            "tiny" -> tiny()
            else -> throw IllegalArgumentException("unknown model `$name` (fastwan-1.3b or tiny)")
        }
    }
}

/** TAEHV (madebyollin/taehv) decoder for Wan 2.1 latents. */
@Serializable
public data class TaeConfig(
    @SerialName("latent_channels") val latentChannels: Int,
    val widths: List<Int>,
    val tgrow: List<Int>,
    @SerialName("mem_blocks") val memBlocks: Int,
    val patch: Int,
    val denormalize: Boolean,
) {
    init {
        require(widths.size == 4) { "tae.widths must have 4 entries, got $widths" }
        require(tgrow.size == 3) { "tae.tgrow must have 3 entries, got $tgrow" }
    }

    public fun tUpscale(): Int = tgrow.fold(1) { a, b -> a * b }

    /** Priming frames emitted first and dropped: `4T − 3` frames for Wan. */
    public fun framesToTrim(): Int = tUpscale() - 1

    public fun spatialScale(): Int = 8 * patch

    public fun outChannels(): Int = 3 * patch * patch

    public companion object {
        public fun taew21(): TaeConfig = TaeConfig(16, listOf(256, 128, 64, 64), listOf(1, 2, 2), 3, 1, false)

        /** LightX2V's LightTAE: TAEHV's network on raw (denormalised) latents. */
        public fun lightTaew21(): TaeConfig = TaeConfig(16, listOf(256, 128, 64, 64), listOf(1, 2, 2), 3, 1, true)

        public fun taew22(): TaeConfig = TaeConfig(48, listOf(256, 128, 64, 64), listOf(1, 2, 2), 3, 2, false)

        public fun tiny(): TaeConfig = TaeConfig(4, listOf(8, 6, 4, 4), listOf(1, 2, 2), 3, 1, false)
    }
}
