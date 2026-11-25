package dev.fastwan.litert.core

/**
 * Memory plan (`plan.py`; fastwan-core `plan.rs`, for the graph cut): what a generation holds at
 * once, before anything is allocated. Hosts show it and refuse what will not fit.
 */
public data class Plan(
    val tokens: Int,
    val chunk: Int,
    val qChunk: Int,
    val weightsTotal: Long,
    val weightsResident: Long,
    val crossKv: Long,
    val activations: Long,
    val attentionScratch: Long,
    val decodePeak: Long,
    val peak: Long,
    val budget: Long,
    val fits: Boolean,
    val notes: List<String> = emptyList(),
) {
    override fun toString(): String {
        fun mib(b: Long): String = String.format(java.util.Locale.ROOT, "%9.1f MiB", b / (1 shl 20).toDouble())
        val lines = mutableListOf(
            "tokens            $tokens",
            "weights (all)     ${mib(weightsTotal)}",
            "weights resident  ${mib(weightsResident)}",
            "cross-attn K/V    ${mib(crossKv)}",
            "DiT activations   ${mib(activations)} (row chunk $chunk)",
            "attention scratch ${mib(attentionScratch)} (query chunk $qChunk)",
            "decode peak       ${mib(decodePeak)}",
            "peak              ${mib(peak)} of ${String.format(java.util.Locale.ROOT, "%.0f", budget / (1 shl 20).toDouble())} MiB  fits: $fits",
        )
        lines += notes.map { "note: $it" }
        return lines.joinToString("\n")
    }

    public companion object {
        public val BYTES_PER_WEIGHT: Map<String, Double> = mapOf("f32" to 4.0, "f16" to 2.0, "int8" to 1.0, "int4" to 0.5)

        public fun compute(
            cfg: WanConfig,
            tae: TaeConfig,
            g: Geometry,
            weights: String,
            chunk: Int,
            qChunk: Int,
            taeFrames: Int,
            budgetBytes: Long,
            residentBlocks: Int = 2,
        ): Plan {
            val s = g.tokens(cfg).toLong()
            val d = cfg.dim.toLong()
            val a = 4L // f32 activations
            val wb = BYTES_PER_WEIGHT[weights]
                ?: throw IllegalArgumentException("unknown weight format `$weights` (${BYTES_PER_WEIGHT.keys})")
            val ffn = cfg.ffnDim.toLong()
            val blockParams = 8 * (d * d + d) + 2 * (d * ffn) + ffn + d + 6 * d
            val otherParams = cfg.params() - cfg.layers * blockParams
            val weightsTotal = (cfg.layers * blockParams * wb + otherParams * 2).toLong()
            val weightsResident = (residentBlocks * blockParams * wb + otherParams * 2).toLong()
            val crossKv = cfg.layers.toLong() * 2 * cfg.textLen * d * a
            // h, q, k, v, o: [S, d] each; the latent, velocity and noise; rope tables.
            val activations = (5 * s * d + 3 * g.latentLen(cfg).toLong() + s * cfg.headDim) * a
            val attentionScratch = cfg.heads.toLong() * qChunk * s * a
            val notes = mutableListOf<String>()
            // TAEHV peak: the last stage's map at 8x resolution for the chunk's 4F frames, plus
            // input/output of the MemBlock convs (x, past, two temporaries, out).
            val lat = g.latent()
            val f = minOf(taeFrames, lat.t).toLong()
            val nf = tae.widths
            val px = lat.h.toLong() * lat.w
            val s0 = f * nf[0] * px
            val s1 = f * tae.tgrow[0] * nf[1] * 4 * px
            val s2 = f * tae.tgrow[0] * tae.tgrow[1] * nf[2] * 16 * px
            val out = f * tae.tUpscale() * nf[3] * 64 * px
            val decodePeak = maxOf(5 * s0, 5 * s1, 5 * s2, s2 + 2 * out) * a
            val peak = maxOf(weightsResident + crossKv + activations + attentionScratch, weightsResident + decodePeak)
            if (residentBlocks < cfg.layers) {
                notes += "$residentBlocks of ${cfg.layers} block graphs resident; the rest stream from disk / the network"
            }
            if (s % chunk != 0L) {
                notes += "$s rows run as ${(s + chunk - 1) / chunk} chunks of $chunk (${chunk - s % chunk} padded rows)"
            }
            return Plan(
                s.toInt(), chunk, qChunk, weightsTotal, weightsResident, crossKv, activations,
                attentionScratch, decodePeak, peak, budgetBytes, peak <= budgetBytes, notes,
            )
        }
    }
}
