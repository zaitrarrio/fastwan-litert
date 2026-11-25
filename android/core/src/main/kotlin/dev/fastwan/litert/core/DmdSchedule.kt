package dev.fastwan.litert.core

import kotlinx.serialization.Serializable
import kotlin.math.abs

/**
 * DMD sampling schedule (`schedule.py`; fastwan-core `schedule.rs`). Each step maps onto the
 * nearest entry of the shift-8 *training* noise table:
 *
 *     x0 = x − σ_t·v
 *     x  = (1 − σ_next)·x0 + σ_next·ε     (fresh ε; skipped on the last step)
 *
 * `sigmas` holds the table sigma of every step (as the f32 table value) and then a trailing 0.0.
 */
@Serializable
public data class DmdSchedule(val timesteps: List<Int>, val sigmas: List<Double>) {
    init {
        require(sigmas.size == timesteps.size + 1) {
            "schedule needs ${timesteps.size + 1} sigmas for ${timesteps.size} timesteps, got ${sigmas.size}"
        }
    }

    public val size: Int get() = timesteps.size

    public data class Coeffs(val sigma: Double, val next: Double?)

    /** `(σ_t, σ_next)`; `next` is null on the last step (no re-noise). */
    public fun coeffs(i: Int): Coeffs = Coeffs(sigmas[i], if (i + 1 < size) sigmas[i + 1] else null)

    public companion object {
        public val FASTWAN_1_3B_STEPS: List<Int> = listOf(1000, 757, 522)
        public const val DMD_TRAINING_NOISE_SHIFT: Float = 8.0f

        /**
         * Float32 sigma / timestep tables, index 0 = timestep N, computed in single precision
         * exactly as the numpy reference (`_training_table`).
         */
        internal fun trainingTable(n: Int, shift: Float): Pair<FloatArray, FloatArray> {
            val nf = n.toFloat()
            val sm1 = shift - 1.0f
            val sig = FloatArray(n)
            val ts = FloatArray(n)
            for (i in 0 until n) {
                var s = (nf - i.toFloat()) / nf
                s = (shift * s) / (1.0f + sm1 * s)
                sig[i] = s
                ts[i] = s * nf
            }
            return sig to ts
        }

        public fun of(steps: List<Int>): DmdSchedule {
            val (sig, ts) = trainingTable(1000, DMD_TRAINING_NOISE_SHIFT)
            val sigmas = ArrayList<Double>(steps.size + 1)
            for (t in steps) {
                var best = 0
                var bestD = Double.POSITIVE_INFINITY
                for (i in ts.indices) {
                    val d = abs(ts[i].toDouble() - t.toDouble())
                    if (d < bestD) { bestD = d; best = i }
                }
                sigmas.add(sig[best].toDouble())
            }
            sigmas.add(0.0)
            return DmdSchedule(steps.toList(), sigmas)
        }

        public fun fastwan13b(): DmdSchedule = of(FASTWAN_1_3B_STEPS)
    }
}
