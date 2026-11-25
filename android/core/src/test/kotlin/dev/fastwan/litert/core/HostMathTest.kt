package dev.fastwan.litert.core

import kotlin.math.abs
import kotlin.math.cos
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The host math against the values fastwan-rs's own tests pin down: every case of
 * `python/tests/test_host.py`, plus values printed by the Python reference for this port.
 */
class HostMathTest {
    @Test
    fun fastwanSigmas() {
        val s = DmdSchedule.fastwan13b()
        assertEquals(1.0, s.sigmas[0])
        assertTrue(abs(s.sigmas[1] - 0.757) < 1e-3, "${s.sigmas[1]}")
        assertTrue(abs(s.sigmas[2] - 0.522) < 1e-3, "${s.sigmas[2]}")
        assertEquals(0.0, s.sigmas[3])
        assertNull(s.coeffs(2).next)
        assertEquals(0.757, s.coeffs(0).next!!, 1e-3)
        // The f32 training table equals numpy's bit for bit: the values the exporter wrote.
        assertEquals(0.7567567229270935, s.sigmas[1])
        assertEquals(0.52173912525177, s.sigmas[2])
    }

    @Test
    fun geometry480p() {
        val cfg = WanConfig.fastwan13b()
        assertEquals(Dims3(21, 60, 104), Geometry.FASTWAN_480P.latent())
        assertEquals(Dims3(21, 30, 52), Geometry.FASTWAN_480P.grid(cfg))
        assertEquals(32760, Geometry.FASTWAN_480P.tokens(cfg))
        for (bad in listOf(Geometry(830, 480, 81), Geometry(832, 480, 80))) {
            assertFailsWith<IllegalArgumentException>("$bad") { bad.validate(cfg) }
        }
        Geometry(832, 480, 1).validate(cfg)
        assertEquals(Geometry(832, 480, 81), Geometry.parse("832x480x81"))
        assertEquals("480x272x33", Geometry.MOBILE_480x272.toString())
        assertEquals(Dims3(9, 34, 60), Geometry.MOBILE_480x272.latent())
        assertEquals(9 * 17 * 30, Geometry.MOBILE_480x272.tokens(cfg))
    }

    @Test
    fun paramsIs14B() {
        val p = WanConfig.fastwan13b().params()
        assertTrue(p in 1_400_000_000L..1_440_000_000L, "$p")
    }

    @Test
    fun ropeSplitAndOrigin() {
        val cfg = WanConfig.fastwan13b()
        assertEquals(Triple(44, 42, 42), cfg.ropeSplit())
        assertEquals(Triple(4, 4, 4), WanConfig.tiny().ropeSplit())
        val table = Rope.table(cfg, Dims3(2, 3, 4), 0)
        assertEquals(listOf(24, 64), table.cos.shape.toList())
        assertEquals(listOf(24, 64), table.sin.shape.toList())
        for (j in 0 until 64) {
            assertEquals(1.0f, table.cos.data[j])
            assertEquals(0.0f, table.sin.data[j])
        }
        val row = 12 // token (t=1, 0, 0): the first temporal pair rotates by 1 rad
        assertTrue(abs(table.cos.data[row * 64 + 0] - cos(1.0)) < 1e-7)
        assertEquals(1.0f, table.cos.data[row * 64 + 22]) // the spatial pairs do not
        assertEquals(1.0f, table.cos.data[row * 64 + 43])
        // token (0, 1, 0): the first height pair rotates by 1 rad, the temporal ones not at all
        val rowH = 4
        assertEquals(1.0f, table.cos.data[rowH * 64 + 0])
        assertTrue(abs(table.cos.data[rowH * 64 + 22] - cos(1.0)) < 1e-7)
        // start_frame shifts the temporal angle
        val shifted = Rope.table(cfg, Dims3(1, 1, 1), 1)
        assertTrue(abs(shifted.cos.data[0] - cos(1.0)) < 1e-7)
    }

    @Test
    fun taehvFrames() {
        val t = TaeConfig.taew21()
        assertEquals(4, t.tUpscale())
        assertEquals(81, 21 * t.tUpscale() - t.framesToTrim())
        assertEquals(8, t.spatialScale())
        assertEquals(3, t.outChannels())
        assertEquals(12, TaeConfig.taew22().outChannels())
    }

    @Test
    fun rngIsSplitMix64BoxMuller() {
        // fastwan-rs seeds the state as `seed ^ GOLDEN`, so the canonical SplitMix64 stream
        // (state 0: first output 0xE220A8397B1DCDAF) appears at seed GOLDEN.
        val r = Rng(0x9E3779B97F4A7C15uL)
        assertEquals(0xE220A8397B1DCDAFuL, r.nextU64())
        assertEquals(0x6E789E6AA1B965F4uL, r.nextU64())
        assertEquals(0xEC779C3693F88501uL, Rng(7L).nextU64())

        val v = Rng(7L).normal(200_000)
        var sum = 0.0
        var sq = 0.0
        for (x in v) { sum += x; sq += x.toDouble() * x }
        val mean = sum / v.size
        val variance = sq / v.size - mean * mean
        assertTrue(abs(mean) < 0.01, "mean $mean")
        assertTrue(abs(variance - 1.0) < 0.02, "var $variance")

        // The scalar definition, pair by pair.
        val a = Rng(3L)
        val scalar = FloatArray(8)
        for (p in 0 until 4) {
            val u1 = a.unit()
            val u2 = a.unit()
            val rr = Math.sqrt(-2.0 * Math.log(u1))
            val th = 2.0 * Math.PI * u2
            scalar[2 * p] = (rr * Math.cos(th)).toFloat()
            scalar[2 * p + 1] = (rr * Math.sin(th)).toFloat()
        }
        val got = Rng(3L).normal(8)
        for (i in 0 until 8) assertTrue(abs(got[i] - scalar[i]) < 1e-6f, "$i: ${got[i]} vs ${scalar[i]}")

        // Values printed by the Python reference (`Rng(3).normal(8)`, `Rng(7).normal(5)`, `Rng(0).normal(3)`).
        assertClose(
            floatArrayOf(0.7532498836517334f, -0.13877740502357483f, -1.1963688135147095f, 0.4369733929634094f,
                0.5278467535972595f, -0.5124930739402771f, -0.25189709663391113f, 1.1102426052093506f),
            got,
        )
        assertClose(
            floatArrayOf(0.3982580900192261f, -0.011231214739382267f, 0.765938401222229f, 1.036062240600586f, -2.5995090007781982f),
            Rng(7L).normal(5),
        )
        assertClose(floatArrayOf(1.2786335945129395f, 0.21434058248996735f, 0.19082410633563995f), Rng(0L).normal(3))
        // An odd count still consumes a whole pair, so the stream stays aligned.
        val odd = Rng(7L)
        odd.normal(5)
        assertEquals(Rng(7L).normal(8).copyOfRange(6, 8).toList(), odd.normal(2).toList())
        // `Rng(7).uniform(3, 0.5)`
        assertClose(floatArrayOf(0.42370009422302246f, 0.49551287293434143f, -0.06396959722042084f), Rng(7L).uniform(3, 0.5f))
    }

    @Test
    fun patchifyRoundTrip() {
        // x[c, t, h, w] = running index, so every element is identifiable.
        val x = Tensor(FloatArray(4 * 3 * 8 * 12) { it.toFloat() }, intArrayOf(4, 3, 8, 12))
        val rows = Host.patchify(x, 2)
        assertEquals(listOf(3 * 4 * 6, 16), rows.shape.toList())
        // feature index c·p² + kh·p + kw
        assertEquals(x.data[((1 * 3 + 0) * 8 + 1) * 12 + 0], rows.data[0 * 16 + 1 * 4 + 1 * 2 + 0])
        // row 0 and row 7 as the numpy reference prints them
        assertEquals(listOf(0f, 1f, 12f, 13f, 288f, 289f, 300f, 301f, 576f, 577f, 588f, 589f, 864f, 865f, 876f, 877f), rows.data.copyOfRange(0, 16).toList())
        assertEquals(listOf(26f, 27f, 38f, 39f, 314f, 315f, 326f, 327f, 602f, 603f, 614f, 615f, 890f, 891f, 902f, 903f), rows.data.copyOfRange(7 * 16, 8 * 16).toList())
        // Diffusers unpatchify reads (kh·p + kw)·c + c': permute the features and go back.
        val perm = FloatArray(72 * 16)
        for (r in 0 until 72) for (c in 0 until 4) for (kh in 0 until 2) for (kw in 0 until 2) {
            perm[r * 16 + (kh * 2 + kw) * 4 + c] = rows.data[r * 16 + c * 4 + kh * 2 + kw]
        }
        val back = Host.unpatchify(Tensor(perm, intArrayOf(72, 16)), 4, 3, 8, 12, 2)
        assertEquals(x.shape.toList(), back.shape.toList())
        assertTrue(back.data.contentEquals(x.data))
        // unpatchify of a running index, as the numpy reference prints it
        val u = Host.unpatchify(Tensor(FloatArray(72 * 16) { it.toFloat() }, intArrayOf(72, 16)), 4, 3, 8, 12, 2)
        fun at(c: Int, t: Int, h: Int, w: Int) = u.data[((c * 3 + t) * 8 + h) * 12 + w]
        assertEquals(1.0f, at(1, 0, 0, 0))
        assertEquals(8.0f, at(0, 0, 1, 0))
        assertEquals(4.0f, at(0, 0, 0, 1))
        assertEquals(384.0f, at(0, 1, 0, 0))
        assertEquals(526.0f, at(2, 1, 3, 5))
    }

    @Test
    fun dmdStepAndPadding() {
        val x = Tensor(floatArrayOf(1f, 2f, 3f, 4f), intArrayOf(2, 2))
        val v = Tensor(floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f), intArrayOf(2, 2))
        val noise = Tensor(floatArrayOf(1f, 1f, 1f, 1f), intArrayOf(2, 2))
        val last = Host.dmdStep(x, v, 0.5f, null, null)
        assertEquals(listOf(0.75f, 1.75f, 2.75f, 3.75f), last.data.toList())
        val mid = Host.dmdStep(x, v, 1.0f, 0.5f, noise)
        // x0 = x − v = [0.5, 1.5, 2.5, 3.5]; 0.5·x0 + 0.5·1
        assertEquals(listOf(0.75f, 1.25f, 1.75f, 2.25f), mid.data.toList())

        val p = Host.padRows(Tensor(FloatArray(18 * 3) { 1f }, intArrayOf(18, 3)), 16)
        assertEquals(18, p.realRows)
        assertEquals(listOf(32, 3), p.tensor.shape.toList())
        assertEquals(18 * 3, p.tensor.data.count { it == 1f })
        val exact = Host.padRows(Tensor(FloatArray(32 * 3), intArrayOf(32, 3)), 16)
        assertEquals(32, exact.tensor.rows)
        assertEquals(32, exact.realRows)
    }

    @Test
    fun rgb8Rounding() {
        // `(clamp(v)·255 + 0.5) as u8`, as the numpy reference prints it for these inputs.
        val f = Tensor(floatArrayOf(0.0f, 0.5f, 1.0f, 1.2f, -0.3f, 0.998f, 0.001f, 0.002f), intArrayOf(1, 1, 1, 8))
        val rgb = Host.toRgb8(Tensor(FloatArray(3 * 8) { f.data[it % 8] }, intArrayOf(1, 3, 1, 8)))
        val expected = listOf(0, 128, 255, 255, 0, 254, 0, 1)
        for (x in 0 until 8) for (ch in 0 until 3) assertEquals(expected[x], rgb[x * 3 + ch].toInt() and 0xFF, "x=$x ch=$ch")
        val rgba = Host.toRgba8(Tensor(FloatArray(3 * 8) { f.data[it % 8] }, intArrayOf(1, 3, 1, 8)))
        for (x in 0 until 8) {
            for (ch in 0 until 3) assertEquals(expected[x], rgba[x * 4 + ch].toInt() and 0xFF)
            assertEquals(255, rgba[x * 4 + 3].toInt() and 0xFF)
        }
        assertTrue(Host.rgbToRgba(rgb).contentEquals(rgba))
    }

    @Test
    fun tensorBasics() {
        val t = Tensor(FloatArray(12) { it.toFloat() }, intArrayOf(3, 2, 2))
        assertEquals(4, t.rowSize)
        assertEquals(listOf(4f, 5f, 6f, 7f), t.sliceRows(1, 2).data.toList())
        assertEquals(listOf(1, 2, 2), t.sliceRows(1, 2).shape.toList())
        val c = Tensor.concatRows(listOf(t.sliceRows(0, 1), t.sliceRows(1, 3)))
        assertTrue(c.data.contentEquals(t.data))
        assertFailsWith<IllegalArgumentException> { Tensor(FloatArray(3), intArrayOf(2, 2)) }
    }

    private fun assertClose(expected: FloatArray, got: FloatArray, tol: Float = 1e-6f) {
        assertEquals(expected.size, got.size)
        for (i in expected.indices) assertTrue(abs(expected[i] - got[i]) <= tol, "[$i] ${got[i]} vs ${expected[i]}")
    }
}
