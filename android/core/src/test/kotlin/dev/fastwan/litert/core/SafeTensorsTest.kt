package dev.fastwan.litert.core

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class SafeTensorsTest {
    @Test
    fun readsTheFixture() {
        val st = Fixture.expected()
        assertEquals(setOf("latent", "prompt_embeds", "frames"), st.names)
        assertEquals(mapOf("geometry" to "32x48x9", "seed" to "7"), st.metadata)

        val embeds = st.tensor("prompt_embeds")
        assertEquals(listOf(8, 16), embeds.shape.toList())
        assertTrue(abs(embeds.data[0] - 0.02227701060473919f) < 1e-7f)
        assertTrue(abs(embeds.data[2] - 1.669760823249817f) < 1e-6f)
        assertEquals(5.684870596975088, embeds.data.sumOf { it.toDouble() }, 1e-4)

        val latent = st.tensor("latent")
        assertEquals(listOf(4, 3, 6, 4), latent.shape.toList())
        assertTrue(abs(latent.data[0] - 0.05838310718536377f) < 1e-7f)
        assertEquals(-47.031720677390695, latent.data.sumOf { it.toDouble() }, 1e-3)

        assertEquals(listOf(9, 48, 32, 3), st.shape("frames").toList())
        assertEquals("U8", st.entry("frames").dtype)
        val frames = st.bytes("frames")
        assertEquals(9 * 48 * 32 * 3, frames.size)
        assertEquals(listOf(0, 20, 34, 0), frames.take(4).map { it.toInt() and 0xFF })
        assertEquals(970235L, frames.sumOf { (it.toInt() and 0xFF).toLong() })
        assertFailsWith<IllegalArgumentException> { st.tensor("frames") }
        assertFailsWith<IllegalArgumentException> { st.tensor("nope") }
    }

    @Test
    fun readsHalfPrecision() {
        // header + one F16 tensor [2]: 1.0 (0x3C00) and -0.5 (0xB800), one BF16 [1]: 2.0 (0x4000)
        val header = """{"h":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[4,6]}}"""
        val hb = header.toByteArray()
        val out = java.io.ByteArrayOutputStream()
        val len = java.nio.ByteBuffer.allocate(8).order(java.nio.ByteOrder.LITTLE_ENDIAN).putLong(hb.size.toLong()).array()
        out.write(len); out.write(hb)
        out.write(byteArrayOf(0x00, 0x3C, 0x00, 0xB8.toByte(), 0x00, 0x40))
        val st = SafeTensors.read(out.toByteArray())
        assertEquals(listOf(1.0f, -0.5f), st.tensor("h").data.toList())
        assertEquals(listOf(2.0f), st.tensor("b").data.toList())
        assertEquals(6.103515625E-5f, SafeTensors.halfToFloat(0x0400.toShort())) // smallest normal, 2^-14
        assertEquals(5.9604645E-8f, SafeTensors.halfToFloat(0x0001.toShort())) // smallest subnormal, 2^-24
        assertEquals(-2.0f, SafeTensors.halfToFloat(0xC000.toShort()))
        assertEquals(65504f, SafeTensors.halfToFloat(0x7BFF.toShort()))
    }
}
