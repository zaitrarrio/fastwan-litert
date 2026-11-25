package dev.fastwan.litert.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ManifestTest {
    @Test
    fun parsesTheTinyManifest() {
        val m = Fixture.manifest()
        assertEquals("tiny", m.model)
        assertEquals(WanConfig.tiny(), m.cfg)
        assertEquals(TaeConfig.tiny(), m.tae)
        assertEquals(DmdSchedule.fastwan13b(), m.schedule)
        assertEquals("f16", m.weights)
        assertEquals(16, m.chunk)
        assertEquals(2, m.graphs.blocks.size)
        assertEquals("block_01_post.tflite", m.graphs.blocks[1].post)
        assertEquals(61912L, m.graphs.blocks[1].bytes)
        assertEquals(listOf(Geometry(32, 48, 9)), m.sizes)
        assertEquals(12, m.files.size)
        assertEquals(2692L, m.files["attn_s18_q8_h2.tflite"]!!.bytes)
        assertEquals(12, m.graphFiles().toSet().size)
        assertEquals(m.files.keys, m.graphFiles().toSet())
    }

    @Test
    fun refusesOtherFormats() {
        val e = assertFailsWith<IllegalArgumentException> {
            Manifest.parse(Fixture.text("manifest.json").replace("fastwan-litert/1", "fastwan-litert/2"))
        }
        assertTrue("fastwan-litert/2" in e.message!! && "fastwan-litert/1" in e.message!!, e.message)
        val missing = assertFailsWith<IllegalArgumentException> { Manifest.parse("""{"model": "x"}""") }
        assertTrue("missing" in missing.message!!, missing.message)
    }

    @Test
    fun lookupsNameTheMismatch() {
        val m = Fixture.manifest()
        assertEquals("attn_s18_q8_h2.tflite", m.attentionFor(18).file)
        val a = assertFailsWith<IllegalArgumentException> { m.attentionFor(32760) }
        assertTrue("32760" in a.message!! && "[18]" in a.message!!, a.message)
        assertEquals("tae_f2_6x4.tflite", m.taeFor(6, 4).file)
        val t = assertFailsWith<IllegalArgumentException> { m.taeFor(60, 104) }
        assertTrue("60x104" in t.message!! && "6x4" in t.message!!, t.message)
    }

    @Test
    fun planOfTheTinyClip() {
        val p = Fixture.manifest().plan(Geometry(32, 48, 9), 64L shl 20)
        assertEquals(18, p.tokens)
        assertEquals(16, p.chunk)
        assertEquals(8, p.qChunk)
        assertTrue(p.fits, p.toString())
        assertTrue(p.notes.any { "18 rows run as 2 chunks of 16 (14 padded rows)" in it }, p.notes.toString())
    }

    @Test
    fun pathNeedsADirectory() {
        val m = Fixture.manifest()
        assertFailsWith<IllegalStateException> { m.path("head.tflite") }
        val withDir = Fixture.manifest(java.io.File("/graphs"))
        assertEquals(java.io.File("/graphs/head.tflite"), withDir.path("head.tflite"))
    }
}
