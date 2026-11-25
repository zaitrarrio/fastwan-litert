package dev.fastwan.litert.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The host loop over the tiny manifest with a fake GraphStore: the call sequence, the chunking
 * (18 tokens, C = 16, Cq = 8), padding rules and the TAEHV frame bookkeeping of `docs/graphs.md`.
 */
class PipelineTest {
    private val manifest = Fixture.manifest()
    private val geometry = Geometry(32, 48, 9)
    private val embeds = Fixture.expected().tensor("prompt_embeds")

    @Test
    fun tinyManifestShapes() {
        assertEquals(18, geometry.tokens(manifest.cfg))
        assertEquals(Dims3(3, 6, 4), geometry.latent())
        assertEquals(16, manifest.chunk)
        assertEquals(8, manifest.attentionFor(18).qChunk)
        assertEquals(2, manifest.taeFor(6, 4).frames)
        assertEquals(listOf(8, 16), embeds.shape.toList())
    }

    @Test
    fun forwardChunksPerBlock() {
        val store = FakeGraphStore(manifest)
        val pipe = Pipeline(store)
        val prompt = pipe.preparePrompt(embeds)
        assertEquals(1, store.count("text_embed.tflite"))
        assertEquals(1, store.count("block_00_kv.tflite"))
        assertEquals(1, store.count("block_01_kv.tflite"))
        assertEquals(listOf(2, 8, 12), prompt.kv[0].first.shape.toList())

        val rope = Rope.table(manifest.cfg, geometry.grid(manifest.cfg))
        val x = Rng(7L).normal(4, 3, 6, 4)
        store.calls.clear()
        val v = pipe.forward(x, prompt, 1000f, rope, manifest.attentionFor(18))

        assertEquals(listOf(4, 3, 6, 4), v.shape.toList())
        assertTrue(v.data.all { it == 1f }, "padded rows leaked into the output (NaN) or markers wrong")

        assertEquals(1, store.count("time_mods.tflite"))
        assertEquals(2, store.count("patch_embed.tflite"))
        for (b in listOf("00", "01")) {
            assertEquals(2, store.count("block_${b}_qkv.tflite"), "qkv chunks of block $b")
            assertEquals(2, store.count("block_${b}_post.tflite"), "post chunks of block $b")
        }
        assertEquals(6, store.count("attn_s18_q8_h2.tflite"), "3 attention chunks per block × 2 blocks")
        assertEquals(2, store.count("head.tflite"))

        // Order: time_mods, patch_embed×2, then per block qkv×2, attn×3, post×2, then head×2.
        val seq = store.calls.map { it.graph }
        val expected = listOf("time_mods.tflite") + List(2) { "patch_embed.tflite" } +
            listOf("00", "01").flatMap { b ->
                List(2) { "block_${b}_qkv.tflite" } + List(3) { "attn_s18_q8_h2.tflite" } + List(2) { "block_${b}_post.tflite" }
            } + List(2) { "head.tflite" }
        assertEquals(expected, seq)

        // Every chunk call sees padded row shapes; keys and values are never padded (S exact).
        for (call in store.calls("block_00_qkv.tflite")) {
            assertEquals(listOf(16, 24), call.inputs["h"])
            assertEquals(listOf(16, 6), call.inputs["cos"])
            assertEquals(listOf(6, 24), call.inputs["mods"])
        }
        for (call in store.calls("attn_s18_q8_h2.tflite")) {
            assertEquals(listOf(8, 2, 12), call.inputs["q"])
            assertEquals(listOf(18, 2, 12), call.inputs["k"])
            assertEquals(listOf(18, 2, 12), call.inputs["v"])
        }
        for (call in store.calls("block_01_post.tflite")) {
            assertEquals(listOf(16, 24), call.inputs["o"])
            assertEquals(listOf(2, 8, 12), call.inputs["k2"])
        }
        assertEquals(listOf(2, 24), store.calls("head.tflite")[0].inputs["mods"])
    }

    @Test
    fun runRowsDropsPaddedRows() {
        val store = FakeGraphStore(manifest)
        val pipe = Pipeline(store)
        val rows = Tensor(FloatArray(18 * 16) { 1f }, intArrayOf(18, 16))
        val out = pipe.runRows(store.fixed("patch_embed.tflite"), 16, 18, mapOf("patches" to rows), emptyMap())
        assertEquals(1, out.size)
        assertEquals(listOf(18, 24), out[0].shape.toList())
        assertTrue(out[0].data.all { it == 1f })
        assertEquals(2, store.count("patch_embed.tflite"))
        // the second chunk's padded rows were zero (the fake returns NaN for them, then they are dropped)
        assertEquals(listOf(16, 16), store.calls("patch_embed.tflite")[1].inputs["patches"])
    }

    @Test
    fun generateEndToEnd() {
        val store = FakeGraphStore(manifest)
        val pipe = Pipeline(store)
        val events = mutableListOf<Event>()
        val (video, timings) = pipe.generate(embeds, geometry, 7L) { events += it }

        // three steps, one event each, then the prepared event first
        assertTrue(events[0] is Event.Prepared)
        assertEquals(18, (events[0] as Event.Prepared).tokens)
        val steps = events.filterIsInstance<Event.Step>()
        assertEquals(listOf(0, 1, 2), steps.map { it.index })
        assertEquals(listOf(3, 3, 3), steps.map { it.total })
        assertEquals(3, timings.stepsS.size)

        // per-step graph calls: 3 time_mods, 6 patch_embed, 6 qkv + 6 post per block, 18 attention, 6 head
        assertEquals(3, store.count("time_mods.tflite"))
        assertEquals(6, store.count("patch_embed.tflite"))
        assertEquals(6, store.count("block_00_qkv.tflite"))
        assertEquals(6, store.count("block_01_post.tflite"))
        assertEquals(18, store.count("attn_s18_q8_h2.tflite"))
        assertEquals(6, store.count("head.tflite"))

        // TAEHV: 3 latent frames in chunks of 2 → 2 calls; 12 decoded frames, 3 priming frames trimmed → 9.
        assertEquals(2, store.count("tae_f2_6x4.tflite"))
        val tae = store.calls("tae_f2_6x4.tflite")
        assertEquals(listOf(2, 4, 6, 4), tae[0].inputs["z"])
        assertEquals(listOf(1, 8, 6, 4), tae[0].inputs["mem_0"])
        assertEquals(listOf(1, 6, 12, 8), tae[0].inputs["mem_3"])
        assertEquals(listOf(1, 4, 24, 16), tae[0].inputs["mem_8"])
        val frames = events.filterIsInstance<Event.Frames>()
        assertEquals(listOf(0 to 5, 5 to 4), frames.map { it.first to it.count })
        assertEquals(9, video.frames)
        assertEquals(32, video.width)
        assertEquals(48, video.height)
        assertEquals(9 * 48 * 32 * 4, video.rgba.size)
        // real frames decode to 1.0 → 255; the padded latent frame's 0.25 outputs (64) were dropped
        assertTrue(video.rgba.all { (it.toInt() and 0xFF) == 255 })
        assertEquals(255, frames[0].frame(4)[3].toInt() and 0xFF)

        // the expected fixture frames have the same geometry
        assertEquals(listOf(9, 48, 32, 3), Fixture.expected().shape("frames").toList())
    }

    @Test
    fun denoiseIsDeterministicPerSeedAndDrawsNoiseInOrder() {
        // With the fake graphs every velocity is 1.0, so the latent is a pure function of the
        // noise stream: x = ε0; then per step x = x − σ·1; re-noise with the next normals.
        val store = FakeGraphStore(manifest)
        val x = Pipeline(store).denoise(embeds, geometry, 7L)
        val rng = Rng(7L)
        var ref = rng.normal(4, 3, 6, 4)
        val sched = manifest.schedule
        val ones = Tensor(FloatArray(ref.size) { 1f }, ref.shape)
        for (i in 0 until sched.size) {
            val (s, n) = sched.coeffs(i)
            val noise = if (n != null) rng.normal(*ref.shape) else null
            ref = Host.dmdStep(ref, ones, s.toFloat(), n?.toFloat(), noise)
        }
        assertTrue(x.data.contentEquals(ref.data))
        assertTrue(x.data.contentEquals(Pipeline(FakeGraphStore(manifest)).denoise(embeds, geometry, 7L).data))
    }

    @Test
    fun refusesWrongPromptAndGeometry() {
        val pipe = Pipeline(FakeGraphStore(manifest))
        val e = assertFailsWith<IllegalArgumentException> { pipe.preparePrompt(Tensor.zeros(8, 15)) }
        assertTrue("[8, 15]" in e.message!! && "[8, 16]" in e.message!!, e.message)
        val g = assertFailsWith<IllegalArgumentException> { pipe.denoise(embeds, Geometry(64, 48, 9), 1L) }
        assertTrue("36 tokens" in g.message!! && "[18]" in g.message!!, g.message)
        val bad = assertFailsWith<IllegalArgumentException> { pipe.denoise(embeds, Geometry(32, 48, 8), 1L) }
        assertTrue("4n+1" in bad.message!!, bad.message)
    }

    @Test
    fun blockGraphsAreLruResident() {
        val store = FakeGraphStore(manifest, resident = 2)
        store.block("block_00_kv.tflite")
        store.block("block_00_qkv.tflite")
        store.block("block_00_kv.tflite") // touch: now most recent
        store.block("block_00_post.tflite") // evicts qkv
        assertEquals(listOf("block_00_qkv.tflite"), store.closed)
        assertEquals(listOf("block_00_kv.tflite", "block_00_post.tflite"), store.residentBlocks())
        store.fixed("head.tflite")
        store.close()
        assertEquals(setOf("block_00_qkv.tflite", "block_00_kv.tflite", "block_00_post.tflite", "head.tflite"), store.closed.toSet())
    }
}
