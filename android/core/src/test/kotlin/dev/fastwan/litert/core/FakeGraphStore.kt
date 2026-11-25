package dev.fastwan.litert.core

/**
 * A GraphStore whose graphs check every input name and shape against `docs/graphs.md` for the
 * manifest's config, record the calls, and return contract-shaped outputs.
 *
 * Row graphs return 1.0 for rows whose primary input row is non-zero and NaN for all-zero rows,
 * so a padded row that leaks into a real output, or a padded key/value, poisons the latent;
 * the TAEHV fake returns 1.0 for frames of real latent frames and 0.25 for padded ones.
 */
class FakeGraphStore(manifest: Manifest, resident: Int = 4) : GraphStore(manifest, resident) {
    class Call(val graph: String, val inputs: Map<String, List<Int>>)

    val calls = mutableListOf<Call>()
    val opened = mutableListOf<String>()
    val closed = mutableListOf<String>()

    fun calls(graph: String) = calls.filter { it.graph == graph }
    fun count(graph: String) = calls.count { it.graph == graph }

    override fun open(file: String): Graph {
        opened += file
        return FakeGraph(file)
    }

    private inner class FakeGraph(override val name: String) : Graph {
        private val c = manifest.cfg
        private val d = c.dim
        private val C = manifest.chunk

        override fun close() { closed += name }

        private fun expect(inputs: Map<String, Tensor>, vararg spec: Pair<String, List<Int>>) {
            val want = spec.toMap()
            if (inputs.keys != want.keys) throw IllegalArgumentException("$name: inputs ${want.keys.sorted()}, got ${inputs.keys.sorted()}")
            for ((k, shape) in want) {
                val got = inputs[k]!!.shape.toList()
                if (got != shape) throw IllegalArgumentException("$name: input `$k` is $got, expected $shape")
            }
        }

        private fun rowMarker(t: Tensor): FloatArray {
            val rs = t.rowSize
            return FloatArray(t.rows) { r ->
                var zero = true
                for (i in 0 until rs) if (t.data[r * rs + i] != 0f) { zero = false; break }
                if (zero) Float.NaN else 1f
            }
        }

        private fun rowsOf(marker: FloatArray, vararg tail: Int): Tensor {
            val rs = Tensor.numel(tail)
            val out = FloatArray(marker.size * rs)
            for (r in marker.indices) java.util.Arrays.fill(out, r * rs, (r + 1) * rs, marker[r])
            return Tensor(out, intArrayOf(marker.size) + tail)
        }

        private fun const(v: Float, vararg shape: Int) = Tensor(FloatArray(Tensor.numel(shape)) { v }, shape)

        override fun run(inputs: Map<String, Tensor>): List<Tensor> {
            calls += Call(name, inputs.mapValues { it.value.shape.toList() })
            val g = manifest.graphs
            val L = c.textLen
            val H = c.heads
            val hd = c.headDim
            return when {
                name == g.timeMods -> {
                    expect(inputs, "t" to listOf(1))
                    listOf(const(0.5f, c.layers, 6, d), const(0.5f, 2, d))
                }
                name == g.textEmbed -> {
                    expect(inputs, "embeds" to listOf(L, c.textDim))
                    listOf(const(1f, L, d))
                }
                g.blocks.any { it.kv == name } -> {
                    expect(inputs, "ctx" to listOf(L, d))
                    listOf(const(1f, H, L, hd), const(1f, H, L, hd))
                }
                name == g.patchEmbed -> {
                    expect(inputs, "patches" to listOf(C, c.patchFeaturesIn()))
                    listOf(rowsOf(rowMarker(inputs["patches"]!!), d))
                }
                g.blocks.any { it.qkv == name } -> {
                    expect(inputs, "h" to listOf(C, d), "mods" to listOf(6, d), "cos" to listOf(C, hd / 2), "sin" to listOf(C, hd / 2))
                    val m = rowMarker(inputs["h"]!!)
                    listOf(rowsOf(m, H, hd), rowsOf(m, H, hd), rowsOf(m, H, hd))
                }
                name.startsWith("attn_") -> {
                    val a = manifest.attention.first { it.file == name }
                    expect(inputs, "q" to listOf(a.qChunk, H, hd), "k" to listOf(a.tokens, H, hd), "v" to listOf(a.tokens, H, hd))
                    val m = rowMarker(inputs["q"]!!)
                    // every query attends to every key: a zero (padded) key poisons every row
                    val kBad = rowMarker(inputs["k"]!!).any { it.isNaN() } || rowMarker(inputs["v"]!!).any { it.isNaN() }
                    listOf(rowsOf(if (kBad) FloatArray(m.size) { Float.NaN } else m, H * hd))
                }
                g.blocks.any { it.post == name } -> {
                    expect(inputs, "h" to listOf(C, d), "o" to listOf(C, d), "mods" to listOf(6, d), "k2" to listOf(H, L, hd), "v2" to listOf(H, L, hd))
                    val mh = rowMarker(inputs["h"]!!)
                    val mo = rowMarker(inputs["o"]!!)
                    listOf(rowsOf(FloatArray(C) { mh[it] + mo[it] }, d))
                }
                name == g.head -> {
                    expect(inputs, "h" to listOf(C, d), "mods" to listOf(2, d))
                    listOf(rowsOf(rowMarker(inputs["h"]!!), c.patchFeaturesOut()))
                }
                name.startsWith("tae_") -> {
                    val e = manifest.taeChunks.first { it.file == name }
                    val tae = manifest.tae
                    val spec = mutableListOf("z" to listOf(e.frames, tae.latentChannels, e.latentH, e.latentW))
                    for (i in 0 until 3 * tae.memBlocks) {
                        val s = i / tae.memBlocks
                        spec += "mem_$i" to listOf(1, tae.widths[s], e.latentH shl s, e.latentW shl s)
                    }
                    expect(inputs, *spec.toTypedArray())
                    val up = tae.tUpscale()
                    val ph = tae.spatialScale() * e.latentH
                    val pw = tae.spatialScale() * e.latentW
                    val z = inputs["z"]!!
                    val perFrame = tae.latentChannels * e.latentH * e.latentW
                    val rgb = Tensor.zeros(e.frames * up, 3, ph, pw)
                    for (fr in 0 until e.frames) {
                        var zero = true
                        for (i in 0 until perFrame) if (z.data[fr * perFrame + i] != 0f) { zero = false; break }
                        val v = if (zero) 0.25f else 1f
                        java.util.Arrays.fill(rgb.data, fr * up * 3 * ph * pw, (fr + 1) * up * 3 * ph * pw, v)
                    }
                    val mems = (0 until 3 * tae.memBlocks).map { i ->
                        val s = i / tae.memBlocks
                        const(2f, 1, tae.widths[s], e.latentH shl s, e.latentW shl s)
                    }
                    listOf(rgb) + mems
                }
                else -> throw IllegalArgumentException("fake store has no graph `$name`")
            }
        }
    }
}
