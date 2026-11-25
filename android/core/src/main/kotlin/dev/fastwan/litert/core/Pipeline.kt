package dev.fastwan.litert.core

/** What the pipeline reports while it runs (`docs/graphs.md` "Events"). */
public sealed class Event {
    /** Prompt K/V, RoPE table and the initial latent are ready. */
    public data class Prepared(val ms: Double, val tokens: Int) : Event()

    /** One DMD step finished. */
    public data class Step(val index: Int, val total: Int, val ms: Double) : Event()

    /**
     * Frames `[first, first + count)` decoded by one TAEHV chunk, as `[count, height, width, 4]`
     * RGBA bytes (alpha 255), ready for an `ARGB_8888` `Bitmap`.
     */
    public class Frames(
        public val first: Int,
        public val count: Int,
        public val width: Int,
        public val height: Int,
        public val rgba: ByteArray,
    ) : Event() {
        /** RGBA bytes of frame `i` (relative to [first]). */
        public fun frame(i: Int): ByteArray {
            val n = width * height * 4
            return rgba.copyOfRange(i * n, (i + 1) * n)
        }

        override fun toString(): String = "Frames(first=$first, count=$count, ${width}x$height)"
    }
}

public class Timings {
    public var prepareS: Double = 0.0
    public val stepsS: MutableList<Double> = mutableListOf()
    public var decodeS: Double = 0.0

    override fun toString(): String =
        "prepare ${fmt(prepareS)} s; steps ${stepsS.joinToString(" / ") { fmt(it) }} s; decode ${fmt(decodeS)} s"

    private fun fmt(v: Double) = String.format(java.util.Locale.ROOT, "%.3f", v)
}

/** A decoded clip: `[frames, height, width, 4]` RGBA bytes. */
public class Video(public val frames: Int, public val width: Int, public val height: Int, public val rgba: ByteArray) {
    public fun frame(i: Int): ByteArray {
        val n = width * height * 4
        return rgba.copyOfRange(i * n, (i + 1) * n)
    }
}

/** Per-prompt cross-attention K/V of every block (`k2`, `v2` of `block_N_kv`). */
public class Prompt(public val kv: List<Pair<Tensor, Tensor>>)

/**
 * Prompt embeddings → DMD denoise → TAEHV decode → RGB frames, driving the graphs exactly as
 * `docs/graphs.md`'s host loop and `pipeline.py` do.
 */
public class Pipeline(public val store: GraphStore, public val manifest: Manifest = store.manifest) {
    public val cfg: WanConfig get() = manifest.cfg

    // ---- prompt -----------------------------------------------------------------------------

    public fun preparePrompt(embeds: Tensor): Prompt {
        val c = cfg
        if (!(embeds.rank == 2 && embeds.shape[0] == c.textLen && embeds.shape[1] == c.textDim)) {
            throw IllegalArgumentException("prompt embeds are ${embeds.shapeString()}, expected [${c.textLen}, ${c.textDim}]")
        }
        val ctx = store.fixed(manifest.graphs.textEmbed).run(mapOf("embeds" to embeds)).single()
        val kv = manifest.graphs.blocks.map { b ->
            val out = store.block(b.kv).run(mapOf("ctx" to ctx))
            require(out.size == 2) { "${b.kv} returned ${out.size} outputs, expected k2, v2" }
            out[0] to out[1]
        }
        return Prompt(kv)
    }

    // ---- one DiT forward ----------------------------------------------------------------------

    /**
     * Run a per-row graph over row chunks of `perRow` (zero-padded to `chunk`), with `fixed`
     * bound unchanged to every call; returns the outputs with the padded rows dropped,
     * concatenated along rows.
     */
    internal fun runRows(graph: Graph, chunk: Int, nReal: Int, perRow: Map<String, Tensor>, fixed: Map<String, Tensor>): List<Tensor> {
        val padded = perRow.mapValues { Host.padRows(it.value, chunk).tensor }
        val n = padded.values.first().rows
        for ((k, v) in padded) require(v.rows == n) { "`$k` has ${v.rows} rows, expected $n" }
        val outs = ArrayList<List<Tensor>>(n / chunk)
        var r = 0
        while (r < n) {
            val inputs = HashMap<String, Tensor>(padded.size + fixed.size)
            for ((k, v) in padded) inputs[k] = v.sliceRows(r, r + chunk)
            inputs.putAll(fixed)
            outs += graph.run(inputs)
            r += chunk
        }
        val nOut = outs[0].size
        return List(nOut) { i -> Tensor.concatRows(outs.map { it[i] }).sliceRows(0, nReal) }
    }

    public fun forward(x: Tensor, prompt: Prompt, t: Float, rope: RopeTable, attn: AttentionEntry): Tensor {
        val c = cfg
        val chunk = manifest.chunk
        require(x.rank == 4) { "latent must be [C, T, H, W], got ${x.shapeString()}" }
        val tt = x.shape[1]
        val hh = x.shape[2]
        val ww = x.shape[3]
        val s = tt * (hh / c.patch) * (ww / c.patch)
        require(rope.tokens == s) { "rope table has ${rope.tokens} tokens, the latent $s" }
        require(prompt.kv.size == manifest.graphs.blocks.size) { "prompt has K/V for ${prompt.kv.size} blocks, the model ${manifest.graphs.blocks.size}" }

        val mods = store.fixed(manifest.graphs.timeMods).run(mapOf("t" to Tensor.scalar(t)))
        require(mods.size == 2) { "${manifest.graphs.timeMods} returned ${mods.size} outputs, expected block_mods, head_mods" }
        val blockMods = mods[0] // [layers, 6, d]
        val headMods = mods[1] // [2, d]
        require(blockMods.rank == 3 && blockMods.shape[0] == manifest.graphs.blocks.size) {
            "block_mods is ${blockMods.shapeString()}, expected [${manifest.graphs.blocks.size}, 6, ${c.dim}]"
        }

        val patches = Host.patchify(x, c.patch)
        var h = runRows(store.fixed(manifest.graphs.patchEmbed), chunk, s, mapOf("patches" to patches), emptyMap()).single()

        val attnGraph = store.fixed(attn.file)
        val cq = attn.qChunk
        for ((l, b) in manifest.graphs.blocks.withIndex()) {
            val m = blockMods.sliceRows(l, l + 1).reshape(blockMods.shape[1], blockMods.shape[2]) // [6, d]
            val qkv = runRows(store.block(b.qkv), chunk, s, mapOf("h" to h, "cos" to rope.cos, "sin" to rope.sin), mapOf("mods" to m))
            require(qkv.size == 3) { "${b.qkv} returned ${qkv.size} outputs, expected q, k, v" }
            val (q, k, v) = qkv
            val o = runRows(attnGraph, cq, s, mapOf("q" to q), mapOf("k" to k, "v" to v)).single()
            val (k2, v2) = prompt.kv[l]
            h = runRows(store.block(b.post), chunk, s, mapOf("h" to h, "o" to o), mapOf("mods" to m, "k2" to k2, "v2" to v2)).single()
        }
        val rows = runRows(store.fixed(manifest.graphs.head), chunk, s, mapOf("h" to h), mapOf("mods" to headMods)).single()
        return Host.unpatchify(rows, c.outChannels, tt, hh, ww, c.patch)
    }

    // ---- denoise ------------------------------------------------------------------------------

    public fun denoise(embeds: Tensor, g: Geometry, seed: Long, on: ((Event) -> Unit)? = null, timings: Timings? = null): Tensor {
        val c = cfg
        g.validate(c)
        val tm = timings ?: Timings()
        val t0 = System.nanoTime()
        val attn = manifest.attentionFor(g.tokens(c))
        val prompt = preparePrompt(embeds)
        val rope = Rope.table(c, g.grid(c), 0)
        val lat = g.latent()
        val rng = Rng(seed)
        var x = rng.normal(c.inChannels, lat.t, lat.h, lat.w)
        tm.prepareS = secs(t0)
        on?.invoke(Event.Prepared(tm.prepareS * 1000.0, g.tokens(c)))
        val sched = manifest.schedule
        for ((i, t) in sched.timesteps.withIndex()) {
            val ts = System.nanoTime()
            val v = forward(x, prompt, t.toFloat(), rope, attn)
            val (sigma, next) = sched.coeffs(i)
            val noise = if (next != null) rng.normal(*x.shape) else null
            x = Host.dmdStep(x, v, sigma.toFloat(), next?.toFloat(), noise)
            tm.stepsS += secs(ts)
            on?.invoke(Event.Step(i, sched.size, tm.stepsS.last() * 1000.0))
        }
        return x
    }

    // ---- decode -------------------------------------------------------------------------------

    public fun decode(latent: Tensor, on: ((Event) -> Unit)? = null, timings: Timings? = null): Video {
        val tcfg = manifest.tae
        val tm = timings ?: Timings()
        val t0 = System.nanoTime()
        require(latent.rank == 4) { "latent must be [C, T, H, W], got ${latent.shapeString()}" }
        val c = latent.shape[0]
        val t = latent.shape[1]
        val h = latent.shape[2]
        val w = latent.shape[3]
        val entry = manifest.taeFor(h, w)
        val graph = store.fixed(entry.file)
        val f = entry.frames
        val nmem = 3 * tcfg.memBlocks
        var mems: List<Tensor> = List(nmem) { i ->
            val s = i / tcfg.memBlocks
            Tensor.zeros(1, tcfg.widths[s], h shl s, w shl s)
        }
        val up = tcfg.tUpscale()
        var toTrim = tcfg.framesToTrim()
        val parts = ArrayList<ByteArray>()
        var total = 0
        var width = 0
        var height = 0
        val plane = h * w
        var f0 = 0
        while (f0 < t) {
            val n = minOf(f, t - f0)
            // z[frame, channel, y, x] = latent[channel, f0 + frame, y, x]; padded frames stay zero.
            val z = Tensor.zeros(f, c, h, w)
            for (fr in 0 until n) for (ch in 0 until c) {
                System.arraycopy(latent.data, (ch * t + f0 + fr) * plane, z.data, (fr * c + ch) * plane, plane)
            }
            val inputs = HashMap<String, Tensor>(nmem + 1)
            inputs["z"] = z
            for (i in 0 until nmem) inputs["mem_$i"] = mems[i]
            val out = graph.run(inputs)
            require(out.size == nmem + 1) { "${entry.file} returned ${out.size} outputs, expected rgb + $nmem memories" }
            val rgbAll = out[0]
            require(rgbAll.rank == 4 && rgbAll.shape[0] == f * up && rgbAll.shape[1] == 3) {
                "${entry.file} rgb is ${rgbAll.shapeString()}, expected [${f * up}, 3, H, W]"
            }
            mems = out.subList(1, out.size)
            val real = n * up
            val skip = minOf(toTrim, real)
            toTrim -= skip
            if (real - skip > 0) {
                val rgb = rgbAll.sliceRows(skip, real)
                val rgba = Host.toRgba8(rgb)
                val count = rgb.shape[0]
                width = rgb.shape[3]
                height = rgb.shape[2]
                parts += rgba
                on?.invoke(Event.Frames(total, count, width, height, rgba))
                total += count
            }
            f0 += f
        }
        tm.decodeS = secs(t0)
        val all = ByteArray(parts.sumOf { it.size })
        var o = 0
        for (p in parts) { System.arraycopy(p, 0, all, o, p.size); o += p.size }
        return Video(total, width, height, all)
    }

    public fun generate(embeds: Tensor, g: Geometry, seed: Long, on: ((Event) -> Unit)? = null): Pair<Video, Timings> {
        val tm = Timings()
        val latent = denoise(embeds, g, seed, on, tm)
        val video = decode(latent, on, tm)
        return video to tm
    }

    private fun secs(since: Long): Double = (System.nanoTime() - since) / 1e9
}
