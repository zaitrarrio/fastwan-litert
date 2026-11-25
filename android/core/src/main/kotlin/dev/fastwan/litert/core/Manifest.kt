package dev.fastwan.litert.core

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.decodeFromJsonElement
import kotlinx.serialization.json.jsonObject
import java.io.File

public const val MANIFEST_FORMAT: String = "fastwan-litert/1"
public const val MANIFEST_FILE: String = "manifest.json"

@Serializable
public data class BlockFiles(
    val qkv: String,
    val post: String,
    val kv: String,
    val bytes: Long = 0,
)

@Serializable
public data class GraphFiles(
    @SerialName("time_mods") val timeMods: String,
    @SerialName("text_embed") val textEmbed: String,
    @SerialName("patch_embed") val patchEmbed: String,
    val head: String,
    val blocks: List<BlockFiles>,
)

@Serializable
public data class AttentionEntry(
    val tokens: Int,
    @SerialName("q_chunk") val qChunk: Int,
    val heads: Int,
    val file: String,
)

@Serializable
public data class TaeChunkEntry(
    val frames: Int,
    @SerialName("latent_h") val latentH: Int,
    @SerialName("latent_w") val latentW: Int,
    val file: String,
)

@Serializable
public data class FileEntry(val bytes: Long, val sha256: String = "")

/**
 * `manifest.json` (`docs/graphs.md`): the contract between the exporter and this runtime.
 * Parse with [parse] or [load]; both refuse any other `format` with the mismatch in the message.
 */
@Serializable
public data class Manifest(
    val format: String,
    val model: String = "",
    val config: WanConfig,
    val tae: TaeConfig,
    val schedule: DmdSchedule,
    val weights: String,
    val chunk: Int,
    val graphs: GraphFiles,
    val attention: List<AttentionEntry>,
    @SerialName("tae_chunks") val taeChunks: List<TaeChunkEntry>,
    val sizes: List<Geometry> = emptyList(),
    val files: Map<String, FileEntry> = emptyMap(),
    /** The directory the graph files live in, when the manifest was read from one. */
    @Transient val directory: File? = null,
) {
    init {
        if (format != MANIFEST_FORMAT) {
            throw IllegalArgumentException("manifest format \"$format\", expected \"$MANIFEST_FORMAT\"")
        }
        require(chunk > 0) { "manifest chunk must be positive, got $chunk" }
        require(config.layers == graphs.blocks.size) {
            "manifest config.layers is ${config.layers} but graphs.blocks lists ${graphs.blocks.size} blocks"
        }
    }

    val cfg: WanConfig get() = config

    public fun attentionFor(tokens: Int): AttentionEntry =
        attention.firstOrNull { it.tokens == tokens } ?: throw IllegalArgumentException(
            "this export has no attention graph for $tokens tokens (it has ${attention.map { it.tokens }.distinct().sorted()}); re-export with --size",
        )

    public fun taeFor(latentH: Int, latentW: Int): TaeChunkEntry =
        taeChunks.firstOrNull { it.latentH == latentH && it.latentW == latentW } ?: throw IllegalArgumentException(
            "this export has no TAEHV graph for latent ${latentH}x$latentW (it has ${
                taeChunks.map { "${it.latentH}x${it.latentW}" }.distinct().sorted()
            }); re-export with --size",
        )

    /** Every graph file this manifest names, in load order. */
    public fun graphFiles(): List<String> = buildList {
        add(graphs.timeMods); add(graphs.textEmbed); add(graphs.patchEmbed); add(graphs.head)
        for (b in graphs.blocks) { add(b.kv); add(b.qkv); add(b.post) }
        for (a in attention) add(a.file)
        for (t in taeChunks) add(t.file)
    }

    /** Absolute path of a graph file; needs a [directory]. */
    public fun path(file: String): File {
        val dir = directory ?: throw IllegalStateException("manifest was not loaded from a directory; cannot resolve `$file`")
        return File(dir, file)
    }

    /** The memory plan of a clip on this export (`plan.py`). */
    public fun plan(g: Geometry, budgetBytes: Long, residentBlocks: Int = 2): Plan {
        g.validate(cfg)
        val attn = attentionFor(g.tokens(cfg))
        val lat = g.latent()
        val tae = taeFor(lat.h, lat.w)
        return Plan.compute(cfg, this.tae, g, weights, chunk, attn.qChunk, tae.frames, budgetBytes, residentBlocks)
    }

    public companion object {
        private val json = Json { ignoreUnknownKeys = true; isLenient = true }

        /** Parse manifest text; `directory` is where its graph files live (optional). */
        public fun parse(text: String, directory: File? = null): Manifest {
            val root = try {
                json.parseToJsonElement(text).jsonObject
            } catch (e: Exception) {
                throw IllegalArgumentException("manifest is not a JSON object: ${e.message}", e)
            }
            val fmt = (root["format"] as? JsonPrimitive)?.contentOrNull
            if (fmt != MANIFEST_FORMAT) {
                throw IllegalArgumentException("manifest format ${fmt?.let { "\"$it\"" } ?: "missing"}, expected \"$MANIFEST_FORMAT\"")
            }
            val m = try {
                json.decodeFromJsonElement<Manifest>(root)
            } catch (e: IllegalArgumentException) {
                throw e
            } catch (e: Exception) {
                throw IllegalArgumentException("manifest does not follow $MANIFEST_FORMAT: ${e.message}", e)
            }
            return if (directory != null) m.copy(directory = directory) else m
        }

        /** Read `<dir>/manifest.json`. */
        public fun load(directory: File): Manifest {
            val f = File(directory, MANIFEST_FILE)
            if (!f.isFile) throw IllegalArgumentException("no $MANIFEST_FILE in $directory")
            return parse(f.readText(), directory)
        }
    }
}
