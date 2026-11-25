package dev.fastwan.litert.app

import android.app.ActivityManager
import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.litert.Accelerator
import dev.fastwan.litert.core.Event
import dev.fastwan.litert.core.Geometry
import dev.fastwan.litert.core.Manifest
import dev.fastwan.litert.core.Pipeline
import dev.fastwan.litert.core.SafeTensors
import dev.fastwan.litert.core.Tensor
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File
import java.nio.ByteBuffer
import java.util.Locale

/** Everything the screen shows. Immutable; the ViewModel replaces it. */
data class UiState(
    val graphsDir: File,
    val galleryDir: File,
    val manifest: Manifest? = null,
    val manifestError: String? = null,
    val gallery: List<File> = emptyList(),
    val prompt: File? = null,
    val sizes: List<Geometry> = emptyList(),
    val size: Geometry? = null,
    val seed: String = "7",
    val useGpu: Boolean = true,
    val budgetBytes: Long = 0,
    val plan: String? = null,
    val planFits: Boolean = false,
    val running: Boolean = false,
    val status: String = "",
    val stepMs: List<Double> = emptyList(),
    val prepareMs: Double? = null,
    val decodeMs: Double? = null,
    val frames: List<Bitmap> = emptyList(),
    val frameWidth: Int = 0,
    val frameHeight: Int = 0,
    val log: List<String> = emptyList(),
    val error: String? = null,
) {
    val canGenerate: Boolean get() = manifest != null && prompt != null && size != null && !running && seed.toLongOrNull() != null
}

/**
 * Runs `Pipeline.generate` on a background dispatcher and streams its events into [state].
 *
 * Files: `<external files>/graphs/` holds `manifest.json` and the `.tflite` graphs,
 * `<external files>/gallery/` holds `*.safetensors` prompt embeddings (`prompt_embeds [512, 4096]`).
 */
class GenerateViewModel(app: Application) : AndroidViewModel(app) {
    private val root: File = app.getExternalFilesDir(null) ?: app.filesDir
    private val _state = MutableStateFlow(UiState(File(root, "graphs"), File(root, "gallery")))
    val state: StateFlow<UiState> = _state

    private var job: Job? = null

    init {
        refresh()
    }

    /** Re-read the manifest and the gallery (after an `adb push`). */
    fun refresh() {
        val s = _state.value
        s.graphsDir.mkdirs()
        s.galleryDir.mkdirs()
        val gallery = (s.galleryDir.listFiles { f -> f.isFile && f.name.endsWith(".safetensors") } ?: emptyArray())
            .sortedBy { it.name.lowercase(Locale.ROOT) }
        val (manifest, err) = try {
            Manifest.load(s.graphsDir) to null
        } catch (e: Exception) {
            null to (e.message ?: e.toString())
        }
        val sizes = manifest?.sizes ?: emptyList()
        val size = s.size?.takeIf { it in sizes }
            ?: sizes.firstOrNull { it == Geometry.MOBILE_480x272 }
            ?: sizes.firstOrNull()
        _state.update {
            it.copy(
                manifest = manifest,
                manifestError = err,
                gallery = gallery,
                prompt = it.prompt?.takeIf { p -> p in gallery } ?: gallery.firstOrNull(),
                sizes = sizes,
                size = size,
                budgetBytes = availableMemory(),
                error = null,
            )
        }
        replan()
    }

    fun selectPrompt(f: File) = _state.update { it.copy(prompt = f) }

    fun selectSize(g: Geometry) {
        _state.update { it.copy(size = g) }
        replan()
    }

    fun setSeed(seed: String) = _state.update { it.copy(seed = seed.filter { c -> c.isDigit() || c == '-' }.take(20)) }

    fun setUseGpu(on: Boolean) = _state.update { it.copy(useGpu = on) }

    private fun replan() {
        val s = _state.value
        val m = s.manifest
        val g = s.size
        if (m == null || g == null) {
            _state.update { it.copy(plan = null, planFits = false) }
            return
        }
        try {
            val p = m.plan(g, s.budgetBytes)
            _state.update { it.copy(plan = p.toString(), planFits = p.fits) }
        } catch (e: Exception) {
            _state.update { it.copy(plan = e.message, planFits = false) }
        }
    }

    private fun availableMemory(): Long {
        val am = getApplication<Application>().getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo()
        am.getMemoryInfo(mi)
        // What the device has free right now; the tier design assumes about 2 GB for the app.
        return mi.availMem
    }

    fun generate() {
        val s = _state.value
        val manifest = s.manifest ?: return
        val prompt = s.prompt ?: return
        val size = s.size ?: return
        val seed = s.seed.toLongOrNull() ?: return
        if (!s.planFits) {
            log("plan says this clip will not fit; generating anyway (see docs/design.md)")
        }
        job?.cancel()
        _state.update {
            it.copy(running = true, status = "opening graphs", stepMs = emptyList(), prepareMs = null, decodeMs = null,
                frames = emptyList(), error = null, log = emptyList())
        }
        job = viewModelScope.launch(Dispatchers.Default) {
            val t0 = System.nanoTime()
            try {
                val embeds = loadEmbeds(prompt, manifest)
                val cacheDir = File(getApplication<Application>().cacheDir, "litert-gpu").apply { mkdirs() }
                LiteRtGraphStore(
                    manifest,
                    accelerator = if (s.useGpu) Accelerator.GPU else Accelerator.CPU,
                    cacheDir = cacheDir,
                    resident = 4,
                    onFallback = { file, e -> log("$file: GPU compile failed, using CPU (${e.message})") },
                ).use { store ->
                    val pipeline = Pipeline(store)
                    log("${manifest.model} ${manifest.weights}, chunk ${manifest.chunk}, ${manifest.graphs.blocks.size} blocks; $size, seed $seed")
                    val (video, timings) = pipeline.generate(embeds, size, seed) { ev ->
                        ensureActive()
                        when (ev) {
                            is Event.Prepared -> _state.update {
                                it.copy(status = "prepared (${ev.tokens} tokens)", prepareMs = ev.ms)
                            }
                            is Event.Step -> _state.update {
                                it.copy(status = "step ${ev.index + 1}/${ev.total}", stepMs = it.stepMs + ev.ms)
                            }
                            is Event.Frames -> {
                                val bitmaps = List(ev.count) { i -> toBitmap(ev.frame(i), ev.width, ev.height) }
                                _state.update {
                                    it.copy(status = "decoded ${ev.first + ev.count} frames", frames = it.frames + bitmaps,
                                        frameWidth = ev.width, frameHeight = ev.height)
                                }
                            }
                        }
                    }
                    val total = (System.nanoTime() - t0) / 1e6
                    _state.update {
                        it.copy(running = false, decodeMs = timings.decodeS * 1000.0,
                            status = "${video.frames} frames ${video.width}x${video.height} in ${fmtMs(total)}")
                    }
                    log("timings: $timings")
                }
            } catch (e: CancellationException) {
                _state.update { it.copy(running = false, status = "cancelled") }
            } catch (e: Throwable) {
                _state.update { it.copy(running = false, status = "failed", error = e.message ?: e.toString()) }
            }
        }
    }

    fun cancel() {
        job?.cancel()
    }

    private fun loadEmbeds(file: File, manifest: Manifest): Tensor {
        val st = SafeTensors.read(file)
        val key = if ("prompt_embeds" in st) "prompt_embeds" else st.names.first()
        var t = st.tensor(key)
        // Tolerate a leading batch axis of 1 (`[1, 512, 4096]`).
        if (t.rank == 3 && t.shape[0] == 1) t = t.reshape(t.shape[1], t.shape[2])
        val c = manifest.cfg
        if (!(t.rank == 2 && t.shape[0] == c.textLen && t.shape[1] == c.textDim)) {
            throw IllegalArgumentException("${file.name}: $key is ${t.shapeString()}, expected [${c.textLen}, ${c.textDim}]")
        }
        return t
    }

    private fun log(line: String) = _state.update { it.copy(log = (it.log + line).takeLast(200)) }

    private fun fmtMs(ms: Double) = if (ms >= 1000) String.format(Locale.ROOT, "%.1f s", ms / 1000) else String.format(Locale.ROOT, "%.0f ms", ms)

    companion object {
        /** `[H, W, 4]` RGBA bytes → an `ARGB_8888` bitmap (its memory order is R, G, B, A). */
        fun toBitmap(rgba: ByteArray, width: Int, height: Int): Bitmap {
            val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            bmp.copyPixelsFromBuffer(ByteBuffer.wrap(rgba))
            return bmp
        }
    }
}
