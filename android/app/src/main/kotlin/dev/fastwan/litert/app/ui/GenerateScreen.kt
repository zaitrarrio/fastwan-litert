package dev.fastwan.litert.app.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.KeyboardOptions
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.fastwan.litert.app.GenerateViewModel
import dev.fastwan.litert.app.UiState
import dev.fastwan.litert.core.Geometry
import kotlinx.coroutines.delay
import java.io.File
import java.util.Locale

private const val PLAYBACK_FPS = 16

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GenerateScreen(vm: GenerateViewModel) {
    val s by vm.state.collectAsStateWithLifecycle()
    Scaffold(topBar = { TopAppBar(title = { Text("FastWan LiteRT") }) }) { padding ->
        Column(
            Modifier.fillMaxSize().padding(padding).verticalScroll(rememberScrollState()).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            FramePlayer(s)
            StatusCard(s)
            GraphsCard(s, onRefresh = vm::refresh)
            PromptPicker(s, onPick = vm::selectPrompt)
            SizeAndSeed(s, onSize = vm::selectSize, onSeed = vm::setSeed, onGpu = vm::setUseGpu)
            PlanCard(s)
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = vm::generate, enabled = s.canGenerate, modifier = Modifier.weight(1f)) { Text("Generate") }
                OutlinedButton(onClick = vm::cancel, enabled = s.running, modifier = Modifier.weight(1f)) { Text("Cancel") }
            }
            LogCard(s)
            Spacer(Modifier.height(24.dp))
        }
    }
}

/** Plays the frames that have arrived so far at 16 fps, looping; shows the last one while decoding. */
@Composable
private fun FramePlayer(s: UiState) {
    var index by remember { mutableIntStateOf(0) }
    val frames = s.frames
    LaunchedEffect(frames.size, s.running) {
        if (frames.isEmpty()) { index = 0; return@LaunchedEffect }
        while (true) {
            delay(1000L / PLAYBACK_FPS)
            index = if (s.running) frames.lastIndex else (index + 1) % frames.size
        }
    }
    val aspect = if (s.frameWidth > 0 && s.frameHeight > 0) s.frameWidth.toFloat() / s.frameHeight else 480f / 272f
    Box(
        Modifier.fillMaxWidth().aspectRatio(aspect).background(MaterialTheme.colorScheme.surfaceVariant),
        contentAlignment = Alignment.Center,
    ) {
        if (frames.isNotEmpty()) {
            val i = index.coerceIn(0, frames.lastIndex)
            Image(
                bitmap = frames[i].asImageBitmap(),
                contentDescription = "frame ${i + 1} of ${frames.size}",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit,
            )
            Text(
                "${i + 1}/${frames.size}  ${s.frameWidth}x${s.frameHeight}",
                Modifier.align(Alignment.BottomEnd).padding(6.dp),
                style = MaterialTheme.typography.labelSmall,
            )
        } else {
            Text(if (s.running) "generating…" else "no frames yet", color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun StatusCard(s: UiState) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            if (s.running) LinearProgressIndicator(Modifier.fillMaxWidth())
            Text(s.status.ifEmpty { "idle" }, style = MaterialTheme.typography.bodyMedium)
            if (s.prepareMs != null || s.stepMs.isNotEmpty() || s.decodeMs != null) {
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.fillMaxWidth()) {
                    s.prepareMs?.let { AssistChip(onClick = {}, label = { Text("prepare ${fmtMs(it)}") }) }
                    s.stepMs.forEachIndexed { i, ms -> AssistChip(onClick = {}, label = { Text("step ${i + 1} ${fmtMs(ms)}") }) }
                    s.decodeMs?.let { AssistChip(onClick = {}, label = { Text("decode ${fmtMs(it)}") }) }
                }
            }
            s.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
        }
    }
}

@Composable
private fun GraphsCard(s: UiState, onRefresh: () -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Graphs", style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                OutlinedButton(onClick = onRefresh, enabled = !s.running) { Text("Rescan") }
            }
            Text(s.graphsDir.path, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
            val m = s.manifest
            if (m != null) {
                Text(
                    "${m.model} · ${m.weights} weights · chunk ${m.chunk} · ${m.graphs.blocks.size} blocks · " +
                        "${m.attention.size} attention / ${m.taeChunks.size} TAEHV graphs",
                    style = MaterialTheme.typography.bodySmall,
                )
            } else {
                Text(
                    s.manifestError ?: "no manifest",
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall,
                )
                Text(
                    "adb push <export dir>/. ${s.graphsDir.path}/  (see android/README.md)",
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun PromptPicker(s: UiState, onPick: (File) -> Unit) {
    var open by remember { mutableStateOf(false) }
    ExposedDropdownMenuBox(expanded = open, onExpandedChange = { if (!s.running) open = it }) {
        OutlinedTextField(
            value = s.prompt?.name ?: if (s.gallery.isEmpty()) "no *.safetensors in ${s.galleryDir.path}" else "",
            onValueChange = {},
            readOnly = true,
            label = { Text("Prompt embedding (gallery/)") },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = open) },
            modifier = Modifier.fillMaxWidth().menuAnchor(MenuAnchorType.PrimaryNotEditable),
            singleLine = true,
        )
        ExposedDropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            for (f in s.gallery) {
                DropdownMenuItem(text = { Text(f.name) }, onClick = { onPick(f); open = false })
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SizeAndSeed(s: UiState, onSize: (Geometry) -> Unit, onSeed: (String) -> Unit, onGpu: (Boolean) -> Unit) {
    var open by remember { mutableStateOf(false) }
    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
        ExposedDropdownMenuBox(expanded = open, onExpandedChange = { if (!s.running) open = it }, modifier = Modifier.weight(1f)) {
            OutlinedTextField(
                value = s.size?.toString() ?: "",
                onValueChange = {},
                readOnly = true,
                label = { Text("Size (manifest sizes)") },
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = open) },
                modifier = Modifier.fillMaxWidth().menuAnchor(MenuAnchorType.PrimaryNotEditable),
                singleLine = true,
            )
            ExposedDropdownMenu(expanded = open, onDismissRequest = { open = false }) {
                for (g in s.sizes) {
                    val tier = if (g == Geometry.MOBILE_480x272) "  (phone tier)" else ""
                    DropdownMenuItem(text = { Text("$g$tier") }, onClick = { onSize(g); open = false })
                }
            }
        }
        OutlinedTextField(
            value = s.seed,
            onValueChange = onSeed,
            label = { Text("Seed") },
            singleLine = true,
            enabled = !s.running,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            modifier = Modifier.width(120.dp),
        )
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text("GPU (LiteRT CompiledModel, fp16)", modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
        Switch(checked = s.useGpu, onCheckedChange = onGpu, enabled = !s.running)
    }
}

@Composable
private fun PlanCard(s: UiState) {
    val plan = s.plan ?: return
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(
                if (s.planFits) "Memory plan: fits" else "Memory plan: does not fit the free memory",
                style = MaterialTheme.typography.titleSmall,
                color = if (s.planFits) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.error,
            )
            Text(plan, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
        }
    }
}

@Composable
private fun LogCard(s: UiState) {
    if (s.log.isEmpty()) return
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            for (line in s.log.takeLast(30)) {
                Text(line, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
            }
        }
    }
}

private fun fmtMs(ms: Double): String =
    if (ms >= 1000) String.format(Locale.ROOT, "%.1f s", ms / 1000) else String.format(Locale.ROOT, "%.0f ms", ms)
