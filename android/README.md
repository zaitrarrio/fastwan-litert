# fastwan-litert on Android

The Android runtime of fastwan-litert: FastWan 2.1 1.3B text-to-video (3 DMD steps, TAEHV decode)
over LiteRT `CompiledModel` graphs, following `docs/graphs.md` exactly. A Kotlin port of the Python
reference in `python/fastwan_litert/`.

## Modules

```
android/
  core/   pure Kotlin/JVM library, package dev.fastwan.litert.core   (builds and tests on any JDK >= 17)
  app/    Android application: LiteRT GPU backend + Compose UI        (needs the Android SDK)
```

**`:core`** has no Android imports. It holds everything that is not a graph call:

| file | what | Python source |
|---|---|---|
| `WanConfig.kt` | `WanConfig`, `TaeConfig` (1.3B and tiny presets, `ropeSplit`, `params`, `tUpscale`, `framesToTrim`) | `config.py` |
| `Geometry.kt` | `Geometry` (`validate`, `latent`, `grid`, `tokens`, `parse`), `Dims3` | `geometry.py` |
| `DmdSchedule.kt` | the shift-8 f32 training table, nearest-timestep sigmas, `coeffs(i)` | `schedule.py` |
| `Rope.kt` | 3-axis RoPE table as `cos`/`sin` `Tensor [S, hd/2]`, f64 angles cast to f32 | `rope.py` |
| `Rng.kt` | SplitMix64 → Box–Muller, `ULong` wrapping arithmetic, Box–Muller in `Double`, stored as `Float` | `rng.py` |
| `Host.kt` | `patchify`, `unpatchify`, `dmdStep`, `padRows`, `toRgb8`, `toRgba8` | `host.py` |
| `Manifest.kt` | `manifest.json` (kotlinx.serialization), `format` check, `attentionFor(tokens)`, `taeFor(h, w)`, `plan(g, budget)` | `runtime.py` |
| `Plan.kt` | the memory plan | `plan.py` |
| `Graph.kt` | `Graph` (inputs by name, outputs by position) and `GraphStore` (fixed graphs + LRU block graphs) | `runtime.py` |
| `Pipeline.kt` | `preparePrompt`, `forward`, `denoise`, `decode`, `generate`; `Event.Prepared / Step / Frames` | `pipeline.py` |
| `SafeTensors.kt` | a tiny safetensors reader (F32 / F16 / BF16 / raw bytes) for prompt embeddings and the fixture | — |
| `Tensor.kt` | `Tensor(data: FloatArray, shape: IntArray)` with row slicing / concatenation | — |

**`:app`** (`dev.fastwan.litert.app`):

- `LiteRtGraphStore` — `GraphStore` over `com.google.ai.edge.litert.CompiledModel`, reading graphs
  from a directory on external storage; `Accelerator.GPU` with fp16 precision and a program cache,
  optional CPU fallback per graph (reported in the log). Every input is checked against the graph's
  own tensor layout, so an export/manifest mismatch fails with both shapes in the message.
- `GenerateViewModel` — runs `Pipeline.generate` on `Dispatchers.Default`, turns `Event.Frames`
  into `Bitmap`s, keeps step timings, the memory `Plan` and a log in a `StateFlow<UiState>`.
- `MainActivity` / `ui/GenerateScreen.kt` — Compose: pick a prompt embedding from `gallery/`,
  a size from the manifest's `sizes` (480×272×33 preselected when present), a seed, GPU on/off;
  shows prepare / step / decode timings and plays the frames as each TAEHV chunk arrives.

## Running the `:core` tests (no Android SDK needed)

```bash
cd android
./gradlew :core:test            # or: gradle :core:test   (Gradle 8.14.3, any JDK >= 17)
```

`settings.gradle.kts` includes `:app` only when an Android SDK is found (`ANDROID_HOME`,
`ANDROID_SDK_ROOT`, or `sdk.dir` in `local.properties`), so a `:core`-only build never resolves the
Android Gradle Plugin. `-Pfastwan.app=true` forces `:app` in.

The tests mirror `python/tests/test_host.py` and `test_plan.py` and add the pipeline orchestration:

- `HostMathTest` — sigmas `1.0 / 0.757 / 0.522 / 0` (and the exact f32 table values the exporter
  wrote), 480p = `(21, 60, 104)` latent and 32,760 tokens, `(44, 42, 42)` RoPE split with the
  identity at the origin and `cos(1 rad)` for token `(1, 0, 0)`, TAEHV `21·4 − 3 = 81`, SplitMix64
  (`Rng(0x9E3779B97F4A7C15)` → `0xE220A8397B1DCDAF`, `0x6E789E6AA1B965F4`), normal moments over
  200k draws, the scalar Box–Muller definition and values printed by the Python `Rng`, patchify /
  unpatchify index conventions, `dmdStep`, `padRows`, the RGB8 rounding.
- `PipelineTest` — a `FakeGraphStore` whose graphs check every input name and shape against the
  contract and record the calls, on the tiny manifest (`fixtures/tiny/manifest.json`, copied to
  `core/src/test/resources/tiny/`): for 18 tokens with `C = 16`, `Cq = 8` every block runs
  2 `qkv`, 3 attention and 2 `post` calls in that order, `k`/`v` are never padded, padded rows are
  dropped (the fakes emit NaN for all-zero rows, and the latent must stay finite), the TAEHV graph
  runs 2 chunks for 3 latent frames and the 12 decoded frames become 9 after trimming the 3 priming
  frames (`Frames` events `(0, 5)`, `(5, 4)`, 32×48 RGBA), noise is drawn in the contract's order,
  wrong prompt / geometry are refused with the mismatch in the message, LRU residency.
- `ManifestTest`, `SafeTensorsTest` (reads `fixtures/tiny/expected.safetensors`), `PlanTest`.

## Building `:app`

Needs the Android SDK with platform 36 (AGP 8.13.2, Kotlin 2.3.0, Compose BOM 2025.11.00,
LiteRT 2.1.6; minSdk 26):

```bash
cd android
echo "sdk.dir=$ANDROID_HOME" > local.properties      # or export ANDROID_HOME
./gradlew :app:installDebug
```

Kotlin must be ≥ 2.2: `litert-api` 2.1.6 is compiled with Kotlin metadata 2.3.0.

### Pushing graphs and prompts

The app reads from its external files directory (no storage permission needed):

```
/sdcard/Android/data/dev.fastwan.litert/files/graphs/    manifest.json + *.tflite
/sdcard/Android/data/dev.fastwan.litert/files/gallery/   *.safetensors with prompt_embeds [512, 4096]
```

Export the phone tier and push it:

```bash
fastwan-litert export --model-dir fastwan --weights int4 --out graphs/int4 --size 480x272x33
fastwan-litert encode --model-dir fastwan --prompt "a red fox trots through fresh snow" --out fox.safetensors

adb shell mkdir -p /sdcard/Android/data/dev.fastwan.litert/files/graphs \
                   /sdcard/Android/data/dev.fastwan.litert/files/gallery
adb push graphs/int4/. /sdcard/Android/data/dev.fastwan.litert/files/graphs/
adb push fox.safetensors /sdcard/Android/data/dev.fastwan.litert/files/gallery/
```

Then press **Rescan** in the app. The tiny fixture (`fixtures/tiny`, 340 KB, size 32×48×9) pushes
the same way and is the quickest way to see the whole loop run on a device; `expected.safetensors`
holds `prompt_embeds` for it.

## The phone tier and memory

The default and only sensible phone geometry is **480×272×33** with **int4** weights
(`docs/design.md`): latent `(9, 34, 60)`, 4,590 tokens, 0.88 GB of weights. Before generating, the
app computes `Manifest.plan(geometry, availMem)` (the port of `plan.py`) and shows it:

```
tokens            4590
weights (all)         713.9 MiB   int4 block linears + f16 everything else
weights resident      138.3 MiB   4 of 30 block graphs open at a time (LiteRtGraphStore's LRU)
cross-attn K/V        180.0 MiB   30 blocks x 2 x 512 x 1536 f32
DiT activations       140.1 MiB   h, q, k, v, o at [S, d]; latent, velocity, noise; rope tables
attention scratch      53.8 MiB   12 x 256 x 4590 f32 (one query chunk against every key)
decode peak           812.8 MiB   one TAEHV chunk of 3 latent frames at 8x resolution
peak                  951.1 MiB of 2048 MiB  fits: true
note: 4590 rows run as 5 chunks of 1024 (530 padded rows)
```

(`fastwan-litert plan --size 480x272x33 --weights int4 --budget-gib 2 --resident 4` prints the
same numbers from the Python reference.)

Everything fits in about 2 GB. 832×480×81 (32,760 tokens) does not fit a phone (the plan says so and
`PlanTest.p480DoesNotFitAPhone` pins it); the app still offers every size the manifest lists and
only warns, since `plan` is a model, not a measurement. Dense attention is ~200 TFLOP per forward at
480p; at the phone tier it is roughly 1/50 of that, still minutes on a mobile GPU.

Memory behaviour of the Android backend: block graphs are opened by file under an LRU of 4
(`resident`), the fixed graphs (embedders, head, attention, TAEHV) stay open, every `TensorBuffer`
is created once per graph input and reused across calls, and the GPU program cache lives in
`cacheDir/litert-gpu` so reopening an evicted block does not recompile its kernels.

## What is not verified here

- **No device, no SDK in this container.** `:app` has not been compiled; `:core` has
  (`gradle :core:test`, 27 tests). Version pins for AGP / Compose / lifecycle were chosen from the
  Google Maven index and the LiteRT 2.1.6 AAR manifests (minSdk 23), not from a build.
- **LiteRT API names** in `LiteRtGraphStore` were checked against the published `litert-api` 2.1.6
  classes (`javap`): `CompiledModel.create(path, options, env)`, `Options(vararg Accelerator)` with
  `gpuOptions`, `createInputBuffer(name)`, `createOutputBuffers()`, `getInputTensorType(name).layout.dimensions`,
  `run(Map, Map)`, `TensorBuffer.writeFloat/readFloat`. Their runtime behaviour (GPU compile of int4
  graphs, fp16 accuracy against the 2e-2 budget of `docs/graphs.md`, whether `output_i` is the
  signature's output name order, the program-cache options) is untested; those lines carry
  `TODO verify` comments.
- **Output names.** The contract says outputs are `output_0, output_1, …` in table order; the
  backend maps `createOutputBuffers()` (positional) onto those names for the by-name `run`. If a
  converter version names outputs differently the first `run` fails with LiteRT's message.
- **Fixture parity.** The Python `check` compares the graph pipeline against
  `expected.safetensors`; on the JVM there is no LiteRT runtime, so the Kotlin pipeline is verified
  structurally (fake graphs, call sequence, shapes, padding, noise order) and numerically only in
  the host math. Running the tiny fixture on a device through `:app` is the first real parity check.
