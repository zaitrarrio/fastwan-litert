# fastwan-litert for iOS

The Swift side of fastwan-litert: the host math and the pipeline of
`docs/graphs.md` as a Swift package, a LiteRT (TensorFlow Lite Swift, Metal)
backend, and a SwiftUI demo. The package mirrors `python/fastwan_litert`
one file at a time; the Python runtime is the reference it is checked
against.

**Status: written without a Swift toolchain.** Nothing in this directory has
been compiled, the tests have not run, and no graph has run on a device. The
code was reviewed by hand for syntax and types; expect a first `swift build`
to turn up small things. See "Not verified" at the end.

## Layout

```
ios/
  Package.swift                       swift-tools-version 5.9; iOS 16 / macOS 13
  Sources/FastWanLiteRTCore/          Foundation only
    Config.swift                      WanConfig, TaeConfig (config.py)
    Geometry.swift                    Geometry: latent / grid / tokens, validate (geometry.py)
    DmdSchedule.swift                 the shift-8 training table and [1000, 757, 522] (schedule.py)
    Rope.swift                        3-axis RoPE cos/sin tables [S, hd/2] (rope.py)
    Rng.swift                         SplitMix64 → Box–Muller, UInt64 wrapping arithmetic (rng.py)
    Tensor.swift                      [Float] + shape; rows / row / concatRows
    Host.swift                        patchify, unpatchify, dmdStep, padRows, toRgb8 / toRgba8 (host.py)
    Graph.swift                       Graph and GraphStore protocols, ResidentGraphCache (LRU of block graphs)
    Manifest.swift                    Codable manifest; format check; attentionFor / taeFor throw on a miss
    Plan.swift                        the memory plan (plan.py)
    Pipeline.swift                    preparePrompt, forward, denoise, decode, generate; events (pipeline.py)
    SafeTensors.swift                 a reader for prompt embeddings and expected.safetensors
    Check.swift                       FixtureCheck: the `check` self-test against expected.safetensors
  Sources/FastWanLiteRTRuntime/
    LiteRTGraph.swift                 LiteRTGraph (Interpreter + SignatureRunner, MetalDelegate), LiteRTGraphStore
  Tests/FastWanLiteRTCoreTests/
    HostTests.swift                   every case of python/tests/test_host.py and test_plan.py, manifest, safetensors
    PipelineTests.swift               the orchestration over a fake GraphStore on the tiny manifest
    Resources/manifest.json           a copy of fixtures/tiny/manifest.json
  Demo/                               SwiftUI app sources and how to make the Xcode project
```

### Core

`Pipeline` drives `Graph`s through a `GraphStore` exactly as
`docs/graphs.md`'s host loop and `pipeline.py` do:

- `preparePrompt(embeds)`: `text_embed`, then `block_N_kv` of every block,
  once per prompt.
- `forward(x:prompt:t:cos:sin:attention:)`: `time_mods`; `patch_embed` over
  row chunks of `chunk` (the last chunk zero-padded, its padded rows'
  outputs dropped); per block `qkv` over row chunks, attention over query
  chunks of `q_chunk` against the exact `S` keys and values, `post` over row
  chunks; `head`; `unpatchify`.
- `denoise(embeds:geometry:seed:onEvent:timings:)`: one `Rng(seed)` draws
  the initial latent first, then the re-noise of every step but the last, in
  step order, in `[16, T, Hl, Wl]` order.
- `decode(latent:onEvent:timings:)`: TAEHV chunks of `frames` latent frames
  (the last zero-padded), nine memories carried, the padded frames' output
  and then the three priming frames dropped.
- Events: `.prepared(seconds:tokens:)`, `.step(index:total:seconds:)`,
  `.frames(Frames)`. `Frames` carries `first`, `count`, `width`, `height`
  and `rgba` (`[count, height, width, 4]`, alpha 255) so a frame goes
  straight into a `CGImage` (`Demo/FrameImage.swift`); `Frames.rgb` gives the
  Python layout for comparisons.

All graph inputs are float32 tensors with exact shapes and no batch
dimension; inputs are bound by signature name, outputs returned by position.

### Runtime

`LiteRTGraph` opens one `.tflite` file with the TensorFlow Lite Swift API:
`Interpreter(modelPath:options:delegates:)` with a `MetalDelegate` when
asked (`isPrecisionLossAllowed` on, so the GPU may run f16), the graph's
single signature through `interpreter.signatureRunner(with:)`, inputs by
name via `SignatureRunner.invoke(with: [String: Data])`, outputs by
`output(named:)` sorted into `output_0, output_1, …` order. Input shapes and
dtypes are checked against the graph before every call and a mismatch is an
error naming the graph, the input and both shapes. `LiteRTGraphStore` reads
graphs from a directory URL, keeps the fixed graphs open and the block graphs
in an LRU of `residentBlocks` (default 2).

The Swift API is distributed as the CocoaPod `TensorFlowLiteSwift`
(2.17.0; GPU delegate in the `Metal` subspec), with `LiteRTSwift` (nightly
pods) as its successor. There is no official Swift package of either, so
`Package.swift` declares no dependency and `LiteRTGraph.swift` is wrapped in
`#if canImport(TensorFlowLite)`: on a Mac without the framework the target
compiles to a stub and `swift test` runs the core tests. In an app, link the
pod and make sure the file compiles where the pod is visible (see
`Demo/README.md` step 4; a Swift package target built by Xcode does not
inherit the app's framework search paths, so adding the file to the app
target is the fallback). Marked `// TODO verify` in the source.

## Adding the package to an Xcode project

File → Add Package Dependencies… → Add Local… → this `ios/` folder; add the
`FastWanLiteRTCore` (and `FastWanLiteRTRuntime`) products to the app target.
Then the pod, as in `Demo/README.md`. The core target has no dependency and
no UIKit / CoreGraphics import; it also builds for macOS 13 so the tests run
on a Mac:

```bash
cd ios && swift test            # host math, manifest, safetensors, pipeline orchestration
```

## Getting the graphs onto the device

The app reads the export from its Documents folder. With
`UIFileSharingEnabled` and `LSSupportsOpeningDocumentsInPlace` in the app's
Info.plist, that folder shows up in the Files app ("On My iPhone → FastWan
Demo") and in Finder (device → Files tab), where you drop:

- `Documents/graphs/`: the whole export directory from
  `fastwan-litert export --weights int4 --size 480x272x33 --out graphs/int4`
  (`manifest.json` and every `.tflite`; `LiteRTGraphStore.missingFiles()`
  lists what is absent),
- `Documents/gallery/<name>.safetensors`: prompt embeddings from
  `fastwan-litert encode` (`prompt_embeds` `[512, 4096]` F32).

The export is the same one the Python, web and Android runtimes use;
nothing is converted on the device.

## The phone tier and memory

The phone tier is 480×272×33 (`Geometry.mobile480x272`: latent
`(9, 34, 60)`, 4,590 tokens) with `int4` block weights, about 0.88 GB of
graphs. `Plan.make(manifest:geometry:budgetBytes:residentBlocks:)` is
`fastwan-litert plan` for the export at hand: weights (all and resident),
cross-attention K/V, the `[S, d]` activations, the `H · Cq · S` attention
score chunk and the TAEHV peak, against a budget. The demo shows it for
2 GiB (what an iOS app can count on) before `Generate`. Two resident blocks
keep ~60 MB of int4 block graphs open at once; the other 28 open and close
through the LRU each step, so the first steps pay for 30 file reads from
flash. At 480p (32,760 tokens) the plan does not fit a phone and dense
attention is minutes of GPU time per step; that size is for the desktop.

## Not verified

- **Nothing compiled.** No Swift toolchain or Xcode was available where this
  was written. Every file was self-reviewed for syntax and types, but the
  first build will be the first compile. The tests are written to reproduce
  the pinned values of `python/tests/test_host.py` (sigmas, 480p geometry,
  RoPE split and origin, SplitMix64 outputs, normal moments, patchify
  conventions, `21·4 − 3 = 81`) and the orchestration counts of the tiny
  manifest (2 qkv / 3 attention / 2 post calls per block per step, 2 TAEHV
  calls, 9 frames); they have not run.
- **No device.** `LiteRTGraph` follows the TensorFlow Lite Swift API as
  published in `tensorflow/lite/swift/Sources` (checked against the source
  while writing), but it has not been linked or run; the Metal delegate path,
  the f16 precision of its output against `expected.safetensors` (the
  contract's 2e-2), speed, and the memory of the LRU under a 30-block int4
  export are all untested. XNNPACK together with the Metal delegate on one
  interpreter is marked `// TODO verify`.
- **Package integration of the pod.** See "Runtime" above; the fallback of
  compiling `LiteRTGraph.swift` inside the app target is documented but not
  tried.
- The `check` path (`FixtureCheck.run`) has not been run against
  `fixtures/tiny/expected.safetensors` through LiteRT.
