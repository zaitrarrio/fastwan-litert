# fastwan-litert

FastWan 2.1 1.3B text-to-video on [LiteRT](https://ai.google.dev/edge/litert),
for the **web** (LiteRT.js, WebGPU), **Android** (LiteRT `CompiledModel`, GPU),
**iOS** (LiteRT, Metal) and the **desktop** (Python on `ai-edge-litert`). A
port of [fastwan-rs](https://github.com/zaitrarrio/fastwan-rs): the same
model, sampler (3 DMD steps, no CFG) and TAEHV decoder; the fused Rust
kernels are replaced by `.tflite` graphs that one exporter writes and four
thin hosts drive.

- **Model:** `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`, decoder TAEHV
  `taew2_1`, prompts from UMT5-XXL precomputed on a desktop (there is no text
  encoder on the device, as in fastwan-rs's browser build).
- **The cut:** per transformer block, a `qkv` graph, a weight-free dense
  attention graph over query chunks, and a `post` graph (cross-attention
  against per-prompt K/V, FFN); timestep conditioning as one tiny graph;
  TAEHV as one chunk graph with carried memories. `docs/design.md` says why,
  `docs/graphs.md` is the contract every runtime follows.
- **Weights:** `f32`, `f16`, `int8` or `int4` (weight-only, block linears),
  exported once per format and reused at every clip size; one block per
  file, so a page or app streams 30 files and keeps a few resident.

| runtime | where | status |
|---|---|---|
| `python/` | desktop (`fastwan-litert export / generate / check / plan / serve`), the exporter, the reference | graphs equal the PyTorch model (rel 4e-7 on the latent, ≤ 1/255 on frames) and the PyTorch model equals Diffusers (rel 2e-7); 15 tests; runs on CPU here, `--accelerator gpu` untested |
| `web/` | LiteRT.js on WebGPU / Wasm, the studio page | 16 vitest cases; the tiny fixture in headless Chromium on the Wasm backend: latent rel 2.9e-7, frames within 1/255; WebGPU path untested (no GPU here) |
| `android/` | Kotlin `:core` + LiteRT 2.1.6 `CompiledModel` `:app` (GPU) | 27 JUnit tests pass (`gradle :core:test`); the LiteRT store compiles against the real AAR; `:app` needs the Android SDK and a device |
| `ios/` | Swift package + TensorFlowLiteSwift / Metal, SwiftUI demo | written and self-reviewed, uncompiled (no Swift toolchain here) |

Nothing here has run on a GPU through a LiteRT delegate yet, and the real
checkpoint has not been exported in CI (1.3 B parameters). See
`docs/design.md` for what is verified and how.

## Use

```bash
cd python && pip install -e ".[export,test]"
pytest                                        # host math, Diffusers parity, graph parity, plan
fastwan-litert export --model tiny --out fixtures/tiny     # the tiny self-test fixture
fastwan-litert check --graphs fixtures/tiny                # the graph pipeline vs the PyTorch reference
```

The real model:

```bash
huggingface-cli download FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
    --include "transformer/*" "text_encoder/*" "tokenizer/*" --local-dir fastwan
huggingface-cli download madebyollin/taehv taew2_1.safetensors --local-dir fastwan
fastwan-litert export --model-dir fastwan --weights int8 --out graphs/int8 \
    --size 832x480x81 --size 480x272x33          # ~1.4 GB, 60 block graphs + 4 + attention + TAEHV
fastwan-litert encode --model-dir fastwan --prompt "a red fox trots through fresh snow" --out fox.safetensors
fastwan-litert generate --graphs graphs/int8 --embeds fox.safetensors --size 832x480x81 --out fox.y4m
ffmpeg -i fox.y4m fox.mp4
fastwan-litert serve --graphs graphs/int8 --gallery gallery/     # http://127.0.0.1:7860/
```

`fastwan-litert plan --size 832x480x81 --weights int8 --budget-gib 8` sizes
a run before anything is allocated. Dense attention at 480p×81 is about
200 TFLOP per forward: on a CPU this is hours, on a desktop GPU minutes. The
phone and default web tiers are 480×272×33 with `int4` weights.

## Layout

```
docs/       design.md (why this cut), graphs.md (the manifest and tensor contract)
python/     fastwan_litert: exporter, desktop runtime, reference pipeline, tests
fixtures/   tiny: the self-test model (12 graphs, 340 KB) and its expected outputs
web/        TypeScript runtime on @litertjs/core + the studio page
android/    Kotlin: :core (host math, pipeline, tests) and :app (LiteRT GPU, Compose)
ios/        Swift package: host math, pipeline, LiteRT backend, SwiftUI demo
scripts/    download.sh
```
