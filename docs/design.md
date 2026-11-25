# fastwan-litert design

FastWan 2.1 1.3B text-to-video (3 DMD steps, TAEHV decode) on
[LiteRT](https://ai.google.dev/edge/litert), Google's on-device runtime, for
the web (LiteRT.js on WebGPU / Wasm), Android (LiteRT `CompiledModel`, GPU),
iOS (LiteRT / TensorFlow Lite Swift, Metal) and the desktop (the Python
runtime on `ai-edge-litert`, which is also the reference every other runtime
is checked against).

It is a port of [fastwan-rs](https://github.com/zaitrarrio/fastwan-rs). The
model, sampler and decoder are unchanged; what changes is *where the math
runs*. fastwan-rs writes every fused op three times (CPU, WGSL, CUDA).
fastwan-litert writes the model once as PyTorch modules, converts them to
`.tflite` graphs with `ai-edge-torch`, and ships the graphs with a manifest
that four thin hosts drive.

## What stays from fastwan-rs

Everything that fastwan-rs computes on the host, or precomputes once, stays
exactly as it is there, and is ported to each host language with its unit
tests:

- the DMD schedule (`[1000, 757, 522]` on the shift-8 training table:
  `σ = 1.0, 0.757, 0.522`, then `0`),
- the geometry (`4n + 1` frames, latent `T = (F − 1)/4 + 1`, `H/8`, `W/8`,
  tokens `T · H/16 · W/16`; 832×480×81 is 32,760 tokens),
- the 3-axis RoPE table (`(44, 42, 42)` of 128, `d/2` angles per token,
  f64 angles cast to f32),
- the deterministic noise (SplitMix64 → Box–Muller; the same seed draws the
  same latent on every platform, so runtimes can be compared end to end),
- patchify / unpatchify and the DMD step `x0 = x − σv; x = (1 − σ')x0 + σ'ε`,
- TAEHV's chunked decode with the per-MemBlock carried frame,
- prompts precomputed by UMT5-XXL (`scripts/encode_prompt.py`, or a gallery):
  there is no text encoder on the device, as in the browser build of
  fastwan-rs.

## What changes: the graph cut

A `.tflite` graph is a fixed-shape, weight-carrying function. The whole DiT
as one graph would be 2.6 GB of f16 with a 32,760×32,760 attention score
matrix inside it: impossible on a phone and useless for streaming weights.
So the DiT is cut along the same lines fastwan-rs chunks its work:

| graph | per | inputs → outputs | weights |
|---|---|---|---|
| `time_mods` | step | `t[1]` → `block_mods[L,6,d]`, `head_mods[2,d]` | time MLP, `time_proj`, every `scale_shift_table` |
| `text_embed` | prompt | `embeds[512,4096]` → `ctx[512,d]` | `text_embedder` |
| `block_N_kv` | prompt | `ctx[512,d]` → `k2[H,512,hd]`, `v2[H,512,hd]` | block N's `attn2.to_k/to_v`, `norm_k` |
| `patch_embed` | row chunk | `patches[C,64]` → `h[C,d]` | `patch_embedding` |
| `block_N_qkv` | row chunk | `h[C,d]`, `mods[6,d]`, `cos/sin[C,64]` → `q,k,v[C,H,hd]` | `attn1.to_q/k/v`, `norm_q/k` |
| `attn_S*_q*` | query chunk | `q[Cq,H,hd]`, `k,v[S,H,hd]` → `o[Cq,d]` | none |
| `block_N_post` | row chunk | `h`, `o`, `mods`, `k2`, `v2` → `h'[C,d]` | `attn1.to_out`, `norm2`, `attn2.to_q/out`, `norm_q`, `ffn` |
| `head` | row chunk | `h[C,d]`, `head_mods[2,d]` → `patches[C,64]` | `proj_out` |
| `tae_f*_H*xW*` | latent chunk | `z[f,16,h,w]`, 9 memories → `rgb[4f,3,8h,8w]`, 9 memories | TAEHV |

`C` is the row chunk (1024 by default, 4096 on a desktop GPU). Every
per-row graph is independent of the clip size, so the 60 block graphs are
exported **once** per weight format and reused at every resolution; only
the weight-free attention graph and the TAEHV chunk graph are exported per
size bucket. A 480p clip runs one block as 32 `qkv` calls, 128 attention
calls (`Cq` = 256 queries against all 32,760 keys), and 32 `post` calls.

Why this cut and not another:

- **Attention is the only op that couples tokens**, so it is the only graph
  that depends on `S`. Chunking its queries bounds the score matrix to
  `H · Cq · S` floats (`12 · 256 · 32760 · 4 B = 400 MB` at 480p; 50 MB at
  the phone tier) without a flash kernel, which LiteRT does not have.
- **Weights live in the graph**, so one graph per block means one download
  per block (an HTTP range on the web, a file on a phone) and only the
  blocks in flight in memory, which is how fastwan-rs streams its
  checkpoint to the browser.
- **The row chunk is the FFN chunk** of fastwan-rs: the `[C, 8960]` hidden
  is the largest activation and never exceeds one chunk.
- **Cross-attention K/V are inputs**, computed once per prompt by the `kv`
  graphs and reused by all 3 steps and 32 chunks, exactly as `prepare_prompt`
  does in Rust.
- **Timestep conditioning is one tiny graph** instead of host code in four
  languages: the sinusoid → MLP → `time_proj` → `+ scale_shift_table` chain
  is a few mat-vecs per step. Its output is the `[6, d]` table per block the
  Rust kernels read.

What LiteRT cannot do and this port does not pretend to:

- **No VSA.** Video sparse attention needs a custom kernel. FastWan's
  checkpoints carry `to_gate_compress`; it is exported but unused, and every
  clip runs dense attention (fastwan-rs runs dense below 81 frames anyway,
  because short clips decay under VSA). Dense attention at 480p×81 is
  ~200 TFLOP per forward: a desktop-GPU job, minutes not seconds.
- **No f16 activations.** Graphs are exported with f32 activations; the
  weight format (`f32`, `f16`, `int8`, `int4` weight-only) is a quantizer
  pass over the exported graphs. LiteRT's GPU delegates may run f16
  internally (`precision: 'fp16'` on the web, `GpuOptions` on Android).
- **No causal / long-video modes** (Self-Forcing, MemRoPE) and no upscaler
  yet. They are host loops over the same block graphs plus a K/V cache; the
  cut above does not preclude them.

## Tiers

| tier | sizes | weights | where |
|---|---|---|---|
| desktop | up to 832×480×81 | f16 / int8 | Python runtime (`fastwan-litert generate`), the same graphs on a Linux / macOS / Windows GPU through LiteRT's delegates |
| web | 832×480×81 on a desktop GPU; 480×272×33 elsewhere | int4 / int8 | LiteRT.js, WebGPU (`jspi` for partitioned graphs) |
| phone | 480×272×33 and below | int4 | Android `CompiledModel` (GPU); iOS LiteRT (Metal) |

The phone tier is fastwan-rs's `MOBILE` tier: the tab or app has about 2 GB,
so 4-bit weights (0.88 GB), ≤ 33 frames, decode one latent chunk at a time.
The `plan` command in every runtime sizes a run before anything is allocated
(weights, K/V, the `[S, 3d]` q/k/v buffers, the attention score chunk, the
TAEHV peak) and refuses what will not fit.

## Verified, and how

The Python package carries the tests; CI runs them on CPU with a tiny model
(`WanConfig.tiny()`, the same numbers as fastwan-rs's).

| claim | evidence |
|---|---|
| the host math equals fastwan-rs | `tests/test_host.py`: sigmas `1.0 / 0.757 / 0.522`, RoPE split `(44, 42, 42)` and the identity at the origin, 480p = `(21, 60, 104)` latent and 32,760 tokens, the SplitMix64 noise moments; the TypeScript, Kotlin and Swift ports carry the same cases |
| the PyTorch DiT equals Diffusers' `WanTransformer3DModel` | `tests/test_parity.py`: one forward on tiny random weights, rel < 1e-4 (the same fixture fastwan-rs uses) |
| the graph cut equals the whole module | `tests/test_graphs.py`: the chunked graph pipeline (every graph through `ai-edge-litert`) against the uncut PyTorch forward, rel < 1e-4; TAEHV chunked through the graph = PyTorch one-shot |
| the pipeline runs end to end | `tests/test_pipeline.py`: 3 DMD steps + decode at tiny scale, finite output, deterministic per seed |
| the real checkpoint converts | `fastwan-litert export --model-dir <download>` (not run in CI: 1.3B parameters) |

Not yet verified: speed and output on a real GPU through any LiteRT
delegate, and the real-checkpoint clip. Those need a machine with a GPU and
the 2.6 GB download; `docs/runs/` is where their records go, as in
fastwan-rs.

## Layout

```
python/     fastwan_litert: config, geometry, schedule, rope, rng (host math);
            modules (PyTorch, the graph cut); weights (safetensors / synthetic);
            export (ai-edge-torch → .tflite + manifest); runtime (ai-edge-litert);
            pipeline; cli (export, generate, check, plan, serve)
web/        @fastwan/litert: the same host math in TypeScript, LiteRT.js pipeline, demo page
android/    :core (pure Kotlin host math + pipeline over a Graph interface, JVM tests);
            :app (LiteRT CompiledModel, Compose UI)
ios/        FastWanLiteRT Swift package (host math + pipeline), LiteRT backend, SwiftUI demo
docs/       this file; graphs.md is the manifest and tensor contract every runtime follows
scripts/    encode_prompt.py (UMT5-XXL embeddings), download.sh
```
