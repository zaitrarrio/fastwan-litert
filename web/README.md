# `@fastwan/litert` — FastWan 2.1 on LiteRT.js

The web runtime of fastwan-litert: the host math of fastwan-rs (schedule,
geometry, RoPE, SplitMix64 noise, patchify, the DMD step, TAEHV chunking) in
TypeScript, the graph pipeline of `docs/graphs.md` over
[LiteRT.js](https://www.npmjs.com/package/@litertjs/core) (`@litertjs/core`
2.6, WebGPU or Wasm), and a browser studio. ESM, strict TypeScript, no
dependency besides `@litertjs/core`.

```
src/config.ts     WanConfig, TaeConfig            (python: config.py)
src/geometry.ts   Geometry, FASTWAN_480P, MOBILE_480x272
src/schedule.ts   DmdSchedule (float32 training table, as the reference)
src/rope.ts       ropeTable: 3-axis RoPE, f64 angles → f32 cos/sin
src/rng.ts        Rng: SplitMix64 → Box–Muller, bit-exact with Rust/Python
src/host.ts       patchify / unpatchify / dmdStep / padRows / toRgba8
src/safetensors.ts a small safetensors reader (prompt galleries, the fixture)
src/manifest.ts   parseManifest / loadManifest: the contract, validated; no fallbacks
src/graphs.ts     Graph (inputs by name, outputs by position), LiteRtGraph,
                  GraphStore (on-demand fetch, LRU of block graphs, progress), UrlSource / MapSource
src/pipeline.ts   Pipeline: preparePrompt / forward / denoise / decode / generate + events
src/plan.ts       plan / formatPlan                (python: plan.py)
demo/             index.html + studio.ts: the browser studio
tests/            host.test.ts (vitest), the browser fixture check, a static server
```

## Build and test

```bash
cd web
npm install
npm run typecheck          # tsc --noEmit
npm run build              # tsc → dist/ (dist/src = the package, dist/demo, dist/tests)
npm test                   # vitest unit tests + the browser fixture check
```

`npm test` runs two things:

- `vitest run` — `tests/host.test.ts`, every case of `python/tests/test_host.py`
  (and `test_plan.py`): sigmas `1.0 / 0.757 / 0.522`, 480p = `(21, 60, 104)`
  latent and 32,760 tokens, the RoPE split `(44, 42, 42)` and the identity at
  the origin, SplitMix64 (`Rng(GOLDEN).nextU64() === 0xE220A8397B1DCDAF` then
  `0x6E789E6AA1B965F4`), the normal moments, the patchify index conventions,
  the float32 DMD step, the manifest validation, the memory plan.
- `node tests/run-fixture.mjs` — launches headless Chromium with Playwright
  (installed globally in this environment; the script does not call
  `playwright install`), serves `web/`, `fixtures/tiny` and
  `node_modules/@litertjs/core/wasm` from a local static server
  (`tests/serve.mjs`), loads the tiny export on the **Wasm** backend and runs
  `denoise` (seed 7, the fixture's `prompt_embeds`) and `decode`, then
  compares with `expected.safetensors` the way `fastwan-litert check` does:
  `rel = max|Δ| / max|ref|` on the latent (tolerance 1e-3 on CPU, 2e-2 on a
  GPU delegate) and `max |Δ|` on the frames (≤ 2/255). It prints the numbers
  and exits non-zero on failure. Measured here:

  ```
  fixture 32x48x9 seed 7 (18 tokens) on wasm: latent rel 2.94e-7 (max |Δ| 9.54e-7, tol 1e-3),
  frames 9/9 max |Δ| 1/255 (mean 0.059), 20 graph downloads, 0.4 s: OK
  ```

  The check runs with `resident: 1`, so the 20 downloads show the LRU
  evicting and re-fetching block graphs between steps, as a 30-block model
  would on a phone.

  Flags: `--accelerator webgpu` (needs a GPU and Chromium's
  `--enable-unsafe-webgpu`; headless here has no adapter, so the fixture is
  judged on Wasm), `--jspi` (the JSPI Wasm build, passes identically),
  `--threads` (the threaded build), `--timeout S`.

### Wasm variants and headless Chromium

`loadLiteRt(dir)` picks a file in the Wasm directory by feature detection:
`litert_wasm_internal.js` (relaxed SIMD, the default), `_compat_` (no relaxed
SIMD), `_jspi_` with `{jspi: true}`, `_threaded_` with `{threads: true}`.
Headless Chromium 158 runs the default and the JSPI build without any flag.
The threaded build needs `SharedArrayBuffer`, i.e. cross-origin isolation:
`tests/serve.mjs` sends `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` for that, but
`node tests/run-fixture.mjs --threads` did not complete in headless Chromium
here (the pthread workers never return), so the threaded build is not
verified and the default (single-threaded, relaxed-SIMD) build is what the
tests and the studio use. JSPI is only needed for WebGPU/Wasm *partitioning*
(per-op fallback when a graph has an op the WebGPU delegate lacks); without it
such a graph falls back to Wasm as a whole with a console warning. The studio
loads with `{jspi: true}` when the browser supports it and WebGPU is in use.

## Use the package

```ts
import { loadLiteRt } from "@litertjs/core";
import { Geometry, GraphStore, Pipeline, UrlSource, litertCompiler, loadManifest } from "@fastwan/litert";

await loadLiteRt("/wasm/");                       // the @litertjs/core wasm directory
const m = await loadManifest("/graphs/int4/");     // format must be fastwan-litert/1
const store = new GraphStore(m, new UrlSource("/graphs/int4/", m.files), litertCompiler({ accelerator: "webgpu" }), {
  resident: 4,                                     // block graphs kept alive (LRU)
  onProgress: (p) => console.log(p.file, p.loaded, p.total),
});
const pipe = new Pipeline(store);
const { video } = await pipe.generate(promptEmbeds /* Float32Array [512·4096] */, Geometry.parse("480x272x33"), 7, (e) => {
  if (e.type === "frames") ctx.putImageData(new ImageData(e.rgba[0], e.width, e.height), 0, 0);
});
```

Events are the contract's: `prepared {ms, tokens}`, one `step {index, total,
ms}` per DMD step, `frames {first, count, width, height, rgba}` per TAEHV
chunk, where `rgba` is one `Uint8ClampedArray` per frame (alpha 255), ready
for `ImageData`. `denoise` returns the latent `[16, T, Hl, Wl]`; `decode`
turns it into frames. Graph inputs are bound by name and checked against the
graph's signature (names and shapes); outputs come back by position
(`output_0`, `output_1`, …). A geometry the export has no graphs for, a
manifest of another format, or a mismatched signature throws with the
mismatch in the message; nothing falls back silently. `plan(...)` sizes a
run before anything is allocated.

## The studio (`demo/`)

```bash
npm run build
npm run serve                      # http://127.0.0.1:8765/demo/
```

`npm run serve` serves `web/` (so `demo/`, `dist/`, `node_modules/` are
reachable) and `../fixtures/`. The page takes a graph base URL (default
`./graphs/`, i.e. `web/demo/graphs/` — symlink or copy an export there, or
type another URL such as `/fixtures/tiny/`), a gallery URL (default
`./gallery/`), a size from the manifest's `sizes`, a seed and the number of
resident block graphs. It shows the memory plan, download progress per block
(from the manifest's `files` sizes), per-step timings and plays the frames on
a canvas at 16 fps as they arrive. It uses `accelerator: 'webgpu'` when
`navigator.gpu` exists, otherwise `wasm`.

**Gallery.** There is no text encoder in the browser (as in fastwan-rs's web
build). `gallery/index.json` lists prompt files, either `["fox.safetensors",
…]` or `[{"name": "a red fox…", "file": "fox.safetensors"}, …]`; each file is
a safetensors with `prompt_embeds` `[512, 4096]` f32 from
`fastwan-litert encode` / `scripts/encode_prompt.py`.

**Phone tier.** On iPhone / iPad / Android (user agent) the studio is
fastwan-rs's MOBILE tier: only `480x272x33` is offered (an export without it
is refused with the sizes it has), only `int4` exports are accepted, and the
plan budget is 2 GiB. Elsewhere the budget is `navigator.deviceMemory` GiB
(8 if unknown); a plan that does not fit is refused before any download.

### Hosting

- Serve `dist/`, `demo/` and the graphs statically. Block graphs are plain
  files (one per block), so any static host or CDN works; `files[*].bytes`
  in the manifest drives the progress bars and is checked after download.
- `@litertjs/core`'s `dist/index.js` imports `@litertjs/wasm-utils` as a bare
  specifier; both are mapped in `demo/index.html`'s import map. Without
  `node_modules` on the server, point the import map at jsDelivr
  (`https://cdn.jsdelivr.net/npm/@litertjs/core@2.6.0/dist/index.js`,
  `https://cdn.jsdelivr.net/npm/@litertjs/wasm-utils@2.6.0/dist/index.js`)
  and the `data-wasm-dir` attribute (or `?wasm=` query) at
  `https://cdn.jsdelivr.net/npm/@litertjs/core@2.6.0/wasm/`.
- The Wasm directory (`node_modules/@litertjs/core/wasm/`, ~45 MB for the four
  variants; only one is fetched) must be served statically or from the CDN.
- The threaded Wasm build needs COOP/COEP headers (see above). Nothing else
  needs special headers.

## What is not verified

- **No GPU here.** Everything was run on the Wasm (XNNPACK) backend in
  headless Chromium 158 on Linux. `accelerator: 'webgpu'`, `precision:
  'fp16'`, WebGPU/Wasm partitioning under JSPI, and the 2e-2 GPU tolerance of
  the contract are untested. The studio's WebGPU path is the same code with a
  different `CompileOptions.accelerator`.
- **The real model.** Only the tiny export (`fixtures/tiny`, 2 blocks, 18
  tokens) has run. Speed and memory at 480×272×33 / 832×480×81 with the
  1.3B int4/int8 graphs (streaming 30 blocks, the 400 MB attention score
  chunk) are from `plan`, not measured.
- **Phones.** The phone tier is a user-agent check and a budget; no device
  was available.
- **The threaded Wasm build** (above).
