/**
 * The browser side of the fixture check: runs the tiny export through the
 * pipeline on the chosen accelerator and compares with `expected.safetensors`
 * the way `fastwan-litert check` does (`rel = max|Δ| / max|ref|` on the
 * latent, `max |Δ|` on the frames). Driven by `run-fixture.mjs` through
 * Playwright, or by hand from `fixture.html`.
 */

import { loadLiteRt } from "@litertjs/core";

import {
  Geometry,
  GraphStore,
  Pipeline,
  UrlSource,
  float32Tensor,
  litertCompiler,
  loadManifest,
  parseSafetensors,
  uint8Tensor,
  type Accelerator,
  type PipelineEvent,
} from "../src/index.js";

export interface FixtureOptions {
  /** Base URL of the export (`manifest.json`, graphs, `expected.safetensors`). */
  base: string;
  /** Directory of the LiteRT.js Wasm files. */
  wasmDir: string;
  accelerator: Accelerator;
  jspi?: boolean;
  threads?: boolean;
  /** Block graphs kept resident; 1 exercises the LRU eviction on the tiny model. */
  resident?: number;
}

export interface FixtureResult {
  ok: boolean;
  accelerator: Accelerator;
  geometry: string;
  seed: number;
  tokens: number;
  latent: { rel: number; maxAbs: number; refMax: number; tol: number };
  frames: { maxDelta: number; meanDelta: number; count: number; width: number; height: number; expected: number };
  timings: { prepareMs: number; stepsMs: number[]; decodeMs: number };
  downloads: number;
  events: string[];
}

export async function runFixture(o: FixtureOptions): Promise<FixtureResult> {
  const events: string[] = [];
  await loadLiteRt(o.wasmDir, { jspi: o.jspi ?? false, threads: o.threads ?? false });
  const m = await loadManifest(o.base);
  let downloads = 0;
  const store = new GraphStore(m, new UrlSource(o.base, m.files), litertCompiler({ accelerator: o.accelerator }), {
    resident: o.resident ?? 1,
    onProgress: (p) => {
      if (p.total && p.loaded === p.total) downloads++;
    },
  });
  const pipe = new Pipeline(store);

  const res = await fetch(`${o.base}expected.safetensors`);
  if (!res.ok) throw new Error(`expected.safetensors: HTTP ${res.status}`);
  const exp = parseSafetensors(await res.arrayBuffer());
  const g = Geometry.parse(exp.metadata.geometry ?? m.sizes.map((s) => `${s.width}x${s.height}x${s.frames}`)[0]);
  const seed = Number(exp.metadata.seed ?? "7");
  const embeds = float32Tensor(exp, "prompt_embeds");
  const refLatent = float32Tensor(exp, "latent");
  const refFrames = uint8Tensor(exp, "frames");

  const on = (e: PipelineEvent): void => {
    if (e.type === "prepared") events.push(`prepared ${e.tokens} tokens in ${e.ms.toFixed(1)} ms`);
    else if (e.type === "step") events.push(`step ${e.index + 1}/${e.total}: ${e.ms.toFixed(1)} ms`);
    else events.push(`frames ${e.first}..${e.first + e.count - 1} ${e.width}x${e.height}`);
  };
  const timings = { prepareMs: 0, stepsMs: [] as number[], decodeMs: 0 };
  const latent = await pipe.denoise(embeds.data, g, seed, on, timings);

  if (latent.data.length !== refLatent.data.length) {
    throw new Error(`latent has ${latent.data.length} values, expected [${refLatent.shape.join(", ")}]`);
  }
  let maxAbs = 0;
  let refMax = 0;
  for (let i = 0; i < latent.data.length; i++) {
    maxAbs = Math.max(maxAbs, Math.abs(latent.data[i] - refLatent.data[i]));
    refMax = Math.max(refMax, Math.abs(refLatent.data[i]));
  }
  const rel = maxAbs / refMax;

  const video = await pipe.decode(latent, on, timings);
  const [nf, fh, fw, fc] = refFrames.shape;
  if (fc !== 3) throw new Error(`expected.frames has ${fc} channels`);
  let maxDelta = 0;
  let sumDelta = 0;
  let n = 0;
  if (video.frames === nf && video.height === fh && video.width === fw) {
    for (let f = 0; f < nf; f++) {
      const px = video.rgba[f];
      for (let p = 0; p < fh * fw; p++) {
        for (let c = 0; c < 3; c++) {
          const d = Math.abs(px[p * 4 + c] - refFrames.data[(f * fh * fw + p) * 3 + c]);
          maxDelta = Math.max(maxDelta, d);
          sumDelta += d;
          n++;
        }
        if (px[p * 4 + 3] !== 255) throw new Error("alpha is not 255");
      }
    }
  } else {
    maxDelta = 255;
  }
  const tol = o.accelerator === "wasm" ? 1e-3 : 2e-2;
  const ok = rel <= tol && video.frames === nf && maxDelta <= 2;
  await store.deleteAll();
  return {
    ok,
    accelerator: o.accelerator,
    geometry: g.toString(),
    seed,
    tokens: g.tokens(m.cfg),
    latent: { rel, maxAbs, refMax, tol },
    frames: { maxDelta, meanDelta: n ? sumDelta / n : NaN, count: video.frames, width: video.width, height: video.height, expected: nf },
    timings,
    downloads,
    events,
  };
}

declare global {
  interface Window {
    runFixture: typeof runFixture;
  }
}

window.runFixture = runFixture;
