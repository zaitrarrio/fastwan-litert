/**
 * The browser studio: pick an export (a base URL with `manifest.json`), a
 * prompt from a gallery of precomputed UMT5 embeddings, a size and a seed;
 * watch the block graphs download, the DMD steps tick and the frames play on
 * a canvas while the clip is still decoding. Dependency-free beyond
 * `@litertjs/core`.
 *
 * Phones (iPhone / iPad / Android) get fastwan-rs's MOBILE tier: only
 * `480x272x33` and only `int4` exports; WebGPU when `navigator.gpu` exists,
 * else Wasm.
 */

import { getWebGpuDevice, loadLiteRt, supportsFeature } from "@litertjs/core";

import {
  Geometry,
  GraphStore,
  MOBILE_480x272,
  Pipeline,
  UrlSource,
  float32Tensor,
  formatPlan,
  joinUrl,
  litertCompiler,
  loadManifest,
  parseSafetensors,
  plan,
  type Accelerator,
  type DownloadProgress,
  type Manifest,
  type PipelineEvent,
  type RgbaFrame,
} from "../src/index.js";

const $ = <T extends HTMLElement>(sel: string): T => {
  const el = document.querySelector<T>(sel);
  if (!el) throw new Error(`missing element ${sel}`);
  return el;
};

const ui = {
  info: $<HTMLElement>("#info"),
  base: $<HTMLInputElement>("#base"),
  galleryUrl: $<HTMLInputElement>("#galleryUrl"),
  load: $<HTMLButtonElement>("#load"),
  prompt: $<HTMLSelectElement>("#prompt"),
  size: $<HTMLSelectElement>("#size"),
  seed: $<HTMLInputElement>("#seed"),
  resident: $<HTMLInputElement>("#resident"),
  go: $<HTMLButtonElement>("#go"),
  stop: $<HTMLButtonElement>("#stop"),
  view: $<HTMLCanvasElement>("#view"),
  log: $<HTMLElement>("#log"),
  blocks: $<HTMLElement>("#blocks"),
  dl: $<HTMLElement>("#dl"),
  steps: $<HTMLElement>("#steps"),
  plan: $<HTMLElement>("#plan"),
};

const FPS = 16;
const GIB = 1024 ** 3;

/** fastwan-rs's MOBILE tier: phones get one size and 4-bit weights. */
export const isPhone = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
/** `webgpu` only when the browser has a GPU adapter (headless Chromium exposes `navigator.gpu` without one). */
let accelerator: Accelerator = "wasm";

async function detectAccelerator(): Promise<Accelerator> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown | null> } }).gpu;
  if (!gpu) return "wasm";
  try {
    return (await gpu.requestAdapter()) ? "webgpu" : "wasm";
  } catch {
    return "wasm";
  }
}

function log(msg: string, cls = ""): void {
  const line = document.createElement("div");
  line.textContent = msg;
  if (cls) line.className = cls;
  ui.log.append(line);
  ui.log.scrollTop = ui.log.scrollHeight;
}

// ---- gallery ------------------------------------------------------------------

interface GalleryEntry {
  name: string;
  file: string;
}

/** `index.json`: `["fox.safetensors", …]` or `[{"name": "fox", "file": "fox.safetensors"}, …]` or `{"prompts": [...]}`. */
async function loadGallery(url: string): Promise<GalleryEntry[]> {
  const res = await fetch(joinUrl(url, "index.json"));
  if (!res.ok) throw new Error(`${joinUrl(url, "index.json")}: HTTP ${res.status}`);
  const json: unknown = await res.json();
  const list = Array.isArray(json) ? json : (json as { prompts?: unknown }).prompts;
  if (!Array.isArray(list)) throw new Error("gallery index.json is not a list");
  return list.map((e): GalleryEntry => {
    if (typeof e === "string") return { name: e.replace(/\.safetensors$/, ""), file: e };
    const o = e as { name?: unknown; file?: unknown };
    const file = typeof o.file === "string" ? o.file : typeof o.name === "string" ? `${o.name}.safetensors` : "";
    if (!file) throw new Error("gallery entry without a file");
    return { name: typeof o.name === "string" ? o.name : file.replace(/\.safetensors$/, ""), file };
  });
}

async function loadEmbeds(url: string, m: Manifest): Promise<Float32Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const st = parseSafetensors(await res.arrayBuffer());
  const name = st.tensors.has("prompt_embeds") ? "prompt_embeds" : [...st.tensors.keys()][0];
  if (!name) throw new Error(`${url}: no tensors`);
  const t = float32Tensor(st, name);
  const want = [m.cfg.textLen, m.cfg.textDim];
  if (t.shape.length !== 2 || t.shape[0] !== want[0] || t.shape[1] !== want[1]) {
    throw new Error(`${url}: ${name} is [${t.shape.join(", ")}], this export wants [${want.join(", ")}]`);
  }
  return t.data;
}

// ---- download panel ----------------------------------------------------------------

class DownloadPanel {
  private readonly cells = new Map<string, { el: HTMLElement; bar: HTMLElement; total: number; loaded: Map<string, number> }>();
  private readonly fileToCell = new Map<string, string>();
  private bytes = 0;

  constructor(m: Manifest) {
    ui.blocks.replaceChildren();
    const sizeOf = (f: string): number => m.bytesOf(f) ?? 0;
    const shared = [m.graphs.time_mods, m.graphs.text_embed, m.graphs.patch_embed, m.graphs.head];
    this.add("shared", shared, shared.reduce((a, f) => a + sizeOf(f), 0));
    m.graphs.blocks.forEach((b, i) => {
      const files = [b.kv, b.qkv, b.post];
      const total = files.reduce((a, f) => a + sizeOf(f), 0) || b.bytes || 0;
      this.add(`block ${String(i).padStart(2, "0")}`, files, total);
    });
    this.add("attention", m.attention.map((a) => a.file), m.attention.reduce((a, e) => a + sizeOf(e.file), 0));
    this.add("TAEHV", m.taeChunks.map((t) => t.file), m.taeChunks.reduce((a, e) => a + sizeOf(e.file), 0));
    this.render();
  }

  private add(label: string, files: string[], total: number): void {
    const el = document.createElement("div");
    el.className = "blk";
    const bar = document.createElement("div");
    bar.className = "bar";
    const span = document.createElement("span");
    span.textContent = label;
    el.append(bar, span);
    el.title = files.join("\n");
    ui.blocks.append(el);
    this.cells.set(label, { el, bar, total, loaded: new Map() });
    for (const f of files) this.fileToCell.set(f, label);
  }

  progress(p: DownloadProgress): void {
    const label = this.fileToCell.get(p.file);
    if (!label) return;
    const c = this.cells.get(label);
    if (!c) return;
    const before = c.loaded.get(p.file) ?? 0;
    c.loaded.set(p.file, p.loaded);
    this.bytes += p.loaded - before;
    const sum = [...c.loaded.values()].reduce((a, b) => a + b, 0);
    const frac = c.total ? Math.min(1, sum / c.total) : p.total ? Math.min(1, p.loaded / p.total) : 0;
    c.bar.style.width = `${(frac * 100).toFixed(0)}%`;
    c.el.classList.toggle("done", frac >= 1);
    for (const [l, o] of this.cells) o.el.classList.toggle("live", l === label && frac < 1);
    this.render();
  }

  private render(): void {
    ui.dl.textContent = `${(this.bytes / 2 ** 20).toFixed(1)} MiB fetched this session`;
  }
}

// ---- player ---------------------------------------------------------------------------

class Player {
  private frames: ImageData[] = [];
  private timer: number | null = null;
  private i = 0;
  private readonly ctx: CanvasRenderingContext2D;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("no 2d context");
    this.ctx = ctx;
  }

  reset(width: number, height: number): void {
    this.stop();
    this.frames = [];
    this.i = 0;
    this.canvas.width = width;
    this.canvas.height = height;
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, width, height);
  }

  add(rgba: RgbaFrame[], width: number, height: number): void {
    if (this.canvas.width !== width || this.canvas.height !== height) this.reset(width, height);
    for (const px of rgba) this.frames.push(new ImageData(px, width, height));
    if (this.timer === null) {
      this.timer = window.setInterval(() => {
        if (!this.frames.length) return;
        this.ctx.putImageData(this.frames[this.i % this.frames.length], 0, 0);
        this.i++;
      }, 1000 / FPS);
    }
  }

  stop(): void {
    if (this.timer !== null) window.clearInterval(this.timer);
    this.timer = null;
  }
}

// ---- the studio -----------------------------------------------------------------------------

interface Loaded {
  manifest: Manifest;
  base: string;
  gallery: GalleryEntry[];
  galleryUrl: string;
}

let loaded: Loaded | null = null;
let litertReady: Promise<void> | null = null;
let running = false;
let stopRequested = false;
const player = new Player(ui.view);

function wasmDir(): string {
  const q = new URLSearchParams(location.search).get("wasm");
  const attr = document.querySelector<HTMLScriptElement>("script[data-wasm-dir]")?.dataset.wasmDir;
  return q ?? attr ?? "../node_modules/@litertjs/core/wasm/";
}

async function ensureLiteRt(): Promise<void> {
  if (!litertReady) {
    litertReady = (async () => {
      // JSPI lets WebGPU graphs fall back to Wasm per op instead of per graph.
      const jspi = accelerator === "webgpu" && (await supportsFeature("jspi").catch(() => false));
      await loadLiteRt(wasmDir(), { jspi });
      if (accelerator === "webgpu" && !getWebGpuDevice()) {
        log("LiteRT.js found no WebGPU device; running on Wasm", "err");
        accelerator = "wasm";
      }
      log(`LiteRT.js loaded (${accelerator}${jspi ? ", jspi" : ""})`);
    })().catch((e: unknown) => {
      litertReady = null;
      throw e;
    });
  }
  return litertReady;
}

function sizesFor(m: Manifest): Geometry[] {
  const all = m.sizes.map((s) => new Geometry(s.width, s.height, s.frames));
  return isPhone ? all.filter((g) => g.equals(MOBILE_480x272)) : all;
}

function budgetBytes(): number {
  if (isPhone) return 2 * GIB;
  const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  return (mem ?? 8) * GIB;
}

async function load(): Promise<void> {
  ui.go.disabled = true;
  ui.log.replaceChildren();
  ui.prompt.replaceChildren();
  ui.size.replaceChildren();
  ui.plan.textContent = "";
  const base = ui.base.value.trim();
  const galleryUrl = ui.galleryUrl.value.trim();
  try {
    const manifest = await loadManifest(base);
    ui.info.textContent = `${manifest.model} · ${manifest.weights} · ${manifest.cfg.layers} blocks · ${accelerator}${isPhone ? " · phone tier" : ""}`;
    if (isPhone && manifest.weights !== "int4") {
      throw new Error(`phones run int4 exports only; this one is ${manifest.weights} (export with --weights int4)`);
    }
    const sizes = sizesFor(manifest);
    if (!sizes.length) {
      throw new Error(
        isPhone
          ? `phones run 480x272x33 only and this export has no graphs for it (it has ${manifest.sizes.map((s) => `${s.width}x${s.height}x${s.frames}`).join(", ")})`
          : "this export lists no sizes",
      );
    }
    for (const g of sizes) {
      const o = document.createElement("option");
      o.value = o.textContent = g.toString();
      ui.size.append(o);
    }
    let gallery: GalleryEntry[] = [];
    try {
      gallery = await loadGallery(galleryUrl);
    } catch (e) {
      log(`no gallery: ${String(e instanceof Error ? e.message : e)}`, "err");
    }
    for (const g of gallery) {
      const o = document.createElement("option");
      o.value = g.file;
      o.textContent = g.name;
      ui.prompt.append(o);
    }
    if (!gallery.length) log("The gallery needs index.json listing prompt .safetensors (prompt_embeds [512, 4096]) — see web/README.md.");
    new DownloadPanel(manifest);
    loaded = { manifest, base, gallery, galleryUrl };
    showPlan();
    ui.go.disabled = !gallery.length;
    log(`loaded ${manifest.model} from ${base}: ${manifest.graphs.blocks.length} blocks, chunk ${manifest.chunk}, sizes ${sizes.join(" ")}`, "ok");
  } catch (e) {
    log(String(e instanceof Error ? e.message : e), "err");
  }
}

function currentPlan(): ReturnType<typeof plan> | null {
  if (!loaded) return null;
  const m = loaded.manifest;
  const g = Geometry.parse(ui.size.value);
  let qChunk = 0;
  let taeFrames = 0;
  try {
    qChunk = m.attentionFor(g.tokens(m.cfg)).q_chunk;
    const [, lh, lw] = g.latent();
    taeFrames = m.taeFor(lh, lw).frames;
  } catch (e) {
    log(String(e instanceof Error ? e.message : e), "err");
    return null;
  }
  const resident = Math.max(1, Number(ui.resident.value) || 4);
  return plan(m.cfg, m.tae, g, m.weights, m.chunk, qChunk, taeFrames, budgetBytes(), resident);
}

function showPlan(): void {
  const p = currentPlan();
  ui.plan.textContent = p ? formatPlan(p) : "";
  ui.plan.className = p && !p.fits ? "err" : "";
}

async function generate(): Promise<void> {
  if (!loaded || running) return;
  const { manifest: m, base, galleryUrl } = loaded;
  const g = Geometry.parse(ui.size.value);
  const seed = Number(ui.seed.value) || 0;
  const p = currentPlan();
  if (!p) return;
  if (!p.fits) {
    log(`refusing: the plan's peak ${(p.peak / 2 ** 20).toFixed(0)} MiB exceeds the ${(p.budget / 2 ** 30).toFixed(0)} GiB budget`, "err");
    return;
  }
  running = true;
  stopRequested = false;
  ui.go.disabled = true;
  ui.stop.disabled = false;
  ui.steps.replaceChildren();
  const [, lh, lw] = g.latent();
  player.reset(lw * m.tae.spatialScale(), lh * m.tae.spatialScale());
  const panel = new DownloadPanel(m);
  const store = new GraphStore(m, new UrlSource(base, m.files), litertCompiler({ accelerator }), {
    resident: Math.max(1, Number(ui.resident.value) || 4),
    onProgress: (pr) => panel.progress(pr),
  });
  const pipe = new Pipeline(store);
  const t0 = performance.now();
  const timing = (label: string, ms: number): void => {
    const row = document.createElement("div");
    const a = document.createElement("span");
    a.textContent = label;
    const b = document.createElement("span");
    b.textContent = ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms.toFixed(0)} ms`;
    row.append(a, b);
    ui.steps.append(row);
  };
  const on = (e: PipelineEvent): void => {
    if (stopRequested) throw new Error("stopped");
    if (e.type === "prepared") {
      timing(`prompt + noise (${e.tokens} tokens)`, e.ms);
    } else if (e.type === "step") {
      timing(`DMD step ${e.index + 1}/${e.total}`, e.ms);
    } else {
      player.add(e.rgba, e.width, e.height);
      log(`frames ${e.first}–${e.first + e.count - 1} (${e.width}x${e.height})`);
    }
  };
  try {
    await ensureLiteRt();
    log(`generating ${g} with seed ${seed} on ${accelerator}…`);
    const embeds = await loadEmbeds(joinUrl(galleryUrl, ui.prompt.value), m);
    const { video, timings } = await pipe.generate(embeds, g, seed, on);
    timing("TAEHV decode", timings.decodeMs);
    timing("total", performance.now() - t0);
    log(`done: ${video.frames} frames of ${video.width}x${video.height} in ${((performance.now() - t0) / 1000).toFixed(1)} s`, "ok");
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    log(msg === "stopped" ? "stopped" : `error: ${msg}`, msg === "stopped" ? "" : "err");
  } finally {
    await store.deleteAll().catch(() => undefined);
    running = false;
    ui.go.disabled = false;
    ui.stop.disabled = true;
  }
}

ui.load.addEventListener("click", () => void load());
ui.go.addEventListener("click", () => void generate());
ui.stop.addEventListener("click", () => {
  stopRequested = true;
});
ui.size.addEventListener("change", showPlan);
ui.resident.addEventListener("change", showPlan);

ui.info.textContent = isPhone ? "phone tier (480x272x33, int4)" : "…";
void (async () => {
  accelerator = await detectAccelerator();
  ui.info.textContent = `${accelerator}${isPhone ? " · phone tier (480x272x33, int4)" : ""}`;
  await load();
})();
