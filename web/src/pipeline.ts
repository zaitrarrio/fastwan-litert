/**
 * Prompt embeddings → DMD denoise → TAEHV decode → RGBA frames, driving the
 * graphs exactly as `docs/graphs.md`'s host loop says (port of
 * `python/fastwan_litert/pipeline.py`).
 */

import type { WanConfig } from "./config.js";
import type { Geometry } from "./geometry.js";
import type { Graph, GraphInput, GraphStore } from "./graphs.js";
import type { Manifest, AttentionEntry } from "./manifest.js";
import { dmdStep, padRows, patchify, toRgba8, unpatchify, type RgbaFrame } from "./host.js";
import { Rng } from "./rng.js";
import { ropeTable, type RopeTable } from "./rope.js";

export interface PreparedEvent {
  type: "prepared";
  ms: number;
  tokens: number;
}

export interface StepEvent {
  type: "step";
  index: number;
  total: number;
  ms: number;
}

export interface FramesEvent {
  type: "frames";
  /** Index of the first frame of this batch in the clip. */
  first: number;
  count: number;
  width: number;
  height: number;
  /** One RGBA buffer per frame (`width·height·4`, alpha 255): `new ImageData(rgba[i], width, height)`. */
  rgba: RgbaFrame[];
}

export type PipelineEvent = PreparedEvent | StepEvent | FramesEvent;
export type OnEvent = (e: PipelineEvent) => void;

export interface Timings {
  prepareMs: number;
  stepsMs: number[];
  decodeMs: number;
}

export function newTimings(): Timings {
  return { prepareMs: 0, stepsMs: [], decodeMs: 0 };
}

/** A latent `[channels, frames, height, width]` row-major. */
export interface Latent {
  data: Float32Array;
  channels: number;
  frames: number;
  height: number;
  width: number;
}

export interface Video {
  frames: number;
  width: number;
  height: number;
  rgba: RgbaFrame[];
}

/** Per-prompt cross-attention K/V of every block, each `[H, L, hd]`. */
export class Prompt {
  constructor(readonly kv: readonly (readonly [Float32Array, Float32Array])[]) {}
}

/** A per-row input of `rows` rows, each of shape `rowShape`. */
interface RowInput {
  data: Float32Array;
  rowShape: readonly number[];
}

function prod(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

function now(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

export class Pipeline {
  readonly m: Manifest;
  readonly cfg: WanConfig;

  constructor(readonly store: GraphStore) {
    this.m = store.m;
    this.cfg = this.m.cfg;
  }

  // ---- prompt -----------------------------------------------------------------

  /** `text_embed` once, then every block's `kv` graph. `embeds` is `[text_len, text_dim]`. */
  async preparePrompt(embeds: Float32Array): Promise<Prompt> {
    const c = this.cfg;
    if (embeds.length !== c.textLen * c.textDim) {
      throw new Error(`prompt embeds have ${embeds.length} values, expected [${c.textLen}, ${c.textDim}]`);
    }
    const textEmbed = await this.store.fixed(this.m.graphs.text_embed);
    const [ctx] = await textEmbed.run({ embeds: { data: embeds, shape: [c.textLen, c.textDim] } });
    const kv: (readonly [Float32Array, Float32Array])[] = [];
    for (const b of this.m.graphs.blocks) {
      const g = await this.store.block(b.kv);
      const [k2, v2] = await g.run({ ctx: { data: ctx, shape: [c.textLen, c.dim] } });
      kv.push([k2, v2]);
    }
    return new Prompt(kv);
  }

  // ---- one DiT forward --------------------------------------------------------

  /**
   * Run a per-row graph over row chunks of the inputs (zero-padded to `chunk`);
   * returns the outputs with the padded rows dropped, concatenated along rows.
   */
  private async rows(
    graph: Graph,
    chunk: number,
    nReal: number,
    rowInputs: Record<string, RowInput>,
    fixed: Record<string, GraphInput>,
  ): Promise<Float32Array[]> {
    const padded: Record<string, { data: Float32Array; rowSize: number; shape: number[] }> = {};
    let full = 0;
    for (const [name, v] of Object.entries(rowInputs)) {
      const rowSize = prod(v.rowShape);
      const [p] = padRows(v.data, nReal, rowSize, chunk);
      padded[name] = { data: p, rowSize, shape: [chunk, ...v.rowShape] };
      full = p.length / rowSize;
    }
    const outs: Float32Array[][] = [];
    for (let r = 0; r < full; r += chunk) {
      const inputs: Record<string, GraphInput> = { ...fixed };
      for (const [name, p] of Object.entries(padded)) {
        inputs[name] = { data: p.data.subarray(r * p.rowSize, (r + chunk) * p.rowSize), shape: p.shape };
      }
      outs.push(await graph.run(inputs));
    }
    const nOut = outs[0].length;
    const result: Float32Array[] = [];
    for (let i = 0; i < nOut; i++) {
      const rowSize = outs[0][i].length / chunk;
      if (!Number.isInteger(rowSize)) {
        throw new Error(`${graph.file}: output ${i} has ${outs[0][i].length} values, not a multiple of ${chunk} rows`);
      }
      const all = new Float32Array(nReal * rowSize);
      let off = 0;
      for (const o of outs) {
        const take = Math.min(o[i].length, all.length - off);
        if (take <= 0) break;
        all.set(o[i].subarray(0, take), off);
        off += take;
      }
      result.push(all);
    }
    return result;
  }

  /** One DiT forward at timestep `t`: the velocity `[out_channels, T, Hl, Wl]`. */
  async forward(x: Latent, prompt: Prompt, t: number, rope: RopeTable, attn: AttentionEntry): Promise<Float32Array> {
    const c = this.cfg;
    const C = this.m.chunk;
    const { channels: ch, frames: tt, height: hh, width: ww } = x;
    const s = tt * (hh / c.patch) * (ww / c.patch);
    if (rope.tokens !== s) throw new Error(`rope table has ${rope.tokens} tokens, the clip has ${s}`);
    const d = c.dim;
    const hd = c.headDim;
    const H = c.heads;

    const timeMods = await this.store.fixed(this.m.graphs.time_mods);
    const [blockMods, headMods] = await timeMods.run({ t: { data: Float32Array.of(t), shape: [1] } });
    if (blockMods.length !== c.layers * 6 * d) {
      throw new Error(`time_mods: block_mods has ${blockMods.length} values, expected [${c.layers}, 6, ${d}]`);
    }

    const patches = patchify(x.data, ch, tt, hh, ww, c.patch);
    const patchEmbed = await this.store.fixed(this.m.graphs.patch_embed);
    let [h] = await this.rows(patchEmbed, C, s, { patches: { data: patches, rowShape: [c.patchFeaturesIn()] } }, {});

    const attnGraph = await this.store.fixed(attn.file);
    const cq = attn.q_chunk;
    const ropeShape = [rope.width];
    for (let l = 0; l < this.m.graphs.blocks.length; l++) {
      const b = this.m.graphs.blocks[l];
      const mods: GraphInput = { data: blockMods.subarray(l * 6 * d, (l + 1) * 6 * d), shape: [6, d] };
      const qkv = await this.store.block(b.qkv);
      const [q, k, v] = await this.rows(
        qkv,
        C,
        s,
        {
          h: { data: h, rowShape: [d] },
          cos: { data: rope.cos, rowShape: ropeShape },
          sin: { data: rope.sin, rowShape: ropeShape },
        },
        { mods },
      );
      const [o] = await this.rows(
        attnGraph,
        cq,
        s,
        { q: { data: q, rowShape: [H, hd] } },
        { k: { data: k, shape: [s, H, hd] }, v: { data: v, shape: [s, H, hd] } },
      );
      const [k2, v2] = prompt.kv[l];
      const post = await this.store.block(b.post);
      [h] = await this.rows(
        post,
        C,
        s,
        { h: { data: h, rowShape: [d] }, o: { data: o, rowShape: [d] } },
        { mods, k2: { data: k2, shape: [H, c.textLen, hd] }, v2: { data: v2, shape: [H, c.textLen, hd] } },
      );
    }
    const head = await this.store.fixed(this.m.graphs.head);
    const [rows] = await this.rows(head, C, s, { h: { data: h, rowShape: [d] } }, { mods: { data: headMods, shape: [2, d] } });
    return unpatchify(rows, c.outChannels, tt, hh, ww, c.patch);
  }

  // ---- denoise ------------------------------------------------------------------

  async denoise(embeds: Float32Array, g: Geometry, seed: number | bigint, on?: OnEvent, timings?: Timings): Promise<Latent> {
    const c = this.cfg;
    g.validate(c);
    const tm = timings ?? newTimings();
    const t0 = now();
    const attn = this.m.attentionFor(g.tokens(c));
    const prompt = await this.preparePrompt(embeds);
    const rope = ropeTable(c, g.grid(c), 0);
    const [lt, lh, lw] = g.latent();
    const rng = new Rng(seed);
    let x: Latent = { data: rng.normal(c.inChannels * lt * lh * lw), channels: c.inChannels, frames: lt, height: lh, width: lw };
    tm.prepareMs = now() - t0;
    on?.({ type: "prepared", ms: tm.prepareMs, tokens: g.tokens(c) });
    const sched = this.m.schedule;
    for (let i = 0; i < sched.length; i++) {
      const ts = now();
      const v = await this.forward(x, prompt, sched.timesteps[i], rope, attn);
      const [sigma, nxt] = sched.coeffs(i);
      const noise = nxt !== null ? rng.normal(x.data.length) : null;
      x = { ...x, data: dmdStep(x.data, v, sigma, nxt, noise) };
      tm.stepsMs.push(now() - ts);
      on?.({ type: "step", index: i, total: sched.length, ms: tm.stepsMs[tm.stepsMs.length - 1] });
    }
    return x;
  }

  // ---- decode ---------------------------------------------------------------------

  async decode(latent: Latent, on?: OnEvent, timings?: Timings): Promise<Video> {
    const tcfg = this.m.tae;
    const tm = timings ?? newTimings();
    const t0 = now();
    const { channels: c, frames: t, height: h, width: w } = latent;
    if (c !== tcfg.latentChannels) throw new Error(`latent has ${c} channels, TAEHV wants ${tcfg.latentChannels}`);
    const entry = this.m.taeFor(h, w);
    const graph = await this.store.fixed(entry.file);
    const f = entry.frames;
    const nmem = 3 * tcfg.memBlocks;
    const memShape = (i: number): number[] => {
      const s = Math.floor(i / tcfg.memBlocks);
      return [1, tcfg.widths[s], h << s, w << s];
    };
    let mems: Float32Array[] = [];
    for (let i = 0; i < nmem; i++) mems.push(new Float32Array(prod(memShape(i))));
    const up = tcfg.tUpscale();
    let toTrim = tcfg.framesToTrim();
    const scale = tcfg.spatialScale();
    const pxH = scale * h;
    const pxW = scale * w;
    const frameSize = 3 * pxH * pxW;
    const frames: RgbaFrame[] = [];
    let total = 0;
    const plane = h * w;
    for (let f0 = 0; f0 < t; f0 += f) {
      const n = Math.min(f, t - f0);
      const z = new Float32Array(f * c * plane);
      for (let fi = 0; fi < n; fi++) {
        for (let ci = 0; ci < c; ci++) {
          const src = (ci * t + f0 + fi) * plane;
          z.set(latent.data.subarray(src, src + plane), (fi * c + ci) * plane);
        }
      }
      const inputs: Record<string, GraphInput> = { z: { data: z, shape: [f, c, h, w] } };
      for (let i = 0; i < nmem; i++) inputs[`mem_${i}`] = { data: mems[i], shape: memShape(i) };
      const out = await graph.run(inputs);
      if (out.length !== 1 + nmem) throw new Error(`${entry.file}: ${out.length} outputs, expected rgb + ${nmem} memories`);
      if (out[0].length !== f * up * frameSize) {
        throw new Error(`${entry.file}: rgb has ${out[0].length} values, expected [${f * up}, 3, ${pxH}, ${pxW}]`);
      }
      mems = out.slice(1);
      const real = n * up;
      const skip = Math.min(toTrim, real);
      toTrim -= skip;
      const count = real - skip;
      if (count > 0) {
        const rgb = out[0].subarray(skip * frameSize, real * frameSize);
        const rgba = toRgba8(rgb, count, pxH, pxW);
        frames.push(...rgba);
        on?.({ type: "frames", first: total, count, width: pxW, height: pxH, rgba });
        total += count;
      }
    }
    tm.decodeMs = now() - t0;
    return { frames: total, width: pxW, height: pxH, rgba: frames };
  }

  async generate(embeds: Float32Array, g: Geometry, seed: number | bigint, on?: OnEvent): Promise<{ video: Video; timings: Timings }> {
    const timings = newTimings();
    const latent = await this.denoise(embeds, g, seed, on, timings);
    const video = await this.decode(latent, on, timings);
    return { video, timings };
  }
}
