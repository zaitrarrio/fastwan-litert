/**
 * `manifest.json`: parsing and validation against `docs/graphs.md`. A manifest
 * of another `format`, or a geometry this export has no graphs for, is an
 * error with the mismatch in its message; nothing falls back silently.
 */

import { TaeConfig, WanConfig, type TaeConfigJson, type WanConfigJson } from "./config.js";
import { DmdSchedule, type DmdScheduleJson } from "./schedule.js";
import type { GeometryJson } from "./geometry.js";

export const FORMAT = "fastwan-litert/1";

export type WeightFormat = "f32" | "f16" | "int8" | "int4";

export interface BlockGraphs {
  qkv: string;
  post: string;
  kv: string;
  bytes?: number;
}

export interface GraphFiles {
  time_mods: string;
  text_embed: string;
  patch_embed: string;
  head: string;
  blocks: BlockGraphs[];
}

export interface AttentionEntry {
  tokens: number;
  q_chunk: number;
  heads: number;
  file: string;
}

export interface TaeChunkEntry {
  frames: number;
  latent_h: number;
  latent_w: number;
  file: string;
}

export interface FileInfo {
  bytes: number;
  sha256?: string;
}

export interface ManifestJson {
  format: string;
  model: string;
  config: WanConfigJson;
  tae: TaeConfigJson;
  schedule: DmdScheduleJson;
  weights: WeightFormat;
  chunk: number;
  graphs: GraphFiles;
  attention: AttentionEntry[];
  tae_chunks: TaeChunkEntry[];
  sizes: GeometryJson[];
  files?: Record<string, FileInfo>;
}

function fail(path: string, msg: string): never {
  throw new Error(`manifest.json: ${path} ${msg}`);
}

function obj(v: unknown, path: string): Record<string, unknown> {
  if (typeof v !== "object" || v === null || Array.isArray(v)) fail(path, "is not an object");
  return v as Record<string, unknown>;
}

function num(v: unknown, path: string): number {
  if (typeof v !== "number" || !Number.isFinite(v)) fail(path, "is not a number");
  return v;
}

function int(v: unknown, path: string): number {
  const n = num(v, path);
  if (!Number.isInteger(n) || n < 0) fail(path, `is ${n}, not a non-negative integer`);
  return n;
}

function str(v: unknown, path: string): string {
  if (typeof v !== "string" || v.length === 0) fail(path, "is not a non-empty string");
  return v;
}

function bool(v: unknown, path: string): boolean {
  if (typeof v !== "boolean") fail(path, "is not a boolean");
  return v;
}

function arr(v: unknown, path: string): unknown[] {
  if (!Array.isArray(v)) fail(path, "is not an array");
  return v;
}

function numArray(v: unknown, path: string): number[] {
  return arr(v, path).map((x, i) => num(x, `${path}[${i}]`));
}

const WEIGHT_FORMATS: readonly WeightFormat[] = ["f32", "f16", "int8", "int4"];

export class Manifest {
  readonly cfg: WanConfig;
  readonly tae: TaeConfig;
  readonly schedule: DmdSchedule;
  readonly chunk: number;
  readonly weights: WeightFormat;
  readonly model: string;
  readonly graphs: GraphFiles;
  readonly attention: readonly AttentionEntry[];
  readonly taeChunks: readonly TaeChunkEntry[];
  readonly sizes: readonly GeometryJson[];
  readonly files: Readonly<Record<string, FileInfo>>;

  constructor(readonly raw: ManifestJson) {
    this.cfg = WanConfig.fromJson(raw.config);
    this.tae = TaeConfig.fromJson(raw.tae);
    this.schedule = new DmdSchedule(raw.schedule.timesteps, raw.schedule.sigmas);
    this.chunk = raw.chunk;
    this.weights = raw.weights;
    this.model = raw.model;
    this.graphs = raw.graphs;
    this.attention = raw.attention;
    this.taeChunks = raw.tae_chunks;
    this.sizes = raw.sizes;
    this.files = raw.files ?? {};
    if (this.graphs.blocks.length !== this.cfg.layers) {
      fail("graphs.blocks", `lists ${this.graphs.blocks.length} blocks but config.layers is ${this.cfg.layers}`);
    }
  }

  /** The attention graph of a clip with exactly `tokens` tokens. */
  attentionFor(tokens: number): AttentionEntry {
    for (const a of this.attention) if (a.tokens === tokens) return a;
    const have = [...new Set(this.attention.map((a) => a.tokens))].sort((x, y) => x - y);
    throw new Error(
      `this export has no attention graph for ${tokens} tokens (it has [${have.join(", ")}]); re-export with --size`,
    );
  }

  /** The TAEHV chunk graph of latent size `latent_h × latent_w`. */
  taeFor(latentH: number, latentW: number): TaeChunkEntry {
    for (const t of this.taeChunks) if (t.latent_h === latentH && t.latent_w === latentW) return t;
    const have = [...new Set(this.taeChunks.map((t) => `${t.latent_h}x${t.latent_w}`))].sort();
    throw new Error(
      `this export has no TAEHV graph for latent ${latentH}x${latentW} (it has [${have.join(", ")}]); re-export with --size`,
    );
  }

  /** Byte size of a graph file when the manifest records it. */
  bytesOf(file: string): number | undefined {
    return this.files[file]?.bytes;
  }

  /** Every graph file this manifest names, in download order (fixed graphs, blocks, attention, TAEHV). */
  allFiles(): string[] {
    const g = this.graphs;
    const out = [g.time_mods, g.text_embed, g.patch_embed, g.head];
    for (const b of g.blocks) out.push(b.kv, b.qkv, b.post);
    for (const a of this.attention) out.push(a.file);
    for (const t of this.taeChunks) out.push(t.file);
    return out;
  }
}

/** Validate a parsed `manifest.json` (any JSON value) into a `Manifest`. */
export function parseManifest(json: unknown): Manifest {
  const m = obj(json, "root");
  const format = m.format;
  if (format !== FORMAT) {
    throw new Error(`manifest.json: format ${JSON.stringify(format)}, expected ${JSON.stringify(FORMAT)}`);
  }
  const c = obj(m.config, "config");
  const config: WanConfigJson = {
    patch: int(c.patch, "config.patch"),
    heads: int(c.heads, "config.heads"),
    head_dim: int(c.head_dim, "config.head_dim"),
    in_channels: int(c.in_channels, "config.in_channels"),
    out_channels: int(c.out_channels, "config.out_channels"),
    text_dim: int(c.text_dim, "config.text_dim"),
    text_len: int(c.text_len, "config.text_len"),
    freq_dim: int(c.freq_dim, "config.freq_dim"),
    ffn_dim: int(c.ffn_dim, "config.ffn_dim"),
    layers: int(c.layers, "config.layers"),
    eps: num(c.eps, "config.eps"),
    rope_theta: num(c.rope_theta, "config.rope_theta"),
  };
  if (c.rope_max_seq_len !== undefined) config.rope_max_seq_len = int(c.rope_max_seq_len, "config.rope_max_seq_len");
  const t = obj(m.tae, "tae");
  const tae: TaeConfigJson = {
    latent_channels: int(t.latent_channels, "tae.latent_channels"),
    widths: numArray(t.widths, "tae.widths"),
    tgrow: numArray(t.tgrow, "tae.tgrow"),
    mem_blocks: int(t.mem_blocks, "tae.mem_blocks"),
    patch: int(t.patch, "tae.patch"),
    denormalize: bool(t.denormalize, "tae.denormalize"),
  };
  const s = obj(m.schedule, "schedule");
  const schedule: DmdScheduleJson = {
    timesteps: numArray(s.timesteps, "schedule.timesteps"),
    sigmas: numArray(s.sigmas, "schedule.sigmas"),
  };
  const weights = str(m.weights, "weights");
  if (!(WEIGHT_FORMATS as readonly string[]).includes(weights)) {
    fail("weights", `is ${JSON.stringify(weights)}, not one of ${WEIGHT_FORMATS.join("/")}`);
  }
  const chunk = int(m.chunk, "chunk");
  if (chunk === 0) fail("chunk", "is 0");
  const g = obj(m.graphs, "graphs");
  const blocks = arr(g.blocks, "graphs.blocks").map((b, i): BlockGraphs => {
    const o = obj(b, `graphs.blocks[${i}]`);
    const out: BlockGraphs = {
      qkv: str(o.qkv, `graphs.blocks[${i}].qkv`),
      post: str(o.post, `graphs.blocks[${i}].post`),
      kv: str(o.kv, `graphs.blocks[${i}].kv`),
    };
    if (o.bytes !== undefined) out.bytes = int(o.bytes, `graphs.blocks[${i}].bytes`);
    return out;
  });
  const graphs: GraphFiles = {
    time_mods: str(g.time_mods, "graphs.time_mods"),
    text_embed: str(g.text_embed, "graphs.text_embed"),
    patch_embed: str(g.patch_embed, "graphs.patch_embed"),
    head: str(g.head, "graphs.head"),
    blocks,
  };
  const attention = arr(m.attention, "attention").map((a, i): AttentionEntry => {
    const o = obj(a, `attention[${i}]`);
    return {
      tokens: int(o.tokens, `attention[${i}].tokens`),
      q_chunk: int(o.q_chunk, `attention[${i}].q_chunk`),
      heads: int(o.heads, `attention[${i}].heads`),
      file: str(o.file, `attention[${i}].file`),
    };
  });
  const tae_chunks = arr(m.tae_chunks, "tae_chunks").map((a, i): TaeChunkEntry => {
    const o = obj(a, `tae_chunks[${i}]`);
    return {
      frames: int(o.frames, `tae_chunks[${i}].frames`),
      latent_h: int(o.latent_h, `tae_chunks[${i}].latent_h`),
      latent_w: int(o.latent_w, `tae_chunks[${i}].latent_w`),
      file: str(o.file, `tae_chunks[${i}].file`),
    };
  });
  const sizes = arr(m.sizes, "sizes").map((a, i): GeometryJson => {
    const o = obj(a, `sizes[${i}]`);
    return {
      width: int(o.width, `sizes[${i}].width`),
      height: int(o.height, `sizes[${i}].height`),
      frames: int(o.frames, `sizes[${i}].frames`),
    };
  });
  const files: Record<string, FileInfo> = {};
  if (m.files !== undefined) {
    for (const [name, v] of Object.entries(obj(m.files, "files"))) {
      const o = obj(v, `files[${name}]`);
      const info: FileInfo = { bytes: int(o.bytes, `files[${name}].bytes`) };
      if (o.sha256 !== undefined) info.sha256 = str(o.sha256, `files[${name}].sha256`);
      files[name] = info;
    }
  }
  return new Manifest({
    format: FORMAT,
    model: str(m.model, "model"),
    config,
    tae,
    schedule,
    weights: weights as WeightFormat,
    chunk,
    graphs,
    attention,
    tae_chunks,
    sizes,
    files,
  });
}

/** Fetch and parse `<baseUrl>manifest.json`. */
export async function loadManifest(baseUrl: string): Promise<Manifest> {
  const url = joinUrl(baseUrl, "manifest.json");
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return parseManifest(await res.json());
}

export function joinUrl(base: string, file: string): string {
  return base.endsWith("/") ? base + file : `${base}/${file}`;
}
