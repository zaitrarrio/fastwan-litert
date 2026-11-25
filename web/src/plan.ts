/**
 * Memory plan (fastwan-core `plan.rs`, for the graph cut): what a generation
 * holds at once, before anything is allocated. Hosts call it before `generate`
 * to refuse what will not fit (port of `python/fastwan_litert/plan.py`).
 */

import type { TaeConfig, WanConfig } from "./config.js";
import type { Geometry } from "./geometry.js";
import type { WeightFormat } from "./manifest.js";

export const BYTES_PER_WEIGHT: Readonly<Record<WeightFormat, number>> = { f32: 4.0, f16: 2.0, int8: 1.0, int4: 0.5 };

export interface Plan {
  tokens: number;
  chunk: number;
  qChunk: number;
  weightsTotal: number;
  weightsResident: number;
  crossKv: number;
  activations: number;
  attentionScratch: number;
  decodePeak: number;
  peak: number;
  budget: number;
  fits: boolean;
  notes: string[];
}

export function plan(
  cfg: WanConfig,
  tae: TaeConfig,
  g: Geometry,
  weights: WeightFormat,
  chunk: number,
  qChunk: number,
  taeFrames: number,
  budgetBytes: number,
  residentBlocks = 2,
): Plan {
  const s = g.tokens(cfg);
  const d = cfg.dim;
  const a = 4; // f32 activations
  const wb = BYTES_PER_WEIGHT[weights];
  const blockParams = 8 * (d * d + d) + 2 * (d * cfg.ffnDim) + cfg.ffnDim + d + 6 * d;
  const otherParams = cfg.params() - cfg.layers * blockParams;
  const weightsTotal = Math.trunc(cfg.layers * blockParams * wb + otherParams * 2);
  const weightsResident = Math.trunc(residentBlocks * blockParams * wb + otherParams * 2);
  const crossKv = cfg.layers * 2 * cfg.textLen * d * a;
  // h, q, k, v, o: [S, d] each; the latent, velocity and noise; rope tables.
  const activations = (5 * s * d + 3 * g.latentLen(cfg) + s * cfg.headDim) * a;
  const attentionScratch = cfg.heads * qChunk * s * a;
  const notes: string[] = [];
  // TAEHV peak: the last stage's 64-channel map at 8x resolution for the chunk's 4F frames,
  // plus input/output of the MemBlock convs (x, past, two temporaries, out).
  const [t, lh, lw] = g.latent();
  const f = Math.min(taeFrames, t);
  const nf = tae.widths;
  const px = lh * lw;
  const s0 = f * nf[0] * px;
  const s1 = f * tae.tgrow[0] * nf[1] * 4 * px;
  const s2 = f * tae.tgrow[0] * tae.tgrow[1] * nf[2] * 16 * px;
  const out = f * tae.tUpscale() * nf[3] * 64 * px;
  const decodePeak = Math.max(5 * s0, 5 * s1, 5 * s2, s2 + 2 * out) * a;
  const peak = Math.max(weightsResident + crossKv + activations + attentionScratch, weightsResident + decodePeak);
  if (residentBlocks < cfg.layers) {
    notes.push(`${residentBlocks} of ${cfg.layers} block graphs resident; the rest stream from disk / the network`);
  }
  if (s % chunk !== 0) {
    notes.push(`${s} rows run as ${Math.ceil(s / chunk)} chunks of ${chunk} (${chunk - (s % chunk)} padded rows)`);
  }
  return {
    tokens: s,
    chunk,
    qChunk,
    weightsTotal,
    weightsResident,
    crossKv,
    activations,
    attentionScratch,
    decodePeak,
    peak,
    budget: budgetBytes,
    fits: peak <= budgetBytes,
    notes,
  };
}

export function formatPlan(p: Plan): string {
  const mib = (b: number): string => (b / (1 << 20)).toFixed(1).padStart(9) + " MiB";
  const lines = [
    `tokens            ${p.tokens}`,
    `weights (all)     ${mib(p.weightsTotal)}`,
    `weights resident  ${mib(p.weightsResident)}`,
    `cross-attn K/V    ${mib(p.crossKv)}`,
    `DiT activations   ${mib(p.activations)} (row chunk ${p.chunk})`,
    `attention scratch ${mib(p.attentionScratch)} (query chunk ${p.qChunk})`,
    `decode peak       ${mib(p.decodePeak)}`,
    `peak              ${mib(p.peak)} of ${(p.budget / (1 << 20)).toFixed(0)} MiB  fits: ${p.fits}`,
  ];
  for (const n of p.notes) lines.push(`note: ${n}`);
  return lines.join("\n");
}
