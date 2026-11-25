/**
 * Architecture constants, as fastwan-rs `fastwan-core/src/config.rs` records
 * them from each checkpoint's Diffusers `config.json` (port of
 * `python/fastwan_litert/config.py`).
 */

export interface WanConfigJson {
  patch: number;
  heads: number;
  head_dim: number;
  in_channels: number;
  out_channels: number;
  text_dim: number;
  text_len: number;
  freq_dim: number;
  ffn_dim: number;
  layers: number;
  eps: number;
  rope_max_seq_len?: number;
  rope_theta: number;
}

/** The Wan 2.1 DiT. Batch is always 1: the DMD-distilled FastWan models run without CFG. */
export class WanConfig {
  constructor(
    readonly patch: number,
    readonly heads: number,
    readonly headDim: number,
    readonly inChannels: number,
    readonly outChannels: number,
    readonly textDim: number,
    readonly textLen: number,
    readonly freqDim: number,
    readonly ffnDim: number,
    readonly layers: number,
    readonly eps: number,
    readonly ropeMaxSeqLen: number,
    readonly ropeTheta: number,
  ) {}

  get dim(): number {
    return this.heads * this.headDim;
  }

  patchFeaturesIn(): number {
    return this.patch * this.patch * this.inChannels;
  }

  patchFeaturesOut(): number {
    return this.patch * this.patch * this.outChannels;
  }

  /** `(t, h, w)` widths of one head (Diffusers `WanRotaryPosEmbed`: `h = w = 2·(d/6)`, `t = d − h − w`; 44/42/42 at d = 128). */
  ropeSplit(): [number, number, number] {
    const hw = 2 * Math.floor(this.headDim / 6);
    return [this.headDim - 2 * hw, hw, hw];
  }

  /** Dense parameter count of the DiT (VSA gates excluded). */
  params(): number {
    const d = this.dim;
    const lin = (i: number, o: number): number => i * o + o;
    const block = 8 * lin(d, d) + 4 * d + 2 * d + lin(d, this.ffnDim) + lin(this.ffnDim, d) + 6 * d;
    return (
      this.layers * block +
      lin(this.patchFeaturesIn(), d) +
      lin(this.freqDim, d) +
      lin(d, d) +
      lin(d, 6 * d) +
      lin(this.textDim, d) +
      lin(d, d) +
      lin(d, this.patchFeaturesOut()) +
      2 * d
    );
  }

  toJson(): WanConfigJson {
    return {
      patch: this.patch,
      heads: this.heads,
      head_dim: this.headDim,
      in_channels: this.inChannels,
      out_channels: this.outChannels,
      text_dim: this.textDim,
      text_len: this.textLen,
      freq_dim: this.freqDim,
      ffn_dim: this.ffnDim,
      layers: this.layers,
      eps: this.eps,
      rope_max_seq_len: this.ropeMaxSeqLen,
      rope_theta: this.ropeTheta,
    };
  }

  static fromJson(v: WanConfigJson): WanConfig {
    return new WanConfig(
      v.patch,
      v.heads,
      v.head_dim,
      v.in_channels,
      v.out_channels,
      v.text_dim,
      v.text_len,
      v.freq_dim,
      v.ffn_dim,
      v.layers,
      v.eps,
      v.rope_max_seq_len ?? 1024,
      v.rope_theta,
    );
  }

  /** `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` (and Wan 2.1 T2V 1.3B). */
  static fastwan13b(): WanConfig {
    return new WanConfig(2, 12, 128, 16, 16, 4096, 512, 256, 8960, 30, 1e-6, 1024, 10000.0);
  }

  /** A graph small enough for the CPU in a unit test (fastwan-rs's `WanConfig::tiny`). */
  static tiny(): WanConfig {
    return new WanConfig(2, 2, 12, 4, 4, 16, 8, 16, 40, 2, 1e-6, 64, 10000.0);
  }

  static fromName(name: string): WanConfig {
    if (["fastwan-1.3b", "fastwan21-1.3b", "fastwan", "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"].includes(name)) {
      return WanConfig.fastwan13b();
    }
    if (name === "tiny") return WanConfig.tiny();
    throw new Error(`unknown model \`${name}\` (fastwan-1.3b or tiny)`);
  }
}

export const WAN21_LATENT_MEAN: readonly number[] = [
  -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922,
  -0.9497, 0.2503, -0.2921,
];
export const WAN21_LATENT_STD: readonly number[] = [
  2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253,
  2.8251, 1.916,
];

export interface TaeConfigJson {
  latent_channels: number;
  widths: number[];
  tgrow: number[];
  mem_blocks: number;
  patch: number;
  denormalize: boolean;
}

/** TAEHV (madebyollin/taehv) decoder for Wan 2.1 latents. */
export class TaeConfig {
  constructor(
    readonly latentChannels: number,
    readonly widths: readonly [number, number, number, number],
    readonly tgrow: readonly [number, number, number],
    readonly memBlocks: number,
    readonly patch: number,
    readonly denormalize: boolean,
  ) {}

  tUpscale(): number {
    let n = 1;
    for (const g of this.tgrow) n *= g;
    return n;
  }

  /** Priming frames emitted first and dropped: `4T − 3` frames for Wan. */
  framesToTrim(): number {
    return this.tUpscale() - 1;
  }

  spatialScale(): number {
    return 8 * this.patch;
  }

  outChannels(): number {
    return 3 * this.patch * this.patch;
  }

  toJson(): TaeConfigJson {
    return {
      latent_channels: this.latentChannels,
      widths: [...this.widths],
      tgrow: [...this.tgrow],
      mem_blocks: this.memBlocks,
      patch: this.patch,
      denormalize: this.denormalize,
    };
  }

  static fromJson(v: TaeConfigJson): TaeConfig {
    if (v.widths.length !== 4) throw new Error(`tae.widths has ${v.widths.length} entries, expected 4`);
    if (v.tgrow.length !== 3) throw new Error(`tae.tgrow has ${v.tgrow.length} entries, expected 3`);
    return new TaeConfig(
      v.latent_channels,
      [v.widths[0], v.widths[1], v.widths[2], v.widths[3]],
      [v.tgrow[0], v.tgrow[1], v.tgrow[2]],
      v.mem_blocks,
      v.patch,
      v.denormalize,
    );
  }

  static taew21(): TaeConfig {
    return new TaeConfig(16, [256, 128, 64, 64], [1, 2, 2], 3, 1, false);
  }

  /** LightX2V's LightTAE: TAEHV's network on raw (denormalised) latents. */
  static lighttaew21(): TaeConfig {
    return new TaeConfig(16, [256, 128, 64, 64], [1, 2, 2], 3, 1, true);
  }

  static taew22(): TaeConfig {
    return new TaeConfig(48, [256, 128, 64, 64], [1, 2, 2], 3, 2, false);
  }

  static tiny(): TaeConfig {
    return new TaeConfig(4, [8, 6, 4, 4], [1, 2, 2], 3, 1, false);
  }
}
