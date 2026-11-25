/** Pixel ↔ latent ↔ token shapes for a Wan 2.1 clip (fastwan-core `geometry.rs`). */

import type { WanConfig } from "./config.js";

export const VAE_SPATIAL = 8;
export const VAE_TEMPORAL = 4;

export interface GeometryJson {
  width: number;
  height: number;
  frames: number;
}

export class Geometry {
  constructor(
    readonly width: number,
    readonly height: number,
    readonly frames: number,
  ) {}

  validate(cfg: WanConfig): void {
    const cell = VAE_SPATIAL * cfg.patch;
    if (this.width <= 0 || this.height <= 0 || this.width % cell !== 0 || this.height % cell !== 0) {
      throw new Error(`${this.width}x${this.height} is not a positive multiple of ${cell}`);
    }
    if (this.frames <= 0 || (this.frames - 1) % VAE_TEMPORAL !== 0) {
      throw new Error(`${this.frames} frames is not 4n+1`);
    }
  }

  /** Latent `(T, H, W)`. */
  latent(): [number, number, number] {
    return [Math.floor((this.frames - 1) / VAE_TEMPORAL) + 1, this.height / VAE_SPATIAL, this.width / VAE_SPATIAL];
  }

  grid(cfg: WanConfig): [number, number, number] {
    const [t, h, w] = this.latent();
    return [t, Math.floor(h / cfg.patch), Math.floor(w / cfg.patch)];
  }

  tokens(cfg: WanConfig): number {
    const [t, h, w] = this.grid(cfg);
    return t * h * w;
  }

  latentLen(cfg: WanConfig): number {
    const [t, h, w] = this.latent();
    return cfg.inChannels * t * h * w;
  }

  toJson(): GeometryJson {
    return { width: this.width, height: this.height, frames: this.frames };
  }

  /** `WxHxF`, e.g. `832x480x81`. */
  static parse(s: string): Geometry {
    const parts = s.toLowerCase().split("x");
    const nums = parts.map((p) => (/^\d+$/.test(p) ? Number(p) : NaN));
    if (nums.length !== 3 || nums.some((n) => !Number.isFinite(n))) {
      throw new Error(`geometry \`${s}\` is not WIDTHxHEIGHTxFRAMES`);
    }
    return new Geometry(nums[0], nums[1], nums[2]);
  }

  equals(o: Geometry): boolean {
    return this.width === o.width && this.height === o.height && this.frames === o.frames;
  }

  toString(): string {
    return `${this.width}x${this.height}x${this.frames}`;
  }
}

/** FastWan 2.1 1.3B's served default: 5 s at 16 fps. */
export const FASTWAN_480P = new Geometry(832, 480, 81);
/** fastwan-rs's phone tier. */
export const MOBILE_480x272 = new Geometry(480, 272, 33);
