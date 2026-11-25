/**
 * DMD sampling schedule (fastwan-core `schedule.rs`; FastVideo `stages/dmd.py`).
 *
 * FastWan 2.1 1.3B denoises at `[1000, 757, 522]`. Each step maps onto the
 * nearest entry of the *training* noise table (a shift-8
 * `FlowMatchEulerDiscreteScheduler` without `set_timesteps`) and
 *
 *     x0 = x − σ_t·v
 *     x  = (1 − σ_next)·x0 + σ_next·ε     (fresh ε; skipped on the last step)
 *
 * The table is formed in float32, as the reference: every operation is
 * rounded with `Math.fround`.
 */

export const FASTWAN_1_3B_STEPS: readonly number[] = [1000, 757, 522];
export const DMD_TRAINING_NOISE_SHIFT = 8.0;

const f = Math.fround;

/** Float32 sigma / timestep tables, index 0 = timestep N (as the reference). */
function trainingTable(n: number, shift: number): { sigmas: Float32Array; timesteps: Float32Array } {
  const nf = f(n);
  const sh = f(shift);
  const sm1 = f(shift - 1.0);
  const sigmas = new Float32Array(n);
  const timesteps = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = f(f(nf - f(i)) / nf);
    s = f(f(sh * s) / f(f(1.0) + f(sm1 * s)));
    sigmas[i] = s;
    timesteps[i] = f(s * nf);
  }
  return { sigmas, timesteps };
}

export interface DmdScheduleJson {
  timesteps: number[];
  sigmas: number[];
}

export class DmdSchedule {
  /** `sigmas`: table sigma for each step, then a trailing 0.0. */
  constructor(
    readonly timesteps: readonly number[],
    readonly sigmas: readonly number[],
  ) {
    if (sigmas.length !== timesteps.length + 1) {
      throw new Error(`schedule has ${timesteps.length} timesteps but ${sigmas.length} sigmas (expected one more)`);
    }
  }

  static create(steps: readonly number[]): DmdSchedule {
    const { sigmas: sig, timesteps: ts } = trainingTable(1000, DMD_TRAINING_NOISE_SHIFT);
    const sigmas: number[] = [];
    for (const t of steps) {
      let best = 0;
      let bestD = Infinity;
      for (let i = 0; i < ts.length; i++) {
        const d = Math.abs(ts[i] - t);
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      }
      sigmas.push(sig[best]);
    }
    sigmas.push(0.0);
    return new DmdSchedule([...steps], sigmas);
  }

  static fastwan13b(): DmdSchedule {
    return DmdSchedule.create(FASTWAN_1_3B_STEPS);
  }

  get length(): number {
    return this.timesteps.length;
  }

  /** `[σ_t, σ_next]`; `σ_next` is null on the last step (no re-noise). */
  coeffs(i: number): [number, number | null] {
    const nxt = i + 1 < this.length ? this.sigmas[i + 1] : null;
    return [this.sigmas[i], nxt];
  }

  toJson(): DmdScheduleJson {
    return { timesteps: [...this.timesteps], sigmas: [...this.sigmas] };
  }
}
