"""DMD sampling schedule (fastwan-core `schedule.rs`; FastVideo `stages/dmd.py`).

FastWan 2.1 1.3B denoises at `[1000, 757, 522]`. Each step maps onto the
nearest entry of the *training* noise table (a shift-8
`FlowMatchEulerDiscreteScheduler` without `set_timesteps`) and

    x0 = x − σ_t·v
    x  = (1 − σ_next)·x0 + σ_next·ε     (fresh ε; skipped on the last step)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FASTWAN_1_3B_STEPS = [1000, 757, 522]
DMD_TRAINING_NOISE_SHIFT = 8.0


def _training_table(n: int, shift: float) -> tuple[np.ndarray, np.ndarray]:
    """Float32 sigma / timestep tables, index 0 = timestep N (as the reference)."""
    nf, sh, sm1 = np.float32(n), np.float32(shift), np.float32(shift - 1.0)
    i = np.arange(n, dtype=np.float32)
    s = (nf - i) / nf
    s = (sh * s) / (np.float32(1.0) + sm1 * s)
    return s.astype(np.float32), (s * nf).astype(np.float32)


@dataclass(frozen=True)
class DmdSchedule:
    timesteps: list[int]
    sigmas: list[float]
    """Table sigma for each step, then a trailing 0.0."""

    @classmethod
    def new(cls, steps: list[int]) -> "DmdSchedule":
        sig, ts = _training_table(1000, DMD_TRAINING_NOISE_SHIFT)
        sigmas = []
        for t in steps:
            i = int(np.argmin(np.abs(ts.astype(np.float64) - float(t))))
            sigmas.append(float(sig[i]))
        sigmas.append(0.0)
        return cls(list(steps), sigmas)

    @classmethod
    def fastwan_1_3b(cls) -> "DmdSchedule":
        return cls.new(FASTWAN_1_3B_STEPS)

    def __len__(self) -> int:
        return len(self.timesteps)

    def coeffs(self, i: int) -> tuple[float, float | None]:
        """`(σ_t, σ_next)`; `σ_next` is None on the last step (no re-noise)."""
        nxt = self.sigmas[i + 1] if i + 1 < len(self) else None
        return self.sigmas[i], nxt

    def to_json(self) -> dict:
        return {"timesteps": self.timesteps, "sigmas": self.sigmas}
