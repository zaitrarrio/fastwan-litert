// DMD sampling schedule (python schedule.py; fastwan-core `schedule.rs`).
//
// FastWan 2.1 1.3B denoises at `[1000, 757, 522]`. Each step maps onto the
// nearest entry of the *training* noise table (a shift-8
// FlowMatchEulerDiscreteScheduler without `set_timesteps`) and
//
//     x0 = x − σ_t·v
//     x  = (1 − σ_next)·x0 + σ_next·ε     (fresh ε; skipped on the last step)

import Foundation

public let fastwan13BSteps = [1000, 757, 522]
public let dmdTrainingNoiseShift: Float = 8.0

public struct DmdSchedule: Codable, Equatable, Sendable {
    public var timesteps: [Int]
    /// Table sigma for each step, then a trailing 0.0.
    public var sigmas: [Double]

    public init(timesteps: [Int], sigmas: [Double]) {
        self.timesteps = timesteps
        self.sigmas = sigmas
    }

    /// Float32 sigma / timestep tables, index 0 = timestep N (as the reference).
    static func trainingTable(n: Int, shift: Float) -> (sigmas: [Float], timesteps: [Float]) {
        let nf = Float(n)
        let sm1 = shift - 1.0
        var sig = [Float](repeating: 0, count: n)
        var ts = [Float](repeating: 0, count: n)
        for i in 0..<n {
            var s = (nf - Float(i)) / nf
            s = (shift * s) / (Float(1.0) + sm1 * s)
            sig[i] = s
            ts[i] = s * nf
        }
        return (sig, ts)
    }

    public static func make(steps: [Int]) -> DmdSchedule {
        let (sig, ts) = trainingTable(n: 1000, shift: dmdTrainingNoiseShift)
        var sigmas: [Double] = []
        for t in steps {
            var best = 0
            var bestDist = Double.infinity
            for i in 0..<ts.count {
                let d = abs(Double(ts[i]) - Double(t))
                if d < bestDist {
                    bestDist = d
                    best = i
                }
            }
            sigmas.append(Double(sig[best]))
        }
        sigmas.append(0.0)
        return DmdSchedule(timesteps: steps, sigmas: sigmas)
    }

    public static var fastwan13B: DmdSchedule { make(steps: fastwan13BSteps) }

    public var count: Int { timesteps.count }

    /// `(σ_t, σ_next)`; `σ_next` is nil on the last step (no re-noise).
    public func coeffs(_ i: Int) -> (sigma: Double, next: Double?) {
        let next: Double? = (i + 1 < count) ? sigmas[i + 1] : nil
        return (sigmas[i], next)
    }
}
