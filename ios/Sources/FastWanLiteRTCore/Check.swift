// The runtime self-test (`fastwan-litert check`): run the tiny fixture through
// this loop on the given store and compare with `expected.safetensors`.

import Foundation

public struct CheckResult: CustomStringConvertible {
    public var latentRel: Double
    public var frameMaxDiff: Int
    public var tolerance: Double
    public var ok: Bool { latentRel <= tolerance && frameMaxDiff <= 2 }

    public var description: String {
        let rel = String(format: "%.2e", latentRel)
        let tol = String(format: "%.0e", tolerance)
        return "latent rel \(rel) (tol \(tol)), frames max |Δ| \(frameMaxDiff)/255: \(ok ? "OK" : "FAIL")"
    }
}

public enum FixtureCheck {
    /// `directory` holds the export and `expected.safetensors` (prompt_embeds, latent, frames; metadata seed).
    public static func run(store: GraphStore, directory: URL, tolerance: Double = 1e-3, onEvent: OnEvent? = nil) throws -> CheckResult {
        let pipe = Pipeline(store: store)
        let m = store.manifest
        let exp = try SafeTensors(url: directory.appendingPathComponent("expected.safetensors"))
        guard let size = m.sizes.first else { throw FastWanError.manifest("manifest lists no sizes") }
        let seed = UInt64(exp.metadata["seed"] ?? "7") ?? 7
        let embeds = try exp.float32("prompt_embeds")
        let latent = try pipe.denoise(embeds: embeds, geometry: size, seed: seed, onEvent: onEvent)
        let want = try exp.float32("latent")
        guard want.shape == latent.shape else {
            throw FastWanError.shape("latent is \(latent.shape), expected \(want.shape)")
        }
        var maxDiff: Float = 0, maxAbs: Float = 0
        for i in 0..<want.count {
            maxDiff = max(maxDiff, abs(latent.data[i] - want.data[i]))
            maxAbs = max(maxAbs, abs(want.data[i]))
        }
        let rel = maxAbs > 0 ? Double(maxDiff / maxAbs) : Double(maxDiff)
        let video = try pipe.decode(latent: latent, onEvent: onEvent)
        let (fshape, frames) = try exp.uint8("frames")
        guard fshape == [video.frames, video.height, video.width, 3] else {
            throw FastWanError.shape("frames are [\(video.frames), \(video.height), \(video.width), 3], expected \(fshape)")
        }
        let got = video.rgb
        var fdiff = 0
        for i in 0..<frames.count {
            fdiff = max(fdiff, abs(Int(got[i]) - Int(frames[i])))
        }
        return CheckResult(latentRel: rel, frameMaxDiff: fdiff, tolerance: tolerance)
    }
}
