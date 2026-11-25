// Memory plan (python plan.py; fastwan-core `plan.rs`, for the graph cut):
// what a generation holds at once, before anything is allocated. Hosts call it
// before `generate` to refuse what will not fit.

import Foundation

public let bytesPerWeight: [String: Double] = ["f32": 4.0, "f16": 2.0, "int8": 1.0, "int4": 0.5]

public struct Plan: CustomStringConvertible {
    public var tokens: Int
    public var chunk: Int
    public var qChunk: Int
    public var weightsTotal: Int
    public var weightsResident: Int
    public var crossKV: Int
    public var activations: Int
    public var attentionScratch: Int
    public var decodePeak: Int
    public var peak: Int
    public var budget: Int
    public var fits: Bool
    public var notes: [String]

    public var description: String {
        func mib(_ b: Int) -> String { String(format: "%9.1f", Double(b) / Double(1 << 20)) }
        var lines = [
            "tokens            \(tokens)",
            "weights (all)     \(mib(weightsTotal)) MiB",
            "weights resident  \(mib(weightsResident)) MiB",
            "cross-attn K/V    \(mib(crossKV)) MiB",
            "DiT activations   \(mib(activations)) MiB (row chunk \(chunk))",
            "attention scratch \(mib(attentionScratch)) MiB (query chunk \(qChunk))",
            "decode peak       \(mib(decodePeak)) MiB",
            "peak              \(mib(peak)) MiB of \(String(format: "%.0f", Double(budget) / Double(1 << 20))) MiB  fits: \(fits)",
        ]
        lines.append(contentsOf: notes.map { "note: \($0)" })
        return lines.joined(separator: "\n")
    }

    public static func make(
        cfg: WanConfig, tae: TaeConfig, geometry g: Geometry, weights: String,
        chunk: Int, qChunk: Int, taeFrames: Int, budgetBytes: Int, residentBlocks: Int = 2
    ) throws -> Plan {
        guard let wb = bytesPerWeight[weights] else {
            throw FastWanError.manifest("unknown weight format `\(weights)` (f32, f16, int8, int4)")
        }
        let s = g.tokens(cfg)
        let d = cfg.dim
        let a = 4  // f32 activations
        let blockParams = 8 * (d * d + d) + 2 * (d * cfg.ffnDim) + cfg.ffnDim + d + 6 * d
        let otherParams = cfg.params - cfg.layers * blockParams
        let weightsTotal = Int(Double(cfg.layers * blockParams) * wb + Double(otherParams * 2))
        let weightsResident = Int(Double(residentBlocks * blockParams) * wb + Double(otherParams * 2))
        let crossKV = cfg.layers * 2 * cfg.textLen * d * a
        // h, q, k, v, o: [S, d] each; the latent, velocity and noise; rope tables.
        let activations = (5 * s * d + 3 * g.latentLen(cfg) + s * cfg.headDim) * a
        let attentionScratch = cfg.heads * qChunk * s * a
        var notes: [String] = []
        // TAEHV peak: the last stage's 64-channel map at 8x resolution for the chunk's 4F frames,
        // plus input/output of the MemBlock convs (x, past, two temporaries, out).
        let lat = g.latent
        let f = min(taeFrames, lat.t)
        let nf = tae.widths
        let px = lat.h * lat.w
        let s0 = f * nf[0] * px
        let s1 = f * tae.tgrow[0] * nf[1] * 4 * px
        let s2 = f * tae.tgrow[0] * tae.tgrow[1] * nf[2] * 16 * px
        let out = f * tae.tUpscale * nf[3] * 64 * px
        let decodePeak = max(max(5 * s0, 5 * s1), max(5 * s2, s2 + 2 * out)) * a
        let peak = max(weightsResident + crossKV + activations + attentionScratch, weightsResident + decodePeak)
        if residentBlocks < cfg.layers {
            notes.append("\(residentBlocks) of \(cfg.layers) block graphs resident; the rest stream from disk / the network")
        }
        if s % chunk != 0 {
            notes.append("\(s) rows run as \((s + chunk - 1) / chunk) chunks of \(chunk) (\(chunk - s % chunk) padded rows)")
        }
        return Plan(
            tokens: s, chunk: chunk, qChunk: qChunk, weightsTotal: weightsTotal, weightsResident: weightsResident,
            crossKV: crossKV, activations: activations, attentionScratch: attentionScratch, decodePeak: decodePeak,
            peak: peak, budget: budgetBytes, fits: peak <= budgetBytes, notes: notes
        )
    }

    /// The plan of a manifest's export for one clip size (what `generate` checks first).
    public static func make(manifest m: Manifest, geometry g: Geometry, budgetBytes: Int, residentBlocks: Int = 2) throws -> Plan {
        let attn = try m.attentionFor(tokens: g.tokens(m.cfg))
        let lat = g.latent
        let tae = try m.taeFor(latentH: lat.h, latentW: lat.w)
        return try make(
            cfg: m.cfg, tae: m.tae, geometry: g, weights: m.weights, chunk: m.chunk,
            qChunk: attn.qChunk, taeFrames: tae.frames, budgetBytes: budgetBytes, residentBlocks: residentBlocks
        )
    }
}
