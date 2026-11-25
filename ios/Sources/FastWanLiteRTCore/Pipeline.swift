// Prompt embeddings → DMD denoise → TAEHV decode → RGB frames, driving the
// graphs exactly as docs/graphs.md's host loop says (python pipeline.py).

import Foundation

/// What the pipeline reports while it runs. Times are seconds (the Python
/// reference reports `secs`; the contract's `ms` is `seconds * 1000`).
public enum PipelineEvent {
    case prepared(seconds: Double, tokens: Int)
    case step(index: Int, total: Int, seconds: Double)
    case frames(Frames)
}

/// One TAEHV chunk's decoded frames, RGBA row-major `[count, height, width, 4]`
/// with alpha 255: ready for `CGImage` / `CGDataProvider`.
public struct Frames {
    public var first: Int
    public var count: Int
    public var width: Int
    public var height: Int
    public var rgba: [UInt8]

    public init(first: Int, count: Int, width: Int, height: Int, rgba: [UInt8]) {
        self.first = first
        self.count = count
        self.width = width
        self.height = height
        self.rgba = rgba
    }

    public var bytesPerFrame: Int { width * height * 4 }

    public func frame(_ i: Int) -> ArraySlice<UInt8> {
        rgba[(i * bytesPerFrame)..<((i + 1) * bytesPerFrame)]
    }

    /// `[count, height, width, 3]` for comparisons against `expected.safetensors`.
    public var rgb: [UInt8] { Host.rgbaToRgb(rgba) }
}

public struct Video {
    public var frames: Int
    public var width: Int
    public var height: Int
    /// `[frames, height, width, 4]` RGBA.
    public var rgba: [UInt8]

    public init(frames: Int, width: Int, height: Int, rgba: [UInt8]) {
        self.frames = frames
        self.width = width
        self.height = height
        self.rgba = rgba
    }

    public var bytesPerFrame: Int { width * height * 4 }

    public func frame(_ i: Int) -> ArraySlice<UInt8> {
        rgba[(i * bytesPerFrame)..<((i + 1) * bytesPerFrame)]
    }

    public var rgb: [UInt8] { Host.rgbaToRgb(rgba) }
}

public struct Timings {
    public var prepareSeconds: Double = 0
    public var stepSeconds: [Double] = []
    public var decodeSeconds: Double = 0

    public init() {}
}

/// Per-prompt cross-attention K/V of every block.
public struct Prompt {
    public var kv: [(k2: Tensor, v2: Tensor)]

    public init(kv: [(k2: Tensor, v2: Tensor)]) {
        self.kv = kv
    }
}

public typealias OnEvent = (PipelineEvent) -> Void

public final class Pipeline {
    public let store: GraphStore
    public let manifest: Manifest
    public let cfg: WanConfig

    public init(store: GraphStore) {
        self.store = store
        self.manifest = store.manifest
        self.cfg = store.manifest.cfg
    }

    // MARK: prompt

    public func preparePrompt(_ embeds: Tensor) throws -> Prompt {
        let c = cfg
        guard embeds.shape == [c.textLen, c.textDim] else {
            throw FastWanError.shape("prompt embeds are \(embeds.shape), expected [\(c.textLen), \(c.textDim)]")
        }
        let ctx = try one(try store.fixed(manifest.graphs.textEmbed).run(["embeds": embeds]), of: manifest.graphs.textEmbed)
        var kv: [(k2: Tensor, v2: Tensor)] = []
        for b in manifest.graphs.blocks {
            let out = try store.block(b.kv).run(["ctx": ctx])
            guard out.count == 2 else {
                throw FastWanError.graph("\(b.kv): expected 2 outputs (k2, v2), got \(out.count)")
            }
            kv.append((out[0], out[1]))
        }
        return Prompt(kv: kv)
    }

    // MARK: one DiT forward

    /// Run a per-row graph over row chunks of `arrays` (zero-padded to a multiple
    /// of `chunk`) with the same `fixed` inputs every call; returns the outputs
    /// with the padded rows dropped, concatenated along rows.
    func rows(_ graph: Graph, chunk: Int, real nReal: Int, arrays: [String: Tensor], fixed: [String: Tensor]) throws -> [Tensor] {
        var padded: [String: Tensor] = [:]
        var n = 0
        for (k, v) in arrays {
            let p = Host.padRows(v, chunk: chunk).padded
            if n == 0 {
                n = p.rows
            } else if p.rows != n {
                throw FastWanError.shape("\(graph.name): row inputs disagree (\(n) vs \(p.rows) rows for `\(k)`)")
            }
            padded[k] = p
        }
        var outs: [[Tensor]] = []
        var r = 0
        while r < n {
            var inputs = fixed
            for (k, v) in padded {
                inputs[k] = v.rows(r..<(r + chunk))
            }
            let o = try graph.run(inputs)
            if let first = outs.first, first.count != o.count {
                throw FastWanError.graph("\(graph.name): output count changed between calls")
            }
            outs.append(o)
            r += chunk
        }
        guard let first = outs.first else {
            throw FastWanError.shape("\(graph.name): no rows to run")
        }
        return (0..<first.count).map { i in
            Tensor.concatRows(outs.map { $0[i] }).rows(0..<nReal)
        }
    }

    func one(_ outs: [Tensor], of name: String) throws -> Tensor {
        guard outs.count == 1 else {
            throw FastWanError.graph("\(name): expected 1 output, got \(outs.count)")
        }
        return outs[0]
    }

    public func forward(x: Tensor, prompt: Prompt, t: Double, cos: Tensor, sin: Tensor, attention: Manifest.AttentionEntry) throws -> Tensor {
        let c = cfg
        let C = manifest.chunk
        guard x.rank == 4 else { throw FastWanError.shape("latent is \(x.shape), expected [C, T, H, W]") }
        let tt = x.shape[1], hh = x.shape[2], ww = x.shape[3]
        let s = tt * (hh / c.patch) * (ww / c.patch)
        guard prompt.kv.count == manifest.graphs.blocks.count else {
            throw FastWanError.shape("prompt has K/V for \(prompt.kv.count) blocks, the model has \(manifest.graphs.blocks.count)")
        }
        let mods = try store.fixed(manifest.graphs.timeMods).run(["t": Tensor(scalar: Float(t))])
        guard mods.count == 2 else {
            throw FastWanError.graph("\(manifest.graphs.timeMods): expected 2 outputs (block_mods, head_mods), got \(mods.count)")
        }
        let blockMods = mods[0], headMods = mods[1]
        guard blockMods.rows == manifest.graphs.blocks.count else {
            throw FastWanError.shape("\(manifest.graphs.timeMods): block_mods is \(blockMods.shape), the manifest has \(manifest.graphs.blocks.count) blocks")
        }
        let patches = Host.patchify(x, patch: c.patch)
        var h = try one(try rows(try store.fixed(manifest.graphs.patchEmbed), chunk: C, real: s, arrays: ["patches": patches], fixed: [:]), of: manifest.graphs.patchEmbed)
        let attnGraph = try store.fixed(attention.file)
        let cq = attention.qChunk
        for (l, b) in manifest.graphs.blocks.enumerated() {
            let m = blockMods.row(l)
            let qkv = try rows(try store.block(b.qkv), chunk: C, real: s, arrays: ["h": h, "cos": cos, "sin": sin], fixed: ["mods": m])
            guard qkv.count == 3 else {
                throw FastWanError.graph("\(b.qkv): expected 3 outputs (q, k, v), got \(qkv.count)")
            }
            let q = qkv[0], k = qkv[1], v = qkv[2]
            let o = try one(try rows(attnGraph, chunk: cq, real: s, arrays: ["q": q], fixed: ["k": k, "v": v]), of: attention.file)
            let (k2, v2) = prompt.kv[l]
            h = try one(try rows(try store.block(b.post), chunk: C, real: s, arrays: ["h": h, "o": o], fixed: ["mods": m, "k2": k2, "v2": v2]), of: b.post)
        }
        let out = try one(try rows(try store.fixed(manifest.graphs.head), chunk: C, real: s, arrays: ["h": h], fixed: ["mods": headMods]), of: manifest.graphs.head)
        return Host.unpatchify(out, channels: c.outChannels, t: tt, h: hh, w: ww, patch: c.patch)
    }

    // MARK: denoise

    public func denoise(embeds: Tensor, geometry g: Geometry, seed: UInt64, onEvent on: OnEvent? = nil, timings: inout Timings) throws -> Tensor {
        let c = cfg
        try g.validate(c)
        let t0 = Date()
        let attn = try manifest.attentionFor(tokens: g.tokens(c))
        let prompt = try preparePrompt(embeds)
        let table = Rope.table(cfg: c, grid: g.grid(c), startFrame: 0)
        let lat = g.latent
        var rng = Rng(seed: seed)
        var x = Tensor(shape: [c.inChannels, lat.t, lat.h, lat.w], data: rng.normal(c.inChannels * lat.t * lat.h * lat.w))
        timings.prepareSeconds = Date().timeIntervalSince(t0)
        on?(.prepared(seconds: timings.prepareSeconds, tokens: g.tokens(c)))
        let sched = manifest.schedule
        for (i, t) in sched.timesteps.enumerated() {
            let ts = Date()
            let v = try forward(x: x, prompt: prompt, t: Double(t), cos: table.cos, sin: table.sin, attention: attn)
            let (sigma, next) = sched.coeffs(i)
            var noise: Tensor? = nil
            if next != nil {
                noise = Tensor(shape: x.shape, data: rng.normal(x.count))
            }
            x = Host.dmdStep(x: x, v: v, sigma: sigma, sigmaNext: next, noise: noise)
            let secs = Date().timeIntervalSince(ts)
            timings.stepSeconds.append(secs)
            on?(.step(index: i, total: sched.count, seconds: secs))
        }
        return x
    }

    public func denoise(embeds: Tensor, geometry g: Geometry, seed: UInt64, onEvent on: OnEvent? = nil) throws -> Tensor {
        var tm = Timings()
        return try denoise(embeds: embeds, geometry: g, seed: seed, onEvent: on, timings: &tm)
    }

    // MARK: decode

    public func decode(latent: Tensor, onEvent on: OnEvent? = nil, timings: inout Timings) throws -> Video {
        let tcfg = manifest.tae
        let t0 = Date()
        guard latent.rank == 4 else { throw FastWanError.shape("latent is \(latent.shape), expected [C, T, H, W]") }
        let c = latent.shape[0], t = latent.shape[1], h = latent.shape[2], w = latent.shape[3]
        let entry = try manifest.taeFor(latentH: h, latentW: w)
        let graph = try store.fixed(entry.file)
        let f = entry.frames
        let nmem = 3 * tcfg.memBlocks
        var mems: [Tensor] = []
        for s in 0..<3 {
            for _ in 0..<tcfg.memBlocks {
                mems.append(Tensor.zeros([1, tcfg.widths[s], h << s, w << s]))
            }
        }
        let up = tcfg.tUpscale
        var toTrim = tcfg.framesToTrim
        var rgba: [UInt8] = []
        var total = 0
        var width = 0, height = 0
        let plane = h * w
        var f0 = 0
        while f0 < t {
            let n = min(f, t - f0)
            var z = Tensor.zeros([f, c, h, w])
            // z[:n] = latent[:, f0:f0+n] transposed to frame-major.
            for fi in 0..<n {
                for ch in 0..<c {
                    let src = ((ch * t) + (f0 + fi)) * plane
                    let dst = ((fi * c) + ch) * plane
                    for p in 0..<plane {
                        z.data[dst + p] = latent.data[src + p]
                    }
                }
            }
            var inputs: [String: Tensor] = ["z": z]
            for i in 0..<nmem {
                inputs["mem_\(i)"] = mems[i]
            }
            let out = try graph.run(inputs)
            guard out.count == 1 + nmem else {
                throw FastWanError.graph("\(entry.file): expected \(1 + nmem) outputs (rgb + \(nmem) memories), got \(out.count)")
            }
            let rgbAll = out[0]
            guard rgbAll.rank == 4, rgbAll.shape[1] == 3, rgbAll.rows >= n * up else {
                throw FastWanError.shape("\(entry.file): rgb is \(rgbAll.shape), expected [\(f * up), 3, H, W]")
            }
            let rgb = rgbAll.rows(0..<(n * up))
            mems = Array(out[1...])
            let skip = min(toTrim, rgb.rows)
            toTrim -= skip
            if rgb.rows > skip {
                let kept = rgb.rows(skip..<rgb.rows)
                let conv = Host.toRgba8(kept)
                width = conv.width
                height = conv.height
                rgba.append(contentsOf: conv.rgba)
                on?(.frames(Frames(first: total, count: conv.frames, width: conv.width, height: conv.height, rgba: conv.rgba)))
                total += conv.frames
            }
            f0 += f
        }
        timings.decodeSeconds = Date().timeIntervalSince(t0)
        return Video(frames: total, width: width, height: height, rgba: rgba)
    }

    public func decode(latent: Tensor, onEvent on: OnEvent? = nil) throws -> Video {
        var tm = Timings()
        return try decode(latent: latent, onEvent: on, timings: &tm)
    }

    // MARK: generate

    public func generate(embeds: Tensor, geometry g: Geometry, seed: UInt64, onEvent on: OnEvent? = nil) throws -> (video: Video, timings: Timings) {
        var tm = Timings()
        let latent = try denoise(embeds: embeds, geometry: g, seed: seed, onEvent: on, timings: &tm)
        let video = try decode(latent: latent, onEvent: on, timings: &tm)
        return (video, tm)
    }
}
