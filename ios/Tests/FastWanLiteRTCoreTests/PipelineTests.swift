// The pipeline's orchestration over a fake `GraphStore` that knows the
// contract's shapes (docs/graphs.md) and records every call, on the tiny
// manifest: 18 tokens, chunk 16, q_chunk 8, TAEHV chunks of 2 latent frames.

import Foundation
import XCTest
@testable import FastWanLiteRTCore

/// A graph that checks its inputs against the contract and returns outputs of
/// the contract's shapes, filled with `fill`.
final class FakeGraph: Graph {
    struct Call {
        var inputs: [String: [Int]]
    }

    let name: String
    let inputShapes: [String: [Int]]
    let outputShapes: [[Int]]
    var fill: (Int, [String: Tensor]) -> Float  // (output index, inputs) → value
    private(set) var calls: [Call] = []

    init(name: String, inputs: [String: [Int]], outputs: [[Int]], fill: @escaping (Int, [String: Tensor]) -> Float = { _, _ in 0 }) {
        self.name = name
        self.inputShapes = inputs
        self.outputShapes = outputs
        self.fill = fill
    }

    func run(_ inputs: [String: Tensor]) throws -> [Tensor] {
        guard Set(inputs.keys) == Set(inputShapes.keys) else {
            throw FastWanError.graph("\(name): inputs \(inputShapes.keys.sorted()), got \(inputs.keys.sorted())")
        }
        for (k, t) in inputs {
            guard t.shape == inputShapes[k]! else {
                throw FastWanError.shape("\(name): input `\(k)` is \(t.shape), the graph wants \(inputShapes[k]!)")
            }
        }
        calls.append(Call(inputs: inputs.mapValues { $0.shape }))
        return outputShapes.enumerated().map { (i, shape) in
            Tensor(shape: shape, data: [Float](repeating: fill(i, inputs), count: shape.reduce(1, *)))
        }
    }
}

/// Every graph of the tiny manifest with the contract's shapes.
final class FakeStore: GraphStore {
    let manifest: Manifest
    private(set) var graphs: [String: FakeGraph] = [:]
    private(set) var blockOpens: [String] = []
    let cache = ResidentGraphCache(capacity: 1)

    init(manifest m: Manifest) {
        self.manifest = m
        let c = m.cfg
        let d = c.dim, H = c.heads, hd = c.headDim, L = c.textLen, C = m.chunk
        let a = m.attention[0]
        let S = a.tokens, cq = a.qChunk
        let t = m.taeChunks[0]
        let F = t.frames, lh = t.latentH, lw = t.latentW
        let tae = m.tae
        func add(_ g: FakeGraph) { graphs[g.name] = g }
        add(FakeGraph(name: m.graphs.timeMods, inputs: ["t": [1]], outputs: [[c.layers, 6, d], [2, d]]))
        add(FakeGraph(name: m.graphs.textEmbed, inputs: ["embeds": [L, c.textDim]], outputs: [[L, d]]))
        add(FakeGraph(name: m.graphs.patchEmbed, inputs: ["patches": [C, c.patchFeaturesIn]], outputs: [[C, d]]))
        add(FakeGraph(name: m.graphs.head, inputs: ["h": [C, d], "mods": [2, d]], outputs: [[C, c.patchFeaturesOut]]))
        for b in m.graphs.blocks {
            add(FakeGraph(name: b.kv, inputs: ["ctx": [L, d]], outputs: [[H, L, hd], [H, L, hd]]))
            add(FakeGraph(name: b.qkv, inputs: ["h": [C, d], "mods": [6, d], "cos": [C, hd / 2], "sin": [C, hd / 2]], outputs: [[C, H, hd], [C, H, hd], [C, H, hd]]))
            add(FakeGraph(name: b.post, inputs: ["h": [C, d], "o": [C, d], "mods": [6, d], "k2": [H, L, hd], "v2": [H, L, hd]], outputs: [[C, d]]))
        }
        add(FakeGraph(name: a.file, inputs: ["q": [cq, H, hd], "k": [S, H, hd], "v": [S, H, hd]], outputs: [[cq, H * hd]]))
        var taeIn: [String: [Int]] = ["z": [F, tae.latentChannels, lh, lw]]
        var taeOut: [[Int]] = [[tae.tUpscale * F, 3, 8 * tae.patch * lh, 8 * tae.patch * lw]]
        for i in 0..<(3 * tae.memBlocks) {
            let s = i / tae.memBlocks
            let shape = [1, tae.widths[s], lh << s, lw << s]
            taeIn["mem_\(i)"] = shape
            taeOut.append(shape)
        }
        add(FakeGraph(name: t.file, inputs: taeIn, outputs: taeOut))
    }

    func fixed(_ file: String) throws -> Graph {
        guard let g = graphs[file] else { throw FastWanError.io("no graph \(file)") }
        return g
    }

    func block(_ file: String) throws -> Graph {
        try cache.get(file) {
            guard let g = graphs[file] else { throw FastWanError.io("no graph \(file)") }
            blockOpens.append(file)
            return g
        }
    }

    func dropBlocks() { cache.clear() }

    func calls(_ file: String) -> [FakeGraph.Call] { graphs[file]?.calls ?? [] }
}

final class PipelineTests: XCTestCase {
    let geometry = Geometry(width: 32, height: 48, frames: 9)

    func makeStore() throws -> FakeStore {
        FakeStore(manifest: try Manifest(json: tinyManifestData()))
    }

    func testDenoiseDrivesTheGraphsAsTheContractSays() throws {
        let store = try makeStore()
        let pipe = Pipeline(store: store)
        let m = store.manifest
        let embeds = Tensor.zeros([8, 16])
        var events: [PipelineEvent] = []
        var tm = Timings()
        let latent = try pipe.denoise(embeds: embeds, geometry: geometry, seed: 7, onEvent: { events.append($0) }, timings: &tm)
        XCTAssertEqual(latent.shape, [4, 3, 6, 4])
        XCTAssertEqual(tm.stepSeconds.count, 3)

        // Once per prompt.
        XCTAssertEqual(store.calls(m.graphs.textEmbed).count, 1)
        XCTAssertEqual(store.calls(m.graphs.blocks[0].kv).count, 1)
        XCTAssertEqual(store.calls(m.graphs.blocks[1].kv).count, 1)
        // Once per step.
        XCTAssertEqual(store.calls(m.graphs.timeMods).count, 3)
        XCTAssertEqual(store.calls(m.graphs.timeMods)[0].inputs["t"], [1])
        // 18 rows in chunks of 16: 2 calls per per-row graph per step.
        XCTAssertEqual(store.calls(m.graphs.patchEmbed).count, 2 * 3)
        XCTAssertEqual(store.calls(m.graphs.head).count, 2 * 3)
        for b in m.graphs.blocks {
            XCTAssertEqual(store.calls(b.qkv).count, 2 * 3, b.qkv)
            XCTAssertEqual(store.calls(b.post).count, 2 * 3, b.post)
        }
        // 18 queries in chunks of 8: 3 attention calls per block per step, keys and values exact (never padded).
        let attn = store.calls(m.attention[0].file)
        XCTAssertEqual(attn.count, 3 * 2 * 3)
        for call in attn {
            XCTAssertEqual(call.inputs["q"], [8, 2, 12])
            XCTAssertEqual(call.inputs["k"], [18, 2, 12])
            XCTAssertEqual(call.inputs["v"], [18, 2, 12])
        }
        // Every per-row call saw full chunks (padding happened) and the fixed inputs.
        for call in store.calls(m.graphs.blocks[0].qkv) {
            XCTAssertEqual(call.inputs["h"], [16, 24])
            XCTAssertEqual(call.inputs["cos"], [16, 6])
            XCTAssertEqual(call.inputs["mods"], [6, 24])
        }
        for call in store.calls(m.graphs.blocks[1].post) {
            XCTAssertEqual(call.inputs["o"], [16, 24])
            XCTAssertEqual(call.inputs["k2"], [2, 8, 12])
        }
        // Events: prepared, then one step per timestep.
        XCTAssertEqual(events.count, 4)
        if case .prepared(_, let tokens) = events[0] {
            XCTAssertEqual(tokens, 18)
        } else {
            XCTFail("first event is not prepared: \(events[0])")
        }
        for i in 0..<3 {
            if case .step(let index, let total, _) = events[i + 1] {
                XCTAssertEqual(index, i)
                XCTAssertEqual(total, 3)
            } else {
                XCTFail("event \(i + 1) is not a step")
            }
        }
        // Block graphs are opened in order and the LRU store (capacity 1) re-opens them each step.
        XCTAssertEqual(store.blockOpens.prefix(2), ["block_00_kv.tflite", "block_01_kv.tflite"])
        XCTAssertEqual(store.blockOpens.count, 2 + 3 * 2 * 2)
    }

    func testNoiseOrderIsLatentThenRenoisePerStep() throws {
        // With zero velocity from the fake head, the latent is a pure function of the noise stream:
        // the initial latent first, then, in step order, the re-noise of every step but the last.
        let store = try makeStore()
        let pipe = Pipeline(store: store)
        let latent = try pipe.denoise(embeds: Tensor.zeros([8, 16]), geometry: geometry, seed: 7)
        var rng = Rng(seed: 7)
        var x = Tensor(shape: [4, 3, 6, 4], data: rng.normal(4 * 3 * 6 * 4))
        let sched = store.manifest.schedule
        for i in 0..<sched.count {
            let (sigma, next) = sched.coeffs(i)
            var noise: Tensor? = nil
            if next != nil { noise = Tensor(shape: x.shape, data: rng.normal(x.count)) }
            x = Host.dmdStep(x: x, v: Tensor.zeros(x.shape), sigma: sigma, sigmaNext: next, noise: noise)
        }
        XCTAssertEqual(latent, x)
        let other = try pipe.denoise(embeds: Tensor.zeros([8, 16]), geometry: geometry, seed: 8)
        XCTAssertNotEqual(other, latent)
        let again = try pipe.denoise(embeds: Tensor.zeros([8, 16]), geometry: geometry, seed: 7)
        XCTAssertEqual(again, latent)
    }

    func testVelocityFlowsThroughUnpatchify() throws {
        // A head that returns 1.0 everywhere gives v = 1 at every latent element.
        let store = try makeStore()
        store.graphs["head.tflite"]!.fill = { _, _ in 1.0 }
        let pipe = Pipeline(store: store)
        let m = store.manifest
        let prompt = try pipe.preparePrompt(Tensor.zeros([8, 16]))
        XCTAssertEqual(prompt.kv.count, 2)
        XCTAssertEqual(prompt.kv[0].k2.shape, [2, 8, 12])
        let table = Rope.table(cfg: m.cfg, grid: geometry.grid(m.cfg))
        XCTAssertEqual(table.cos.shape, [18, 6])
        let x = Tensor.zeros([4, 3, 6, 4])
        let v = try pipe.forward(x: x, prompt: prompt, t: 1000, cos: table.cos, sin: table.sin, attention: try m.attentionFor(tokens: 18))
        XCTAssertEqual(v.shape, [4, 3, 6, 4])
        XCTAssert(v.data.allSatisfy { $0 == 1.0 })
        XCTAssertThrowsError(try pipe.preparePrompt(Tensor.zeros([512, 4096])))
    }

    func testDecodeChunksTrimsAndCarriesMemory() throws {
        let store = try makeStore()
        let m = store.manifest
        let tae = store.graphs[m.taeChunks[0].file]!
        // rgb = 0.5 everywhere; memory outputs carry the call number so the carry can be checked.
        var callNo = 0
        tae.fill = { i, _ in
            if i == 0 { return 0.5 }
            return Float(callNo)
        }
        let pipe = Pipeline(store: store)
        var events: [PipelineEvent] = []
        var tm = Timings()
        // Wrap the fill so it counts calls: the fake calls `fill` once per output, so count via the recorded calls.
        let latent = Tensor(shape: [4, 3, 6, 4], data: (0..<288).map { Float($0) * 0.01 })
        tae.fill = { i, inputs in
            if i == 0 { return 0.5 }
            // mem_0 of the second call must be the first call's mem_out_0 (= 1.0); first call sees zeros.
            callNo = Int(inputs["mem_0"]!.data[0]) + 1
            return Float(callNo)
        }
        let video = try pipe.decode(latent: latent, onEvent: { events.append($0) }, timings: &tm)
        // 3 latent frames in chunks of 2 → 2 calls; 12 decoded frames, 3 priming frames dropped → 9.
        XCTAssertEqual(tae.calls.count, 2)
        XCTAssertEqual(video.frames, 9)
        XCTAssertEqual(video.width, 32)
        XCTAssertEqual(video.height, 48)
        XCTAssertEqual(video.rgba.count, 9 * 48 * 32 * 4)
        XCTAssertEqual(video.rgba[0], 128)
        XCTAssertEqual(video.rgba[3], 255)
        XCTAssertEqual(video.rgb.count, 9 * 48 * 32 * 3)
        XCTAssertEqual(events.count, 2)
        if case .frames(let f) = events[0] {
            XCTAssertEqual(f.first, 0)
            XCTAssertEqual(f.count, 5)  // 2 real frames → 8 decoded, minus 3 priming
            XCTAssertEqual(f.width, 32)
            XCTAssertEqual(f.height, 48)
            XCTAssertEqual(f.rgba.count, 5 * 48 * 32 * 4)
            XCTAssertEqual(f.frame(4).count, 48 * 32 * 4)
        } else {
            XCTFail("first event is not frames")
        }
        if case .frames(let f) = events[1] {
            XCTAssertEqual(f.first, 5)
            XCTAssertEqual(f.count, 4)  // 1 real frame (padded to 2) → 4 decoded, padding's 4 dropped
        } else {
            XCTFail("second event is not frames")
        }
        XCTAssertEqual(callNo, 2)
        // The z input is frame-major: z[fi, ch] = latent[ch, f0 + fi]; the padded frame is zero.
        XCTAssertEqual(tae.calls[1].inputs["z"], [2, 4, 6, 4])
        XCTAssertGreaterThanOrEqual(tm.decodeSeconds, 0)
    }

    func testDecodeZIsFrameMajorAndZeroPadded() throws {
        let store = try makeStore()
        let m = store.manifest
        let tae = store.graphs[m.taeChunks[0].file]!
        var seen: [Tensor] = []
        tae.fill = { i, inputs in
            if i == 0, let z = inputs["z"], !seen.contains(z) { seen.append(z) }
            return 0
        }
        let latent = Tensor(shape: [4, 3, 6, 4], data: (0..<288).map { Float($0) })
        _ = try Pipeline(store: store).decode(latent: latent)
        XCTAssertEqual(seen.count, 2)
        let plane = 24
        // first chunk: frames 0, 1
        for fi in 0..<2 {
            for ch in 0..<4 {
                XCTAssertEqual(seen[0].data[(fi * 4 + ch) * plane], latent.data[(ch * 3 + fi) * plane])
            }
        }
        // second chunk: frame 2 then zeros
        for ch in 0..<4 {
            XCTAssertEqual(seen[1].data[ch * plane], latent.data[(ch * 3 + 2) * plane])
        }
        XCTAssert(seen[1].data[(1 * 4) * plane..<(2 * 4) * plane].allSatisfy { $0 == 0 })
    }

    func testGenerateReportsEverythingInOrder() throws {
        let store = try makeStore()
        let pipe = Pipeline(store: store)
        var kinds: [String] = []
        let (video, tm) = try pipe.generate(embeds: Tensor.zeros([8, 16]), geometry: geometry, seed: 7) { ev in
            switch ev {
            case .prepared: kinds.append("prepared")
            case .step: kinds.append("step")
            case .frames: kinds.append("frames")
            }
        }
        XCTAssertEqual(kinds, ["prepared", "step", "step", "step", "frames", "frames"])
        XCTAssertEqual(video.frames, 9)
        XCTAssertEqual(tm.stepSeconds.count, 3)
    }

    func testMissingGeometryIsAnError() throws {
        let store = try makeStore()
        let pipe = Pipeline(store: store)
        XCTAssertThrowsError(try pipe.denoise(embeds: Tensor.zeros([8, 16]), geometry: Geometry(width: 64, height: 48, frames: 9), seed: 7)) { e in
            XCTAssert("\(e)".contains("36 tokens"), "\(e)")
        }
        XCTAssertThrowsError(try pipe.denoise(embeds: Tensor.zeros([8, 16]), geometry: Geometry(width: 30, height: 48, frames: 9), seed: 7))
        XCTAssertThrowsError(try pipe.decode(latent: Tensor.zeros([4, 3, 8, 4]))) { e in
            XCTAssert("\(e)".contains("8x4"), "\(e)")
        }
    }

    func testResidentGraphCacheIsLRU() throws {
        let cache = ResidentGraphCache(capacity: 2)
        var opens: [String] = []
        func get(_ f: String) throws -> Graph {
            try cache.get(f) {
                opens.append(f)
                return FakeGraph(name: f, inputs: [:], outputs: [])
            }
        }
        _ = try get("a")
        _ = try get("b")
        _ = try get("a")  // a is now the most recent
        _ = try get("c")  // evicts b
        XCTAssertEqual(cache.residentFiles, ["a", "c"])
        _ = try get("b")  // evicts a
        XCTAssertEqual(cache.residentFiles, ["c", "b"])
        XCTAssertEqual(opens, ["a", "b", "c", "b"])
        cache.clear()
        XCTAssertEqual(cache.residentFiles, [])
    }
}
