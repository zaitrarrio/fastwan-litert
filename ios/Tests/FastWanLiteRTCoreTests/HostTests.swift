// The host math against the values fastwan-rs's own tests pin down
// (python/tests/test_host.py, test_plan.py), plus the manifest and the
// safetensors reader.

import Foundation
import XCTest
@testable import FastWanLiteRTCore

final class HostTests: XCTestCase {
    func testFastwanSigmas() {
        let s = DmdSchedule.fastwan13B
        XCTAssertEqual(s.sigmas[0], 1.0)
        XCTAssertEqual(s.sigmas[1], 0.757, accuracy: 1e-3)
        XCTAssertEqual(s.sigmas[2], 0.522, accuracy: 1e-3)
        XCTAssertEqual(s.sigmas[3], 0.0)
        XCTAssertNil(s.coeffs(2).next)
        XCTAssertEqual(s.coeffs(0).next ?? 0, 0.757, accuracy: 1e-3)
        XCTAssertEqual(s.timesteps, [1000, 757, 522])
    }

    func test480pGeometry() throws {
        let cfg = WanConfig.fastwan13B
        let g = Geometry.fastwan480p
        XCTAssertEqual(g.latent.t, 21)
        XCTAssertEqual(g.latent.h, 60)
        XCTAssertEqual(g.latent.w, 104)
        XCTAssertEqual(g.grid(cfg).t, 21)
        XCTAssertEqual(g.grid(cfg).h, 30)
        XCTAssertEqual(g.grid(cfg).w, 52)
        XCTAssertEqual(g.tokens(cfg), 32760)
        XCTAssertThrowsError(try Geometry(width: 830, height: 480, frames: 81).validate(cfg))
        XCTAssertThrowsError(try Geometry(width: 832, height: 480, frames: 80).validate(cfg))
        XCTAssertNoThrow(try Geometry(width: 832, height: 480, frames: 1).validate(cfg))
        XCTAssertNoThrow(try Geometry.mobile480x272.validate(cfg))
        XCTAssertEqual(Geometry.mobile480x272.tokens(cfg), 9 * 17 * 30)
        XCTAssertEqual(try Geometry.parse("832x480x81"), g)
        XCTAssertThrowsError(try Geometry.parse("832x480"))
    }

    func testParamsIs14B() {
        let p = WanConfig.fastwan13B.params
        XCTAssert(p >= 1_400_000_000 && p <= 1_440_000_000, "\(p)")
    }

    func testRopeSplitAndOrigin() {
        let cfg = WanConfig.fastwan13B
        XCTAssertEqual(cfg.ropeSplit.t, 44)
        XCTAssertEqual(cfg.ropeSplit.h, 42)
        XCTAssertEqual(cfg.ropeSplit.w, 42)
        XCTAssertEqual(WanConfig.tiny.ropeSplit.t, 4)
        XCTAssertEqual(WanConfig.tiny.ropeSplit.h, 4)
        XCTAssertEqual(WanConfig.tiny.ropeSplit.w, 4)
        let table = Rope.table(cfg: cfg, grid: (t: 2, h: 3, w: 4), startFrame: 0)
        let cosT = table.cos, sinT = table.sin
        XCTAssertEqual(cosT.shape, [24, 64])
        XCTAssertEqual(sinT.shape, [24, 64])
        for j in 0..<64 {
            XCTAssertEqual(cosT.data[j], 1.0)
            XCTAssertEqual(sinT.data[j], 0.0)
        }
        let row = 12  // token (t=1, 0, 0): the first temporal pair rotates by 1 rad
        XCTAssertEqual(Double(cosT.data[row * 64 + 0]), cos(1.0), accuracy: 1e-7)
        XCTAssertEqual(cosT.data[row * 64 + 22], 1.0)  // the spatial pairs do not
        // token (0, 0, 1): the first w pair rotates by 1 rad, t and h pairs do not
        XCTAssertEqual(cosT.data[1 * 64 + 0], 1.0)
        XCTAssertEqual(cosT.data[1 * 64 + 22], 1.0)
        XCTAssertEqual(Double(cosT.data[1 * 64 + 22 + 21]), cos(1.0), accuracy: 1e-7)
        // start_frame shifts the temporal angle only
        let shifted = Rope.table(cfg: cfg, grid: (t: 1, h: 1, w: 1), startFrame: 1)
        XCTAssertEqual(Double(shifted.cos.data[0]), cos(1.0), accuracy: 1e-7)
        XCTAssertEqual(shifted.cos.data[22], 1.0)
    }

    func testTaehvFrames() {
        let t = TaeConfig.taew21
        XCTAssertEqual(t.tUpscale, 4)
        XCTAssertEqual(21 * t.tUpscale - t.framesToTrim, 81)
        XCTAssertEqual(TaeConfig.tiny.framesToTrim, 3)
    }

    func testRngIsSplitMix64BoxMuller() {
        // fastwan-rs seeds the state as `seed ^ GOLDEN`, so the canonical
        // SplitMix64 stream (state 0: first output 0xE220A8397B1DCDAF) appears at seed GOLDEN.
        var r = Rng(seed: 0x9E37_79B9_7F4A_7C15)
        XCTAssertEqual(r.nextU64(), 0xE220_A839_7B1D_CDAF)
        XCTAssertEqual(r.nextU64(), 0x6E78_9E6A_A1B9_65F4)
        var g = Rng(seed: 7)
        let v = g.normal(200_000)
        XCTAssertEqual(v.count, 200_000)
        var mean = 0.0
        for x in v { mean += Double(x) }
        mean /= Double(v.count)
        var variance = 0.0
        for x in v { variance += (Double(x) - mean) * (Double(x) - mean) }
        variance /= Double(v.count)
        XCTAssertLessThan(abs(mean), 0.01)
        XCTAssertLessThan(abs(variance - 1.0), 0.02)
        // The stream equals the scalar definition (u1 then u2 per pair; cos then sin).
        var a = Rng(seed: 3)
        var scalar: [Float] = []
        for _ in 0..<4 {
            let u1 = a.unit(), u2 = a.unit()
            let rr = (-2.0 * log(u1)).squareRoot()
            let th = 2.0 * Double.pi * u2
            scalar.append(Float(rr * cos(th)))
            scalar.append(Float(rr * sin(th)))
        }
        var b = Rng(seed: 3)
        let got = b.normal(8)
        for i in 0..<8 {
            XCTAssertEqual(got[i], scalar[i], accuracy: 1e-6)
        }
        // An odd count drops the second member of the last pair and consumes the pair.
        var c1 = Rng(seed: 3)
        let odd = c1.normal(7)
        XCTAssertEqual(odd.count, 7)
        XCTAssertEqual(odd[6], got[6])
        var c2 = Rng(seed: 3)
        _ = c2.normal(7)
        XCTAssertEqual(c1.state, c2.state)
    }

    func testPatchifyRoundTrip() {
        var rng = Rng(seed: 0)
        let x = Tensor(shape: [4, 3, 8, 12], data: rng.normal(4 * 3 * 8 * 12))
        let rows = Host.patchify(x, patch: 2)
        XCTAssertEqual(rows.shape, [3 * 4 * 6, 16])
        // feature index c·p² + kh·p + kw
        XCTAssertEqual(rows.data[0 * 16 + 1 * 4 + 1 * 2 + 0], x.data[((1 * 3 + 0) * 8 + 1) * 12 + 0])
        // row (t=2, h'=1, w'=3), feature (c=3, kh=0, kw=1) is x[3, 2, 2, 7]
        let row = (2 * 4 + 1) * 6 + 3
        XCTAssertEqual(rows.data[row * 16 + 3 * 4 + 0 * 2 + 1], x.data[((3 * 3 + 2) * 8 + 2) * 12 + 7])
        // Diffusers unpatchify reads (kh·p + kw)·c + c'
        var diffusers = [Float](repeating: 0, count: 72 * 16)
        for r in 0..<72 {
            for c in 0..<4 {
                for kh in 0..<2 {
                    for kw in 0..<2 {
                        diffusers[r * 16 + (kh * 2 + kw) * 4 + c] = rows.data[r * 16 + c * 4 + kh * 2 + kw]
                    }
                }
            }
        }
        let back = Host.unpatchify(Tensor(shape: [72, 16], data: diffusers), channels: 4, t: 3, h: 8, w: 12, patch: 2)
        XCTAssertEqual(back, x)
    }

    func testDmdStepAndPadding() {
        let x = Tensor(shape: [2], data: [1.0, -2.0])
        let v = Tensor(shape: [2], data: [0.5, 0.5])
        let last = Host.dmdStep(x: x, v: v, sigma: 0.522, sigmaNext: nil, noise: nil)
        XCTAssertEqual(last.data[0], 1.0 - Float(0.522) * 0.5)
        XCTAssertEqual(last.data[1], -2.0 - Float(0.522) * 0.5)
        let noise = Tensor(shape: [2], data: [2.0, 3.0])
        let mid = Host.dmdStep(x: x, v: v, sigma: 1.0, sigmaNext: 0.757, noise: noise)
        let s = Float(0.757)
        XCTAssertEqual(mid.data[0], (1 - s) * 0.5 + s * 2.0)
        XCTAssertEqual(mid.data[1], (1 - s) * (-2.5) + s * 3.0)

        let rows = Tensor(shape: [18, 3], data: (0..<54).map { Float($0) })
        let (padded, real) = Host.padRows(rows, chunk: 16)
        XCTAssertEqual(real, 18)
        XCTAssertEqual(padded.shape, [32, 3])
        XCTAssertEqual(Array(padded.data[0..<54]), rows.data)
        XCTAssertEqual(padded.data[54...].allSatisfy { $0 == 0 }, true)
        let exact = Host.padRows(rows, chunk: 9)
        XCTAssertEqual(exact.padded.shape, [18, 3])
        XCTAssertEqual(padded.rows(0..<18), rows)
    }

    func testToRgb8Rounding() {
        // [F=1, 3, H=1, W=2]
        let f = Tensor(shape: [1, 3, 1, 2], data: [0.5, 1.0, -1.0, 0.999, 0.0, 2.0])
        let rgb = Host.toRgb8(f)
        XCTAssertEqual(rgb.height, 1)
        XCTAssertEqual(rgb.width, 2)
        // pixel 0: r=0.5 → 128, g=-1 → 0, b=0 → 0; pixel 1: r=1 → 255, g=0.999 → 255, b=2 → 255
        XCTAssertEqual(rgb.rgb, [128, 0, 0, 255, 255, 255])
        let rgba = Host.toRgba8(f)
        XCTAssertEqual(rgba.rgba, [128, 0, 0, 255, 255, 255, 255, 255])
        XCTAssertEqual(Host.rgbaToRgb(rgba.rgba), rgb.rgb)
    }

    func testTensorRows() {
        let t = Tensor(shape: [3, 2, 2], data: (0..<12).map { Float($0) })
        XCTAssertEqual(t.row(1), Tensor(shape: [2, 2], data: [4, 5, 6, 7]))
        XCTAssertEqual(t.rows(1..<3).shape, [2, 2, 2])
        XCTAssertEqual(Tensor.concatRows([t.rows(0..<1), t.rows(1..<3)]), t)
        XCTAssertEqual(t.rowStride, 4)
    }

    func testManifestDecodesAndRefusesMismatches() throws {
        let m = try Manifest(json: tinyManifestData())
        XCTAssertEqual(m.format, "fastwan-litert/1")
        XCTAssertEqual(m.model, "tiny")
        XCTAssertEqual(m.config, WanConfig.tiny)
        XCTAssertEqual(m.tae, TaeConfig.tiny)
        XCTAssertEqual(m.chunk, 16)
        XCTAssertEqual(m.weights, "f16")
        XCTAssertEqual(m.graphs.blocks.count, 2)
        XCTAssertEqual(m.graphs.blocks[1].qkv, "block_01_qkv.tflite")
        XCTAssertEqual(m.schedule.timesteps, [1000, 757, 522])
        XCTAssertEqual(m.schedule.sigmas[1], 0.757, accuracy: 1e-3)
        XCTAssertEqual(try m.attentionFor(tokens: 18), Manifest.AttentionEntry(tokens: 18, qChunk: 8, heads: 2, file: "attn_s18_q8_h2.tflite"))
        XCTAssertEqual(try m.taeFor(latentH: 6, latentW: 4).file, "tae_f2_6x4.tflite")
        XCTAssertEqual(m.sizes, [Geometry(width: 32, height: 48, frames: 9)])
        XCTAssertEqual(m.files?["tae_f2_6x4.tflite"]?.bytes, 104280)
        XCTAssertEqual(m.graphFiles.count, 12)
        XCTAssertThrowsError(try m.attentionFor(tokens: 32760)) { e in
            XCTAssert("\(e)".contains("32760") && "\(e)".contains("[18]"), "\(e)")
        }
        XCTAssertThrowsError(try m.taeFor(latentH: 60, latentW: 104)) { e in
            XCTAssert("\(e)".contains("60x104"), "\(e)")
        }
        var raw = try JSONSerialization.jsonObject(with: tinyManifestData()) as! [String: Any]
        raw["format"] = "fastwan-litert/2"
        let other = try JSONSerialization.data(withJSONObject: raw)
        XCTAssertThrowsError(try Manifest(json: other)) { e in
            XCTAssert("\(e)".contains("fastwan-litert/2") && "\(e)".contains("fastwan-litert/1"), "\(e)")
        }
    }

    func testPlanMatchesTheReference() throws {
        let p = try Plan.make(cfg: .fastwan13B, tae: .taew21, geometry: .fastwan480p, weights: "f16", chunk: 1024, qChunk: 256, taeFrames: 3, budgetBytes: 16 << 30)
        XCTAssert(p.fits, p.description)
        XCTAssertEqual(p.tokens, 32760)
        XCTAssertEqual(p.attentionScratch, 12 * 256 * 32760 * 4)
        let phone = try Plan.make(cfg: .fastwan13B, tae: .taew21, geometry: .mobile480x272, weights: "int4", chunk: 1024, qChunk: 256, taeFrames: 3, budgetBytes: 2 << 30)
        XCTAssert(phone.fits, phone.description)
        XCTAssertLessThan(phone.weightsTotal, 1 << 30)
        let big = try Plan.make(cfg: .fastwan13B, tae: .taew21, geometry: .fastwan480p, weights: "int4", chunk: 1024, qChunk: 256, taeFrames: 3, budgetBytes: 2 << 30)
        XCTAssertFalse(big.fits)
        XCTAssertThrowsError(try Plan.make(cfg: .fastwan13B, tae: .taew21, geometry: .fastwan480p, weights: "int2", chunk: 1024, qChunk: 256, taeFrames: 3, budgetBytes: 1))
        let tiny = try Plan.make(manifest: Manifest(json: tinyManifestData()), geometry: Geometry(width: 32, height: 48, frames: 9), budgetBytes: 1 << 30)
        XCTAssertEqual(tiny.tokens, 18)
        XCTAssert(tiny.notes.contains { $0.contains("14 padded rows") }, "\(tiny.notes)")
    }

    func testSafeTensorsReader() throws {
        // Hand-built file: header length, JSON header, then F32 [2, 2] and U8 [3] data.
        let header = "{\"__metadata__\":{\"seed\":\"7\"},\"a\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},\"b\":{\"dtype\":\"U8\",\"shape\":[3],\"data_offsets\":[16,19]}}"
        var file = Data()
        var n = UInt64(header.utf8.count).littleEndian
        file.append(Data(bytes: &n, count: 8))
        file.append(header.data(using: .utf8)!)
        let floats: [Float] = [1.0, -2.5, 3.25, 0.0]
        for f in floats {
            var bits = f.bitPattern.littleEndian
            file.append(Data(bytes: &bits, count: 4))
        }
        file.append(contentsOf: [7, 8, 9])
        let st = try SafeTensors(data: file)
        XCTAssertEqual(st.metadata["seed"], "7")
        XCTAssertEqual(st.names, ["a", "b"])
        XCTAssertEqual(try st.float32("a"), Tensor(shape: [2, 2], data: floats))
        let b = try st.uint8("b")
        XCTAssertEqual(b.shape, [3])
        XCTAssertEqual(b.data, [7, 8, 9])
        XCTAssertThrowsError(try st.float32("b"))
        XCTAssertThrowsError(try st.float32("zzz"))
        XCTAssertEqual(SafeTensors.halfToFloat(0x3C00), 1.0)
        XCTAssertEqual(SafeTensors.halfToFloat(0xC000), -2.0)
        XCTAssertEqual(SafeTensors.halfToFloat(0x3555), 0.333251953125)
        XCTAssertEqual(SafeTensors.halfToFloat(0x0001), 5.960464477539063e-08)
    }
}

/// The tiny fixture's manifest (fixtures/tiny/manifest.json), copied into the test resources.
func tinyManifestData() throws -> Data {
    guard let url = Bundle.module.url(forResource: "manifest", withExtension: "json") else {
        throw FastWanError.io("manifest.json is missing from the test resources")
    }
    return try Data(contentsOf: url)
}
