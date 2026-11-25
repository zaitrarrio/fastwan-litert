// `manifest.json` of an export (docs/graphs.md). Decoding validates the
// format; lookups by geometry throw with the mismatch, never fall back.

import Foundation

public let manifestFormat = "fastwan-litert/1"

public struct Manifest: Codable, Equatable {
    public struct BlockGraphs: Codable, Equatable {
        public var kv: String
        public var qkv: String
        public var post: String
        public var bytes: Int?

        public init(kv: String, qkv: String, post: String, bytes: Int? = nil) {
            self.kv = kv
            self.qkv = qkv
            self.post = post
            self.bytes = bytes
        }
    }

    public struct Graphs: Codable, Equatable {
        public var timeMods: String
        public var textEmbed: String
        public var patchEmbed: String
        public var head: String
        public var blocks: [BlockGraphs]

        enum CodingKeys: String, CodingKey {
            case head, blocks
            case timeMods = "time_mods"
            case textEmbed = "text_embed"
            case patchEmbed = "patch_embed"
        }

        public init(timeMods: String, textEmbed: String, patchEmbed: String, head: String, blocks: [BlockGraphs]) {
            self.timeMods = timeMods
            self.textEmbed = textEmbed
            self.patchEmbed = patchEmbed
            self.head = head
            self.blocks = blocks
        }
    }

    public struct AttentionEntry: Codable, Equatable {
        public var tokens: Int
        public var qChunk: Int
        public var heads: Int
        public var file: String

        enum CodingKeys: String, CodingKey {
            case tokens, heads, file
            case qChunk = "q_chunk"
        }

        public init(tokens: Int, qChunk: Int, heads: Int, file: String) {
            self.tokens = tokens
            self.qChunk = qChunk
            self.heads = heads
            self.file = file
        }
    }

    public struct TaeChunkEntry: Codable, Equatable {
        public var frames: Int
        public var latentH: Int
        public var latentW: Int
        public var file: String

        enum CodingKeys: String, CodingKey {
            case frames, file
            case latentH = "latent_h"
            case latentW = "latent_w"
        }

        public init(frames: Int, latentH: Int, latentW: Int, file: String) {
            self.frames = frames
            self.latentH = latentH
            self.latentW = latentW
            self.file = file
        }
    }

    public struct FileInfo: Codable, Equatable {
        public var bytes: Int
        public var sha256: String?
    }

    public var format: String
    public var model: String
    public var config: WanConfig
    public var tae: TaeConfig
    public var schedule: DmdSchedule
    public var weights: String
    public var chunk: Int
    public var graphs: Graphs
    public var attention: [AttentionEntry]
    public var taeChunks: [TaeChunkEntry]
    public var sizes: [Geometry]
    public var files: [String: FileInfo]?

    enum CodingKeys: String, CodingKey {
        case format, model, config, tae, schedule, weights, chunk, graphs, attention, sizes, files
        case taeChunks = "tae_chunks"
    }

    /// Decode and validate `manifest.json` bytes.
    public init(json: Data, source: String = "manifest.json") throws {
        let decoded: Manifest
        do {
            decoded = try JSONDecoder().decode(Manifest.self, from: json)
        } catch {
            throw FastWanError.manifest("\(source): \(error)")
        }
        guard decoded.format == manifestFormat else {
            throw FastWanError.manifest("\(source): format \"\(decoded.format)\", expected \"\(manifestFormat)\"")
        }
        self = decoded
    }

    /// Read `<directory>/manifest.json`.
    public init(directory: URL) throws {
        let url = directory.appendingPathComponent("manifest.json")
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw FastWanError.io("cannot read \(url.path): \(error)")
        }
        try self.init(json: data, source: url.path)
    }

    public var cfg: WanConfig { config }

    public func attentionFor(tokens: Int) throws -> AttentionEntry {
        if let a = attention.first(where: { $0.tokens == tokens }) {
            return a
        }
        let have = Array(Set(attention.map { $0.tokens })).sorted()
        throw FastWanError.noAttentionGraph(tokens: tokens, have: have)
    }

    public func taeFor(latentH h: Int, latentW w: Int) throws -> TaeChunkEntry {
        if let t = taeChunks.first(where: { $0.latentH == h && $0.latentW == w }) {
            return t
        }
        let have = Array(Set(taeChunks.map { "\($0.latentH)x\($0.latentW)" })).sorted()
        throw FastWanError.noTaeGraph(latentH: h, latentW: w, have: have)
    }

    /// Every graph file the manifest names, in a stable order.
    public var graphFiles: [String] {
        var out = [graphs.timeMods, graphs.textEmbed, graphs.patchEmbed, graphs.head]
        for b in graphs.blocks {
            out.append(contentsOf: [b.kv, b.qkv, b.post])
        }
        out.append(contentsOf: attention.map { $0.file })
        out.append(contentsOf: taeChunks.map { $0.file })
        return out
    }
}
