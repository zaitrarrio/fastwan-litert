// Architecture constants (python/fastwan_litert/config.py; fastwan-rs `config.rs`).

import Foundation

/// The Wan 2.1 DiT. Batch is always 1: the DMD-distilled FastWan models run
/// without classifier-free guidance.
public struct WanConfig: Codable, Equatable, Sendable {
    public var patch: Int
    public var heads: Int
    public var headDim: Int
    public var inChannels: Int
    public var outChannels: Int
    public var textDim: Int
    public var textLen: Int
    public var freqDim: Int
    public var ffnDim: Int
    public var layers: Int
    public var eps: Double
    public var ropeMaxSeqLen: Int
    public var ropeTheta: Double

    enum CodingKeys: String, CodingKey {
        case patch, heads, layers, eps
        case headDim = "head_dim"
        case inChannels = "in_channels"
        case outChannels = "out_channels"
        case textDim = "text_dim"
        case textLen = "text_len"
        case freqDim = "freq_dim"
        case ffnDim = "ffn_dim"
        case ropeMaxSeqLen = "rope_max_seq_len"
        case ropeTheta = "rope_theta"
    }

    public init(
        patch: Int, heads: Int, headDim: Int, inChannels: Int, outChannels: Int,
        textDim: Int, textLen: Int, freqDim: Int, ffnDim: Int, layers: Int,
        eps: Double, ropeMaxSeqLen: Int, ropeTheta: Double
    ) {
        self.patch = patch
        self.heads = heads
        self.headDim = headDim
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.textDim = textDim
        self.textLen = textLen
        self.freqDim = freqDim
        self.ffnDim = ffnDim
        self.layers = layers
        self.eps = eps
        self.ropeMaxSeqLen = ropeMaxSeqLen
        self.ropeTheta = ropeTheta
    }

    /// `d = heads · head_dim`.
    public var dim: Int { heads * headDim }

    public var patchFeaturesIn: Int { patch * patch * inChannels }

    public var patchFeaturesOut: Int { patch * patch * outChannels }

    /// `(t, h, w)` widths of one head (Diffusers `WanRotaryPosEmbed`:
    /// `h = w = 2·(d/6)`, `t = d − h − w`; 44/42/42 at d = 128).
    public var ropeSplit: (t: Int, h: Int, w: Int) {
        let hw = 2 * (headDim / 6)
        return (headDim - 2 * hw, hw, hw)
    }

    /// Dense parameter count of the DiT (VSA gates excluded).
    public var params: Int {
        let d = dim
        func lin(_ i: Int, _ o: Int) -> Int { i * o + o }
        let block = 8 * lin(d, d) + 4 * d + 2 * d + lin(d, ffnDim) + lin(ffnDim, d) + 6 * d
        return layers * block
            + lin(patchFeaturesIn, d)
            + lin(freqDim, d)
            + lin(d, d)
            + lin(d, 6 * d)
            + lin(textDim, d)
            + lin(d, d)
            + lin(d, patchFeaturesOut)
            + 2 * d
    }

    /// `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` (and Wan 2.1 T2V 1.3B).
    public static let fastwan13B = WanConfig(
        patch: 2, heads: 12, headDim: 128, inChannels: 16, outChannels: 16,
        textDim: 4096, textLen: 512, freqDim: 256, ffnDim: 8960, layers: 30,
        eps: 1e-6, ropeMaxSeqLen: 1024, ropeTheta: 10000.0
    )

    /// A graph small enough for the CPU in a unit test (fastwan-rs's
    /// `WanConfig::tiny`): head_dim a multiple of 6 so the 3-axis RoPE split
    /// is non-trivial.
    public static let tiny = WanConfig(
        patch: 2, heads: 2, headDim: 12, inChannels: 4, outChannels: 4,
        textDim: 16, textLen: 8, freqDim: 16, ffnDim: 40, layers: 2,
        eps: 1e-6, ropeMaxSeqLen: 64, ropeTheta: 10000.0
    )
}

/// Wan 2.1 latent statistics (LightTAE denormalisation).
public enum Wan21Latent {
    public static let mean: [Float] = [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    ]
    public static let std: [Float] = [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
    ]
}

/// TAEHV (madebyollin/taehv) decoder for Wan 2.1 latents.
public struct TaeConfig: Codable, Equatable, Sendable {
    public var latentChannels: Int
    public var widths: [Int]
    public var tgrow: [Int]
    public var memBlocks: Int
    public var patch: Int
    public var denormalize: Bool

    enum CodingKeys: String, CodingKey {
        case widths, tgrow, patch, denormalize
        case latentChannels = "latent_channels"
        case memBlocks = "mem_blocks"
    }

    public init(latentChannels: Int, widths: [Int], tgrow: [Int], memBlocks: Int, patch: Int, denormalize: Bool) {
        self.latentChannels = latentChannels
        self.widths = widths
        self.tgrow = tgrow
        self.memBlocks = memBlocks
        self.patch = patch
        self.denormalize = denormalize
    }

    public var tUpscale: Int { tgrow.reduce(1, *) }

    /// Priming frames emitted first and dropped: `4T − 3` frames for Wan.
    public var framesToTrim: Int { tUpscale - 1 }

    public var spatialScale: Int { 8 * patch }

    public var outChannels: Int { 3 * patch * patch }

    public static let taew21 = TaeConfig(latentChannels: 16, widths: [256, 128, 64, 64], tgrow: [1, 2, 2], memBlocks: 3, patch: 1, denormalize: false)

    /// LightX2V's LightTAE: TAEHV's network on raw (denormalised) latents.
    public static let lighttaew21 = TaeConfig(latentChannels: 16, widths: [256, 128, 64, 64], tgrow: [1, 2, 2], memBlocks: 3, patch: 1, denormalize: true)

    public static let taew22 = TaeConfig(latentChannels: 48, widths: [256, 128, 64, 64], tgrow: [1, 2, 2], memBlocks: 3, patch: 2, denormalize: false)

    public static let tiny = TaeConfig(latentChannels: 4, widths: [8, 6, 4, 4], tgrow: [1, 2, 2], memBlocks: 3, patch: 1, denormalize: false)
}
