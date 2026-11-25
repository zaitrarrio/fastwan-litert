// Pixel ↔ latent ↔ token shapes for a Wan 2.1 clip (python geometry.py; fastwan-core `geometry.rs`).

import Foundation

public let vaeSpatial = 8
public let vaeTemporal = 4

public struct Geometry: Codable, Equatable, Hashable, Sendable, CustomStringConvertible {
    public var width: Int
    public var height: Int
    public var frames: Int

    public init(width: Int, height: Int, frames: Int) {
        self.width = width
        self.height = height
        self.frames = frames
    }

    public func validate(_ cfg: WanConfig) throws {
        let cell = vaeSpatial * cfg.patch
        if width <= 0 || height <= 0 || width % cell != 0 || height % cell != 0 {
            throw FastWanError.invalidGeometry("\(width)x\(height) is not a positive multiple of \(cell)")
        }
        if frames <= 0 || (frames - 1) % vaeTemporal != 0 {
            throw FastWanError.invalidGeometry("\(frames) frames is not 4n+1")
        }
    }

    /// Latent `(T, H, W)`.
    public var latent: (t: Int, h: Int, w: Int) {
        ((frames - 1) / vaeTemporal + 1, height / vaeSpatial, width / vaeSpatial)
    }

    public func grid(_ cfg: WanConfig) -> (t: Int, h: Int, w: Int) {
        let l = latent
        return (l.t, l.h / cfg.patch, l.w / cfg.patch)
    }

    public func tokens(_ cfg: WanConfig) -> Int {
        let g = grid(cfg)
        return g.t * g.h * g.w
    }

    public func latentLen(_ cfg: WanConfig) -> Int {
        let l = latent
        return cfg.inChannels * l.t * l.h * l.w
    }

    /// `WxHxF`, e.g. `832x480x81`.
    public static func parse(_ s: String) throws -> Geometry {
        let parts = s.lowercased().split(separator: "x").map { Int($0) }
        guard parts.count == 3, let w = parts[0], let h = parts[1], let f = parts[2] else {
            throw FastWanError.invalidGeometry("geometry `\(s)` is not WIDTHxHEIGHTxFRAMES")
        }
        return Geometry(width: w, height: h, frames: f)
    }

    public var description: String { "\(width)x\(height)x\(frames)" }

    /// FastWan 2.1 1.3B's served default: 5 s at 16 fps.
    public static let fastwan480p = Geometry(width: 832, height: 480, frames: 81)
    /// fastwan-rs's phone tier.
    public static let mobile480x272 = Geometry(width: 480, height: 272, frames: 33)
}
