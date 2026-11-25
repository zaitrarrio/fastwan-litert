// RGBA bytes → CGImage, for `Image(decorative:scale:)`.

import CoreGraphics
import Foundation
import FastWanLiteRTCore

enum FrameImage {
    /// One frame of a `Frames` event or a `Video` as a `CGImage` (copies the bytes).
    static func make(rgba: ArraySlice<UInt8>, width: Int, height: Int) -> CGImage? {
        guard width > 0, height > 0, rgba.count == width * height * 4 else { return nil }
        let data = Data(rgba)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: info,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )
    }

    static func images(of frames: Frames) -> [CGImage] {
        (0..<frames.count).compactMap { make(rgba: frames.frame($0), width: frames.width, height: frames.height) }
    }
}
