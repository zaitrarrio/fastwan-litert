// swift-tools-version: 5.9
// FastWanLiteRT: FastWan 2.1 1.3B text-to-video on LiteRT, the iOS runtime.
//
// - FastWanLiteRTCore    pure Swift (Foundation only): host math, manifest, pipeline over `Graph`.
// - FastWanLiteRTRuntime `Graph` / `GraphStore` over the TensorFlow Lite Swift API (LiteRT for iOS).
//   Its sources are behind `#if canImport(TensorFlowLite)`: link the `TensorFlowLiteSwift`
//   CocoaPod (with the `Metal` subspec) into the app, see README.md. Without it the target
//   compiles to a stub, so `swift test` of the core runs on any Mac.
// - FastWanLiteRTCoreTests XCTest: the pinned host-math values and the pipeline orchestration.

import PackageDescription

let package = Package(
    name: "FastWanLiteRT",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        .library(name: "FastWanLiteRTCore", targets: ["FastWanLiteRTCore"]),
        .library(name: "FastWanLiteRTRuntime", targets: ["FastWanLiteRTRuntime"]),
    ],
    dependencies: [
        // TODO verify: TensorFlow Lite Swift is published as the CocoaPod `TensorFlowLiteSwift`
        // (2.17.0, subspec `Metal` for the GPU delegate), not as a Swift package, and the newer
        // `LiteRTSwift` pod (nightly) is the successor. If a Swift package of either appears,
        // add it here and to FastWanLiteRTRuntime's dependencies, e.g.
        //   .package(url: "https://github.com/<mirror>/TensorFlowLiteSwift.git", from: "2.17.0"),
    ],
    targets: [
        .target(
            name: "FastWanLiteRTCore",
            dependencies: [],
            path: "Sources/FastWanLiteRTCore"
        ),
        .target(
            name: "FastWanLiteRTRuntime",
            dependencies: ["FastWanLiteRTCore"],
            path: "Sources/FastWanLiteRTRuntime"
        ),
        .testTarget(
            name: "FastWanLiteRTCoreTests",
            dependencies: ["FastWanLiteRTCore"],
            path: "Tests/FastWanLiteRTCoreTests",
            resources: [.copy("Resources/manifest.json")]
        ),
    ]
)
