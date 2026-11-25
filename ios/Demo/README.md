# Demo app

SwiftUI sources for a small iOS app around the package: pick a prompt
embedding, a size and a seed, run `Pipeline.generate` off the main actor, show
the step timings, show frames as the TAEHV chunks decode, play the clip at
16 fps when done.

The sources are here, not an `.xcodeproj`: create the project in Xcode.

1. Xcode → File → New → Project → iOS → App. Interface SwiftUI, language
   Swift. Name it `FastWanDemo`. Minimum deployment iOS 16.
2. Delete the generated `FastWanDemoApp.swift` and `ContentView.swift`; add
   the four files of this folder to the app target
   (`FastWanDemoApp.swift`, `ContentView.swift`, `GenerationModel.swift`,
   `FrameImage.swift`).
3. File → Add Package Dependencies… → Add Local… → choose the `ios/` folder
   of this repository. Add the `FastWanLiteRTCore` and `FastWanLiteRTRuntime`
   products to the app target.
4. The LiteRT backend. The TensorFlow Lite Swift API is published as a
   CocoaPod, so close Xcode and in the project folder:

   ```ruby
   # Podfile
   platform :ios, '16.0'
   use_frameworks!
   target 'FastWanDemo' do
     pod 'TensorFlowLiteSwift', '~> 2.17.0', :subspecs => ['Metal']
   end
   ```

   `pod install`, open the `.xcworkspace`. The `FastWanLiteRTRuntime`
   sources compile their LiteRT code only where `import TensorFlowLite`
   resolves. A Swift package target built by Xcode does not see the pod's
   framework search paths, so if `LiteRTGraphStore` is reported missing, add
   `ios/Sources/FastWanLiteRTRuntime/LiteRTGraph.swift` directly to the app
   target instead of (or as well as) the package product: the file then
   compiles inside the app, where the pod is linked. `// TODO verify` on a
   real Xcode: this is the integration we could not run here.
5. Info.plist: add `UIFileSharingEnabled = YES` and
   `LSSupportsOpeningDocumentsInPlace = YES` so the app's Documents folder
   appears in the Files app and in Finder (device → Files tab).
6. Copy an export to `Documents/graphs/` (the whole directory:
   `manifest.json` and every `.tflite`) and prompt embeddings
   (`<name>.safetensors` with `prompt_embeds` `[512, 4096]` F32, written by
   `fastwan-litert encode`) to `Documents/gallery/`. "Rescan Documents" in
   the app.

The Metal delegate is on by default (`MetalDelegate`,
`isPrecisionLossAllowed = true`: the contract judges the f16 GPU path at
2e-2). Turn it off to run on the CPU (XNNPACK).

For the tiny fixture (`fixtures/tiny`): copy its directory to
`Documents/graphs` and write its `prompt_embeds` to
`Documents/gallery/tiny.safetensors` (`expected.safetensors` itself works:
the reader picks `prompt_embeds` by name), size 32×48×9, seed 7.
`FixtureCheck.run(store:directory:)` compares against `expected.safetensors`
the way `fastwan-litert check` does.
