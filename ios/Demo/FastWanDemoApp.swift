// The demo app's entry point. Create an iOS App project in Xcode (SwiftUI,
// Swift), delete its generated App/ContentView files, add these four files,
// add the `FastWanLiteRT` package (Core + Runtime products) and the
// `TensorFlowLiteSwift` pod; see Demo/README.md.

import SwiftUI

@main
struct FastWanDemoApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
