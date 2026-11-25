// Prompt, size and seed in; step timings and frames out. Frames show as they
// arrive and play at 16 fps when the clip is done.

import SwiftUI
import FastWanLiteRTCore

struct ContentView: View {
    @StateObject private var model = GenerationModel()

    var body: some View {
        NavigationStack {
            Form {
                Section("Export") {
                    TextField("Documents folder of the graphs", text: $model.graphsFolder)
                        .autocorrectionDisabled()
                        .onSubmit { model.refresh() }
                    Text(model.status).font(.footnote).foregroundStyle(.secondary)
                    Button("Rescan Documents") { model.refresh() }
                }
                Section("Prompt") {
                    if model.gallery.isEmpty {
                        Text("No prompt embeddings in Documents/gallery (fastwan-litert encode writes them).")
                            .font(.footnote).foregroundStyle(.secondary)
                    } else {
                        Picker("Embedding", selection: $model.prompt) {
                            ForEach(model.gallery, id: \.self) { Text($0).tag($0) }
                        }
                    }
                    Picker("Size", selection: $model.size) {
                        ForEach(model.sizes, id: \.self) { Text($0.description).tag($0) }
                    }
                    .onChange(of: model.size) { _ in model.updatePlan() }
                    HStack {
                        Text("Seed")
                        Spacer()
                        TextField("seed", text: $model.seedText)
                            .keyboardType(.numberPad)
                            .multilineTextAlignment(.trailing)
                    }
                    Toggle("Metal delegate (GPU)", isOn: $model.useMetal)
                }
                if !model.plan.isEmpty {
                    Section("Plan") {
                        Text(model.plan).font(.system(.footnote, design: .monospaced))
                    }
                }
                Section {
                    if model.running {
                        HStack {
                            ProgressView()
                            Text(model.status).padding(.leading, 8)
                            Spacer()
                            Button("Cancel") { model.cancel() }
                        }
                    } else {
                        Button("Generate") { model.generate() }
                            .disabled(model.manifest == nil || model.prompt.isEmpty)
                    }
                }
                if !model.timings.isEmpty {
                    Section("Timings") {
                        ForEach(model.timings) { t in
                            HStack {
                                Text(t.label)
                                Spacer()
                                Text(String(format: "%.0f ms", t.milliseconds)).monospacedDigit()
                            }
                        }
                    }
                }
                if !model.frames.isEmpty {
                    Section(model.done ? "Clip (\(model.frames.count) frames, 16 fps)" : "Frames so far (\(model.frames.count))") {
                        let index = min(model.frameIndex, model.frames.count - 1)
                        Image(decorative: model.frames[index], scale: 1.0)
                            .resizable()
                            .interpolation(.none)
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: .infinity)
                        if model.done {
                            HStack {
                                Button(model.playing ? "Pause" : "Play") {
                                    if model.playing { model.stopPlayback() } else { model.startPlayback() }
                                }
                                Spacer()
                                Text("frame \(index + 1)/\(model.frames.count)").monospacedDigit().foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
            .navigationTitle("FastWan LiteRT")
            .onAppear { model.refresh() }
        }
    }
}
