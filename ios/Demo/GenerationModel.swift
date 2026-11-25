// The demo's state: the export and gallery in Documents, the chosen prompt,
// size and seed, and one generation at a time running off the main actor.

import Combine
import CoreGraphics
import Foundation
import FastWanLiteRTCore
#if canImport(TensorFlowLite)
import FastWanLiteRTRuntime
#endif

@MainActor
final class GenerationModel: ObservableObject {
    struct StepTiming: Identifiable {
        let id: Int
        let label: String
        let milliseconds: Double
    }

    /// Documents/<graphs>: the export the user copied with the Files app or Finder.
    @Published var graphsFolder = "graphs"
    @Published var manifest: Manifest?
    @Published var gallery: [String] = []
    @Published var prompt: String = ""
    @Published var size: Geometry = .mobile480x272
    @Published var seedText: String = "7"
    var seed: UInt64 { UInt64(seedText.trimmingCharacters(in: .whitespaces)) ?? 7 }
    @Published var useMetal = true
    @Published var status = "Copy an export (manifest.json + .tflite) into Documents/graphs and prompt embeddings (.safetensors) into Documents/gallery."
    @Published var plan: String = ""
    @Published var timings: [StepTiming] = []
    @Published var frames: [CGImage] = []
    @Published var running = false
    @Published var done = false
    @Published var playing = false
    @Published var frameIndex = 0

    private var playTimer: Timer?
    private var task: Task<Void, Never>?

    var documents: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    var graphsURL: URL { documents.appendingPathComponent(graphsFolder) }
    var galleryURL: URL { documents.appendingPathComponent("gallery") }

    var sizes: [Geometry] {
        if let m = manifest, !m.sizes.isEmpty { return m.sizes }
        return [.mobile480x272]
    }

    func refresh() {
        do {
            let m = try Manifest(directory: graphsURL)
            manifest = m
            if !m.sizes.contains(size) {
                size = m.sizes.contains(.mobile480x272) ? .mobile480x272 : (m.sizes.first ?? .mobile480x272)
            }
            status = "\(m.model) export, \(m.weights) weights, \(m.graphs.blocks.count) blocks, chunk \(m.chunk)."
            updatePlan()
        } catch {
            manifest = nil
            status = "\(error)"
        }
        let files = (try? FileManager.default.contentsOfDirectory(atPath: galleryURL.path)) ?? []
        gallery = files.filter { $0.hasSuffix(".safetensors") }.map { String($0.dropLast(".safetensors".count)) }.sorted()
        if prompt.isEmpty || !gallery.contains(prompt) {
            prompt = gallery.first ?? ""
        }
    }

    func updatePlan() {
        guard let m = manifest else {
            plan = ""
            return
        }
        do {
            // The phone tier: about 2 GB for the app.
            let p = try Plan.make(manifest: m, geometry: size, budgetBytes: 2 << 30, residentBlocks: 2)
            plan = p.description
        } catch {
            plan = "\(error)"
        }
    }

    func generate() {
        guard !running else { return }
        guard let m = manifest else {
            status = "No export at \(graphsURL.path)."
            return
        }
        guard !prompt.isEmpty else {
            status = "Pick a prompt embedding from Documents/gallery."
            return
        }
        stopPlayback()
        running = true
        done = false
        frames = []
        timings = []
        frameIndex = 0
        status = "Opening graphs…"
        let embedsURL = galleryURL.appendingPathComponent("\(prompt).safetensors")
        let graphsURL = self.graphsURL
        let size = self.size
        let seed = self.seed
        let useMetal = self.useMetal
        task = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let embeds = try SafeTensors(url: embedsURL).float32("prompt_embeds")
                let store = try GenerationModel.openStore(manifest: m, directory: graphsURL, useMetal: useMetal)
                let pipe = Pipeline(store: store)
                let (video, tm) = try pipe.generate(embeds: embeds, geometry: size, seed: seed) { event in
                    Task { @MainActor [weak self] in self?.handle(event) }
                }
                store.dropBlocks()
                await MainActor.run { [weak self] in
                    guard let self = self else { return }
                    self.timings.append(StepTiming(id: 1000, label: "decode", milliseconds: tm.decodeSeconds * 1000))
                    let total = tm.prepareSeconds + tm.stepSeconds.reduce(0, +) + tm.decodeSeconds
                    self.status = "\(video.frames) frames \(video.width)x\(video.height) in \(String(format: "%.1f", total)) s"
                    self.running = false
                    self.done = true
                    self.startPlayback()
                }
            } catch {
                await MainActor.run { [weak self] in
                    self?.status = "Failed: \(error)"
                    self?.running = false
                }
            }
        }
    }

    nonisolated static func openStore(manifest: Manifest, directory: URL, useMetal: Bool) throws -> GraphStore {
        #if canImport(TensorFlowLite)
        var options = LiteRTOptions()
        options.useMetal = useMetal
        options.residentBlocks = 2
        let store = LiteRTGraphStore(manifest: manifest, directory: directory, options: options)
        let missing = store.missingFiles()
        guard missing.isEmpty else {
            throw FastWanError.io("missing graphs in \(directory.lastPathComponent): \(missing.prefix(5).joined(separator: ", "))\(missing.count > 5 ? ", …" : "")")
        }
        return store
        #else
        throw FastWanError.graph("this build has no LiteRT backend: link the TensorFlowLiteSwift pod (see ios/README.md)")
        #endif
    }

    private func handle(_ event: PipelineEvent) {
        switch event {
        case .prepared(let secs, let tokens):
            timings.append(StepTiming(id: -1, label: "prepare (\(tokens) tokens)", milliseconds: secs * 1000))
            status = "Denoising…"
        case .step(let index, let total, let secs):
            timings.append(StepTiming(id: index, label: "step \(index + 1)/\(total)", milliseconds: secs * 1000))
            status = index + 1 < total ? "Denoising… step \(index + 1)/\(total) done" : "Decoding…"
        case .frames(let f):
            frames.append(contentsOf: FrameImage.images(of: f))
            frameIndex = max(0, frames.count - 1)
        }
    }

    func cancel() {
        task?.cancel()  // the pipeline has no cancellation points; this only drops the result
        stopPlayback()
        running = false
        status = "Cancelled (the current step runs to its end)."
    }

    func startPlayback() {
        guard frames.count > 1 else { return }
        playing = true
        frameIndex = 0
        playTimer?.invalidate()
        playTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 16.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self = self, self.playing, !self.frames.isEmpty else { return }
                self.frameIndex = (self.frameIndex + 1) % self.frames.count
            }
        }
    }

    func stopPlayback() {
        playing = false
        playTimer?.invalidate()
        playTimer = nil
    }
}
