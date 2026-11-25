import Foundation

/// Every error the core raises. Messages name the mismatch; nothing falls back silently.
public enum FastWanError: Error, CustomStringConvertible, LocalizedError {
    case invalidGeometry(String)
    case manifest(String)
    case noAttentionGraph(tokens: Int, have: [Int])
    case noTaeGraph(latentH: Int, latentW: Int, have: [String])
    case shape(String)
    case graph(String)
    case safetensors(String)
    case io(String)

    public var description: String {
        switch self {
        case .invalidGeometry(let m): return m
        case .manifest(let m): return m
        case .noAttentionGraph(let tokens, let have):
            return "this export has no attention graph for \(tokens) tokens (it has \(have)); re-export with --size"
        case .noTaeGraph(let h, let w, let have):
            return "this export has no TAEHV graph for latent \(h)x\(w) (it has \(have)); re-export with --size"
        case .shape(let m): return m
        case .graph(let m): return m
        case .safetensors(let m): return m
        case .io(let m): return m
        }
    }

    public var errorDescription: String? { description }
}
