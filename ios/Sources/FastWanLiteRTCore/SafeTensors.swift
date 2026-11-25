// A tiny safetensors reader: 8-byte little-endian header length, a JSON
// header (`{name: {dtype, shape, data_offsets}, "__metadata__": {...}}`), then
// the raw little-endian tensor bytes. Enough for prompt embeddings
// (`prompt_embeds` F32 [512, 4096]) and the fixture's `expected.safetensors`.

import Foundation

public struct SafeTensors {
    public struct Entry {
        public var dtype: String
        public var shape: [Int]
        public var begin: Int
        public var end: Int
    }

    public let metadata: [String: String]
    public let entries: [String: Entry]
    private let data: Data
    private let dataStart: Int

    public init(data: Data) throws {
        guard data.count >= 8 else { throw FastWanError.safetensors("file is shorter than its 8-byte header length") }
        var n: UInt64 = 0
        for i in 0..<8 {
            n |= UInt64(data[data.startIndex + i]) << (8 * UInt64(i))
        }
        guard n <= UInt64(data.count - 8) else { throw FastWanError.safetensors("header length \(n) exceeds the file") }
        let headerLen = Int(n)
        let headerData = data.subdata(in: (data.startIndex + 8)..<(data.startIndex + 8 + headerLen))
        let json: Any
        do {
            json = try JSONSerialization.jsonObject(with: headerData, options: [])
        } catch {
            throw FastWanError.safetensors("header is not JSON: \(error)")
        }
        guard let dict = json as? [String: Any] else { throw FastWanError.safetensors("header is not a JSON object") }
        var meta: [String: String] = [:]
        var entries: [String: Entry] = [:]
        for (k, v) in dict {
            if k == "__metadata__" {
                if let m = v as? [String: Any] {
                    for (mk, mv) in m {
                        meta[mk] = "\(mv)"
                    }
                }
                continue
            }
            guard let e = v as? [String: Any],
                let dtype = e["dtype"] as? String,
                let shapeAny = e["shape"] as? [Any],
                let offsets = e["data_offsets"] as? [Any], offsets.count == 2
            else {
                throw FastWanError.safetensors("tensor `\(k)` has a malformed header entry")
            }
            let shape = shapeAny.compactMap { ($0 as? NSNumber)?.intValue }
            guard shape.count == shapeAny.count,
                let b = (offsets[0] as? NSNumber)?.intValue, let en = (offsets[1] as? NSNumber)?.intValue, b <= en
            else {
                throw FastWanError.safetensors("tensor `\(k)` has a malformed shape or data_offsets")
            }
            entries[k] = Entry(dtype: dtype, shape: shape, begin: b, end: en)
        }
        self.metadata = meta
        self.entries = entries
        self.data = data
        self.dataStart = data.startIndex + 8 + headerLen
    }

    public init(url: URL) throws {
        let d: Data
        do {
            d = try Data(contentsOf: url)
        } catch {
            throw FastWanError.io("cannot read \(url.path): \(error)")
        }
        try self.init(data: d)
    }

    public var names: [String] { entries.keys.sorted() }

    func bytes(of name: String) throws -> (Entry, Data) {
        guard let e = entries[name] else {
            throw FastWanError.safetensors("no tensor `\(name)` (it has \(names))")
        }
        let lo = dataStart + e.begin, hi = dataStart + e.end
        guard hi <= data.endIndex else {
            throw FastWanError.safetensors("tensor `\(name)` runs past the end of the file")
        }
        return (e, data.subdata(in: lo..<hi))
    }

    /// An F32 (or F16) tensor as float32. Assumes a little-endian host (every Apple platform).
    public func float32(_ name: String) throws -> Tensor {
        let (e, raw) = try bytes(of: name)
        let count = e.shape.reduce(1, *)
        switch e.dtype {
        case "F32":
            guard raw.count == count * 4 else {
                throw FastWanError.safetensors("tensor `\(name)`: \(raw.count) bytes for shape \(e.shape) F32")
            }
            var out = [Float](repeating: 0, count: count)
            _ = out.withUnsafeMutableBytes { dst in
                raw.copyBytes(to: dst)
            }
            return Tensor(shape: e.shape, data: out)
        case "F16":
            guard raw.count == count * 2 else {
                throw FastWanError.safetensors("tensor `\(name)`: \(raw.count) bytes for shape \(e.shape) F16")
            }
            var out = [Float](repeating: 0, count: count)
            raw.withUnsafeBytes { (buf: UnsafeRawBufferPointer) in
                for i in 0..<count {
                    let bits = UInt16(buf[2 * i]) | (UInt16(buf[2 * i + 1]) << 8)
                    out[i] = SafeTensors.halfToFloat(bits)
                }
            }
            return Tensor(shape: e.shape, data: out)
        default:
            throw FastWanError.safetensors("tensor `\(name)` is \(e.dtype); only F32 and F16 are read as float")
        }
    }

    /// A U8 tensor and its shape.
    public func uint8(_ name: String) throws -> (shape: [Int], data: [UInt8]) {
        let (e, raw) = try bytes(of: name)
        guard e.dtype == "U8" else {
            throw FastWanError.safetensors("tensor `\(name)` is \(e.dtype), expected U8")
        }
        let count = e.shape.reduce(1, *)
        guard raw.count == count else {
            throw FastWanError.safetensors("tensor `\(name)`: \(raw.count) bytes for shape \(e.shape) U8")
        }
        return (e.shape, [UInt8](raw))
    }

    /// IEEE binary16 → Float, integer math only (portable to every Swift target).
    static func halfToFloat(_ h: UInt16) -> Float {
        let sign = UInt32(h & 0x8000) << 16
        let exp = UInt32((h >> 10) & 0x1F)
        var mant = UInt32(h & 0x03FF)
        var bits: UInt32
        if exp == 0 {
            if mant == 0 {
                bits = sign
            } else {
                // subnormal: normalise
                var e: UInt32 = 127 - 15 + 1
                while (mant & 0x0400) == 0 {
                    mant <<= 1
                    e -= 1
                }
                mant &= 0x03FF
                bits = sign | (e << 23) | (mant << 13)
            }
        } else if exp == 0x1F {
            bits = sign | 0x7F80_0000 | (mant << 13)
        } else {
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13)
        }
        return Float(bitPattern: bits)
    }
}
