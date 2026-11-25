// A dense row-major float32 tensor: `[Float]` plus a shape. Every graph input
// and output in the contract is one of these (no batch dimension).

import Foundation

public struct Tensor: Equatable {
    public var shape: [Int]
    public var data: [Float]

    public init(shape: [Int], data: [Float]) {
        precondition(shape.reduce(1, *) == data.count, "tensor data \(data.count) does not match shape \(shape)")
        self.shape = shape
        self.data = data
    }

    public init(scalar: Float) {
        self.init(shape: [1], data: [scalar])
    }

    public static func zeros(_ shape: [Int]) -> Tensor {
        Tensor(shape: shape, data: [Float](repeating: 0, count: shape.reduce(1, *)))
    }

    public var count: Int { data.count }

    public var rank: Int { shape.count }

    /// Elements per row (everything but the first axis).
    public var rowStride: Int { shape.dropFirst().reduce(1, *) }

    /// The first axis.
    public var rows: Int { shape.first ?? 1 }

    public func reshaped(_ newShape: [Int]) -> Tensor {
        precondition(newShape.reduce(1, *) == data.count, "cannot reshape \(shape) to \(newShape)")
        return Tensor(shape: newShape, data: data)
    }

    /// Rows `r` of the first axis (the shape keeps its trailing dims).
    public func rows(_ r: Range<Int>) -> Tensor {
        precondition(r.lowerBound >= 0 && r.upperBound <= rows, "rows \(r) out of \(rows)")
        let s = rowStride
        let slice = Array(data[(r.lowerBound * s)..<(r.upperBound * s)])
        return Tensor(shape: [r.count] + Array(shape.dropFirst()), data: slice)
    }

    /// Row `i` of the first axis with that axis dropped: `[a, b, c][i]` is `[b, c]`.
    public func row(_ i: Int) -> Tensor {
        precondition(i >= 0 && i < rows, "row \(i) out of \(rows)")
        let s = rowStride
        return Tensor(shape: Array(shape.dropFirst()), data: Array(data[(i * s)..<((i + 1) * s)]))
    }

    /// Concatenate along the first axis; the trailing dims must agree.
    public static func concatRows(_ parts: [Tensor]) -> Tensor {
        precondition(!parts.isEmpty, "concat of nothing")
        let tail = Array(parts[0].shape.dropFirst())
        var total = 0
        var data: [Float] = []
        data.reserveCapacity(parts.reduce(0) { $0 + $1.count })
        for p in parts {
            precondition(Array(p.shape.dropFirst()) == tail, "concat shapes \(parts[0].shape) and \(p.shape)")
            total += p.rows
            data.append(contentsOf: p.data)
        }
        return Tensor(shape: [total] + tail, data: data)
    }

    public subscript(flat i: Int) -> Float {
        get { data[i] }
        set { data[i] = newValue }
    }

    public func shapeEquals(_ other: [Int]) -> Bool { shape == other }
}
