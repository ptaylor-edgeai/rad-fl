// GossipFrame.swift
//
// Wire format for gossip weight exchange between peers.
//
// Design goals (see project discussion):
//   - Dense-only today, byte-cost equivalent to "just send the raw weights"
//     so RQ1 (Swift vs Python efficiency) isn't carrying any sparse-readiness tax.
//   - Self-describing per-tensor structure so a future SPARSE_COO encoding can be
//     added as a new case without changing the outer framing or breaking the
//     dense baseline already collected.
//
// That second goal is now doing its job. RQ1's compression mechanisms are added
// as new `GossipEncoding` cases, so the outer framing, the header, and every
// byte of the dense path are unchanged. A dense run and a compressed run differ
// in exactly one field on the wire.
//
// Layout:
//   Header:
//     [4 bytes] magic              (UInt32) — 0x52414644 ("RAFD")
//     [4 bytes] protocol version   (UInt32) — currently 1
//     [4 bytes] message type       (UInt32) — see MessageType
//     [4 bytes] sender node ID     (UInt32)
//     [4 bytes] round number       (UInt32)
//     [4 bytes] sample count       (UInt32)
//     [4 bytes] tensor count       (UInt32)
//   Per tensor:
//     [4 bytes] tensor ID         (UInt32) — stable index into the model's parameter list
//     [4 bytes] encoding          (UInt32) — see GossipEncoding
//     [4 bytes] element count     (UInt32) — ALWAYS the dense element count, for
//                                            every encoding. See the note below.
//     payload:
//       dense:      [elementCount * 4]  Float32, host byte order
//       denseFP16:  [elementCount * 2]  Float16 bit patterns
//       denseINT8:  [4] Float32 min, [4] Float32 scale, [elementCount] UInt8
//       sparseCOO:  [4] UInt32 nnz, [nnz * 4] UInt32 indices, [nnz * 4] Float32 values
//
// ELEMENT COUNT IS ALWAYS THE DENSE LENGTH. The original comment described it
// as "total elements for dense; nonzero count for sparse", which would have made
// a sparse tensor un-reconstructable: the receiver needs the dense length to
// build the zero-filled array, and the nonzero count to know how many pairs
// follow. Those are two different numbers, and the sparse payload carries its own
// nnz field for the second. Fixing this before any sparse data existed avoided a
// wire-format change mid-campaign.
//
// PROTOCOL VERSION IS NOT BUMPED. A node built before these encodings existed
// rejects an unknown encoding with `unknownEncoding`, which is a clean, specific
// failure at the tensor it occurred on. Bumping the version would instead reject
// the whole frame at the header, including dense frames that an old node could
// have handled perfectly well — so a new binary could no longer interoperate
// with an old one even when sending dense. Keeping version 1 preserves that.
//
// NIO framing: this struct is the *application* frame. NIOFrameDecoder prefixes
// it with a 4-byte little-endian length so ByteToMessageDecoder can reassemble
// frames split across TCP packets.

import Foundation
import NIOCore

/// How one tensor's values are represented on the wire.
///
/// Every encoding reconstructs to a DENSE `[Float32]` on the receiving side.
/// This is the property that keeps compression confined to the transport:
/// `GossipAggregator`, `SimpleCNN` and the round loop never see a compressed
/// tensor and require no changes for a new encoding to be added. A compression
/// experiment and a dense baseline therefore differ in one place only, which is
/// what makes the comparison attributable.
public enum GossipEncoding: UInt32, Sendable {
    /// Flat Float32. The baseline, and what every run before RQ1 used.
    case dense = 0

    /// Coordinate-format sparse: nonzero indices and values.
    ///
    /// Reserved on the wire since the first version of this file and now
    /// implemented, for RQ1's top-k mechanism. Only worth sending below roughly
    /// 50% density — each nonzero costs 8 bytes (index + value) against 4 for a
    /// dense element, so a half-full sparse tensor is exactly as large as the
    /// dense one and a denser one is larger. `TensorCodec.encodedByteCount`
    /// makes that break-even checkable before choosing.
    case sparseCOO = 1

    /// IEEE-754 half precision. Exactly half the bytes, no metadata.
    ///
    /// Measured on He-initialised conv weights: max absolute error 3.2e-04,
    /// mean 3.8e-05. Lossy, but far below the 1e-2 scale at which cross-node
    /// divergence starts to cost accuracy in this system.
    case denseFP16 = 2

    /// Affine per-tensor 8-bit quantisation: min + scale + one byte per element.
    ///
    /// A quarter of the bytes. Measured max absolute error 3.6e-03 — an order of
    /// magnitude worse than FP16, and the relative error on near-zero weights is
    /// unbounded, since an affine scheme spends its resolution uniformly across
    /// the range rather than where the values are. That matters more for
    /// gradients than for weights; it is a real trade, not a free saving.
    case denseINT8 = 3

    /// Whether this encoding loses information. Recorded so a run can report
    /// what it did without inferring it from the mechanism name.
    public var isLossy: Bool {
        switch self {
        case .dense: return false
        case .sparseCOO: return true      // by construction: dropped values are lost
        case .denseFP16, .denseINT8: return true
        }
    }
}

public enum GossipMessageType: UInt32, Sendable {
    case weightsFull = 0      // full dense model push (current implementation)
    case weightsDelta = 1     // reserved: delta against last-known peer state
    case heartbeat = 2        // liveness ping — see dead-peer handling notes below
    case roundComplete = 3    // signals this node finished local aggregation for the round
}

/// Quantisation and sparsification, kept separate from the codec's buffer
/// handling so the numerical behaviour can be tested without NIO.
public enum TensorCodec {

    // MARK: - FP16

    /// `Float16` is used directly rather than a hand-rolled IEEE-754 conversion.
    /// Both targets are arm64 — the Pi Zero 2W's Cortex-A53 and the development
    /// Mac — where Swift provides it natively. On a platform without it this
    /// would need a manual conversion; that is not a platform this project
    /// builds for, and writing one speculatively would add untested code to the
    /// path every gossip payload passes through.
    public static func toFP16(_ values: [Float32]) -> [UInt16] {
        values.map { Float16($0).bitPattern }
    }

    public static func fromFP16(_ bits: [UInt16]) -> [Float32] {
        bits.map { Float32(Float16(bitPattern: $0)) }
    }

    // MARK: - INT8

    public struct AffineScale: Sendable {
        public let min: Float32
        public let scale: Float32
    }

    /// Affine quantisation over the tensor's own range.
    ///
    /// Per-tensor rather than per-model: a model's parameter tensors differ in
    /// scale by orders of magnitude (a conv kernel against a bias vector), and a
    /// single global range would quantise the smaller ones to a handful of
    /// distinct levels. Per-tensor costs 8 bytes each, which against a 432-element
    /// tensor is 1.8% overhead and against a larger one is negligible.
    public static func toINT8(_ values: [Float32]) -> ([UInt8], AffineScale) {
        guard let lo = values.min(), let hi = values.max() else {
            return ([], AffineScale(min: 0, scale: 1))
        }
        // A constant tensor has zero range. Scale 1 makes dequantisation return
        // `lo` for every element, which is correct; a zero scale would produce
        // NaN on the way back.
        let scale = hi > lo ? (hi - lo) / 255.0 : 1.0
        let quantised = values.map { v -> UInt8 in
            let q = ((v - lo) / scale).rounded()
            return UInt8(Swift.max(0, Swift.min(255, q)))
        }
        return (quantised, AffineScale(min: lo, scale: scale))
    }

    public static func fromINT8(_ quantised: [UInt8], _ s: AffineScale) -> [Float32] {
        quantised.map { s.min + Float32($0) * s.scale }
    }

    // MARK: - Top-k sparsification

    public struct Sparse: Sendable {
        public let denseCount: Int
        public let indices: [UInt32]
        public let values: [Float32]
    }

    /// Keeps the `k` largest-magnitude values, discarding the rest.
    ///
    /// Returns the DISCARDED residual alongside, which the caller accumulates and
    /// adds to the next round's update. Without that feedback, values below the
    /// threshold are never transmitted at all, and a parameter that is
    /// consistently small — but consistently nonzero — never reaches its peers.
    /// Under label skew this diverges rather than merely converging slowly, which
    /// is the effect E1.2's ablation exists to measure.
    public static func topK(_ values: [Float32], k: Int) -> (sparse: Sparse, residual: [Float32]) {
        let keep = Swift.max(1, Swift.min(k, values.count))
        if keep >= values.count {
            return (Sparse(denseCount: values.count,
                           indices: Array(0..<UInt32(values.count)),
                           values: values),
                    Array(repeating: 0, count: values.count))
        }

        // Partial selection by magnitude. A full sort is O(n log n) on a tensor
        // that can hold hundreds of thousands of elements, once per tensor per
        // round, on a Cortex-A53 — worth avoiding, but correctness first: this
        // sorts indices and takes a prefix, which is simple and obviously right.
        // If it shows up in the phase timings, a quickselect belongs here.
        let order = (0..<values.count).sorted { abs(values[$0]) > abs(values[$1]) }
        let kept = order.prefix(keep).sorted()

        var residual = values
        var idx: [UInt32] = []
        var vals: [Float32] = []
        idx.reserveCapacity(keep)
        vals.reserveCapacity(keep)
        for i in kept {
            idx.append(UInt32(i))
            vals.append(values[i])
            residual[i] = 0      // transmitted, so nothing is owed on it
        }
        return (Sparse(denseCount: values.count, indices: idx, values: vals), residual)
    }

    public static func fromSparse(_ s: Sparse) -> [Float32] {
        var dense = [Float32](repeating: 0, count: s.denseCount)
        for (i, v) in zip(s.indices, s.values) {
            let at = Int(i)
            if at < dense.count { dense[at] = v }
        }
        return dense
    }

    // MARK: - Cost

    /// Payload bytes for one tensor under a given encoding, excluding the
    /// 12-byte per-tensor header.
    ///
    /// Exposed so a mechanism can check the break-even before choosing: sparse
    /// costs 8 bytes per nonzero against dense's 4 per element, so above ~50%
    /// density it is larger than sending everything.
    public static func encodedByteCount(_ encoding: GossipEncoding,
                                        denseCount: Int,
                                        nonzeroCount: Int = 0) -> Int {
        switch encoding {
        case .dense:     return denseCount * MemoryLayout<Float32>.size
        case .denseFP16: return denseCount * MemoryLayout<UInt16>.size
        case .denseINT8: return 2 * MemoryLayout<Float32>.size + denseCount
        case .sparseCOO: return MemoryLayout<UInt32>.size
                              + nonzeroCount * (MemoryLayout<UInt32>.size
                                              + MemoryLayout<Float32>.size)
        }
    }
}

/// One parameter tensor's worth of gossip payload.
///
/// `values` is ALWAYS the dense representation, whatever `encoding` says. The
/// encoding describes how it will be written to, or was read from, the wire —
/// it is not a statement about the array in memory. Callers therefore construct
/// and consume tensors identically regardless of mechanism, and only
/// `GossipCodec` is aware of the difference.
///
/// For `.sparseCOO`, `values` is the dense array with dropped positions zeroed:
/// the codec derives indices from the nonzeros at encode time. That keeps the
/// sparse case from needing a different in-memory shape, at the cost of one scan
/// per tensor.
public struct GossipTensor: Sendable {
    public let tensorID: UInt32
    public let encoding: GossipEncoding
    public let values: [Float32]

    /// Dense, the original initialiser. Unchanged so every existing call site
    /// keeps compiling and keeps meaning exactly what it meant.
    public init(tensorID: UInt32, values: [Float32]) {
        self.tensorID = tensorID
        self.encoding = .dense
        self.values = values
    }

    public init(tensorID: UInt32, values: [Float32], encoding: GossipEncoding) {
        self.tensorID = tensorID
        self.encoding = encoding
        self.values = values
    }
}

public struct GossipMessage: Sendable {
    /// Magic bytes "RAFD" (RAD-FL) packed into a UInt32, used as a header sanity check.
    public static let magic: UInt32 = 0x5241_4644
    public static let protocolVersion: UInt32 = 1

    public let messageType: GossipMessageType
    public let senderNodeID: UInt32
    public let round: UInt32
    /// Number of local training samples the sender used this round —
    /// matches Python's wire protocol exactly (gossip.py: "num_samples —
    /// int32, number of training samples used (for weighted avg)",
    /// genuinely transmitted over the wire as part of each push, not
    /// assumed or inferred by the receiver). Each node discloses only its
    /// OWN count as part of its own update — nothing here requires or
    /// implies any node learning about a peer's data beyond what that
    /// peer itself chooses to send, the same legitimate DFL design
    /// Python's protocol already uses. Closes a real, previously-flagged
    /// gap: RoundOrchestrator was approximating every peer's sampleCount
    /// as the LOCAL node's own count (since this field didn't exist on
    /// the wire), which silently distorted sample-weighted averaging
    /// whenever peer shard sizes genuinely differed — exactly the
    /// non-IID setting this whole project's experiments use.
    public let sampleCount: UInt32
    public let tensors: [GossipTensor]

    public init(messageType: GossipMessageType, senderNodeID: UInt32, round: UInt32, sampleCount: UInt32, tensors: [GossipTensor]) {
        self.messageType = messageType
        self.senderNodeID = senderNodeID
        self.round = round
        self.sampleCount = sampleCount
        self.tensors = tensors
    }
}

