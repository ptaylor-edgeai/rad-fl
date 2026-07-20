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
// Layout:
//   Header:
//     [4 bytes] magic              (UInt32) — 0x52414644 ("RAFD")
//     [4 bytes] protocol version   (UInt32) — currently 1
//     [4 bytes] message type       (UInt32) — see MessageType
//     [4 bytes] sender node ID     (UInt32)
//     [4 bytes] round number       (UInt32)
//     [4 bytes] tensor count       (UInt32)
//   Per tensor:
//     [4 bytes] tensor ID         (UInt32) — stable index into the model's parameter list
//     [4 bytes] encoding         (UInt32) — 0 = dense, 1 = sparseCOO (reserved, unimplemented)
//     [4 bytes] element count     (UInt32) — total elements for dense; nonzero count for sparse
//     payload:
//       dense:      [elementCount * 4 bytes]  Float32, big-endian-free (host byte order, same-arch cluster)
//       sparseCOO:  [count * 4 bytes indices][count * 4 bytes values]  (reserved)
//
// NIO framing: this struct is the *application* frame. NIOFrameDecoder prefixes
// it with a 4-byte little-endian length so ByteToMessageDecoder can reassemble
// frames split across TCP packets.

import Foundation
import NIOCore

public enum GossipEncoding: UInt32, Sendable {
    case dense = 0
    case sparseCOO = 1   // reserved for future sparse-update work; not yet implemented
}

public enum GossipMessageType: UInt32, Sendable {
    case weightsFull = 0      // full dense model push (current implementation)
    case weightsDelta = 1     // reserved: delta against last-known peer state
    case heartbeat = 2        // liveness ping — see dead-peer handling notes below
    case roundComplete = 3    // signals this node finished local aggregation for the round
}

/// One parameter tensor's worth of gossip payload.
/// `values` is always dense (flat row-major Float32) in the current implementation;
/// `encoding` is carried on the wire now so a sparse variant can be introduced later
/// without changing the outer message shape.
public struct GossipTensor: Sendable {
    public let tensorID: UInt32
    public let encoding: GossipEncoding
    public let values: [Float32]

    public init(tensorID: UInt32, values: [Float32]) {
        self.tensorID = tensorID
        self.encoding = .dense
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

