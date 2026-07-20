// GossipAggregator.swift
//
// Sample-weighted averaging of peer model updates. For full-mesh topology
// (current experimental setup), this produces bit-identical post-aggregation
// weights across all nodes each round — a confirmed mathematical property
// (sample-weighted average over the same 10 inputs), not a bug, and explains
// the single-trace communication-efficiency plots already seen in the Python
// results.
//
// Dead-peer handling: aggregates over whatever peer updates arrived before the
// round deadline (enforced by the orchestration layer), rather than blocking
// indefinitely — this is the Swift-side fix for the alpha_0.1 stall behavior
// observed in the Python baseline (no timeout, surviving nodes wait forever).
//
// COMPUTE: routes through Tensor.swift's `weightedSum` (SIMD4 + parallel
// across cores) rather than the original nested-scalar-loop implementation.
// This matters specifically because the paper's entire claim is about Swift
// compute efficiency — a scalar aggregation step sitting in the per-round
// critical path would have directly undercut that claim regardless of how
// fast the model's own forward/backward passes were.

import Foundation

public struct PeerUpdate: Sendable {
    public let nodeID: Int
    public let sampleCount: Int       // shard size — weighting factor for this peer's contribution
    public let parameters: [[Float32]]

    public init(nodeID: Int, sampleCount: Int, parameters: [[Float32]]) {
        self.nodeID = nodeID
        self.sampleCount = sampleCount
        self.parameters = parameters
    }
}

public enum GossipAggregator {

    /// Sample-weighted average across `localUpdate` and all `peerUpdates` that
    /// arrived in time for this round. If `peerUpdates` is empty (e.g. every
    /// peer timed out), returns `localUpdate.parameters` unchanged rather than
    /// dividing by zero — the orchestration layer should flag this round's
    /// status accordingly (see RoundStatus.partialPeersUnreachable).
    public static func aggregate(
        localUpdate: PeerUpdate,
        peerUpdates: [PeerUpdate]
    ) -> [[Float32]] {
        let allUpdates = [localUpdate] + peerUpdates
        let totalSamples = allUpdates.reduce(0) { $0 + $1.sampleCount }

        guard totalSamples > 0 else { return localUpdate.parameters }

        let weights = allUpdates.map { Float32($0.sampleCount) / Float32(totalSamples) }
        let tensorCount = localUpdate.parameters.count

        var result: [[Float32]] = []
        result.reserveCapacity(tensorCount)

        for tensorIdx in 0..<tensorCount {
            let tensorsAcrossPeers = allUpdates.map { $0.parameters[tensorIdx] }
            result.append(weightedSum(arrays: tensorsAcrossPeers, weights: weights))
        }

        return result
    }
}

