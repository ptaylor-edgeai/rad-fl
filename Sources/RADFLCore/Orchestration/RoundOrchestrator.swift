// RoundOrchestrator.swift
//
// Drives a fixed-N-round federated learning loop on one node: train locally,
// push updated parameters to every peer, wait for every peer's update for
// THIS round to arrive, aggregate via GossipAggregator, replace local
// parameters with the aggregated result, log RoundMetrics, repeat.
//
// FAILURE POLICY (explicit project decision, not the earlier "tolerate dead
// peers" design documented in GossipAggregator.swift/RoundMetrics.swift):
// for this development phase, the orchestrator waits INDEFINITELY for every
// expected peer's update each round — no timeout. If a peer is genuinely
// down, the round (and therefore the whole node) stalls, rather than
// silently proceeding without that peer's contribution. This is a
// deliberate reversal of the partial-peer-tolerance design built earlier in
// the project for THIS phase specifically: tolerant-of-failure makes sense
// once the system is trusted and resilience matters more than catching
// bugs; right now, silently completing a round "successfully" using only
// 9 of 10 nodes would be a much worse outcome than the process visibly
// stalling so a real problem gets noticed and fixed. RoundStatus.
// partialPeersUnreachable / the timeout-and-degrade behavior described in
// GossipAggregator's doc comments remain valid future behavior — this is a
// "for now" project decision, not a verdict that the old design was wrong.
//
// Since there is no timeout, there is no deadline-timer machinery here at
// all — "wait for all peers" really does mean wait, with a periodic
// heartbeat (matching the `test-connectivity` command's same lesson:
// silence while genuinely waiting is indistinguishable from a hang) so a
// human watching the process can tell it's alive and see exactly which
// peer(s) it's still blocked on.
//
// METRICS: this version produces REAL values for nearly every field —
// round/nodeID/condition, every timing field (trainTimeS, fwdS/bwdS/optS,
// gossipPushS/gossipAggS, roundTotalS, evalTestS/evalLocalS/evalTotalS),
// the model fields (trainLoss/trainAcc/testAcc/testLoss/localAcc/localLoss),
// effectiveCores and peakRSSMB (via ResourceUsage.swift's getrusage
// wrapper), and bytesSent (via messageByteCount, computed once per round
// since the pushed message is identical for every peer). The ONE remaining
// genuine placeholder is bytesReceived (see RoundMetricsPlaceholders below)
// — nothing instruments GossipServer's receive path to track cumulative
// bytes received from peers across a round. Note also ResourceUsage.swift's
// own caveat: it could not be compiled/verified against a real Swift
// toolchain while being written, since none was available in the authoring
// environment — the C-level getrusage struct layout is well-documented and
// not in question, but the exact Swift-bridged numeric types were not
// directly confirmed.

import Foundation
import NIOCore

/// Honest, clearly-named placeholder values for metrics this version of the
/// orchestrator cannot yet measure for real. Using named constants rather
/// than scattering literal 0/-1 values through RoundMetrics construction —
/// makes it easy to grep for every place a placeholder is used, and to find
/// every one of them later when the real instrumentation is built.
///
/// effectiveCores and peakRSSBytes/peakRSSMB are NO LONGER placeholders as
/// of ResourceUsage.swift — both now come from real getrusage() data (see
/// that file's own caveat about unverified-by-compilation Swift-bridged
/// types, since no Swift toolchain was available while writing it).
/// bytesSent is also no longer a placeholder — messageByteCount (computed
/// once per round from the real outgoing GossipMessage) times peer count
/// gives the real total. bytesReceived remains a genuine, unaddressed gap:
/// nothing instruments GossipServer's receive path to track cumulative
/// bytes received from peers across a round.
private enum RoundMetricsPlaceholders {
    /// bytesReceived requires the transport layer's RECEIVE path
    /// (GossipServer / GossipFrameDecoder) to track cumulative byte counts
    /// per inbound message — not instrumented yet. bytesSent (the analogous
    /// send-side metric) IS now real, computed from messageByteCount.
    static let bytesReceived: Int64 = -1
}

public enum RoundOrchestratorError: Error, CustomStringConvertible {
    case localNodeIDNotNumeric(any Error)
    case malformedPeerMessage(senderID: UInt32, expectedTensorCount: Int, gotTensorCount: Int)

    public var description: String {
        switch self {
        case .localNodeIDNotNumeric(let underlying):
            return "RoundOrchestrator: local node's topology ID could not be converted to the wire protocol's numeric ID: \(underlying)"
        case .malformedPeerMessage(let senderID, let expected, let got):
            return "RoundOrchestrator: peer \(senderID)'s message has \(got) tensors, expected \(expected) — refusing to aggregate a malformed/partial update rather than risk an out-of-bounds crash in GossipAggregator"
        }
    }
}

/// Collects inbound peer GossipMessages, keyed by (round, senderNodeID),
/// and lets the round loop wait for a specific round's messages to arrive
/// from a specific set of expected peers. An actor specifically because
/// this state is written from GossipServer's network callback (fires on a
/// NIO event-loop thread) and read from the round loop's waiting logic —
/// genuinely concurrent access to shared mutable state, exactly the case
/// actors exist for.
public actor InboundUpdateCollector {
    private var messagesByRound: [UInt32: [UInt32: GossipMessage]] = [:]  // round -> senderNodeID -> message

    public init() {}

    public func record(_ message: GossipMessage) {
        messagesByRound[message.round, default: [:]][message.senderNodeID] = message
    }

    /// Returns the messages received so far for `round` from the given
    /// `expectedSenderIDs`, plus which of those senders are still missing.
    public func currentStatus(round: UInt32, expectedSenderIDs: Set<UInt32>) -> (received: [UInt32: GossipMessage], missing: Set<UInt32>) {
        let received = messagesByRound[round] ?? [:]
        let receivedIDs = Set(received.keys).intersection(expectedSenderIDs)
        let missing = expectedSenderIDs.subtracting(receivedIDs)
        let filteredReceived = received.filter { expectedSenderIDs.contains($0.key) }
        return (filteredReceived, missing)
    }

    /// Discards buffered messages for rounds strictly before `round` —
    /// called once a round completes, so the buffer doesn't grow
    /// unboundedly across a long multi-round run. Messages for the
    /// CURRENT or future rounds are kept (a fast peer's early push for a
    /// round we haven't reached yet is legitimate and should be honored
    /// when we get there, not discarded).
    public func pruneRoundsBefore(_ round: UInt32) {
        messagesByRound = messagesByRound.filter { $0.key >= round }
    }
}

public struct RoundOrchestratorConfig: Sendable {
    public let totalRounds: Int
    public let condition: String          // e.g. "alpha_0.5" — for RoundMetrics.condition
    public let baseSeed: UInt64           // mixed with round number to derive each round's shuffle seed
    public let learningRate: Float
    public let heartbeatIntervalSeconds: Double   // how often to print "still waiting on..." while blocked

    /// Seconds to wait for peer updates before proceeding without them.
    /// `nil` (the default) preserves the original indefinite wait exactly.
    ///
    /// The indefinite wait was the right call for development — a stalled
    /// round should be visible and investigated, not silently completed with a
    /// subset of peers. It also makes every failure regime unmeasurable: a
    /// crashed peer blocks its neighbours forever, so no experiment involving
    /// node loss, churn or partition can run at all.
    ///
    /// Opt-in rather than a new default so that every run recorded so far
    /// remains reproducible with the same binary, and so a deadline can never
    /// mask a genuine stall in a run that did not intend to tolerate one.
    ///
    /// FIXED seconds, not a multiple of the node's own round time. A relative
    /// deadline would adapt to heterogeneous hardware — which will matter once
    /// cgroup profiles are in play, since a deliberately slowed node under a
    /// fixed deadline is permanently late through no fault of the topology.
    /// Recorded here as the known next step rather than built speculatively.
    public let peerDeadlineSeconds: Double?

    /// Per-peer, per-round probability that this node ignores a peer's update
    /// even though it arrived — synthetic churn for the failure regimes.
    ///
    /// Injected at the RECEIVING node rather than by silencing a sender, so
    /// churn is per-link rather than per-node: a peer can be dropped by one
    /// neighbour and heard by another, which is what intermittent connectivity
    /// actually looks like. Node-level failure is a separate mechanism.
    public let churnDropProbability: Double

    /// Seed for the churn RNG, kept separate from `baseSeed`.
    ///
    /// Separate because the churn realisation is itself a random variable worth
    /// sampling independently: three churn draws at ONE partition seed isolates
    /// failure-pattern variance from partition variance, and mixing the two
    /// into one seed would make that impossible.
    public let churnSeed: UInt64

    /// How long to keep retrying a failed push before giving up, in seconds.
    ///
    /// Defaults to 15s rather than the full peer deadline. The failure it
    /// exists for — a peer whose event loop is starved because it is mid-
    /// training — resolves the moment that peer's training phase ends, and in
    /// practice within a few seconds of a normal round boundary. Spending the
    /// entire peer deadline on it means a genuinely dead peer costs the
    /// deadline TWICE per round, once retrying and again waiting.
    ///
    /// Ignored entirely when no peer deadline is set, where the first failure
    /// throws as it always did.
    public let pushRetrySeconds: Double

    public init(totalRounds: Int, condition: String, baseSeed: UInt64,
                learningRate: Float, heartbeatIntervalSeconds: Double = 20,
                peerDeadlineSeconds: Double? = nil,
                churnDropProbability: Double = 0.0,
                churnSeed: UInt64 = 0,
                pushRetrySeconds: Double = 15) {
        self.totalRounds = totalRounds
        self.condition = condition
        self.baseSeed = baseSeed
        self.learningRate = learningRate
        self.heartbeatIntervalSeconds = heartbeatIntervalSeconds
        self.peerDeadlineSeconds = peerDeadlineSeconds
        self.churnDropProbability = churnDropProbability
        self.churnSeed = churnSeed
        self.pushRetrySeconds = pushRetrySeconds
    }

    /// Deterministic churn decision for one (round, peer) pair.
    ///
    /// Derived from (churnSeed, round, peer) rather than drawn from a running
    /// RNG, so the decision does not depend on how many other draws happened
    /// first. That matters because peer updates arrive in nondeterministic
    /// order: a sequential RNG would give a different drop pattern on every
    /// run of the same configuration, making failure regimes irreproducible in
    /// exactly the way seeds exist to prevent.
    func shouldDropPeer(round: Int, peerID: UInt32) -> Bool {
        guard churnDropProbability > 0 else { return false }
        var rng = SplitMix64(seed: churnSeed
                             &+ (UInt64(round) &* 0x9E37_79B9_7F4A_7C15)
                             &+ (UInt64(peerID) &* 0xBF58_476D_1CE4_E5B9))
        _ = rng.next()   // discard the first output; SplitMix64's first value
                         // correlates visibly with similar seeds
        return Double(rng.next() >> 11) * (1.0 / 9007199254740992.0)
               < churnDropProbability
    }

    /// Derives a per-round shuffle seed from the base seed and round
    /// number, so each round reshuffles differently (see FederatedModel's
    /// doc comment on trainEpoch's seed parameter for why this matters)
    /// while still being fully reproducible from (baseSeed, round) alone.
    ///
    /// Uses SplitMix64, NOT Swift's Hasher — Hasher is explicitly
    /// documented as randomly seeded per PROCESS execution ("Hasher is
    /// usually randomly seeded, which means it will return different
    /// values on every new execution of your program" — Apple's own
    /// Hasher documentation). An earlier draft of this function used
    /// Hasher and would have silently broken reproducibility across runs —
    /// the exact bug this whole seed-as-an-explicit-parameter design was
    /// built to prevent. SplitMix64 (already used elsewhere in this
    /// project for the same reason) is genuinely deterministic: the same
    /// (baseSeed, round) always produces the same derived seed, in this
    /// process or any other.
    /// Derives a deterministic shuffle seed for a given round, purely from
    /// baseSeed and round number — same across all nodes, matching Python's
    /// effective behavior (all nodes use the same literal seed=42, diverging
    /// only because different shard sizes consume different amounts of RNG
    /// state, not because of deliberate per-node seed differentiation).
    ///
    /// A per-node seed variant (mixing nodeID into baseSeed) was tried and
    /// experimentally reverted: it made post-aggregation local_acc worse
    /// (0.088 → 0.042) rather than better, disproving the hypothesis that
    /// identical shuffle seeds were causing the convergence gap. The actual
    /// root cause was missing momentum, not shuffle-seed sharing.
    func seedForRound(_ round: Int) -> UInt64 {
        var rng = SplitMix64(seed: baseSeed)
        for _ in 0..<round { _ = rng.next() }
        return rng.next()
    }
}

/// Ties together a FederatedModel, a Topology, and gossip transport to run
/// `config.totalRounds` rounds of federated training on one node, logging
/// RoundMetrics for each round via MetricsLogger.
public final class RoundOrchestrator {
    private var model: any FederatedModel
    private let topology: Topology
    private let config: RoundOrchestratorConfig
    private let client: GossipClient
    private let collector: InboundUpdateCollector
    private let metricsLogger: MetricsLogger
    private let localNumericID: UInt32
    private var peerByNumericID: [UInt32: TopologyNode]
    // `var`, not `let`, because exclusion is decided AFTER construction. The
    // orchestrator has to exist before the startup readiness gate runs — it is
    // what handles inbound messages while this node waits — so which peers
    // never showed up is not knowable at init time.
    private var expectedSenderIDs: Set<UInt32>
    /// Peers this node will actually push to — topology.peers minus any
    /// excluded at startup. Used instead of `topology.peers` everywhere a
    /// round iterates peers, so an excluded node is neither pushed to nor
    /// waited for.
    private var activePeers: [TopologyNode]
    private let onTrainingProgress: (@Sendable (_ round: Int, _ batchIndex: Int, _ totalBatches: Int) -> Void)?

    public init(
        model: any FederatedModel,
        topology: Topology,
        config: RoundOrchestratorConfig,
        client: GossipClient,
        collector: InboundUpdateCollector,
        metricsLogger: MetricsLogger,
        onTrainingProgress: (@Sendable (_ round: Int, _ batchIndex: Int, _ totalBatches: Int) -> Void)? = nil
    ) throws {
        self.model = model
        self.topology = topology
        self.config = config
        self.client = client
        self.collector = collector
        self.metricsLogger = metricsLogger
        self.onTrainingProgress = onTrainingProgress

        do {
            self.localNumericID = try topology.localNode.numericID()
        } catch {
            throw RoundOrchestratorError.localNodeIDNotNumeric(error)
        }

        // Starts with every peer in the topology. Peers that never come up are
        // removed later by `excludePeers`, once the startup readiness gate has
        // determined which those are — see that method for why exclusion
        // cannot happen here.
        var byID: [UInt32: TopologyNode] = [:]
        for peer in topology.peers {
            if let id = try? peer.numericID() {
                byID[id] = peer
            }
        }
        self.peerByNumericID = byID
        self.expectedSenderIDs = Set(byID.keys)
        self.activePeers = topology.peers
    }


    /// Writes one peer_push_log.jsonl line per lost peer, naming it.
    ///
    /// Best-effort: a logging failure must not abort a round that otherwise
    /// succeeded, and the counts in training_log.csv remain authoritative for
    /// how many were lost. These entries add the identities.
    private func logPeerLosses(round: Int, outcome: PeerWaitOutcome, waitedS: Double) {
        for peerID in outcome.timedOut {
            guard let node = peerByNumericID[peerID] else { continue }
            try? metricsLogger.log(.timeout(
                round: round, fromNode: Int(localNumericID), toNode: Int(peerID),
                peerID: node.id, waitedS: waitedS))
        }
        for peerID in outcome.churnDropped {
            guard let node = peerByNumericID[peerID] else { continue }
            try? metricsLogger.log(.churnDropped(
                round: round, fromNode: Int(localNumericID), toNode: Int(peerID),
                peerID: node.id))
        }
    }

    /// Pushes to one peer, retrying with backoff until the peer deadline.
    ///
    /// A refused connection means the peer is BUSY, not dead. Training
    /// saturates all cores via `concurrentPerform`, which starves the NIO
    /// event loop, so a node in the middle of its training phase cannot accept
    /// connections. Observed directly: a node received weights FROM a peer in
    /// the same round that it failed to push TO that peer, and a late-starting
    /// node failed to push to seven peers while all ten were healthy.
    ///
    /// This never surfaced before the failure regimes because nodes normally
    /// finish training within a second or two of each other, so everyone
    /// pushes while nobody is training. Any skew — a node that started late, a
    /// slower device, a longer round — breaks that assumption, and it will
    /// break routinely once heterogeneity profiles deliberately slow nodes down.
    ///
    /// The retry budget is `pushRetrySeconds` (default 15s), NOT the peer
    /// deadline. A busy peer becomes reachable as soon as its training phase
    /// ends, so a short budget covers the real failure mode; spending the full
    /// deadline here would make a genuinely dead peer cost the deadline twice
    /// per round — once retrying, then again waiting.
    ///
    /// With no deadline configured there is no retry and the first failure
    /// throws, preserving the original fail-loud behaviour exactly.
    private func pushWithRetry(_ message: GossipMessage,
                               to address: GossipNodeAddress,
                               peerName: String,
                               round: Int) async throws {
        guard config.peerDeadlineSeconds != nil else {
            try await client.push(message, to: address)   // fail-loud, no retry
            return
        }

        let giveUpAt = Date().addingTimeInterval(config.pushRetrySeconds)
        var attempt = 0
        var backoff: UInt64 = 500_000_000   // 0.5s, doubling to a 5s ceiling

        while true {
            do {
                try await client.push(message, to: address)
                if attempt > 0 {
                    print("[round \(round)] push to \(peerName) succeeded on attempt "
                          + "\(attempt + 1) — peer was busy, not down")
                }
                return
            } catch {
                attempt += 1
                if Date() >= giveUpAt {
                    throw error
                }
                try await Task.sleep(nanoseconds: backoff)
                backoff = min(backoff * 2, 5_000_000_000)
            }
        }
    }

    /// Removes peers that never became reachable from this run.
    ///
    /// Called after the startup readiness gate, which is the earliest point at
    /// which the answer is known: the orchestrator must already exist to
    /// receive inbound messages while the gate is waiting, so this cannot be an
    /// init parameter.
    ///
    /// Excluded peers are removed from the expected set entirely rather than
    /// left to time out each round. A node that never joined is a permanent
    /// absence: waiting the full peer deadline on it every round would cost
    /// more than the rest of the round put together, and would report one
    /// permanent failure as N transient ones.
    ///
    /// Must be called before `run()`; calling it later would change the
    /// expected set mid-run, which is a different experiment (mid-run failure)
    /// and is not what this is for.
    public func excludePeers(_ ids: Set<String>) {
        guard !ids.isEmpty else { return }
        // Logged before filtering, while the numeric IDs are still resolvable.
        for (numericID, node) in peerByNumericID where ids.contains(node.id) {
            try? metricsLogger.log(.excludedAtStartup(
                fromNode: Int(localNumericID), toNode: Int(numericID),
                peerID: node.id))
        }
        peerByNumericID = peerByNumericID.filter { !ids.contains($0.value.id) }
        expectedSenderIDs = Set(peerByNumericID.keys)
        activePeers = activePeers.filter { !ids.contains($0.id) }
    }

    /// Feeds an inbound GossipMessage into this orchestrator's collector —
    /// wire up as the `onMessage` callback when constructing the
    /// GossipServer this node listens on. Kept as a thin pass-through
    /// rather than having RoundOrchestrator own the GossipServer itself,
    /// so the caller controls server lifecycle (start/stop, port, logging
    /// of raw received messages) independently of round-loop logic.
    public func handleInboundMessage(_ message: GossipMessage) async {
        await collector.record(message)
    }

    /// Runs all `config.totalRounds` rounds, training on `trainShard` and
    /// evaluating on `testShard` each round. Returns once every round has
    /// completed — or never returns if a peer never responds (see this
    /// file's header comment on the no-timeout failure policy).
    ///
    /// After all rounds complete, saves the model's final parameters as
    /// individual .npy files into `outputDirectory` — conv1W.npy, conv1B.npy,
    /// fcW.npy, fcB.npy (matching SimpleCNN's own documented tensorID
    /// ordering/naming) — mirroring Python's run_experiment.py, which
    /// saves final_weights.npz only after the full round loop finishes,
    /// not per-round. Individual .npy files rather than a real .npz (zip)
    /// container — see NPYWriter.swift's header comment for why: the
    /// actual need is "Swift can load these back to seed/resume a future
    /// Swift experiment," and NPYReader already exists and reads exactly
    /// this format; a real .npz writer was explicitly deferred as
    /// unnecessary work for that purpose.
    public func run(trainShard: DataShard, testShard: DataShard, outputDirectory: URL) async throws {
        for round in 1...config.totalRounds {
            let metrics = try await runOneRound(round: round, trainShard: trainShard, testShard: testShard)
            try metricsLogger.log(metrics)
            await collector.pruneRoundsBefore(UInt32(round))

            print("[round \(round)/\(config.totalRounds)] train_loss=\(metrics.trainLoss.map { String(format: "%.4f", $0) } ?? "n/a") test_acc=\(metrics.testAcc.map { String(format: "%.4f", $0) } ?? "n/a") round_total_s=\(String(format: "%.2f", metrics.roundTotalS))")
        }

        try saveFinalWeights(to: outputDirectory)
    }

    /// Saves each of the model's parameter tensors as its own .npy file —
    /// see `run`'s doc comment for the naming/format rationale. Tensor
    /// shapes are NOT recorded in these .npy files themselves beyond
    /// element count (a flat parameter array has no shape metadata of its
    /// own) — but this is no longer an unaddressed gap: the CALLER (see
    /// RADFLMain.swift's `run-round` case) separately saves a config.json
    /// containing the model's SimpleCNNConfig into the same output
    /// directory, which is enough to reconstruct the exact architecture
    /// (and therefore the exact tensor shapes) these weights belong to.
    /// RoundOrchestrator itself doesn't do this — it's built against the
    /// abstract FederatedModel protocol and deliberately doesn't know
    /// about SimpleCNNConfig (or SimpleCNN) specifically, so saving that
    /// concrete config is the caller's responsibility, not this class's.
    private func saveFinalWeights(to outputDirectory: URL) throws {
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

        // Names match SimpleCNN's FederatedModel conformance's documented
        // FIXED tensor order exactly (see SimpleCNN.swift's "FIXED ORDER
        // (tensorID 0-7)" comment) — updated from the earlier 4-tensor
        // single-conv-layer naming (["convW","convB","fcW","fcB"]) when
        // SimpleCNN was rebuilt to 3 conv layers matching Python's
        // SimpleCNNCIFAR. This file wasn't touched during that rebuild —
        // a real gap, caught when asked directly whether weight-saving had
        // been updated, rather than something verified proactively at the
        // time of the architecture change itself. Worth noting: the
        // fallback guard below (generic tensor_<N>.npy names on a count
        // mismatch) meant this gap was never a silent-WRONG-DATA bug —
        // weights were always saved correctly and were still fully
        // reconstructable via config.json's documented tensor order, just
        // under less readable names than intended.
        let tensorNames = ["conv1W", "conv1B", "conv2W", "conv2B", "conv3W", "conv3B", "fcW", "fcB"]
        let parameters = model.parameters
        guard parameters.count == tensorNames.count else {
            print("[RoundOrchestrator] WARNING: model.parameters.count (\(parameters.count)) != expected \(tensorNames.count) — saving with generic tensor_<N> names instead, since the fixed naming assumption doesn't hold for this model")
            for (index, values) in parameters.enumerated() {
                let url = outputDirectory.appendingPathComponent("tensor_\(index).npy")
                try NPYWriter.writeFloat32Array(values, shape: [values.count], to: url)
            }
            return
        }

        for (name, values) in zip(tensorNames, parameters) {
            let url = outputDirectory.appendingPathComponent("\(name).npy")
            try NPYWriter.writeFloat32Array(values, shape: [values.count], to: url)
        }
        let savedFiles = tensorNames.map { "\($0).npy" }.joined(separator: ", ")
        print("[RoundOrchestrator] final weights saved to \(outputDirectory.path): \(savedFiles)")
    }

    private func runOneRound(round: Int, trainShard: DataShard, testShard: DataShard) async throws -> RoundMetrics {
        let roundStart = Date()

        // --- Local training ---
        // CPU snapshot brackets ONLY the training phase, matching Python's
        // exact boundary: trainer.py's _cpu_snapshot() calls are inside
        // train_round(), bracketing t0 (before the epoch loop) and t1
        // (after the post-training accuracy pass) — NOT the outer eval
        // calls in run_experiment.py's round loop. effectiveCores is
        // therefore cpu_seconds (during training) / trainTimeS, the same
        // ratio Python computes the same way.
        let cpuBefore = ResourceUsage.cpuSnapshot()
        let trainStart = Date()
        let seed = config.seedForRound(round)
        let trainResult = try model.trainEpoch(
            shard: trainShard, seed: seed, learningRate: config.learningRate,
            onBatchProgress: onTrainingProgress.map { callback in
                { @Sendable (batchIndex: Int, totalBatches: Int) in callback(round, batchIndex, totalBatches) }
            }
        )
        let trainTimeS = Date().timeIntervalSince(trainStart)
        let cpuAfter = ResourceUsage.cpuSnapshot()
        let cpuSecondsThisRound = (cpuAfter.userSeconds - cpuBefore.userSeconds) + (cpuAfter.systemSeconds - cpuBefore.systemSeconds)
        let effectiveCores = trainTimeS > 0 ? cpuSecondsThisRound / trainTimeS : 0.0

        // --- Push parameters to peers ---
        let gossipStart = Date()
        let myParameters = model.parameters
        // localSampleCount: this node's own real shard size, disclosed
        // honestly as part of its own update — moved earlier than the
        // aggregation step below (where it was previously only defined)
        // specifically so the OUTGOING message can carry it. trainShard is
        // a function parameter already in scope from the start of this
        // method, so this is a safe reordering, not a new dependency.
        let localSampleCount = trainShard.sampleCount
        let outgoingMessage = GossipMessage(
            messageType: .weightsFull,
            senderNodeID: localNumericID,
            round: UInt32(round),
            sampleCount: UInt32(localSampleCount),
            tensors: myParameters.enumerated().map { GossipTensor(tensorID: UInt32($0.offset), values: $0.element) }
        )

        // Byte count computed ONCE per round (not per-peer) — the message
        // is identical for every peer this round, so its encoded size is
        // the same regardless of destination. This does mean encoding the
        // message twice overall (once here just for the byte count, once
        // again inside client.push's own NIO pipeline when it's actually
        // sent) — a small amount of real redundant work, accepted as a
        // reasonable tradeoff rather than threading a byte-count return
        // value through GossipClient.push's public API just for this.
        let messageByteCount = Int64(GossipCodec.encode(outgoingMessage, allocator: ByteBufferAllocator()).readableBytes)

        // Closes a previously-flagged gap: peer_push_log.jsonl existed
        // (MetricsLogger creates it on construction) but nothing ever
        // wrote to it — the original push loop here used
        // withThrowingTaskGroup(of: Void.self), discarding everything
        // except success/failure per peer, with no per-peer timing
        // captured at all. Now logs one PeerPushLogEntry per peer, timed
        // individually (not just the aggregate gossipAggS phase already
        // captured in RoundMetrics).
        //
        // A failed push is always logged to peer_push_log.jsonl, so a failure
        // is visible rather than silently absent. What happens NEXT depends on
        // whether the run opted into failure tolerance.
        //
        // WITHOUT a peer deadline the error is re-thrown, preserving this
        // project's fail-loud policy: a peer failure should stall or abort the
        // round rather than be silently tolerated, so a real problem gets
        // noticed and fixed instead of quietly degrading results.
        //
        // WITH a peer deadline set, the run has explicitly declared that peer
        // loss is the thing being measured, and a push failure must not abort
        // it. Killing a node mid-run previously took down every surviving node
        // with `Connection refused` propagated to the top level — the receive
        // side had a deadline while the SEND side had none, so the round never
        // reached the point where the deadline would have applied.
        //
        // Peers that could not be pushed to are also treated as immediately
        // unreachable for this round rather than being waited on: this node
        // has direct evidence they are down, so spending the full deadline
        // confirming it wastes the entire deadline every round for as long as
        // the failure lasts.
        // Pushes run CONCURRENTLY with the wait for peer updates, not before
        // it. Both concern the same peers and the same failure, so running them
        // in sequence made a dead peer cost the full budget twice per round.
        // Overlapping them makes the round's gossip cost max(push, wait) rather
        // than their sum.
        //
        // Safe because gossip is asynchronous by construction: a peer sends
        // when its own round completes, and nothing about receiving requires
        // this node to have finished sending. The barrier that matters is the
        // wait, which is unchanged.
        //
        // Consequence for the metrics: gossipPushS and gossipWaitS now overlap
        // in wall-clock terms and no longer sum to the gossip phase. Both are
        // still individually meaningful — push duration and wait duration — but
        // adding them together would double-count. Round wall-clock remains
        // authoritative.
        async let pushOutcome: Set<String> = withThrowingTaskGroup(of: String?.self) { taskGroup -> Set<String> in
            for peer in activePeers {
                taskGroup.addTask {
                    let address = GossipNodeAddress(host: peer.host, port: peer.port)
                    let pushStart = Date()
                    let peerNumericID = (try? peer.numericID()) ?? 0

                    do {
                        try await self.pushWithRetry(outgoingMessage, to: address,
                                                     peerName: peer.id, round: round)
                        let entry = PeerPushLogEntry(
                            round: round, fromNode: Int(self.localNumericID), toNode: Int(peerNumericID),
                            bytes: messageByteCount, pushDurationS: Date().timeIntervalSince(pushStart),
                            success: true, timestamp: pushStart.timeIntervalSince1970,
                            kind: .push, peerID: peer.id
                        )
                        try self.metricsLogger.log(entry)
                        return nil
                    } catch {
                        let entry = PeerPushLogEntry(
                            round: round, fromNode: Int(self.localNumericID), toNode: Int(peerNumericID),
                            bytes: messageByteCount, pushDurationS: Date().timeIntervalSince(pushStart),
                            success: false, timestamp: pushStart.timeIntervalSince1970,
                            kind: .push, peerID: peer.id
                        )
                        try? self.metricsLogger.log(entry)  // best-effort log even on failure; don't let a logging error mask the real push error below
                        if self.config.peerDeadlineSeconds == nil {
                            throw error      // fail-loud, as before
                        }
                        return peer.id       // tolerated; reported, not fatal
                    }
                }
            }
            var failed: Set<String> = []
            for try await id in taskGroup {
                if let id { failed.insert(id) }
            }
            return failed
        }
        let gossipPushS = Date().timeIntervalSince(gossipStart)  // push-phase-only wall-clock — Python's gossip_push_s

        // --- Wait for every peer's update for THIS round, while pushes run ---
        let waitStart = Date()
        let waitOutcome = try await waitForAllPeerUpdates(round: UInt32(round))
        let gossipWaitS = Date().timeIntervalSince(waitStart)

        // Record WHICH peers were lost, not just how many. training_log.csv
        // carries counts; RQ2's targeted-failure arm needs identities, because
        // "killing a head costs more than killing a hub" cannot be verified
        // from a number.
        logPeerLosses(round: round, outcome: waitOutcome, waitedS: gossipWaitS)

        let failedPushes = try await pushOutcome
        if !failedPushes.isEmpty {
            print("[round \(round)] push failed to \(failedPushes.count) peer(s) after "
                  + "retrying: \(failedPushes.sorted().joined(separator: ", ")). "
                  + "Still waiting for their updates — a peer we cannot reach may "
                  + "well be able to reach us.")
        }
        // --- (push/wait now overlap; see above) ---
        //
        // TIMING NOTE, and a correction to what the previous comment here
        // claimed. `gossipAggS` was documented as "wait+aggregate wall-clock",
        // but it is computed on the line immediately after the wait returns,
        // while the aggregation itself does not run until ~45 lines below
        // (PeerUpdate construction, tensor-count validation,
        // GossipAggregator.aggregate, setParameters). So this timer has always
        // measured the WAIT ALONE, and every second of the aggregation has
        // been landing outside all logged phases.
        //
        // That was not visible until per-round wall-clock was compared against
        // the sum of the logged phases: they disagreed by ~11s per round on a
        // 25-round IID run, 13% of the round, larger than the entire gossip
        // phase as reported. The power trace made the shape of it obvious —
        // one deep dip per round to ~1.0W, just above the 0.824W platform
        // floor, exactly at the round boundary.
        //
        // Fixed by measuring the two separately, WITHOUT changing what
        // `gossipAggS` reports. Its value stays exactly as before, so runs
        // recorded under the old schema remain directly comparable with new
        // ones. `gossipWaitS` carries the same number under an honest name,
        // and `gossipAggregateS` adds the measurement that was missing.
        // The wait itself is issued above, concurrently with the pushes.
        //
        // failedPushes is deliberately NOT used to skip waiting for those
        // peers. A push failure means this node could not reach the peer at
        // that instant; it does not mean the peer is down, and it says nothing
        // about whether the peer can reach US — a peer whose update has already
        // arrived may be one we just failed to push to. Treating push failure
        // as proof of death converted transient busy-ness into whole rounds of
        // spurious exclusion. Let the deadline decide, from actual silence.
        let peerUpdates = waitOutcome.updates

        // Unchanged in value and meaning from every previous run: time from
        // the end of the push phase until every peer's update has arrived.
        // Retained rather than redefined so `gossip_agg_s` means the same
        // thing across the whole campaign. Prefer `gossipWaitS` in new
        // analysis; this column is kept for continuity, not because it is
        // well named.
        let gossipAggS = gossipWaitS

        // Marks the start of the aggregation proper. Everything between here
        // and `setParameters` is the work that was previously unattributed:
        // building PeerUpdates from wire messages, validating tensor counts,
        // the sample-weighted average itself, and adopting the result.
        let aggregateStart = Date()

        // --- Aggregate and adopt the result ---
        // localSampleCount was already defined earlier (before the push),
        // not redeclared here — see that declaration's comment for why.
        let localUpdate = PeerUpdate(nodeID: Int(localNumericID), sampleCount: localSampleCount, parameters: myParameters)

        // Validate tensor counts BEFORE constructing PeerUpdate from wire
        // data — GossipAggregator.aggregate indexes every peer's
        // parameters[tensorIdx] using localUpdate's tensor count as
        // authoritative, with no bounds check of its own (reasonable for
        // that function, which is also called directly with
        // hand-constructed test data elsewhere — see test-aggregator).
        // This is the first place untrusted wire data becomes a
        // PeerUpdate, so it's the right place to catch a malformed/partial
        // peer message with a clear error rather than let it reach
        // GossipAggregator and crash with an out-of-bounds fault.
        let expectedTensorCount = myParameters.count
        var peerUpdateList: [PeerUpdate] = []
        for (senderID, message) in peerUpdates {
            guard message.tensors.count == expectedTensorCount else {
                throw RoundOrchestratorError.malformedPeerMessage(
                    senderID: senderID, expectedTensorCount: expectedTensorCount, gotTensorCount: message.tensors.count
                )
            }
            peerUpdateList.append(PeerUpdate(
                nodeID: Int(senderID),
                // REAL per-peer sample count, self-disclosed by the
                // sender as part of its own message — closes a real,
                // previously-flagged gap where every peer was approximated
                // as having this LOCAL node's own count, silently
                // distorting sample-weighted averaging whenever peer shard
                // sizes genuinely differed (exactly the non-IID setting
                // this project's experiments use). Matches Python's
                // gossip.py exactly: num_samples is transmitted over the
                // wire by each sender, never assumed by the receiver.
                sampleCount: Int(message.sampleCount),
                parameters: message.tensors.sorted { $0.tensorID < $1.tensorID }.map(\.values)
            ))
        }
        // Real per-peer sample counts (see comment above) — this gap is
        // now closed; GossipAggregator.aggregate performs a genuinely
        // correctly-weighted average using each node's own disclosed
        // shard size, matching Python's gossip.py exactly.
        let aggregated = GossipAggregator.aggregate(localUpdate: localUpdate, peerUpdates: peerUpdateList)
        model.setParameters(aggregated)

        // Closes the window opened at `aggregateStart`. Deliberately includes
        // PeerUpdate construction and validation as well as the arithmetic:
        // those are the cost of turning received wire data into an aggregate,
        // they scale with peer count exactly as the aggregation does, and
        // splitting them further would attribute time more precisely than the
        // question requires.
        let gossipAggregateS = Date().timeIntervalSince(aggregateStart)

        // --- Evaluate both test set and local shard (AFTER aggregation) ---
        // Matches Python's run_experiment.py exactly: both test_metrics and
        // local_metrics are evaluated AFTER gossip.aggregate(), with the
        // aggregated weights. The comment in the previous version of this
        // code was wrong — Python does NOT evaluate test_acc pre-aggregation.
        // Confirmed by reading run_experiment.py lines 360-376:
        //   weights = gossip.aggregate(weights, ...)   # aggregation
        //   test_metrics  = trainer.evaluate(weights, split="test")   # AFTER
        //   local_metrics = trainer.evaluate(weights, split="local")  # AFTER
        // Evaluating test_acc pre-aggregation produced a systematic gap
        // (~0.18 lower than Python) because each node's locally-trained model
        // is less generalizable than the aggregated model.
        let testEval  = try model.evaluate(shard: testShard)
        let localEval = try model.evaluate(shard: trainShard)

        // roundTotalS EXCLUDES both eval passes — matching Python's documented
        // convention exactly (round_total_s deliberately excludes eval;
        // true total is round_total_s + eval_total_s where
        // eval_total_s = eval_test_s + eval_local_s).
        let evalTotalS = testEval.evalS + localEval.evalS
        // Captured once and reused as both the roundTotalS anchor and the
        // RoundMetrics timestamp, rather than calling Date() a second time
        // a few lines later — keeps the two values referring to the exact
        // same instant instead of introducing a (negligible but pointless)
        // gap between them. Matches Python's write-order exactly:
        // run_experiment.py computes round_total_s, then immediately writes
        // `timestamp` via datetime.now().isoformat(timespec="seconds") — see
        // its CSVLogger row-building code around line 417.
        let roundEndTime = Date()
        let roundTotalS = roundEndTime.timeIntervalSince(roundStart) - evalTotalS

        return RoundMetrics(
            round: round,
            nodeID: Int(localNumericID),
            condition: config.condition,
            trainTimeS: trainTimeS,
            fwdS: trainResult.fwdS,
            bwdS: trainResult.bwdS,
            optS: trainResult.optS,
            evalTotalS: evalTotalS,
            evalTestS: testEval.evalS,
            evalLocalS: localEval.evalS,
            gossipPushS: gossipPushS,
            gossipAggS: gossipAggS,
            roundTotalS: roundTotalS,
            timestamp: roundEndTime,
            effectiveCores: effectiveCores,
            shufS: trainResult.shufS,
            peakRSSMB: ResourceUsage.peakRSSMB(),
            bytesSent: messageByteCount * Int64(activePeers.count),  // one push per peer, all the same size — see messageByteCount's own comment for why it's computed once per round rather than per-peer
            bytesReceived: RoundMetricsPlaceholders.bytesReceived,
            peersReached: peerUpdates.count,
            peersExpected: expectedSenderIDs.count,
            throttled: "unavailable",  // Pi-specific vcgencmd output; always unavailable on non-Pi hardware; metrics.py handles this gracefully (returns None, skips those rounds in throttled-state plots)
            nSamples: trainShard.sampleCount,
            trainLoss: trainResult.meanLoss,
            trainAcc: trainResult.trainAcc,
            testAcc: testEval.accuracy,
            testLoss: testEval.loss,
            localAcc: localEval.accuracy,
            localLoss: localEval.loss,
            // A round that aggregated over fewer peers than expected is NOT
            // `.completed`. Recording it as such would make a degraded round
            // indistinguishable from a healthy one in the results — and under
            // the failure regimes that distinction IS the measurement.
            status: (waitOutcome.timedOut.isEmpty
                     && waitOutcome.churnDropped.isEmpty)
                    ? .completed : .partialPeersUnreachable,
            // Schema v3 arguments go LAST, matching the order they are
            // declared in RoundMetrics.init. They were appended there (with
            // -1 defaults) so that adding them could not break any existing
            // call site — but that same choice means Swift requires them at
            // the end of the argument list here, not beside the gossip
            // parameters they logically belong with. Placing them next to
            // gossipAggS reads better and does not compile.
            // Argument order must match RoundMetrics.init's declaration order:
            // peersTimedOut is a schema-v2 field and therefore precedes the v3
            // gossip timings, while peersChurnDropped is v4 and follows them.
            peersTimedOut: waitOutcome.timedOut.count,
            gossipWaitS: gossipWaitS,
            gossipAggregateS: gossipAggregateS,
            peersChurnDropped: waitOutcome.churnDropped.count
        )
    }

    /// Outcome of one round's wait for peer updates.
    struct PeerWaitOutcome {
        let updates: [UInt32: GossipMessage]
        /// Peers whose update never arrived before the deadline.
        let timedOut: Set<UInt32>
        /// Peers whose update DID arrive but was dropped by churn injection.
        let churnDropped: Set<UInt32>
    }

    /// Waits until every ID in `expectedSenderIDs` has a recorded message for
    /// `round`, or until `config.peerDeadlineSeconds` elapses.
    ///
    /// With no deadline configured this blocks indefinitely, exactly as before
    /// — the original behaviour, preserved by default. See
    /// `RoundOrchestratorConfig.peerDeadlineSeconds` for why that was the right
    /// default and why it nonetheless has to be overridable.
    ///
    /// A heartbeat every `config.heartbeatIntervalSeconds` names the missing
    /// peers, so a human watching can tell the process is alive and see what it
    /// is blocked on rather than staring at silence.
    ///
    /// LATE UPDATES ARE DISCARDED, not carried into the next round. Accepting
    /// them would make the protocol partially asynchronous — a different
    /// algorithm, not a failure-tolerance policy, and one whose convergence
    /// behaviour would confound every topology comparison it appeared in. The
    /// collector keys messages by round, so a late arrival is simply never
    /// consulted again.
    private func waitForAllPeerUpdates(round: UInt32) async throws -> PeerWaitOutcome {
        guard !expectedSenderIDs.isEmpty else {
            return PeerWaitOutcome(updates: [:], timedOut: [], churnDropped: [])
        }

        let deadline = config.peerDeadlineSeconds.map { Date().addingTimeInterval($0) }
        var lastHeartbeat = Date()

        while true {
            let (received, missing) = await collector.currentStatus(
                round: round, expectedSenderIDs: expectedSenderIDs)

            if missing.isEmpty {
                return applyChurn(to: received, round: round, timedOut: [])
            }

            if let deadline, Date() >= deadline {
                let names = missing.compactMap { peerByNumericID[$0]?.id }.sorted()
                print("[round \(round)] deadline reached after "
                      + "\(config.peerDeadlineSeconds!.rounded())s — proceeding "
                      + "with \(received.count)/\(expectedSenderIDs.count) peer(s). "
                      + "Missing: \(names.joined(separator: ", "))")
                return applyChurn(to: received, round: round, timedOut: missing)
            }

            if Date().timeIntervalSince(lastHeartbeat) >= config.heartbeatIntervalSeconds {
                let missingNames = missing.compactMap { peerByNumericID[$0]?.id }
                var msg = "[round \(round)] still waiting on \(missing.count) peer(s): "
                        + "\(missingNames.joined(separator: ", "))"
                if let deadline {
                    msg += String(format: " (%.0fs until deadline)",
                                  deadline.timeIntervalSinceNow)
                }
                print(msg)
                lastHeartbeat = Date()
            }

            try await Task.sleep(nanoseconds: 500_000_000)  // 0.5s poll interval — cheap, frequent enough that the heartbeat interval itself (not this) governs perceived responsiveness
        }
    }

    /// Drops received updates according to the configured churn probability.
    ///
    /// Applied AFTER the wait rather than by refusing to record the message,
    /// so a churned peer still counts as having arrived on time. That keeps the
    /// two failure modes separable in the logs: `timed_out` means the update
    /// never came, `churn_dropped` means it came and was deliberately ignored.
    /// Conflating them would make a slow network indistinguishable from an
    /// unreliable one.
    private func applyChurn(to received: [UInt32: GossipMessage],
                            round: UInt32,
                            timedOut: Set<UInt32>) -> PeerWaitOutcome {
        guard config.churnDropProbability > 0 else {
            return PeerWaitOutcome(updates: received, timedOut: timedOut,
                                   churnDropped: [])
        }
        var kept: [UInt32: GossipMessage] = [:]
        var dropped: Set<UInt32> = []
        for (peerID, msg) in received {
            if config.shouldDropPeer(round: Int(round), peerID: peerID) {
                dropped.insert(peerID)
            } else {
                kept[peerID] = msg
            }
        }
        if !dropped.isEmpty {
            let names = dropped.compactMap { peerByNumericID[$0]?.id }.sorted()
            print("[round \(round)] churn dropped \(dropped.count) peer update(s): "
                  + "\(names.joined(separator: ", "))")
        }
        return PeerWaitOutcome(updates: kept, timedOut: timedOut,
                               churnDropped: dropped)
    }
}




