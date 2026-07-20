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

    public init(totalRounds: Int, condition: String, baseSeed: UInt64, learningRate: Float, heartbeatIntervalSeconds: Double = 20) {
        self.totalRounds = totalRounds
        self.condition = condition
        self.baseSeed = baseSeed
        self.learningRate = learningRate
        self.heartbeatIntervalSeconds = heartbeatIntervalSeconds
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
    private let peerByNumericID: [UInt32: TopologyNode]
    private let expectedSenderIDs: Set<UInt32>
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

        var byID: [UInt32: TopologyNode] = [:]
        for peer in topology.peers {
            if let id = try? peer.numericID() {
                byID[id] = peer
            }
        }
        self.peerByNumericID = byID
        self.expectedSenderIDs = Set(byID.keys)
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
        // IMPORTANT: a failed push still gets logged (so a failure is
        // visible in peer_push_log.jsonl, not just silently absent), but
        // is then RE-THROWN — matching this project's explicit fail-loud
        // policy (see this file's header comment: a peer failure should
        // stall/abort the round, not be silently tolerated). An earlier
        // draft of this change caught the push error internally and never
        // re-threw it, which would have silently changed that policy
        // without flagging it — a push failure would have been logged but
        // the round would have continued as if nothing went wrong. Fixed
        // before this was ever sent.
        try await withThrowingTaskGroup(of: Void.self) { taskGroup in
            for peer in topology.peers {
                taskGroup.addTask {
                    let address = GossipNodeAddress(host: peer.host, port: peer.port)
                    let pushStart = Date()
                    let peerNumericID = (try? peer.numericID()) ?? 0

                    do {
                        try await self.client.push(outgoingMessage, to: address)
                        let entry = PeerPushLogEntry(
                            round: round, fromNode: Int(self.localNumericID), toNode: Int(peerNumericID),
                            bytes: messageByteCount, pushDurationS: Date().timeIntervalSince(pushStart),
                            success: true, timestamp: pushStart.timeIntervalSince1970
                        )
                        try self.metricsLogger.log(entry)
                    } catch {
                        let entry = PeerPushLogEntry(
                            round: round, fromNode: Int(self.localNumericID), toNode: Int(peerNumericID),
                            bytes: messageByteCount, pushDurationS: Date().timeIntervalSince(pushStart),
                            success: false, timestamp: pushStart.timeIntervalSince1970
                        )
                        try? self.metricsLogger.log(entry)  // best-effort log even on failure; don't let a logging error mask the real push error below
                        throw error
                    }
                }
            }
            try await taskGroup.waitForAll()
        }
        let gossipPushS = Date().timeIntervalSince(gossipStart)  // push-phase-only wall-clock — Python's gossip_push_s

        // --- Wait (indefinitely) for every peer's update for THIS round ---
        let aggStart = Date()
        let peerUpdates = try await waitForAllPeerUpdates(round: UInt32(round))
        let gossipAggS = Date().timeIntervalSince(aggStart)  // wait+aggregate wall-clock only — Python's gossip_agg_s

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
            bytesSent: messageByteCount * Int64(topology.peers.count),  // one push per peer, all the same size — see messageByteCount's own comment for why it's computed once per round rather than per-peer
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
            status: .completed
        )
    }

    /// Waits, with NO timeout, until every ID in `expectedSenderIDs` has a
    /// recorded message for `round`. Prints a heartbeat every
    /// `config.heartbeatIntervalSeconds` naming exactly which peer(s) are
    /// still missing, so a human watching the process can tell it's alive
    /// and see what it's blocked on rather than staring at silence.
    private func waitForAllPeerUpdates(round: UInt32) async throws -> [UInt32: GossipMessage] {
        guard !expectedSenderIDs.isEmpty else { return [:] }

        var lastHeartbeat = Date()
        while true {
            let (received, missing) = await collector.currentStatus(round: round, expectedSenderIDs: expectedSenderIDs)
            if missing.isEmpty {
                return received
            }

            if Date().timeIntervalSince(lastHeartbeat) >= config.heartbeatIntervalSeconds {
                let missingNames = missing.compactMap { peerByNumericID[$0]?.id }
                print("[round \(round)] still waiting on \(missing.count) peer(s): \(missingNames.joined(separator: ", "))")
                lastHeartbeat = Date()
            }

            try await Task.sleep(nanoseconds: 500_000_000)  // 0.5s poll interval — cheap, frequent enough that the heartbeat interval itself (not this) governs perceived responsiveness
        }
    }
}


