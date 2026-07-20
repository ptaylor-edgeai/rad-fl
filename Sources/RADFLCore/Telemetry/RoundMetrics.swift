// RoundMetrics.swift
//
// Per-round telemetry, field-matched to the Python baseline's instrumentation
// (metrics.py / power_analysis.py / log_power.py) so the results table can do
// same-condition, same-seed Python vs Swift pairs without unit/field mismatches.
//
// Fields intentionally mirror Python naming (snake_case in the CSV/JSONL output,
// even though Swift convention is camelCase internally) so downstream analysis
// scripts (pandas-based) don't need per-runtime column-name branching.

import Foundation

public struct RoundMetrics: Sendable, Codable {
    public let round: Int
    public let nodeID: Int
    public let condition: String          // e.g. "alpha_0.1", "alpha_0.5", "iid"

    // Timing
    public let trainTimeS: Double
    // fwdS/bwdS/optS: per-phase training timing, matching Python's
    // fwd_s/bwd_s/opt_s exactly (see run_experiment.py's real CSV_COLUMNS) —
    // these sum to approximately trainTimeS (plus shufS and any
    // unaccounted overhead), giving the same phase-level granularity
    // Python reports rather than one undifferentiated training-time figure.
    public let fwdS: Double
    public let bwdS: Double
    public let optS: Double
    // evalTotalS: time spent in model.evaluate(testSet:), timed as ITS OWN
    // phase, deliberately EXCLUDED from roundTotalS — matching Python's
    // explicit, documented convention (run_experiment.py's module
    // docstring: "round_total_s deliberately excludes eval... When
    // reconstructing total per-round wall-clock time ... use
    // round_total_s + eval_total_s, not round_total_s alone"). An earlier
    // version of this project's round-timing computation included eval
    // inside roundTotalS, which would have been a real, silent
    // boundary mismatch against Python's own metric of the same name.
    public let evalTotalS: Double
    public let gossipPushS: Double         // time spent pushing parameters to peers (outbound only) — Python's gossip_push_s
    public let gossipAggS: Double          // time spent waiting for peer updates + aggregating them — Python's gossip_agg_s
    public let roundTotalS: Double        // EXCLUDES evalTotalS — see evalTotalS's doc comment; true total wall-clock is roundTotalS + evalTotalS

    // Wall-clock round-end marker — Python's `timestamp` column, written by
    // run_experiment.py right after round_total_s and eval timing are both
    // computed (see power_analysis.py's energy_by_round() docstring:
    // "'timestamp' is assumed to mark round END"). This field was missing
    // entirely from RoundMetrics/MetricsLogger despite the header comment's
    // claim of exact column parity with Python. Its absence is the actual
    // root cause of two separate-looking failures: energy_by_round() hard-
    // requires this column and would always skip round-boundary energy for
    // Swift runs, and — before metrics.py's plot_power was fixed to check
    // runtime — a lookup keyed only on node_id/condition could silently
    // borrow a DIFFERENT run's (Python's) timestamps instead of erroring,
    // producing a nonsense multi-day clock_offset_s.
    //
    // Whoever constructs RoundMetrics (RoundOrchestrator, presumably) must
    // capture Date() at actual round-end time — after round_total_s and
    // eval_total_s are both finalized — not reconstruct it later from
    // durations, matching Python's convention exactly.
    public let timestamp: Date

    // Compute utilization
    public let effectiveCores: Double     // cpu_seconds / elapsed_s, wall-clock inclusive
    public let shufS: Double?             // time inside extractBatch — Python's shuf_s

    // Memory
    public let peakRSSMB: Double           // high-water-mark RSS in MB, from ResourceUsage.peakRSSMB() — already unit-corrected for macOS(bytes)/Linux(KB), see that file's doc comment

    // Communication
    public let bytesSent: Int64
    public let bytesReceived: Int64
    public let peersReached: Int
    public let peersExpected: Int

    // Pi hardware state (always "unavailable" on Mac) — Python's throttled column
    public let throttled: String

    // Shard size for this node — Python's n_samples column
    public let nSamples: Int

    // Model
    public let trainLoss: Double?
    public let trainAcc: Double?     // accuracy on local shard using pre-aggregation weights, computed after training inside train_round — Python's train_acc
    public let testAcc: Double?      // accuracy on shared test set using post-aggregation weights — Python's test_acc
    public let testLoss: Double?     // loss on shared test set using post-aggregation weights — Python's test_loss
    public let localAcc: Double?     // accuracy on this node's local shard using post-aggregation weights ("in-distribution check") — Python's local_acc, the one metrics.py actually plots
    public let localLoss: Double?    // loss on this node's local shard using post-aggregation weights — Python's local_loss

    // Eval timing (excluded from roundTotalS per Python's convention)
    public let evalTestS: Double     // time spent evaluating on test set — Python's eval_test_s
    public let evalLocalS: Double    // time spent evaluating on local shard — Python's eval_local_s
    // NOTE: evalTotalS = evalTestS + evalLocalS, same as Python's eval_total_s = eval_test_s + eval_local_s

    // Status — explicit rather than silently averaging over incomplete rounds
    // (see alpha_0.1 OOM/stall handling decision).
    public let status: RoundStatus

    public init(
        round: Int,
        nodeID: Int,
        condition: String,
        trainTimeS: Double,
        fwdS: Double,
        bwdS: Double,
        optS: Double,
        evalTotalS: Double,
        evalTestS: Double,
        evalLocalS: Double,
        gossipPushS: Double,
        gossipAggS: Double,
        roundTotalS: Double,
        timestamp: Date,
        effectiveCores: Double,
        shufS: Double?,
        peakRSSMB: Double,
        bytesSent: Int64,
        bytesReceived: Int64,
        peersReached: Int,
        peersExpected: Int,
        throttled: String,
        nSamples: Int,
        trainLoss: Double?,
        trainAcc: Double?,
        testAcc: Double?,
        testLoss: Double?,
        localAcc: Double?,
        localLoss: Double?,
        status: RoundStatus
    ) {
        self.round = round
        self.nodeID = nodeID
        self.condition = condition
        self.trainTimeS = trainTimeS
        self.fwdS = fwdS
        self.bwdS = bwdS
        self.optS = optS
        self.evalTotalS = evalTotalS
        self.evalTestS = evalTestS
        self.evalLocalS = evalLocalS
        self.gossipPushS = gossipPushS
        self.gossipAggS = gossipAggS
        self.roundTotalS = roundTotalS
        self.timestamp = timestamp
        self.effectiveCores = effectiveCores
        self.shufS = shufS
        self.peakRSSMB = peakRSSMB
        self.bytesSent = bytesSent
        self.bytesReceived = bytesReceived
        self.peersReached = peersReached
        self.peersExpected = peersExpected
        self.throttled = throttled
        self.nSamples = nSamples
        self.trainLoss = trainLoss
        self.trainAcc = trainAcc
        self.testAcc = testAcc
        self.testLoss = testLoss
        self.localAcc = localAcc
        self.localLoss = localLoss
        self.status = status
    }
}

public enum RoundStatus: String, Sendable, Codable {
    case completed
    case partialPeersUnreachable = "partial_peers_unreachable"  // some peers timed out; round proceeded with available updates
    case oom = "oom"
    case crashed
}

/// One line in peer_push_log.jsonl — mirrors the Python field for per-push
/// communication diagnostics (RQ2/RQ3 instrumentation, captured now per the
/// project's "instrument now, analyze later" philosophy).
public struct PeerPushLogEntry: Sendable, Codable {
    public let round: Int
    public let fromNode: Int
    public let toNode: Int
    public let bytes: Int64
    public let pushDurationS: Double
    public let success: Bool
    public let timestamp: Double  // unix epoch seconds, matches Python's time.time() convention

    public init(round: Int, fromNode: Int, toNode: Int, bytes: Int64, pushDurationS: Double, success: Bool, timestamp: Double) {
        self.round = round
        self.fromNode = fromNode
        self.toNode = toNode
        self.bytes = bytes
        self.pushDurationS = pushDurationS
        self.success = success
        self.timestamp = timestamp
    }
}


