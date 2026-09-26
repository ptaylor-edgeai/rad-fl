// RoundMetrics.swift
//
// Per-round telemetry, field-matched to the Python baseline's instrumentation
// (metrics.py / power_analysis.py / log_power.py) so the results table can do
// same-condition, same-seed Python vs Swift pairs without unit/field mismatches.
//
// Fields intentionally mirror Python naming (snake_case in the CSV/JSONL output,
// even though Swift convention is camelCase internally) so downstream analysis
// scripts (pandas-based) don't need per-runtime column-name branching.
//
// ── Schema versioning and the sentinel convention (added at schema v2) ──────
//
// All experiments from here are Swift-only — the Python baseline is frozen as
// submitted, so this struct is no longer constrained by cross-runtime column
// parity. It IS still constrained by metrics.py/power_analysis.py, which read
// these CSVs as analysis tools; those parse by column NAME and skip unknown
// columns, so new fields are always APPENDED to the header, never inserted.
//
// Fields are being added ahead of the code that populates them, deliberately.
// The alternative — adding columns mid-campaign as each feature lands — leaves
// runs from the same campaign with different CSV shapes, which metrics.py then
// has to branch on. Adding them now with an explicit "not instrumented"
// sentinel keeps every run from this point forward structurally identical.
//
// SENTINEL CONVENTION, and it matters:
//     -1   means NOT INSTRUMENTED — this build could not measure the value
//      0   means MEASURED AS ZERO — a real observation
//
// These are NOT interchangeable. `peersTimedOut == 0` means the deadline was
// active and nothing timed out (an RQ2 result); `peersTimedOut == -1` means no
// deadline mechanism existed in the build that produced the row. Conflating
// them turns "the topology was robust" into "we weren't measuring", which is
// exactly the kind of silent ambiguity that ruins a robustness claim after the
// cluster time has already been spent. Analysis code MUST filter -1 explicitly
// rather than treating it as a numeric value; a mean over a column containing
// -1 sentinels is meaningless.
//
// Schema history:
//   v1 — original: Python-parity columns + node_id/condition/bytes_received/
//        peers_reached/peers_expected/status.
//   v2 — adds schema_version, the wire/payload byte split, peers_timed_out,
//        compute_threads, achieved_freq_khz. All default to the -1 sentinel
//        so existing call sites compile and run unchanged.
//   v3 — adds gossip_wait_s and gossip_aggregate_s. NON-BREAKING:
//        gossip_agg_s keeps exactly the value it has always had, so v2 runs
//        stay directly comparable with v3 ones. See those fields for what
//        the previous instrumentation was actually measuring.
//   v5 — adds train_acc_s, the post-training accuracy pass. That pass has run
//        since the project began and was never timed; it appeared only as an
//        unattributed remainder in every phase breakdown. train_acc itself is
//        unchanged and remains nil when the pass is skipped.
//   v4 — adds peers_churn_dropped, for the failure regimes. peers_timed_out
//        (v2) becomes populated at the same time, having been a -1 sentinel
//        until the round deadline existed to produce it.

import Foundation

public struct RoundMetrics: Sendable, Codable {
    /// Sentinel meaning "this build did not instrument this value". Distinct
    /// from a measured zero — see the file header. Used as the default for
    /// every field added after schema v1.
    public static let notInstrumented: Int64 = -1

    /// CSV/struct schema version. Bump when columns are added so analysis code
    /// can dispatch on an explicit number rather than sniffing for the presence
    /// of a column and guessing.
    public static let schemaVersion: Int = 5

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

    /// Python's `gossip_agg_s`, kept bit-for-bit as it has always been
    /// recorded so that runs from before schema v3 stay directly comparable.
    ///
    /// Its documented meaning was "waiting for peer updates + aggregating
    /// them". That was never what it measured. RoundOrchestrator computed it
    /// on the line immediately after the peer wait returned, roughly 45 lines
    /// before the aggregation actually ran, so it has always been WAIT ONLY.
    ///
    /// Prefer `gossipWaitS` in new analysis — same number, honest name. This
    /// field is retained for continuity across the campaign, not because it is
    /// well named.
    public let gossipAggS: Double

    /// Time blocked waiting for every peer's update for this round.
    ///
    /// Numerically identical to `gossipAggS`; it exists so the quantity has a
    /// name that matches what it measures. This is straggler-synchronisation
    /// cost, and it is what topology choice and any deadline-driven local-work
    /// policy are trying to reduce — so it needs to be addressable by name
    /// rather than inferred from a mis-named column.
    ///
    /// The node is near-idle throughout: measured at ~1.0 W against a 0.824 W
    /// platform floor, i.e. roughly 0.18 W of recoverable headroom against a
    /// ~2.5 W loaded draw. Worth knowing before designing any policy that
    /// proposes to reclaim power during this window — there is far less there
    /// than the duration suggests.
    public let gossipWaitS: Double

    /// Time spent turning received peer messages into an adopted model:
    /// PeerUpdate construction, tensor-count validation, the sample-weighted
    /// average, and setParameters.
    ///
    /// Previously unmeasured, and therefore invisible. It surfaced as part of
    /// a ~11s per-round gap between round wall-clock and the sum of all logged
    /// phases — 13% of the round on a 25-node-round IID run, larger than the
    /// entire gossip phase as it was then reported.
    ///
    /// Scales with peer count, so it matters more as topology density rises,
    /// which is exactly when a topology comparison needs it attributed rather
    /// than lost.
    public let gossipAggregateS: Double

    /// Peers whose update ARRIVED but was deliberately discarded by churn
    /// injection.
    ///
    /// Kept distinct from `peersTimedOut`, which counts updates that never
    /// came. Conflating them would make an unreliable network
    /// indistinguishable from a slow one — and separating those is the whole
    /// point of running a churn regime alongside a crash regime.
    public let peersChurnDropped: Int

    /// Seconds spent on the post-training accuracy pass; 0 when skipped, -1 in
    /// runs from before this was instrumented.
    ///
    /// With this the logged phases finally account for the whole round. The
    /// pass costs ~11s on a 5,000-sample shard — about 12% of an 88-second
    /// round, and the largest single item remaining once evaluation cadence has
    /// been reduced.
    public let trainAccS: Double
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

    // ── Communication, schema v2 ────────────────────────────────────────────
    //
    // The wire/payload split exists because of connect-per-push. Every gossip
    // message pays a TCP handshake plus RAFD framing plus per-message headers
    // on top of its tensor payload. At dense FP32 that overhead is a rounding
    // error against a model-sized payload and nobody would notice. Under RQ1's
    // compression mechanisms — top-k at k=1%, INT8 quantisation — the payload
    // shrinks by one to two orders of magnitude while the fixed cost does not
    // move at all, so the overhead can come to DOMINATE the transfer.
    //
    // Reporting only one figure gets this wrong in one of two ways: report
    // payload alone and the compression saving is overstated, because the bytes
    // that actually crossed the network didn't fall nearly as far; report wire
    // alone and the mechanism's real effect on the tensor is invisible. RQ1's
    // primary metric is bytes-to-target, so this is not a diagnostic nicety —
    // getting it wrong misstates the headline result.
    //
    // The gap between them is itself a finding about this architecture, and one
    // worth reporting: it quantifies what connect-per-push costs, and it is the
    // evidence that would justify persistent connections if the numbers warrant.
    //
    // wire    = everything written to / read from the socket, framing included
    // payload = serialised tensor bytes only
    // Invariant when both are instrumented: wire >= payload.
    public let wireBytesSent: Int64
    public let wireBytesReceived: Int64
    public let payloadBytesSent: Int64
    public let payloadBytesReceived: Int64

    /// Peers whose update did not arrive before the round deadline. Requires
    /// the round-level peer deadline (currently RoundOrchestrator waits
    /// indefinitely, so this is -1 until that lands). Distinct from
    /// `peersExpected - peersReached`, which cannot tell a timeout apart from a
    /// peer that was never reachable in the first place — a distinction RQ2's
    /// churn and targeted-failure arms depend on.
    public let peersTimedOut: Int

    // ── Resource state, schema v2 ───────────────────────────────────────────

    /// Compute width actually used by Tensor.swift's concurrentPerform calls.
    ///
    /// Recorded because libdispatch on Linux sizes its worker pool from
    /// sysconf(_SC_NPROCESSORS_ONLN), which does NOT respect a cgroup cpuset —
    /// so a node restricted to one CPU may still run four workers timeslicing
    /// on that core. That is an oversubscribed workload with a materially
    /// different cache profile from a genuine single-core device, and it would
    /// silently corrupt every heterogeneity-profile comparison. Logging the
    /// requested width per round makes the manipulation auditable from the data
    /// rather than assumed from the launch command.
    public let computeThreads: Int

    /// CPU clock actually achieved, in kHz, from `vcgencmd measure_clock arm`
    /// (NOT the requested scaling_max_freq).
    ///
    /// These differ under thermal throttling, and the direction of the error is
    /// the damaging one: the fast profile is the one that throttles, so an
    /// unlogged drop biases results AGAINST the high-frequency condition and
    /// compresses the heterogeneity gradient. Pairs with `throttled` — that
    /// field says whether a limit was hit at all, this one says what the clock
    /// actually was when it happened.
    public let achievedFreqKHz: Int

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
        status: RoundStatus,
        // ── schema v2 ───────────────────────────────────────────────────────
        // Appended at the END of the parameter list, with defaults, so every
        // existing call site (RoundOrchestrator) compiles and runs unchanged.
        // Inserting them next to their related v1 fields would have read more
        // naturally but would break every caller, which is not worth it for
        // parameters that are about to be filled in one at a time as each
        // feature lands.
        //
        // Defaults are the -1 "not instrumented" sentinel, NOT 0 — see the
        // file header. A default of 0 would be a lie: it would claim a
        // measurement of zero for a quantity nothing measured.
        wireBytesSent: Int64 = RoundMetrics.notInstrumented,
        wireBytesReceived: Int64 = RoundMetrics.notInstrumented,
        payloadBytesSent: Int64 = RoundMetrics.notInstrumented,
        payloadBytesReceived: Int64 = RoundMetrics.notInstrumented,
        peersTimedOut: Int = -1,
        computeThreads: Int = -1,
        achievedFreqKHz: Int = -1,
        // ── schema v3 ───────────────────────────────────────────────────────
        // Default to -1 (not instrumented) rather than 0, so a build that has
        // not yet been updated records honestly instead of claiming a measured
        // zero for a phase it never timed.
        gossipWaitS: Double = -1,
        gossipAggregateS: Double = -1,
        // ── schema v4 ───────────────────────────────────────────────────────
        peersChurnDropped: Int = -1,
        // ── schema v5 ───────────────────────────────────────────────────────
        trainAccS: Double = -1
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
        self.wireBytesSent = wireBytesSent
        self.wireBytesReceived = wireBytesReceived
        self.payloadBytesSent = payloadBytesSent
        self.payloadBytesReceived = payloadBytesReceived
        self.peersTimedOut = peersTimedOut
        self.computeThreads = computeThreads
        self.achievedFreqKHz = achievedFreqKHz
        self.gossipWaitS = gossipWaitS
        self.gossipAggregateS = gossipAggregateS
        self.peersChurnDropped = peersChurnDropped
        self.trainAccS = trainAccS
    }
}

public enum RoundStatus: String, Sendable, Codable {
    case completed
    case partialPeersUnreachable = "partial_peers_unreachable"  // some peers timed out; round proceeded with available updates
    case oom = "oom"
    case crashed
}

/// What a peer_push_log.jsonl line describes.
///
/// The file began as a push-only log, which is why the type is named for
/// pushes. Under the failure regimes it also has to record what happened on
/// the RECEIVE side, because `training_log.csv` carries only counts —
/// `peers_timed_out: 1` says a peer was lost, not WHICH peer.
///
/// That distinction is the measurement in RQ2's targeted-failure arm: the
/// claim is that killing a cluster head costs more than killing a high-degree
/// hub, and verifying it requires knowing the failure was the head. Push
/// failures alone cannot supply it — a peer this node failed to push to is not
/// necessarily the peer that failed to send, which is precisely the asymmetry
/// observed when a node received weights from a peer it could not reach.
public enum PeerEventKind: String, Sendable, Codable {
    /// A push attempt, successful or not. `success` carries which.
    case push
    /// A peer's update never arrived before the round deadline.
    case timeout
    /// A peer's update arrived and was deliberately discarded by churn
    /// injection. Distinct from `timeout`: an unreliable link and a slow one
    /// are different failures, and separating them is the point of running a
    /// churn regime alongside a crash regime.
    case churnDropped = "churn_dropped"
    /// A peer was unreachable at startup and excluded from the whole run.
    case excludedAtStartup = "excluded_at_startup"
}

/// One line in peer_push_log.jsonl — per-peer communication diagnostics.
///
/// `kind` distinguishes the event types; older files contain only pushes and
/// have no `kind` field, so readers should default a missing value to `.push`.
public struct PeerPushLogEntry: Sendable, Codable {
    public let round: Int
    public let fromNode: Int
    public let toNode: Int
    public let bytes: Int64
    public let pushDurationS: Double
    public let success: Bool
    public let timestamp: Double  // unix epoch seconds, matches Python's time.time() convention
    public let kind: PeerEventKind
    /// Peer's topology ID (e.g. "pi-10"). Numeric IDs are what the wire
    /// protocol uses, but every analysis groups by topology ID, and resolving
    /// one to the other after the fact needs the topology file that produced
    /// the run — which is exactly what goes missing.
    public let peerID: String?

    public init(round: Int, fromNode: Int, toNode: Int, bytes: Int64,
                pushDurationS: Double, success: Bool, timestamp: Double,
                kind: PeerEventKind = .push, peerID: String? = nil) {
        self.round = round
        self.fromNode = fromNode
        self.toNode = toNode
        self.bytes = bytes
        self.pushDurationS = pushDurationS
        self.success = success
        self.timestamp = timestamp
        self.kind = kind
        self.peerID = peerID
    }

    /// A peer whose update never arrived before the deadline.
    public static func timeout(round: Int, fromNode: Int, toNode: Int,
                               peerID: String, waitedS: Double) -> PeerPushLogEntry {
        PeerPushLogEntry(round: round, fromNode: fromNode, toNode: toNode,
                         bytes: 0, pushDurationS: waitedS, success: false,
                         timestamp: Date().timeIntervalSince1970,
                         kind: .timeout, peerID: peerID)
    }

    /// A peer's update that arrived and was discarded by churn injection.
    public static func churnDropped(round: Int, fromNode: Int, toNode: Int,
                                    peerID: String) -> PeerPushLogEntry {
        PeerPushLogEntry(round: round, fromNode: fromNode, toNode: toNode,
                         bytes: 0, pushDurationS: 0, success: false,
                         timestamp: Date().timeIntervalSince1970,
                         kind: .churnDropped, peerID: peerID)
    }

    /// A peer excluded before round 1 because it never became reachable.
    public static func excludedAtStartup(fromNode: Int, toNode: Int,
                                         peerID: String) -> PeerPushLogEntry {
        PeerPushLogEntry(round: 0, fromNode: fromNode, toNode: toNode,
                         bytes: 0, pushDurationS: 0, success: false,
                         timestamp: Date().timeIntervalSince1970,
                         kind: .excludedAtStartup, peerID: peerID)
    }
}



