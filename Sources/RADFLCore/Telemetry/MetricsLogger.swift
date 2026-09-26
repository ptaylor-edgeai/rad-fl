// MetricsLogger.swift
//
// Writes RoundMetrics to CSV and PeerPushLogEntry to JSONL, matching the
// Python baseline's output conventions so existing analysis scripts
// (power_analysis.py, plotting) need minimal/no changes to consume Swift runs.
//
// Naming convention (matches established pattern, e.g. power_<runtime>_<condition>_<tag>_<timestamp>.csv):
//   results_swift_<condition>_node<N>_<timestamp>.csv
//   peer_push_log_swift_<condition>_node<N>_<timestamp>.jsonl

import Foundation

public enum MetricsLoggerError: Error, CustomStringConvertible {
    case fileCreationFailed(URL)

    public var description: String {
        switch self {
        case .fileCreationFailed(let url):
            return "MetricsLogger: failed to create file at \(url.path) — check disk space and write permissions for that directory"
        }
    }
}

/// `@unchecked Sendable`: genuinely safe now that `writeLock` serializes
/// every access to this class's mutable state (csvHandle, jsonlHandle,
/// wroteCSVHeader) — see that property's own doc comment for why this
/// needed to be added (RoundOrchestrator's concurrent per-peer push
/// logging is the first caller that actually requires this). Same pattern
/// used elsewhere in this project for verified-safe mutable classes (the
/// NIO channel handlers in GossipChannelHandlers.swift/GossipTransport.swift).
public final class MetricsLogger: @unchecked Sendable {
    private let csvURL: URL
    private let jsonlURL: URL
    private var csvHandle: FileHandle?
    private var jsonlHandle: FileHandle?
    private var wroteCSVHeader = false

    // Serializes actual write operations. Added specifically because
    // RoundOrchestrator's per-peer push logging (PeerPushLogEntry, one
    // call per peer from a withThrowingTaskGroup) is the first caller that
    // invokes log(_:) CONCURRENTLY from multiple tasks at once — every
    // earlier use of MetricsLogger was sequential (one round at a time).
    // Without this lock, concurrent FileHandle.write calls from different
    // tasks have no guarantee against interleaving their actual byte
    // writes, which could corrupt the JSONL file by merging two lines'
    // bytes together mid-write (each line is supposed to be one complete,
    // separate JSON object). A plain NSLock rather than converting this
    // class to an actor — actor isolation would require every existing
    // call site (which calls log(_:) synchronously, no await) to change,
    // a much bigger ripple than this targeted fix.
    private let writeLock = NSLock()

    // Matches run_experiment.py's `timestamp` column EXACTLY:
    //   "timestamp": datetime.now().isoformat(timespec="seconds")
    // — confirmed by reading run_experiment.py directly, not inferred.
    // That's naive local time (no timezone designator at all) truncated to
    // whole seconds (no fractional component). This is deliberately a plain
    // DateFormatter, not ISO8601DateFormatter: ISO8601DateFormatter's
    // .withInternetDateTime option ALWAYS appends a timezone designator
    // ("Z" or "+HH:MM") — there's no option to omit it — which would make
    // this column tz-AWARE while Python's is tz-NAIVE. That mismatch isn't
    // cosmetic: power_analysis.py's energy_by_round() does
    // pd.to_datetime(tlog["timestamp"]).values.astype("datetime64[ns]"),
    // and a tz-aware Series can't cleanly astype to naive datetime64[ns]
    // the same way — it either raises or silently mishandles the
    // conversion, which would corrupt every round-boundary energy figure
    // for Swift runs even though the column would exist and "look" fine at
    // a glance. TimeZone.current (not UTC) matches Python's naive
    // datetime.now(), which reflects whatever local timezone the machine
    // running it is set to — same principle energy_by_round()'s own
    // clock_offset_s auto-estimation already accounts for between the
    // training log and the power meter's clock.
    //
    // en_US_POSIX locale: standard practice for machine-parseable date
    // strings in Swift — without it, formatting can silently vary by the
    // device's current locale/calendar settings (e.g. non-Gregorian
    // calendars), which a fixed format string alone doesn't protect
    // against.
    //
    // Not to be confused with the `timestamp: String` constructor
    // parameter below — that one is an unrelated filename tag for this
    // whole run, not a per-round value.
    //
    // Thread safety: DateFormatter.string(from:) is called only from
    // log(_:) below, which already holds writeLock for its entire body —
    // so calls here are inherently serialized already; no separate
    // synchronization needed for this formatter.
    private static let timestampFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone.current
        return f
    }()

    private static let csvHeader = [
        // Column names match run_experiment.py's CSV_COLUMNS exactly so that
        // the same metrics.py can process Swift and Python output without
        // modification. Ordering also matches Python's for readability when
        // comparing CSVs side by side.
        //
        // "timestamp" was missing here until this fix, despite the claim
        // above — this list didn't actually match Python's columns exactly.
        // Its absence made power_analysis.py's energy_by_round() unable to
        // compute round-boundary energy for any Swift run at all (hard-
        // required column, immediate skip), and was the root cause behind
        // a separate, worse-looking symptom: metrics.py's plot_power, before
        // its own fix, could silently pair a Swift power trace with a
        // DIFFERENT run's (Python's) timestamps for the same node_id/
        // condition instead of erroring, producing a nonsense multi-day
        // clock-offset warning.
        "round",
        "train_loss", "train_acc",
        "test_acc", "test_loss", "local_acc", "local_loss",
        "eval_test_s", "eval_local_s", "eval_total_s",
        "fwd_s", "bwd_s", "opt_s", "shuf_s",
        "gossip_push_s", "gossip_agg_s",
        "bytes_sent",
        "peak_rss_mb",
        "effective_cores",
        "throttled",
        "n_samples",
        "elapsed_s",
        "round_total_s",
        "timestamp",
        // Swift-only columns (harmlessly ignored by metrics.py's _parse_row
        // since it skips columns not in _NUMERIC_COLUMNS; kept because they're
        // useful for Swift-specific analysis)
        "node_id", "condition",
        "bytes_received", "peers_reached", "peers_expected",
        "status",
        // ── schema v2 ───────────────────────────────────────────────────────
        // APPENDED, never inserted. metrics.py parses by column name and skips
        // names it doesn't recognise, so appending is invisible to it; inserting
        // would shift every position-indexed reader downstream.
        //
        // These are written from the first run of the campaign onward, even
        // though most are the -1 sentinel until their instrumentation lands.
        // The point is that every CSV produced from here has an IDENTICAL
        // shape — one schema for the whole campaign instead of analysis code
        // branching on which columns a given run happened to have.
        //
        // -1 in any of these means NOT INSTRUMENTED, not measured-as-zero.
        // Analysis MUST filter these out rather than averaging over them.
        "schema_version",
        "wire_bytes_sent", "wire_bytes_received",
        "payload_bytes_sent", "payload_bytes_received",
        "peers_timed_out",
        "compute_threads",
        "achieved_freq_khz",
        // ── schema v3 ───────────────────────────────────────────────────────
        // gossip_agg_s above is unchanged in value, so v2 and v3 runs remain
        // comparable on it. These two split what was previously conflated:
        // gossip_wait_s is the same number under a name that matches what it
        // measures, and gossip_aggregate_s is the aggregation work that was
        // never timed at all.
        "gossip_wait_s",
        "gossip_aggregate_s",
        // ── schema v4 ───────────────────────────────────────────────────────
        "peers_churn_dropped",
        // ── schema v5 ───────────────────────────────────────────────────────
        "train_acc_s",
    ].joined(separator: ",")

    public init(outputDirectory: URL, condition: String, nodeID: Int, timestamp: String) throws {
        // Filename matches metrics.py's expected path exactly:
        //   <results_root>/<node_id>/<condition>/training_log.csv
        // The earlier name (results_swift_<condition>_node<N>_<timestamp>.csv)
        // was a real blocker — metrics.py scans for training_log.csv
        // specifically and would never find a differently-named file.
        // The timestamp that was in the old name is now just in the JSONL
        // filename, where it's still useful for multiple runs producing
        // distinct push-log files without overwriting each other.
        self.csvURL = outputDirectory.appendingPathComponent("training_log.csv")
        self.jsonlURL = outputDirectory.appendingPathComponent(
            "peer_push_log_swift_\(condition.replacingOccurrences(of: " ", with: "_"))_node\(nodeID)_\(timestamp).jsonl"
        )

        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

        // ALWAYS (re)create both files from empty, even if a stale one from
        // a previous run already exists at this same path.
        //
        // This mirrors Python's CSVLogger, which opens with `open(path, "w",
        // ...)` — "w" truncates unconditionally on every process start, so a
        // retry always begins from a clean file. The previous version of this
        // initializer only created the file `if !fileExists`, then opened it
        // with `FileHandle(forWritingTo:)`, which seeks to offset 0 but does
        // NOT truncate. On a node/condition that gets re-run in place (e.g.
        // a Pi reboot mid-experiment, or simply re-launching after a crash —
        // training_log.csv's path is stable across runs, unlike the
        // timestamp-tagged jsonl name, so it's the one that actually collides)
        // a shorter new run would overwrite only the first N bytes and leave
        // the old file's longer tail dangling past that point. Whatever line
        // was mid-write when the new run stopped got spliced directly onto
        // whatever text used to occupy that same byte offset in the old file
        // — producing one torn, column-misaligned CSV line. Downstream,
        // metrics.py's _parse_row has no way to detect this; it just tries to
        // float() whatever stray text landed in a numeric column's slot and
        // raises `ValueError: could not convert string to float: '...'`.
        // Confirmed by reproducing this exact failure mode against the real
        // metrics.py with a synthetic torn file.
        //
        // Deleting-then-recreating (rather than opening with a truncating
        // flag) keeps this a single obvious FileManager call and matches the
        // createFile-based error handling already below.
        if FileManager.default.fileExists(atPath: csvURL.path) {
            try FileManager.default.removeItem(at: csvURL)
        }
        guard FileManager.default.createFile(atPath: csvURL.path, contents: nil) else {
            throw MetricsLoggerError.fileCreationFailed(csvURL)
        }
        if FileManager.default.fileExists(atPath: jsonlURL.path) {
            try FileManager.default.removeItem(at: jsonlURL)
        }
        guard FileManager.default.createFile(atPath: jsonlURL.path, contents: nil) else {
            throw MetricsLoggerError.fileCreationFailed(jsonlURL)
        }

        self.csvHandle = try FileHandle(forWritingTo: csvURL)
        self.jsonlHandle = try FileHandle(forWritingTo: jsonlURL)
    }

    public func log(_ metrics: RoundMetrics) throws {
        writeLock.lock()
        defer { writeLock.unlock() }

        if !wroteCSVHeader {
            try write(line: Self.csvHeader, to: csvHandle)
            wroteCSVHeader = true
        }

        // Built as individual statements rather than one large array literal.
        // The original single-literal version (16 elements, several involving
        // Optional.map(String.init) ?? "") hit a real Swift type-checker
        // limitation: stacking multiple overloaded String.init resolutions
        // inside one array literal can make the checker give up on the whole
        // expression with "ambiguous without a type annotation" rather than
        // pointing at the actual problem line. Each line below is its own
        // statement with an explicit type, so there's nothing left to infer.
        let shufSField: String = metrics.shufS.map { String($0) } ?? ""
        let trainLossField: String = metrics.trainLoss.map { String($0) } ?? ""
        let trainAccField: String = metrics.trainAcc.map { String($0) } ?? ""
        let testAccField: String = metrics.testAcc.map { String($0) } ?? ""
        let testLossField: String = metrics.testLoss.map { String($0) } ?? ""
        let localAccField: String = metrics.localAcc.map { String($0) } ?? ""
        let localLossField: String = metrics.localLoss.map { String($0) } ?? ""
        // peak_rss_mb: ResourceUsage.peakRSSMB() already returns MB,
        // correctly unit-converted for the platform (macOS bytes vs Linux
        // KB — see ResourceUsage.swift's doc comment) — written directly,
        // no further conversion. An earlier version of this code assumed
        // the RoundMetrics field held raw bytes and divided by 1024*1024
        // here; that would have silently double-converted the now-real
        // getrusage data and produced numbers wrong by another factor of
        // ~1e6, the exact class of unit bug Python's own trainer.py
        // comment specifically warns about.
        let peakRSSMBField: String = metrics.peakRSSMB < 0
            ? String(metrics.peakRSSMB)
            : String(format: "%.3f", metrics.peakRSSMB)
        let timestampField: String = Self.timestampFormatter.string(from: metrics.timestamp)

        let fields: [String] = [
            // Python-compatible columns in Python's order
            String(metrics.round),
            trainLossField, trainAccField,
            testAccField, testLossField, localAccField, localLossField,
            String(metrics.evalTestS), String(metrics.evalLocalS), String(metrics.evalTotalS),
            String(metrics.fwdS), String(metrics.bwdS), String(metrics.optS), shufSField,
            String(metrics.gossipPushS),   // gossip_push_s — push phase only
            String(metrics.gossipAggS),    // gossip_agg_s — wait-for-peers + aggregate phase
            String(metrics.bytesSent),
            peakRSSMBField,                // peak_rss_mb — in MB, matching Python's unit
            String(metrics.effectiveCores),
            metrics.throttled,             // "unavailable" on non-Pi hardware
            String(metrics.nSamples),      // this node's local shard size
            String(metrics.trainTimeS),    // elapsed_s — total training wall-clock
            String(metrics.roundTotalS),
            timestampField,                // timestamp — wall-clock round end, ISO8601
            // Swift-only columns (harmlessly ignored by metrics.py)
            String(metrics.nodeID), metrics.condition,
            String(metrics.bytesReceived),
            String(metrics.peersReached), String(metrics.peersExpected),
            metrics.status.rawValue as String,
            // schema v2 — order must match csvHeader's tail exactly
            String(RoundMetrics.schemaVersion),
            String(metrics.wireBytesSent), String(metrics.wireBytesReceived),
            String(metrics.payloadBytesSent), String(metrics.payloadBytesReceived),
            String(metrics.peersTimedOut),
            String(metrics.computeThreads),
            String(metrics.achievedFreqKHz),
            // schema v3
            String(metrics.gossipWaitS),
            String(metrics.gossipAggregateS),
            // schema v4
            String(metrics.peersChurnDropped),
            // schema v5
            String(metrics.trainAccS),
        ]

        // Column-count guard. The header and this array are two separate
        // literals that have to stay in lockstep, and a mismatch produces a
        // silently misaligned CSV rather than an error — every column after the
        // divergence point shifts, so a numeric column ends up holding a
        // neighbour's value and still parses as a valid float. That is precisely
        // the failure this file's own header comment describes hitting before,
        // via a different route (the torn-file truncation bug). Checking here
        // costs one comparison per round and turns a silent data-corruption bug
        // into an immediate, obvious failure at the first logged round.
        let headerCount = Self.csvHeader.split(separator: ",", omittingEmptySubsequences: false).count
        precondition(
            fields.count == headerCount,
            "MetricsLogger: CSV column count mismatch — header declares \(headerCount) "
            + "columns but the row has \(fields.count). The csvHeader array and the "
            + "fields array in log(_:) have diverged; every column after the "
            + "divergence point would be silently misaligned."
        )

        try write(line: fields.joined(separator: ","), to: csvHandle)
    }

    public func log(_ entry: PeerPushLogEntry) throws {
        writeLock.lock()
        defer { writeLock.unlock() }

        let data = try JSONEncoder().encode(entry)
        guard let line = String(data: data, encoding: .utf8) else { return }
        try write(line: line, to: jsonlHandle)
    }

    private func write(line: String, to handle: FileHandle?) throws {
        guard let handle, let data = (line + "\n").data(using: .utf8) else { return }
        try handle.write(contentsOf: data)
    }

    public func close() throws {
        writeLock.lock()
        defer { writeLock.unlock() }

        try csvHandle?.close()
        try jsonlHandle?.close()
        csvHandle = nil
        jsonlHandle = nil
    }
}


