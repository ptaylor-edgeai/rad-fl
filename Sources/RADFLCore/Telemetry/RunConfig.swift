// RunConfig.swift
//
// Writes run_config.json — the complete, machine-readable record of what a
// `run-round` invocation actually was.
//
// ── Why this exists ─────────────────────────────────────────────────────────
//
// config.json already records SimpleCNNConfig, i.e. the ARCHITECTURE. That is
// genuinely useful (it lets saved .npy weights be reloaded without guessing
// layer shapes) but it is only a fraction of what defines a run. Seed, learning
// rate, rounds, topology mode, which shards were loaded, which binary was
// running, and what the CPU was doing at the time are all absent — and every
// one of them is needed either to reproduce the run or to interpret its
// numbers.
//
// The concrete consequence: an existing set of runs cannot be told apart by
// seed, cannot confirm which governor was in effect, and cannot be traced to a
// source revision. Those facts are recoverable only from whoever ran them,
// while they still remember. This file makes each run self-describing at the
// moment it starts, so that stops being true.
//
// ── Cross-platform constraint, applied strictly ─────────────────────────────
//
// Everything here is pure Foundation. No subprocess spawning, no C interop, no
// vcgencmd. The reasoning is the same one that moved ResourceUsage off
// sched_getaffinity after it failed to resolve under the Debian Trixie SDK's
// Glibc modulemap: a metadata feature must not be able to break a
// cross-compile, and must not need a macOS-specific branch to stay buildable
// on the dev machine.
//
// So CPU frequency and governor come from /sys reads, which return nil on
// macOS. A nil there is CORRECT, not a gap — a Mac dev run genuinely has no Pi
// frequency state, and recording "unknown" is more truthful than fabricating a
// value.
//
// The one thing deliberately NOT captured here is `vcgencmd get_throttled`.
// That is the authoritative throttling signal — scaling_cur_freq reports the
// governor's intent, not what the SoC actually did — but reading it needs a
// subprocess. It belongs in the collection script, which already runs on the
// Pi over SSH and can capture it once per run into node state. Noting the
// distinction explicitly so nobody later mistakes scaling_cur_freq for proof
// that throttling did not occur.

import Foundation

/// Point-in-time CPU and OS state, read from /sys and /proc.
///
/// All fields optional: nil means "this platform does not expose it", which is
/// the expected result on macOS for every frequency field.
public struct SystemState: Codable, Sendable {
    public let governor: String?
    public let scalingMaxFreqKHz: Int?
    public let scalingMinFreqKHz: Int?
    public let scalingCurFreqKHz: Int?
    public let availableFrequenciesKHz: [Int]?

    /// Cores this process may run on (respects taskset AND cgroup cpuset).
    public let cpuAffinityCount: Int
    /// Cores the OS reports online (IGNORES cgroup cpuset).
    public let cpuOnlineCount: Int
    /// What Foundation reports — the number an unconfigured
    /// `concurrentPerform` is most likely to follow.
    public let activeProcessorCount: Int

    public let osRelease: String?
    public let kernelVersion: String?

    private static func readTrimmed(_ path: String) -> String? {
        guard let data = FileManager.default.contents(atPath: path),
              let text = String(data: data, encoding: .utf8) else { return nil }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private static func readInt(_ path: String) -> Int? {
        readTrimmed(path).flatMap { Int($0) }
    }

    private static let cpufreq = "/sys/devices/system/cpu/cpu0/cpufreq"

    public static func capture() -> SystemState {
        let available = readTrimmed("\(cpufreq)/scaling_available_frequencies")
            .map { line in
                line.split(separator: " ").compactMap { Int($0) }
            }

        // /etc/os-release's PRETTY_NAME, for recording which Raspberry Pi OS
        // release a node was on. Parsed rather than read wholesale so the JSON
        // stays one short line instead of a dozen shell-style assignments.
        var pretty: String? = nil
        if let osr = readTrimmed("/etc/os-release") {
            for line in osr.split(separator: "\n") where line.hasPrefix("PRETTY_NAME=") {
                pretty = line
                    .dropFirst("PRETTY_NAME=".count)
                    .trimmingCharacters(in: CharacterSet(charactersIn: "\""))
                break
            }
        }

        return SystemState(
            governor: readTrimmed("\(cpufreq)/scaling_governor"),
            scalingMaxFreqKHz: readInt("\(cpufreq)/scaling_max_freq"),
            scalingMinFreqKHz: readInt("\(cpufreq)/scaling_min_freq"),
            scalingCurFreqKHz: readInt("\(cpufreq)/scaling_cur_freq"),
            availableFrequenciesKHz: available,
            cpuAffinityCount: ResourceUsage.affinityCoreCount(),
            cpuOnlineCount: ResourceUsage.onlineProcessorCount(),
            activeProcessorCount: ResourceUsage.activeProcessorCount(),
            osRelease: pretty,
            kernelVersion: readTrimmed("/proc/sys/kernel/osrelease")
        )
    }
}

/// Partition family and skew level, parsed from the condition string.
///
/// The condition string is already the single source of truth for which shards
/// were loaded and where results are written, so deriving these from it keeps
/// one authority rather than adding flags that could disagree with it. Parsing
/// is best-effort: an unrecognised condition records the raw string with nil
/// family/alpha rather than failing the run, since a custom condition name is a
/// legitimate thing to want and is not worth aborting over.
public struct ConditionSpec: Codable, Sendable {
    public let raw: String
    /// "eq" (P-eq, equal cardinality) or "dir" (P-dir, natural Dirichlet).
    public let family: String?
    /// Nominal Dirichlet alpha, or nil for IID.
    public let alpha: Double?
    public let isIID: Bool

    /// Parses conditions of the forms produced by extract_cifar10_shards.py:
    ///   "iid", "alpha_0p1"            -> P-dir
    ///   "peq_iid", "peq_alpha_0p1"    -> P-eq
    ///
    /// NOTE on nominal vs realised alpha: under P-eq the equal-cardinality
    /// constraint projects the drawn Dirichlet distribution onto a feasible
    /// set, so the alpha recorded here is what was REQUESTED, not what the
    /// nodes received. Realised skew (label entropy, TV distance) lives in the
    /// partition manifest and is the figure that belongs in the paper. This
    /// field is for identifying the condition, not for characterising it.
    public static func parse(_ condition: String) -> ConditionSpec {
        var rest = condition
        var family: String? = nil

        if rest.hasPrefix("peq_") {
            family = "eq"
            rest = String(rest.dropFirst("peq_".count))
        } else if rest.hasPrefix("iid") || rest.hasPrefix("alpha_") {
            family = "dir"
        }

        if rest == "iid" {
            return ConditionSpec(raw: condition, family: family, alpha: nil, isIID: true)
        }

        if rest.hasPrefix("alpha_") {
            // Take only the alpha token, stopping at the next underscore.
            // Conditions can legitimately carry trailing suffixes — the
            // seed-separated form "peq_alpha_0p1_s43" exists precisely so a
            // second seed's shards do not overwrite the first's — and a naive
            // parse of the whole remainder yields "0.1_s43", which is not a
            // number, silently producing alpha=nil for a perfectly valid
            // condition. The seed itself is recorded explicitly from --seed,
            // so nothing here needs to interpret the suffix; it only needs to
            // stop before it.
            let token = rest
                .dropFirst("alpha_".count)
                .split(separator: "_", maxSplits: 1, omittingEmptySubsequences: false)
                .first
                .map(String.init) ?? ""
            let value = token.replacingOccurrences(of: "p", with: ".")
            return ConditionSpec(raw: condition, family: family,
                                 alpha: Double(value), isIID: false)
        }

        // "iid" with a trailing suffix, e.g. "peq_iid_s43".
        if rest.hasPrefix("iid") {
            return ConditionSpec(raw: condition, family: family, alpha: nil, isIID: true)
        }

        return ConditionSpec(raw: condition, family: nil, alpha: nil, isIID: false)
    }
}

/// The full record of one `run-round` invocation on one node.
///
/// Written once at startup, BEFORE training begins — deliberately, matching the
/// existing reasoning for config.json. A run that is interrupted partway still
/// leaves behind a complete description of what was being attempted, which is
/// exactly when that description is most useful.
///
/// Codable but deliberately NOT Sendable: it embeds SimpleCNNConfig, and
/// declaring Sendable here would silently require that type to be Sendable too
/// — a conformance this file has no business demanding. Nothing needs it: the
/// value is constructed and written synchronously at startup and never crosses
/// a concurrency boundary. SystemState and ConditionSpec hold only primitives
/// and stay Sendable on their own merits.
public struct RunConfig: Codable {
    /// Bump when fields are added, so analysis code dispatches on a number
    /// rather than probing for key presence.
    public static let schemaVersion = 1

    public let schemaVersion: Int

    // Identity
    public let nodeID: String
    public let nodeNumericID: Int
    public let nodeIndex: Int
    public let hostname: String
    public let startedAtUTC: String

    // Experiment
    public let condition: ConditionSpec
    public let seed: UInt64
    public let rounds: Int
    public let learningRate: Float
    public let batchSize: Int

    // Failure regime. nil / 0 means the default: wait indefinitely for peers,
    // drop nothing. Recorded because a run that tolerated peer loss and one
    // that did not are different experiments, and the difference is invisible
    // in the results otherwise — a clean run under a deadline looks identical
    // to a clean run without one.
    public let peerDeadlineSeconds: Double?
    public let churnDropProbability: Double
    public let churnSeed: UInt64

    /// Compression mechanism, as configured. Declared rather than measured —
    /// unlike link shaping, this is genuinely under the binary's control, so
    /// what it was told to do is what it did. The realised byte ratio is in
    /// training_log.csv (wire_bytes_sent / payload_bytes_sent), which is the
    /// number to report: it is smaller than the configured ratio, because
    /// message and per-tensor headers are not compressed.
    public let compression: String

    /// Evaluate every N rounds; 1 means every round. Needed to interpret
    /// rounds-to-target, which resolves only to N rounds under this setting.
    public let evalEvery: Int

    /// Whether the post-training accuracy pass was skipped. Recorded because a
    /// run without it is ~12% cheaper per round, and nothing else in the
    /// artefacts distinguishes the two.
    public let skipTrainAcc: Bool

    // Data
    public let dataDirectory: String
    public let outputDirectory: String
    public let trainSampleCount: Int
    public let testSampleCount: Int
    public let trainImageShape: [Int]

    // Topology
    public let topologyPath: String
    public let topologyMode: String
    public let peersExpected: Int
    public let peerIDs: [String]

    /// Peers that were unreachable when the run started and were excluded from
    /// it, per --startup-deadline-s.
    ///
    /// Recorded because the graph a run actually used is not the graph in the
    /// topology file it was handed. Without this, a ten-node run that silently
    /// proceeded on nine would be indistinguishable from a healthy one, and
    /// per-node degree derived from the topology file would be wrong for every
    /// neighbour of the missing node.
    public let excludedPeerIDs: [String]

    // Model architecture (same content as config.json, embedded here too so
    // this file alone is sufficient to describe the run — config.json stays
    // for backward compatibility and for the weight-reloading path that
    // already depends on it).
    public let modelConfig: SimpleCNNConfig

    // Provenance
    public let gitCommit: String
    public let buildTimestampUTC: String
    public let binarySHA256: String
    public let metricsSchemaVersion: Int

    // Hardware state at start
    public let systemState: SystemState

    /// Compute width requested for Tensor.swift's concurrentPerform calls.
    /// -1 when not explicitly set, meaning libdispatch chose for itself.
    ///
    /// Worth comparing against systemState.cpuAffinityCount when reading
    /// results: if affinity is 1 but this is 4, the node was oversubscribed
    /// and its timings do not represent a genuine single-core device.
    public let computeThreads: Int

    public init(
        nodeID: String,
        nodeNumericID: Int,
        nodeIndex: Int,
        condition: String,
        seed: UInt64,
        rounds: Int,
        learningRate: Float,
        batchSize: Int,
        peerDeadlineSeconds: Double? = nil,
        churnDropProbability: Double = 0,
        churnSeed: UInt64 = 0,
        compression: String = "none",
        evalEvery: Int = 1,
        skipTrainAcc: Bool = false,
        dataDirectory: String,
        outputDirectory: String,
        trainSampleCount: Int,
        testSampleCount: Int,
        trainImageShape: [Int],
        topologyPath: String,
        topologyMode: String,
        peerIDs: [String],
        excludedPeerIDs: [String] = [],
        modelConfig: SimpleCNNConfig,
        computeThreads: Int = -1
    ) {
        self.schemaVersion = RunConfig.schemaVersion
        self.nodeID = nodeID
        self.nodeNumericID = nodeNumericID
        self.nodeIndex = nodeIndex
        self.hostname = ProcessInfo.processInfo.hostName
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd'T'HH:mm:ss'Z'"
        fmt.locale = Locale(identifier: "en_US_POSIX")
        fmt.timeZone = TimeZone(identifier: "UTC")
        self.startedAtUTC = fmt.string(from: Date())

        self.condition = ConditionSpec.parse(condition)
        self.seed = seed
        self.rounds = rounds
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.peerDeadlineSeconds = peerDeadlineSeconds
        self.churnDropProbability = churnDropProbability
        self.churnSeed = churnSeed
        self.compression = compression
        self.evalEvery = evalEvery
        self.skipTrainAcc = skipTrainAcc

        self.dataDirectory = dataDirectory
        self.outputDirectory = outputDirectory
        self.trainSampleCount = trainSampleCount
        self.testSampleCount = testSampleCount
        self.trainImageShape = trainImageShape

        self.topologyPath = topologyPath
        self.topologyMode = topologyMode
        self.excludedPeerIDs = excludedPeerIDs.sorted()
        let active = peerIDs.filter { !excludedPeerIDs.contains($0) }
        self.peersExpected = active.count
        self.peerIDs = active

        self.modelConfig = modelConfig

        self.gitCommit = BuildInfo.gitCommit
        self.buildTimestampUTC = BuildInfo.buildTimestampUTC
        self.binarySHA256 = BuildInfo.binarySHA256
        self.metricsSchemaVersion = RoundMetrics.schemaVersion

        self.systemState = SystemState.capture()
        self.computeThreads = computeThreads
    }

    /// Writes run_config.json into `directory`, pretty-printed with sorted keys
    /// so the file diffs cleanly between runs — the fields that changed should
    /// be visible in a diff without reformatting first.
    public func write(to directory: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: directory.appendingPathComponent("run_config.json"),
                       options: .atomic)
    }

    /// One-line summary for the run log, so a mismatched governor or an
    /// oversubscribed cpuset is visible in the terminal at launch rather than
    /// only in the JSON afterwards.
    public var summary: String {
        let gov = systemState.governor ?? "n/a"
        let freq = systemState.scalingCurFreqKHz.map { "\($0 / 1000)MHz" } ?? "n/a"
        let cores = "\(systemState.cpuAffinityCount)/\(systemState.cpuOnlineCount)"
        let regime = peerDeadlineSeconds.map { "deadline=\($0)s" } ?? "no-deadline"
        let comp = compression == "none" ? "" : " compression=\(compression)"
            + (evalEvery > 1 ? " eval-every=\(evalEvery)" : "")
            + (skipTrainAcc ? " skip-train-acc" : "")
        let excl = excludedPeerIDs.isEmpty ? ""
                 : " excluded=\(excludedPeerIDs.joined(separator: ","))"
        let churn = churnDropProbability > 0 ? " churn=\(churnDropProbability)" : ""
        return "seed=\(seed) rounds=\(rounds) lr=\(learningRate) batch=\(batchSize) "
            + "| \(regime)\(churn)\(comp)\(excl) "
            + "| \(condition.raw) n=\(trainSampleCount) "
            + "| \(topologyMode) peers=\(peersExpected) "
            + "| gov=\(gov) freq=\(freq) cores=\(cores) "
            + "| commit=\(gitCommit)"
    }
}

