// ResourceUsage.swift
//
// Wraps POSIX getrusage(RUSAGE_SELF) to measure CPU time and peak resident
// set size — the data effective_cores and peak_rss_mb need, neither of
// which existed anywhere in this project before this file. Mirrors
// Python's trainer.py functions exactly:
//   _cpu_snapshot()  -> (user_seconds, system_seconds) cumulative since
//                        process start, across all cores combined
//   _peak_rss_mb()   -> high-water-mark RSS in MB, with the documented
//                        macOS-reports-bytes vs Linux-reports-KB unit
//                        difference handled explicitly (Python's own
//                        comment: "a real ~400-500MB process showed up as
//                        ~400,000-500,000 'MB' before this fix" — the
//                        exact same platform split applies here, since
//                        this project explicitly develops/smoke-tests on
//                        Mac but targets Pi/Linux for real experiments).
//
// IMPORTANT CAVEAT, stated plainly rather than glossed over: this file
// could not be compiled or run against a real Swift toolchain while being
// written (no Swift compiler available in the authoring environment) —
// every other file change this session was at least syntax/brace-balance
// checked, but the EXACT Swift-bridged numeric types of struct rusage's
// fields (e.g. whether ru_maxrss bridges as Int, Int64, or something
// platform-specific; whether timeval's tv_sec/tv_usec are Int or a typedef
// like __time_t) could not be verified directly. The C-level struct layout
// itself (ru_utime/ru_stime as timeval, ru_maxrss as a platform-sized
// integer, RUSAGE_SELF as the `who` argument) is well-documented and
// stable POSIX/Linux-manual-page material, not in question — only the
// precise Swift bridging of those C types is unverified. Every numeric
// access below goes through an explicit Int()/Double() conversion at the
// point of use specifically so that IF a bridged type differs from what's
// assumed here, the fix is a single line at that conversion, not a
// cascading change through this file's logic.

#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif
import Foundation

public enum ResourceUsage {
    // RUSAGE_SELF's bridged Swift type genuinely differs by platform —
    // confirmed by REAL build results on both, not guessed:
    //   - Darwin (Mac): bare `RUSAGE_SELF` compiled and ran successfully
    //     as-is, passed directly to getrusage — so on Darwin it already
    //     matches getrusage's expected parameter type.
    //   - Glibc (Linux/Pi): bare `RUSAGE_SELF` failed with "cannot convert
    //     value of type '__rusage_who' to expected argument type
    //     '__rusage_who_t' (aka 'Int32')" — confirming Glibc imports it as
    //     a RawRepresentable-conforming enum, needing `.rawValue` to reach
    //     the actual Int32 getrusage wants.
    // Branching by platform here, using what's ACTUALLY CONFIRMED to work
    // for each, rather than trying to find one expression that happens to
    // satisfy both — an earlier attempt at a single shared expression
    // risked breaking the already-confirmed-working Mac build while fixing
    // Linux, which would have been a worse outcome than the original bug.
    #if canImport(Glibc)
    private static let rusageSelf: Int32 = RUSAGE_SELF.rawValue
    #else
    private static let rusageSelf = RUSAGE_SELF
    #endif

    /// (user CPU seconds, system CPU seconds) consumed by this process so
    /// far, cumulative since process start, summed across however many
    /// cores were actually busy — NOT wall-clock time. Taking a snapshot
    /// before and after some operation and summing the deltas gives total
    /// CPU-seconds spent in that operation across all cores, which is
    /// exactly what effective_cores divides by wall-clock to get a
    /// multi-core-utilization figure. Matches Python's _cpu_snapshot()
    /// exactly in purpose and the (user, system) -> sum convention.
    public static func cpuSnapshot() -> (userSeconds: Double, systemSeconds: Double) {
        var usage = rusage()
        let result = getrusage(Self.rusageSelf, &usage)
        guard result == 0 else {
            // getrusage failing for RUSAGE_SELF would be highly unusual
            // (no documented failure mode for this specific call short of
            // a corrupt argument, which can't happen here) — returning
            // zero rather than crashing, since a metrics-collection
            // function misbehaving shouldn't take down a training run.
            return (0, 0)
        }
        let userSeconds = Double(usage.ru_utime.tv_sec) + Double(usage.ru_utime.tv_usec) / 1_000_000.0
        let systemSeconds = Double(usage.ru_stime.tv_sec) + Double(usage.ru_stime.tv_usec) / 1_000_000.0
        return (userSeconds, systemSeconds)
    }

    /// Peak resident set size (high-water mark) in MB for the current
    /// process, for the process's entire lifetime so far (NOT a per-call
    /// snapshot — ru_maxrss is monotonically increasing). Handles the
    /// documented platform unit difference exactly like Python's
    /// _peak_rss_mb(): macOS reports ru_maxrss in BYTES, Linux reports it
    /// in KB. Getting this wrong silently produces numbers wrong by a
    /// factor of 1024 — Python's own comment confirms this was a real,
    /// previously-hit bug on their side, not a theoretical concern.
    public static func peakRSSMB() -> Double {
        var usage = rusage()
        let result = getrusage(Self.rusageSelf, &usage)
        guard result == 0 else { return 0 }

        let raw = Double(usage.ru_maxrss)
        #if os(macOS)
        return raw / (1024.0 * 1024.0)   // bytes -> MB
        #else
        return raw / 1024.0              // KB -> MB (Linux/Pi)
        #endif
    }

    // ── CPU topology reporting ──────────────────────────────────────────────
    //
    // These exist because of a specific hazard in the heterogeneity-profile
    // design. Device classes are simulated by restricting nodes to 1, 2 or 4
    // cores via cgroup cpuset. But libdispatch on Linux sizes its worker pool
    // from the machine's ONLINE core count, which takes no account of a cgroup
    // cpuset restriction. If DispatchQueue.concurrentPerform follows that
    // number, a node pinned to one CPU still spawns four workers, which then
    // timeslice on that single core.
    //
    // That is NOT the same workload as a genuine single-core device. Four
    // workers on one core add context-switch cost, and — worse for this project
    // specifically — each worker's tile evicts the others' from a 32KB L1 that
    // Tensor.swift's 8x8 GEMM tiling was sized against. The observable signature
    // is a SUPERLINEAR slowdown at one core rather than the expected ~4x, which
    // would look like a device-class effect while actually being an artefact of
    // the measurement apparatus.
    //
    // ── Why procfs/sysfs rather than sched_getaffinity ──────────────────────
    //
    // The obvious implementation is sched_getaffinity(2) plus CPU_COUNT. Both
    // are unavailable here, for separate reasons:
    //
    //   - CPU_COUNT / CPU_ISSET are C MACROS. Macros do not bridge into Swift
    //     at all, so the bit counting would have to be done by hand over
    //     cpu_set_t's raw bytes regardless.
    //   - sched_getaffinity itself is a GNU extension, declared in <sched.h>
    //     only when _GNU_SOURCE is defined. The Debian Trixie aarch64 SDK's
    //     Glibc modulemap does not expose it, so it fails to resolve with
    //     "cannot find 'sched_getaffinity' in scope" when cross-compiling for
    //     the Pi — CONFIRMED by a real build failure, not anticipated.
    //
    // The workarounds would be a C shim target compiled with _GNU_SOURCE, or
    // syscall(2) directly (also unavailable — variadic C functions don't
    // import). Both add a build-system dependency to work around a modulemap
    // gap, in service of a diagnostic.
    //
    // Reading /proc and /sys instead is pure Foundation with zero C interop, so
    // it cannot break on a toolchain or SDK change, and it happens to be
    // strictly more informative: the kernel reports the actual CPU LIST, not
    // just a count. It also reflects taskset and cgroup cpuset restrictions
    // identically, which is exactly the equivalence being tested.

    /// Parses a Linux CPU list ("0-3", "0,2", "0-1,3", "0") into a count.
    /// Returns -1 if the string cannot be parsed, so a malformed read is
    /// distinguishable from a genuine zero.
    static func parseCPUList(_ raw: String) -> Int {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return -1 }

        var count = 0
        for part in trimmed.split(separator: ",") {
            let piece = part.trimmingCharacters(in: .whitespaces)
            if piece.isEmpty { continue }

            if let dash = piece.firstIndex(of: "-") {
                let loStr = String(piece[piece.startIndex..<dash])
                let hiStr = String(piece[piece.index(after: dash)...])
                guard let lo = Int(loStr), let hi = Int(hiStr), hi >= lo else {
                    return -1
                }
                count += hi - lo + 1
            } else {
                guard Int(piece) != nil else { return -1 }
                count += 1
            }
        }
        return count
    }

    private static func readFirstLine(_ path: String) -> String? {
        guard let data = FileManager.default.contents(atPath: path),
              let text = String(data: data, encoding: .utf8) else { return nil }
        return text.split(separator: "\n", omittingEmptySubsequences: false).first
            .map(String.init)
    }

    /// Cores this process is actually permitted to run on, read from
    /// /proc/self/status's `Cpus_allowed_list`. Reflects BOTH `taskset` and
    /// cgroup cpuset restriction — which is the point, since the profiles use
    /// cgroups and the quick diagnostic uses taskset, and they need to agree.
    ///
    /// Returns -1 where /proc is unavailable (Darwin), which is expected on the
    /// Mac and not a failure.
    public static func affinityCoreCount() -> Int {
        guard let data = FileManager.default.contents(atPath: "/proc/self/status"),
              let text = String(data: data, encoding: .utf8) else { return -1 }

        for line in text.split(separator: "\n") {
            guard line.hasPrefix("Cpus_allowed_list:") else { continue }
            let value = line.dropFirst("Cpus_allowed_list:".count)
            return parseCPUList(String(value))
        }
        return -1
    }

    /// Cores the OS reports as online, from /sys/devices/system/cpu/online.
    /// This is the number that IGNORES cgroup restriction — the suspected
    /// source of libdispatch's pool sizing, and therefore the one that should
    /// stay at 4 even on a node pinned to a single CPU.
    ///
    /// Falls back to ProcessInfo where sysfs is unavailable. sysconf(3) is
    /// deliberately not used: _SC_NPROCESSORS_ONLN's Swift bridging varies by
    /// platform in the same way RUSAGE_SELF's did (see the top of this file),
    /// and this diagnostic is not worth another cross-compile failure.
    public static func onlineProcessorCount() -> Int {
        if let line = readFirstLine("/sys/devices/system/cpu/online") {
            let n = parseCPUList(line)
            if n > 0 { return n }
        }
        return ProcessInfo.processInfo.activeProcessorCount
    }

    /// What Foundation reports, and therefore what an unconfigured
    /// `concurrentPerform` is most likely to follow.
    public static func activeProcessorCount() -> Int {
        return ProcessInfo.processInfo.activeProcessorCount
    }

    /// All three counts together, for the `probe-cpu` diagnostic.
    ///
    /// Interpretation under a restriction such as `taskset -c 0` or
    /// `systemd-run --scope -p AllowedCPUs=0`:
    ///
    ///   affinity == 1 and active == 1
    ///       Foundation follows the process's CPU mask. Restriction works as-is
    ///       and an explicit --compute-threads flag is hardening, not a fix.
    ///
    ///   affinity == 1 but active == 4
    ///       Foundation follows the online count and ignores the cpuset. Every
    ///       concurrentPerform call site in Tensor.swift must be given an
    ///       explicit width before any heterogeneity profile is trusted.
    ///
    ///   affinity == -1
    ///       No /proc — expected on the Mac, not a failure.
    ///
    /// Worth running under BOTH taskset and cgroup restriction: they are
    /// different mechanisms and there is no guarantee a runtime honours both.
    public static func cpuTopologyReport() -> (affinity: Int, online: Int, active: Int) {
        return (affinityCoreCount(), onlineProcessorCount(), activeProcessorCount())
    }
}


