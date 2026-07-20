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
}

