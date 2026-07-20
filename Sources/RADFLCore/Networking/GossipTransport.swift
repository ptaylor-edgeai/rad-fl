// GossipTransport.swift
//
// Server/client bootstrap wrapping the frame codec + timeout handling into a
// simple async API for the orchestration layer. One GossipServer per node
// (listens for incoming peer pushes); one GossipClient connection per outbound
// peer push. Connections are not persistently held open across rounds in this
// first implementation — each push dials, sends, closes. This trades a little
// per-round TCP handshake cost for much simpler peer-failure semantics (a dead
// peer just fails to connect/times out, no stale-connection bookkeeping).
// Revisit if handshake overhead shows up materially in round_total_s.

import Foundation
import NIOCore
import NIOPosix

public struct GossipNodeAddress: Sendable, Hashable {
    public let host: String
    public let port: Int

    public init(host: String, port: Int) {
        self.host = host
        self.port = port
    }
}

public enum GossipTransportError: Error {
    case connectTimeout(GossipNodeAddress)
    case peerTimedOut(GossipNodeAddress)
}

/// Receives incoming gossip pushes from peers. Hand it a closure that gets
/// called for each fully-decoded GossipMessage.
public final class GossipServer {
    private let group: EventLoopGroup
    private var channel: Channel?
    private let port: Int
    private let peerTimeout: TimeAmount
    private let onMessage: @Sendable (GossipMessage, GossipNodeAddress) -> Void

    public init(
        port: Int,
        group: EventLoopGroup,
        peerTimeout: TimeAmount = .seconds(30),
        onMessage: @escaping @Sendable (GossipMessage, GossipNodeAddress) -> Void
    ) {
        self.port = port
        self.group = group
        self.peerTimeout = peerTimeout
        self.onMessage = onMessage
    }

    public func start() async throws {
        let bootstrap = ServerBootstrap(group: group)
            .serverChannelOption(ChannelOptions.backlog, value: 256)
            .childChannelInitializer { [onMessage, peerTimeout] channel in
                // Using `makeCompletedFuture { ... }` + `pipeline.syncOperations.addHandler`
                // rather than the plain `async` `pipeline.addHandlers([...])` API.
                // `ByteToMessageHandler<GossipFrameDecoder>` (and several other NIO
                // handler types) are deliberately marked
                // `@available(*, unavailable) extension ...: Sendable` upstream — this
                // is swift-nio's own established pattern (see e.g. LineBasedFrameDecoder,
                // HTTPServerProtocolErrorHandler in swift-nio/swift-nio-extras), not a
                // bug in our types. No `@unchecked Sendable` on our own decoder fixes
                // this, because the unavailable conformance lives on NIO's wrapper type,
                // not ours. `syncOperations.addHandler`, called synchronously from
                // inside a closure that already runs on the channel's own event loop,
                // sidesteps the Sendable-across-async-boundary requirement entirely —
                // this mirrors NIO's own example code (NIOChatServer, NIOTCPEchoClient).
                let remoteAddr = channel.remoteAddress
                let peerAddress = GossipNodeAddress(
                    host: remoteAddr?.ipAddress ?? "unknown",
                    port: remoteAddr?.port ?? 0
                )
                return channel.eventLoop.makeCompletedFuture {
                    try channel.pipeline.syncOperations.addHandler(ByteToMessageHandler(GossipFrameDecoder()))
                    try channel.pipeline.syncOperations.addHandler(
                        GossipPeerTimeoutHandler(timeout: peerTimeout) { ctx in
                            ctx.eventLoop.execute {
                                // Connection-level timeout. Round-level dead-peer
                                // decisions belong to the orchestration loop, which
                                // observes "no message received for round N" rather
                                // than relying solely on TCP-level liveness.
                            }
                        }
                    )
                    try channel.pipeline.syncOperations.addHandler(
                        GossipMessageSink(peerAddress: peerAddress, onMessage: onMessage)
                    )
                }
            }
            .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)

        self.channel = try await bootstrap.bind(host: "0.0.0.0", port: port).get()
    }

    public func shutdown() async throws {
        try await channel?.close()
    }
}

/// Terminal handler that just forwards decoded messages to the server's callback.
/// `@unchecked Sendable`: all stored properties are immutable `let`s, and
/// `onMessage` is itself constrained to `@Sendable`, so this is safe to share
/// across NIO's event-loop threads.
final class GossipMessageSink: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = GossipMessage

    private let peerAddress: GossipNodeAddress
    private let onMessage: @Sendable (GossipMessage, GossipNodeAddress) -> Void

    init(peerAddress: GossipNodeAddress, onMessage: @escaping @Sendable (GossipMessage, GossipNodeAddress) -> Void) {
        self.peerAddress = peerAddress
        self.onMessage = onMessage
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        let message = unwrapInboundIn(data)
        onMessage(message, peerAddress)
    }
}

/// Sends a single GossipMessage to one peer. Connect-per-push (see file header
/// for rationale). Throws on connect failure or timeout — caller (orchestration
/// loop) decides how to treat that peer for this round.
public struct GossipClient {
    private let group: EventLoopGroup
    private let connectTimeout: TimeAmount

    public init(group: EventLoopGroup, connectTimeout: TimeAmount = .seconds(10)) {
        self.group = group
        self.connectTimeout = connectTimeout
    }

    public func push(_ message: GossipMessage, to address: GossipNodeAddress) async throws {
        let bootstrap = ClientBootstrap(group: group)
            .connectTimeout(connectTimeout)
            .channelInitializer { channel in
                channel.pipeline.addHandler(GossipFrameEncoder())
            }

        let channel = try await bootstrap.connect(host: address.host, port: address.port).get()

        do {
            try await channel.writeAndFlush(message)
            try await channel.close()
        } catch {
            try? await channel.close()
            throw error
        }
    }

    /// Bare TCP connect-and-close, no message sent. Separate from `push`
    /// rather than reusing it with an empty message, because:
    ///   1. A reachability probe wants a short timeout regardless of what
    ///      `connectTimeout` is configured to for real pushes (which may
    ///      legitimately need to be longer for a loaded/slow real Pi node).
    ///   2. Conflating "TCP connect failed" with "TCP connected but the
    ///      write failed" would make failure reporting less precise — this
    ///      method can only fail at the connect step, so a thrown error here
    ///      unambiguously means "couldn't reach this host:port at all."
    /// This is a SINGLE attempt — no retry. For the common case of peers
    /// that may not be started yet (e.g. starting 10 terminals one at a
    /// time), use `waitUntilReachable` instead.
    public func checkReachable(_ address: GossipNodeAddress, timeout: TimeAmount = .seconds(3)) async throws {
        let bootstrap = ClientBootstrap(group: group)
            .connectTimeout(timeout)

        let channel = try await bootstrap.connect(host: address.host, port: address.port).get()
        try await channel.close()
    }

    /// Polls `checkReachable` continuously until it succeeds or `deadline`
    /// passes. Unlike a fixed attempt-count retry, this is meant for the
    /// realistic case of "keep trying until this peer comes up, however
    /// long that takes, up to some sane outer ceiling" — the same thing a
    /// real gossip round needs to do before pushing to a peer, not just a
    /// one-off connectivity smoke test. Throws the most recent failure if
    /// `deadline` passes without success.
    public func waitUntilReachable(
        _ address: GossipNodeAddress,
        deadline: ContinuousClock.Instant,
        pollInterval: TimeAmount = .seconds(2),
        perAttemptTimeout: TimeAmount = .seconds(3),
        onAttemptFailure: (@Sendable (GossipNodeAddress, any Error) -> Void)? = nil
    ) async throws {
        let clock = ContinuousClock()
        var lastError: Error?
        while clock.now < deadline {
            do {
                try await checkReachable(address, timeout: perAttemptTimeout)
                return
            } catch {
                lastError = error
                onAttemptFailure?(address, error)
                let remaining = deadline - clock.now
                guard remaining > .zero else { break }
                let sleepDuration = min(Duration.nanoseconds(pollInterval.nanoseconds), remaining)
                guard sleepDuration > .zero else { break }
                try await Task.sleep(for: sleepDuration)
            }
        }
        throw lastError ?? GossipTransportError.connectTimeout(address)
    }
}

/// Per-peer reachability state, used by `waitForAllReachable`'s
/// `onStatusChange` callback to report transitions as they happen, rather
/// than only a final pass/fail summary at the end.
public enum PeerReachability: Sendable, Equatable {
    case pending          // not yet successfully reached
    case reachable        // successfully reached
    case timedOut         // overall deadline passed without ever reaching it
}

/// Polls every address in `addresses` concurrently until ALL of them are
/// reachable or the overall `timeout` elapses, calling `onStatusChange`
/// exactly once per address the moment its state changes (pending ->
/// reachable, or pending -> timedOut at the deadline). This is the shared
/// primitive behind `test-connectivity`'s live status display AND what a
/// real gossip round needs before pushing — "wait for my peers to be ready,
/// show progress as they come up, give up after a sane ceiling" is the same
/// operation in both cases.
///
/// `onHeartbeat`, if provided, fires roughly every `heartbeatInterval`
/// regardless of whether anything has changed — reporting (reachableCount,
/// totalCount). This exists because `onStatusChange` alone can leave a
/// caller looking at total silence for the entire `timeout` if peers simply
/// haven't been started yet (a real, common case — see project history):
/// there's no way to distinguish "genuinely stuck" from "correctly still
/// waiting on peers you haven't started" without some periodic signal that
/// the loop is alive and what it's currently waiting on.
///
/// Returns the final reachability state for every address. Does NOT throw
/// on partial failure — a caller doing a real gossip round needs to know
/// exactly which peers timed out and proceed with whoever's left (matching
/// the project's existing dead-peer-handling philosophy: a stalled/missing
/// peer shouldn't block forever), not have the whole operation abort.
public func waitForAllReachable(
    client: GossipClient,
    addresses: [GossipNodeAddress],
    timeout: TimeAmount,
    pollInterval: TimeAmount = .seconds(2),
    perAttemptTimeout: TimeAmount = .seconds(3),
    heartbeatInterval: TimeAmount = .seconds(15),
    onStatusChange: @escaping @Sendable (GossipNodeAddress, PeerReachability) -> Void,
    onHeartbeat: (@Sendable (_ reachableCount: Int, _ totalCount: Int) -> Void)? = nil,
    onAttemptFailure: (@Sendable (GossipNodeAddress, any Error) -> Void)? = nil
) async -> [GossipNodeAddress: PeerReachability] {
    let deadline = ContinuousClock().now + Duration.nanoseconds(timeout.nanoseconds)
    let totalCount = addresses.count
    let progress = ReachableCounter()

    return await withTaskGroup(of: (GossipNodeAddress, PeerReachability)?.self) { taskGroup in
        for address in addresses {
            taskGroup.addTask {
                do {
                    try await client.waitUntilReachable(
                        address,
                        deadline: deadline,
                        pollInterval: pollInterval,
                        perAttemptTimeout: perAttemptTimeout,
                        onAttemptFailure: onAttemptFailure
                    )
                    await progress.markReachable()
                    onStatusChange(address, .reachable)
                    return (address, .reachable)
                } catch {
                    onStatusChange(address, .timedOut)
                    return (address, .timedOut)
                }
            }
        }

        // Heartbeat task: not a real peer result (hence the `nil` return,
        // filtered out below) — periodically reports progress so the
        // caller isn't staring at silence while correctly waiting on peers
        // that simply haven't been started yet (mirrors Python's
        // wait_for_peers(), which logs "Still waiting for: ..." on every
        // poll_interval retry — see gossip.py). Reads the shared `progress`
        // actor (updated by the real per-peer tasks above) rather than
        // tracking its own counter, so the numbers it reports are accurate.
        //
        // Checks for completion on a TIGHT cadence (`pollInterval`, default
        // 2s — matching Python's wait_for_peers default poll_interval
        // exactly), but only PRINTS on the slower `heartbeatInterval`
        // cadence. These are deliberately decoupled: an earlier version
        // checked AND printed on the same (slow, default 15s)
        // heartbeatInterval cadence, which meant the function could take
        // up to a full heartbeatInterval to notice all peers were already
        // reachable and return — a real, measurable startup-latency
        // mismatch against Python's tighter loop, not just a cosmetic
        // logging difference. Checking every `pollInterval` instead means
        // this returns roughly as promptly as Python's equivalent wait
        // does, while still only printing status at the more
        // human-readable `heartbeatInterval` cadence.
        if let onHeartbeat {
            taskGroup.addTask {
                let clock = ContinuousClock()
                var lastPrint = clock.now
                while clock.now < deadline {
                    let sleepDuration = min(Duration.nanoseconds(pollInterval.nanoseconds), deadline - clock.now)
                    guard sleepDuration > .zero else { break }
                    try? await Task.sleep(for: sleepDuration)

                    let count = await progress.count
                    if count >= totalCount {
                        onHeartbeat(count, totalCount)
                        break
                    }
                    if clock.now - lastPrint >= Duration.nanoseconds(heartbeatInterval.nanoseconds) {
                        onHeartbeat(count, totalCount)
                        lastPrint = clock.now
                    }
                }
                return nil
            }
        }

        var results: [GossipNodeAddress: PeerReachability] = [:]
        for await result in taskGroup {
            guard let (address, status) = result else { continue }
            results[address] = status
        }
        return results
    }
}

/// Thread-safe counter shared between `waitForAllReachable`'s per-peer
/// tasks (which increment it on success) and its heartbeat task (which
/// reads it) — an actor rather than a plain `var` specifically because
/// multiple per-peer tasks can complete concurrently from different
/// threads, and a heartbeat reporting a stale or racy count would defeat
/// the entire point of the heartbeat (accurate "here's where things stand"
/// progress).
private actor ReachableCounter {
    private(set) var count = 0

    func markReachable() {
        count += 1
    }
}

/// Current liveness of a peer under continuous monitoring. Unlike
/// `PeerReachability` (used by the one-shot `waitForAllReachable`, which has
/// a genuine terminal "gave up" state once its deadline passes),
/// `monitorReachability` never gives up — a peer is always either currently
/// `.up` or currently `.down`, and can transition between the two
/// indefinitely as processes start and stop.
public enum PeerLiveness: Sendable, Equatable {
    case up
    case down
}

/// Continuously monitors every address in `addresses`, forever, calling
/// `onStatusChange` each time a peer's liveness actually flips (up -> down
/// or down -> up) — including DOWN transitions after a peer was previously
/// up. This is the real difference from `waitForAllReachable`: that function
/// treats "reachable" as a one-shot terminal state and stops checking a peer
/// once reached, so it can never notice a peer going away again later. This
/// function never stops checking anyone, for exactly that reason — built
/// after observing that a peer process exiting mid-run needs to show as gone,
/// not stay marked reachable forever from a single past success.
///
/// Has no overall deadline and does not return on its own — it runs until
/// the calling `Task` is cancelled (e.g. the process receives Ctrl-C/SIGINT
/// and the enclosing Task tree gets cancelled, or a caller explicitly stores
/// and cancels the `Task` running this). `withTaskGroup`'s child tasks check
/// `Task.isCancelled` each loop iteration and exit promptly rather than
/// blocking a cancellation request.
public func monitorReachability(
    client: GossipClient,
    addresses: [GossipNodeAddress],
    pollInterval: TimeAmount = .seconds(2),
    perAttemptTimeout: TimeAmount = .seconds(3),
    heartbeatInterval: TimeAmount = .seconds(15),
    onStatusChange: @escaping @Sendable (GossipNodeAddress, PeerLiveness) -> Void,
    onHeartbeat: (@Sendable (_ upCount: Int, _ totalCount: Int) -> Void)? = nil
) async {
    let totalCount = addresses.count
    let liveness = LivenessTracker(addresses: addresses)

    await withTaskGroup(of: Void.self) { taskGroup in
        for address in addresses {
            taskGroup.addTask {
                while !Task.isCancelled {
                    let isUp: Bool
                    do {
                        try await client.checkReachable(address, timeout: perAttemptTimeout)
                        isUp = true
                    } catch {
                        isUp = false
                    }

                    let changed = await liveness.update(address, isUp: isUp)
                    if changed {
                        onStatusChange(address, isUp ? .up : .down)
                    }

                    // Sleep regardless of up/down — this is a continuous
                    // monitor, not a "stop once reached" wait. Checking a
                    // peer that's currently up costs about the same as
                    // checking one that's down, and is the only way to
                    // notice it later going away.
                    if Task.isCancelled { break }
                    try? await Task.sleep(for: .nanoseconds(pollInterval.nanoseconds))
                }
            }
        }

        if let onHeartbeat {
            taskGroup.addTask {
                while !Task.isCancelled {
                    try? await Task.sleep(for: .nanoseconds(heartbeatInterval.nanoseconds))
                    if Task.isCancelled { break }
                    let upCount = await liveness.upCount
                    onHeartbeat(upCount, totalCount)
                }
            }
        }

        await taskGroup.waitForAll()
    }
}

/// Tracks current up/down state per address for `monitorReachability`,
/// reporting whether a given update actually represents a CHANGE (so the
/// caller only gets `onStatusChange` calls on real transitions, not a call
/// every single poll cycle regardless of whether anything changed).
private actor LivenessTracker {
    private var state: [GossipNodeAddress: Bool]

    init(addresses: [GossipNodeAddress]) {
        // Start every address as "down" (not yet confirmed up) rather than
        // some unknown/optional third state — simpler, and the first
        // successful check for any address will correctly report it as a
        // change to .up, same as the old one-shot design's first transition.
        self.state = Dictionary(uniqueKeysWithValues: addresses.map { ($0, false) })
    }

    /// Returns true if this update represents an actual change from the
    /// previously recorded state.
    func update(_ address: GossipNodeAddress, isUp: Bool) -> Bool {
        let previous = state[address] ?? false
        state[address] = isUp
        return previous != isUp
    }

    var upCount: Int {
        state.values.filter { $0 }.count
    }
}

