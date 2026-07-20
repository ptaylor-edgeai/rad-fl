// GossipChannelHandlers.swift
//
// NIO pipeline: raw TCP bytes <-> length-prefixed frames <-> GossipMessage.
//
// Framing: 4-byte little-endian length prefix (NIOCore's LengthFieldBasedFrameDecoder
// convention) followed by the GossipCodec-encoded payload.
//
// Dead-peer handling: alpha_0.1 experiments on the Python baseline showed gossip
// aggregation stalling indefinitely when a peer OOM-crashes mid-round (no timeout,
// no dead-peer detection — see project notes). The Swift implementation builds
// this in from the start via:
//   1. A per-peer read timeout (IdleStateHandler-style) that fires if no data
//      arrives within `peerTimeout`.
//   2. A round-level deadline at the orchestration layer (see Orchestration/)
//      that proceeds with whatever peer updates *did* arrive once the deadline
//      passes, rather than waiting forever for a crashed node.
// This file implements (1); (2) is the orchestration loop's responsibility and
// is out of scope for the transport layer itself.

import Foundation
import NIOCore
import NIOPosix
import NIOConcurrencyHelpers

/// Encodes outbound GossipMessages with a 4-byte length prefix.
///
/// `@unchecked Sendable`: this class has no stored mutable state (no `var`
/// properties at all), so it's genuinely safe to share across NIO's
/// event-loop threads. Note this is unrelated to the separate
/// `ByteToMessageHandler<GossipFrameDecoder>` Sendable warning seen on the
/// decoder side — that one comes from NIO's own wrapper type, not from
/// anything fixable via annotating our own decoder/encoder (see the comment
/// on GossipFrameDecoder below, and GossipTransport.swift's
/// `childChannelInitializer` for the actual fix to that issue).
public final class GossipFrameEncoder: ChannelOutboundHandler, @unchecked Sendable {
    public typealias OutboundIn = GossipMessage
    public typealias OutboundOut = ByteBuffer

    public init() {}

    public func write(context: ChannelHandlerContext, data: NIOAny, promise: EventLoopPromise<Void>?) {
        let message = unwrapOutboundIn(data)
        let payload = GossipCodec.encode(message, allocator: context.channel.allocator)

        var framed = context.channel.allocator.buffer(capacity: payload.readableBytes + 4)
        framed.writeInteger(UInt32(payload.readableBytes), endianness: .little)
        var payloadCopy = payload
        framed.writeBuffer(&payloadCopy)

        context.write(wrapOutboundOut(framed), promise: promise)
    }
}

/// Decodes inbound length-prefixed frames into GossipMessages.
/// Buffers partial frames across multiple `channelRead` calls — TCP gives no
/// guarantee a frame arrives in one read.
///
/// `@unchecked Sendable`: `maxFrameSize` is an immutable `let`, and this class
/// has no other stored state, so it's genuinely safe to share across NIO's
/// event-loop threads.
///
/// IMPORTANT: this conformance does NOT fix the
/// "Conformance of 'ByteToMessageHandler<Decoder>' to 'Sendable' is
/// unavailable" warning seen when this type is wrapped in
/// `ByteToMessageHandler(GossipFrameDecoder())`. That warning comes from
/// `ByteToMessageHandler` itself — swift-nio deliberately marks several of
/// its own handler wrapper types `@available(*, unavailable) extension ...:
/// Sendable {}` (see e.g. `LineBasedFrameDecoder`,
/// `HTTPServerProtocolErrorHandler` in the swift-nio/swift-nio-extras source)
/// specifically so users can't paper over it with an external `@unchecked
/// Sendable`. No annotation on `GossipFrameDecoder` can change that. The
/// actual fix is in GossipTransport.swift's `childChannelInitializer`, which
/// adds handlers via `channel.eventLoop.makeCompletedFuture { ... }` +
/// `pipeline.syncOperations.addHandler(...)` (matching NIO's own example
/// code) instead of the plain `async` `pipeline.addHandlers([...])`, which is
/// what actually required the wrapper to be `Sendable` in the first place.
public final class GossipFrameDecoder: ByteToMessageDecoder, @unchecked Sendable {
    public typealias InboundOut = GossipMessage

    private let maxFrameSize: Int

    public init(maxFrameSize: Int = 64 * 1024 * 1024) {
        // 64 MiB ceiling — generous for a CNN's full weight set even uncompressed;
        // exists primarily to reject corrupt/malicious length fields rather than
        // to constrain legitimate traffic.
        self.maxFrameSize = maxFrameSize
    }

    public func decode(context: ChannelHandlerContext, buffer: inout ByteBuffer) throws -> DecodingState {
        guard buffer.readableBytes >= 4 else { return .needMoreData }

        let lengthFieldSize = 4
        guard let length: UInt32 = buffer.getInteger(at: buffer.readerIndex, endianness: .little) else {
            return .needMoreData
        }

        guard Int(length) <= maxFrameSize else {
            throw GossipCodecError.truncatedFrame(expectedAtLeast: Int(length), got: maxFrameSize)
        }

        let totalFrameSize = lengthFieldSize + Int(length)
        guard buffer.readableBytes >= totalFrameSize else { return .needMoreData }

        buffer.moveReaderIndex(forwardBy: lengthFieldSize)
        var payload = buffer.readSlice(length: Int(length))!

        let message = try GossipCodec.decode(&payload)
        context.fireChannelRead(wrapInboundOut(message))

        return .continue
    }

    public func decodeLast(context: ChannelHandlerContext, buffer: inout ByteBuffer, seenEOF: Bool) throws -> DecodingState {
        try decode(context: context, buffer: &buffer)
    }
}

/// Tracks per-connection liveness. Fires `onPeerTimeout` if no readable data
/// arrives within `timeout` of the last successful read. This is connection-level
/// liveness, distinct from the round-level "did this peer's gossip update arrive
/// in time" decision made by the orchestration loop — a peer can be TCP-alive
/// but still miss a round deadline under load, which this handler alone won't catch.
///
/// CONCURRENCY: this class has mutable state (`lastReadTime`, `timeoutTask`)
/// that is NOT independently thread-safe on its own — it relies on NIO's
/// guarantee that all of a channel's handlers execute on that channel's
/// single EventLoop thread, so these properties are only ever read/written
/// from that one thread. This is a standard, correct NIO pattern.
/// `@unchecked Sendable` asserts that safety explicitly, since the compiler
/// can't verify "only touched from one specific EventLoop thread" on its own
/// — it's a real guarantee from NIO's execution model, not a skipped check.
public final class GossipPeerTimeoutHandler: ChannelInboundHandler, @unchecked Sendable {
    public typealias InboundIn = GossipMessage
    public typealias InboundOut = GossipMessage

    private let timeout: TimeAmount
    private let onPeerTimeout: @Sendable (ChannelHandlerContext) -> Void
    private var lastReadTime: NIODeadline
    private var timeoutTask: Scheduled<Void>?

    public init(timeout: TimeAmount, onPeerTimeout: @escaping @Sendable (ChannelHandlerContext) -> Void) {
        self.timeout = timeout
        self.onPeerTimeout = onPeerTimeout
        self.lastReadTime = .now()
    }

    public func channelActive(context: ChannelHandlerContext) {
        scheduleTimeoutCheck(context: context)
        context.fireChannelActive()
    }

    public func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        lastReadTime = .now()
        context.fireChannelRead(data)
    }

    public func channelInactive(context: ChannelHandlerContext) {
        timeoutTask?.cancel()
        context.fireChannelInactive()
    }

    private func scheduleTimeoutCheck(context: ChannelHandlerContext) {
        timeoutTask = context.eventLoop.scheduleTask(in: timeout) { [weak self] in
            guard let self else { return }
            let elapsed = NIODeadline.now() - self.lastReadTime
            if elapsed >= self.timeout {
                self.onPeerTimeout(context)
                context.close(promise: nil)
            } else {
                self.scheduleTimeoutCheck(context: context)
            }
        }
    }
}
