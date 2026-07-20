// GossipCodec.swift
//
// Encodes/decodes GossipMessage <-> NIOCore.ByteBuffer.
//
// All multi-byte integers and floats are written in host byte order. This is
// deliberate, not an oversight: every node in the cluster is the same
// architecture (aarch64, Pi Zero 2W), so byte-order normalization would be
// pure overhead with zero portability benefit for this deployment. If the
// cluster ever becomes heterogeneous-architecture, this is the place to add
// explicit little-endian framing.

import Foundation
import NIOCore

public enum GossipCodecError: Error, CustomStringConvertible {
    case badMagic(found: UInt32)
    case unsupportedVersion(found: UInt32)
    case unknownMessageType(raw: UInt32)
    case unknownEncoding(raw: UInt32)
    case truncatedFrame(expectedAtLeast: Int, got: Int)

    public var description: String {
        switch self {
        case .badMagic(let found):
            return "GossipCodec: bad magic 0x\(String(found, radix: 16)) — not a RAD-FL frame"
        case .unsupportedVersion(let found):
            return "GossipCodec: unsupported protocol version \(found)"
        case .unknownMessageType(let raw):
            return "GossipCodec: unknown message type \(raw)"
        case .unknownEncoding(let raw):
            return "GossipCodec: unknown tensor encoding \(raw)"
        case .truncatedFrame(let expected, let got):
            return "GossipCodec: truncated frame, expected at least \(expected) bytes, buffer has \(got)"
        }
    }
}

public enum GossipCodec {

    // MARK: - Encode

    /// Encodes a GossipMessage into a fresh ByteBuffer (header + all tensor payloads,
    /// NOT including the outer 4-byte length prefix — that's added by the NIO pipeline's
    /// length-field framing, see GossipChannelHandlers.swift).
    public static func encode(_ message: GossipMessage, allocator: ByteBufferAllocator) -> ByteBuffer {
        var buffer = allocator.buffer(capacity: estimatedSize(of: message))

        buffer.writeInteger(GossipMessage.magic, endianness: .host)
        buffer.writeInteger(GossipMessage.protocolVersion, endianness: .host)
        buffer.writeInteger(message.messageType.rawValue, endianness: .host)
        buffer.writeInteger(message.senderNodeID, endianness: .host)
        buffer.writeInteger(message.round, endianness: .host)
        buffer.writeInteger(message.sampleCount, endianness: .host)
        buffer.writeInteger(UInt32(message.tensors.count), endianness: .host)

        for tensor in message.tensors {
            buffer.writeInteger(tensor.tensorID, endianness: .host)
            buffer.writeInteger(tensor.encoding.rawValue, endianness: .host)
            buffer.writeInteger(UInt32(tensor.values.count), endianness: .host)

            // Dense payload: flat Float32 array, written via withUnsafeBufferPointer
            // to avoid a per-element function-call write — matters at model scale
            // (hundreds of thousands of params) on a CPU this constrained.
            tensor.values.withUnsafeBufferPointer { ptr in
                let byteCount = ptr.count * MemoryLayout<Float32>.size
                buffer.writeBytes(UnsafeRawBufferPointer(start: ptr.baseAddress, count: byteCount))
            }
        }

        return buffer
    }

    private static func estimatedSize(of message: GossipMessage) -> Int {
        let headerSize = 7 * MemoryLayout<UInt32>.size  // magic, version, messageType, senderNodeID, round, sampleCount, tensor count
        let tensorOverhead = message.tensors.count * (3 * MemoryLayout<UInt32>.size)
        let payloadSize = message.tensors.reduce(0) { $0 + $1.values.count * MemoryLayout<Float32>.size }
        return headerSize + tensorOverhead + payloadSize
    }

    // MARK: - Decode

    /// Decodes a complete application frame from `buffer`. Assumes the outer
    /// length-prefix has already been stripped by the NIO decoder and that
    /// `buffer` contains exactly one full message.
    public static func decode(_ buffer: inout ByteBuffer) throws -> GossipMessage {
        // 7 fields now, not 6: magic, version, messageType, senderNodeID,
        // round, sampleCount, tensorCount — sampleCount is a new field
        // (see GossipMessage.swift's doc comment for why), and this
        // headerSize constant is the MINIMUM-bytes-required check used by
        // every truncation guard below, not just the first one — getting
        // it wrong here would make every one of those guards check against
        // a stale, too-small minimum.
        let headerSize = 7 * MemoryLayout<UInt32>.size
        guard buffer.readableBytes >= headerSize else {
            throw GossipCodecError.truncatedFrame(expectedAtLeast: headerSize, got: buffer.readableBytes)
        }

        guard let magic: UInt32 = buffer.readInteger(endianness: .host) else {
            throw GossipCodecError.truncatedFrame(expectedAtLeast: headerSize, got: buffer.readableBytes)
        }
        guard magic == GossipMessage.magic else {
            throw GossipCodecError.badMagic(found: magic)
        }
        guard let version: UInt32 = buffer.readInteger(endianness: .host) else {
            throw GossipCodecError.truncatedFrame(expectedAtLeast: headerSize, got: buffer.readableBytes)
        }
        guard version == GossipMessage.protocolVersion else {
            throw GossipCodecError.unsupportedVersion(found: version)
        }
        guard let rawType: UInt32 = buffer.readInteger(endianness: .host) else {
            throw GossipCodecError.truncatedFrame(expectedAtLeast: headerSize, got: buffer.readableBytes)
        }
        guard let messageType = GossipMessageType(rawValue: rawType) else {
            throw GossipCodecError.unknownMessageType(raw: rawType)
        }
        guard let senderNodeID: UInt32 = buffer.readInteger(endianness: .host) else {
            throw GossipCodecError.truncatedFrame(expectedAtLeast: headerSize, got: buffer.readableBytes)
        }
        guard let round: UInt32 = buffer.readInteger(endianness: .host) else {
            throw GossipCodecError.truncatedFrame(expectedAtLeast: headerSize, got: buffer.readableBytes)
        }
        guard let sampleCount: UInt32 = buffer.readInteger(endianness: .host) else {
            throw GossipCodecError.truncatedFrame(expectedAtLeast: headerSize, got: buffer.readableBytes)
        }
        guard let tensorCount: UInt32 = buffer.readInteger(endianness: .host) else {
            throw GossipCodecError.truncatedFrame(expectedAtLeast: headerSize, got: buffer.readableBytes)
        }

        var tensors: [GossipTensor] = []
        tensors.reserveCapacity(Int(tensorCount))

        for _ in 0..<tensorCount {
            guard let tensorID: UInt32 = buffer.readInteger(endianness: .host),
                  let rawEncoding: UInt32 = buffer.readInteger(endianness: .host),
                  let encoding = GossipEncoding(rawValue: rawEncoding),
                  let elementCount: UInt32 = buffer.readInteger(endianness: .host) else {
                throw GossipCodecError.truncatedFrame(expectedAtLeast: 12, got: buffer.readableBytes)
            }

            switch encoding {
            case .dense:
                let byteCount = Int(elementCount) * MemoryLayout<Float32>.size
                guard buffer.readableBytes >= byteCount else {
                    throw GossipCodecError.truncatedFrame(expectedAtLeast: byteCount, got: buffer.readableBytes)
                }
                let values: [Float32] = buffer.readBytes(length: byteCount)!.withUnsafeBytes { raw in
                    Array(raw.bindMemory(to: Float32.self))
                }
                tensors.append(GossipTensor(tensorID: tensorID, values: values))

            case .sparseCOO:
                // Reserved for future work — not yet implemented on the wire.
                throw GossipCodecError.unknownEncoding(raw: rawEncoding)
            }
        }

        return GossipMessage(messageType: messageType, senderNodeID: senderNodeID, round: round, sampleCount: sampleCount, tensors: tensors)
    }
}

