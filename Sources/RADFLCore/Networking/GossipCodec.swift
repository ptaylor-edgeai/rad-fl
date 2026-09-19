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
//
// COMPRESSION LIVES HERE AND NOWHERE ELSE. Every encoding decodes back to a
// dense `[Float32]`, so `GossipAggregator`, `SimpleCNN` and the round loop are
// untouched by RQ1's mechanisms and require no change when one is added. A
// compressed run and a dense run differ in exactly one field on the wire, which
// is what makes the comparison attributable to the mechanism rather than to
// anything else that moved with it.

import Foundation
import NIOCore

public enum GossipCodecError: Error, CustomStringConvertible {
    case badMagic(found: UInt32)
    case unsupportedVersion(found: UInt32)
    case unknownMessageType(raw: UInt32)
    case unknownEncoding(raw: UInt32)
    case truncatedFrame(expectedAtLeast: Int, got: Int)
    case malformedSparse(nonzeroCount: Int, denseCount: Int)

    public var description: String {
        switch self {
        case .badMagic(let found):
            return "GossipCodec: bad magic 0x\(String(found, radix: 16)) — not a RAD-FL frame"
        case .unsupportedVersion(let found):
            return "GossipCodec: unsupported protocol version \(found)"
        case .unknownMessageType(let raw):
            return "GossipCodec: unknown message type \(raw)"
        case .unknownEncoding(let raw):
            return "GossipCodec: unknown tensor encoding \(raw) — a peer is sending "
                 + "a compression mechanism this binary does not implement. Check that "
                 + "every node is running the same build."
        case .truncatedFrame(let expected, let got):
            return "GossipCodec: truncated frame, expected at least \(expected) bytes, buffer has \(got)"
        case .malformedSparse(let nnz, let dense):
            return "GossipCodec: sparse tensor claims \(nnz) nonzeros in a \(dense)-element "
                 + "tensor — more nonzeros than elements is impossible, so the frame is corrupt "
                 + "or the sender's encoder is wrong"
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
            // ALWAYS the dense element count, for every encoding — the receiver
            // needs it to size the reconstructed array. Sparse carries its own
            // nonzero count in the payload.
            buffer.writeInteger(UInt32(tensor.values.count), endianness: .host)

            switch tensor.encoding {
            case .dense:
                // Flat Float32, written via withUnsafeBufferPointer to avoid a
                // per-element function-call write — matters at model scale
                // (hundreds of thousands of params) on a CPU this constrained.
                tensor.values.withUnsafeBufferPointer { ptr in
                    let byteCount = ptr.count * MemoryLayout<Float32>.size
                    buffer.writeBytes(UnsafeRawBufferPointer(start: ptr.baseAddress, count: byteCount))
                }

            case .denseFP16:
                let bits = TensorCodec.toFP16(tensor.values)
                bits.withUnsafeBufferPointer { ptr in
                    let byteCount = ptr.count * MemoryLayout<UInt16>.size
                    buffer.writeBytes(UnsafeRawBufferPointer(start: ptr.baseAddress, count: byteCount))
                }

            case .denseINT8:
                let (quantised, scale) = TensorCodec.toINT8(tensor.values)
                // Scale and offset precede the payload so the decoder can read
                // them without knowing the element count first.
                buffer.writeInteger(scale.min.bitPattern, endianness: .host)
                buffer.writeInteger(scale.scale.bitPattern, endianness: .host)
                buffer.writeBytes(quantised)

            case .sparseCOO:
                // Indices are derived from the nonzeros of the dense array. The
                // caller has already zeroed whatever it chose to drop — see
                // GossipTensor's doc comment for why the in-memory shape stays
                // dense regardless of encoding.
                var indices: [UInt32] = []
                var values: [Float32] = []
                for (i, v) in tensor.values.enumerated() where v != 0 {
                    indices.append(UInt32(i))
                    values.append(v)
                }
                buffer.writeInteger(UInt32(indices.count), endianness: .host)
                indices.withUnsafeBufferPointer { ptr in
                    buffer.writeBytes(UnsafeRawBufferPointer(
                        start: ptr.baseAddress, count: ptr.count * MemoryLayout<UInt32>.size))
                }
                values.withUnsafeBufferPointer { ptr in
                    buffer.writeBytes(UnsafeRawBufferPointer(
                        start: ptr.baseAddress, count: ptr.count * MemoryLayout<Float32>.size))
                }
            }
        }

        return buffer
    }

    private static func estimatedSize(of message: GossipMessage) -> Int {
        let headerSize = 7 * MemoryLayout<UInt32>.size  // magic, version, messageType, senderNodeID, round, sampleCount, tensor count
        let tensorOverhead = message.tensors.count * (3 * MemoryLayout<UInt32>.size)
        // Sized for the dense case regardless of encoding: every other encoding
        // is smaller, so this over-allocates rather than forcing a reallocation
        // mid-write. Sparse is the exception — above 50% density it exceeds
        // dense — but a mechanism sending sparse at that density has chosen
        // wrongly, and a reallocation is the least of that problem.
        let payloadSize = message.tensors.reduce(0) { $0 + $1.values.count * MemoryLayout<Float32>.size }
        return headerSize + tensorOverhead + payloadSize
    }

    // MARK: - Decode

    /// Decodes a complete application frame from `buffer`. Assumes the outer
    /// length-prefix has already been stripped by the NIO decoder and that
    /// `buffer` contains exactly one full message.
    ///
    /// Every encoding reconstructs to a dense `[Float32]`, so callers never see
    /// a compressed tensor.
    public static func decode(_ buffer: inout ByteBuffer) throws -> GossipMessage {
        // 7 fields: magic, version, messageType, senderNodeID, round,
        // sampleCount, tensorCount. This headerSize constant is the
        // MINIMUM-bytes-required check used by every truncation guard below,
        // not just the first one — getting it wrong here would make every one
        // of those guards check against a stale, too-small minimum.
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
                  let elementCount: UInt32 = buffer.readInteger(endianness: .host) else {
                throw GossipCodecError.truncatedFrame(expectedAtLeast: 12, got: buffer.readableBytes)
            }
            guard let encoding = GossipEncoding(rawValue: rawEncoding) else {
                throw GossipCodecError.unknownEncoding(raw: rawEncoding)
            }

            let dense = Int(elementCount)
            let values: [Float32]

            switch encoding {
            case .dense:
                let byteCount = dense * MemoryLayout<Float32>.size
                guard buffer.readableBytes >= byteCount else {
                    throw GossipCodecError.truncatedFrame(expectedAtLeast: byteCount, got: buffer.readableBytes)
                }
                values = buffer.readBytes(length: byteCount)!.withUnsafeBytes { raw in
                    Array(raw.bindMemory(to: Float32.self))
                }

            case .denseFP16:
                let byteCount = dense * MemoryLayout<UInt16>.size
                guard buffer.readableBytes >= byteCount else {
                    throw GossipCodecError.truncatedFrame(expectedAtLeast: byteCount, got: buffer.readableBytes)
                }
                let bits: [UInt16] = buffer.readBytes(length: byteCount)!.withUnsafeBytes { raw in
                    Array(raw.bindMemory(to: UInt16.self))
                }
                values = TensorCodec.fromFP16(bits)

            case .denseINT8:
                let metaBytes = 2 * MemoryLayout<Float32>.size
                guard buffer.readableBytes >= metaBytes + dense else {
                    throw GossipCodecError.truncatedFrame(
                        expectedAtLeast: metaBytes + dense, got: buffer.readableBytes)
                }
                guard let minBits: UInt32 = buffer.readInteger(endianness: .host),
                      let scaleBits: UInt32 = buffer.readInteger(endianness: .host),
                      let quantised = buffer.readBytes(length: dense) else {
                    throw GossipCodecError.truncatedFrame(
                        expectedAtLeast: metaBytes + dense, got: buffer.readableBytes)
                }
                let scale = TensorCodec.AffineScale(min: Float32(bitPattern: minBits),
                                                    scale: Float32(bitPattern: scaleBits))
                values = TensorCodec.fromINT8(quantised, scale)

            case .sparseCOO:
                guard let nnz32: UInt32 = buffer.readInteger(endianness: .host) else {
                    throw GossipCodecError.truncatedFrame(
                        expectedAtLeast: MemoryLayout<UInt32>.size, got: buffer.readableBytes)
                }
                let nnz = Int(nnz32)
                // Checked before allocating: a corrupt nnz larger than the dense
                // count would otherwise size an array from an untrusted field.
                guard nnz <= dense else {
                    throw GossipCodecError.malformedSparse(nonzeroCount: nnz, denseCount: dense)
                }
                let idxBytes = nnz * MemoryLayout<UInt32>.size
                let valBytes = nnz * MemoryLayout<Float32>.size
                guard buffer.readableBytes >= idxBytes + valBytes else {
                    throw GossipCodecError.truncatedFrame(
                        expectedAtLeast: idxBytes + valBytes, got: buffer.readableBytes)
                }
                let indices: [UInt32] = buffer.readBytes(length: idxBytes)!.withUnsafeBytes { raw in
                    Array(raw.bindMemory(to: UInt32.self))
                }
                let sparseValues: [Float32] = buffer.readBytes(length: valBytes)!.withUnsafeBytes { raw in
                    Array(raw.bindMemory(to: Float32.self))
                }
                values = TensorCodec.fromSparse(TensorCodec.Sparse(
                    denseCount: dense, indices: indices, values: sparseValues))
            }

            // Reconstructed tensors are always dense in memory. The encoding is
            // carried through so a receiver can report what it was sent — the
            // payload-byte accounting RQ1 needs cannot be recovered from the
            // dense array alone.
            tensors.append(GossipTensor(tensorID: tensorID, values: values, encoding: encoding))
        }

        return GossipMessage(messageType: messageType, senderNodeID: senderNodeID, round: round, sampleCount: sampleCount, tensors: tensors)
    }
}

