// NPYWriter.swift
//
// Writes NumPy's .npy binary format (v1.0) — the write-side counterpart to
// NPYReader. Built specifically so Swift can save a trained model's
// parameters in a format Swift itself (via NPYReader) can load back later
// to seed/resume a future experiment — see project decision: individual
// .npy files per tensor, NOT a real .npz (zip) container, since the actual
// need is "Swift loads Swift's own saved weights," and NPYReader already
// exists, is already verified, and reads exactly this format. A real .npz
// writer (implementing the zip format) was considered and explicitly
// deferred — only worth the real new work if Python needed to load Swift's
// saved weights directly, which isn't the current requirement.
//
// FORMAT: mirrors NPYReader's own documented spec exactly (see that file's
// header comment) — v1.0 layout: 6-byte magic, 2 version bytes, 2-byte
// little-endian header length, ASCII Python-dict-literal header
// (newline-terminated, space-padded so total header length is divisible by
// 16), then raw little-endian data. Writing to this exact spec, rather than
// inventing a simplified one, means files this writer produces are real
// .npy files — readable by NPYReader (round-trip within this project) AND
// by actual NumPy (e.g. for manual inspection via `np.load` on the Mac, or
// future cross-checking against Python), not a Swift-only lookalike format.

import Foundation

public enum NPYWriterError: Error, CustomStringConvertible {
    case writeFailed(URL, any Error)

    public var description: String {
        switch self {
        case .writeFailed(let url, let error):
            return "NPYWriter: failed to write \(url.path): \(error)"
        }
    }
}

public enum NPYWriter {
    /// Writes `values` as a .npy file with the given `shape` — dtype is
    /// always '<f4' (little-endian float32), matching every float array
    /// this project produces/consumes elsewhere (CIFAR-10 shards, model
    /// parameters). `values.count` must equal the product of `shape`.
    public static func writeFloat32Array(_ values: [Float32], shape: [Int], to url: URL) throws {
        precondition(values.count == shape.reduce(1, *), "NPYWriter: values.count (\(values.count)) != product of shape \(shape)")
        let data = encode(values: values, shape: shape, descr: "<f4")
        try write(data, to: url)
    }

    /// Writes `values` as a .npy file — dtype '<i8' (little-endian int64),
    /// matching the label-array convention used elsewhere in this project.
    public static func writeInt64Array(_ values: [Int64], shape: [Int], to url: URL) throws {
        precondition(values.count == shape.reduce(1, *), "NPYWriter: values.count (\(values.count)) != product of shape \(shape)")
        let data = encode(values: values, shape: shape, descr: "<i8")
        try write(data, to: url)
    }

    private static func write(_ data: Data, to url: URL) throws {
        do {
            try data.write(to: url, options: .atomic)
        } catch {
            throw NPYWriterError.writeFailed(url, error)
        }
    }

    /// Builds the complete .npy file bytes (header + data) for a Float32 array.
    private static func encode(values: [Float32], shape: [Int], descr: String) -> Data {
        var data = Data()
        data.append(header(shape: shape, descr: descr))
        values.withUnsafeBufferPointer { buffer in
            data.append(Data(buffer: buffer))
        }
        return data
    }

    /// Builds the complete .npy file bytes (header + data) for an Int64 array.
    private static func encode(values: [Int64], shape: [Int], descr: String) -> Data {
        var data = Data()
        data.append(header(shape: shape, descr: descr))
        values.withUnsafeBufferPointer { buffer in
            data.append(Data(buffer: buffer))
        }
        return data
    }

    /// Builds just the header bytes (magic + version + length field +
    /// padded dict string) — shared by both encode(values:) overloads,
    /// since the header format doesn't depend on the element type beyond
    /// the `descr` string already passed in.
    private static func header(shape: [Int], descr: String) -> Data {
        // Python tuple literal: "(50000, 3, 32, 32)" for multi-dim,
        // "(3,)" for 1-D (note required trailing comma — this is valid
        // Python tuple syntax and is exactly what NPYReader's
        // shape-parsing regex already expects and correctly handles, per
        // that file's own confirmed-against-real-NumPy-output testing).
        let shapeStr: String
        if shape.count == 1 {
            shapeStr = "(\(shape[0]),)"
        } else {
            shapeStr = "(" + shape.map(String.init).joined(separator: ", ") + ")"
        }

        let dictBody = "{'descr': '\(descr)', 'fortran_order': False, 'shape': \(shapeStr), }"

        // Pad with spaces (and a final newline) so that
        // magic(6) + version(2) + lenfield(2) + header(dictBody + padding + "\n")
        // is divisible by 16 — the v1.0 alignment rule (see NPYReader's
        // header comment / the original npy-format spec). NPYReader itself
        // doesn't actually enforce this alignment on read (it's a
        // write-side performance convention, not something a reader must
        // validate), but writing it correctly keeps these files genuinely
        // standard .npy files, not just "happens to work with our reader."
        let fixedPrefixLength = 6 + 2 + 2  // magic + version + length field
        let unpaddedLength = dictBody.count + 1  // +1 for the trailing newline
        let totalBeforePadding = fixedPrefixLength + unpaddedLength
        let remainder = totalBeforePadding % 16
        let paddingLength = remainder == 0 ? 0 : (16 - remainder)
        let paddedDictBody = dictBody + String(repeating: " ", count: paddingLength) + "\n"

        var data = Data()
        data.append(contentsOf: [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59])  // \x93NUMPY
        data.append(contentsOf: [0x01, 0x00])  // version 1.0

        let headerLen = UInt16(paddedDictBody.utf8.count)
        data.append(UInt8(headerLen & 0xFF))
        data.append(UInt8((headerLen >> 8) & 0xFF))

        data.append(paddedDictBody.data(using: .ascii)!)

        return data
    }
}
