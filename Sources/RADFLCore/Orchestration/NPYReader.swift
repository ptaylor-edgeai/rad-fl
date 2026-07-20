// NPYReader.swift
//
// Reads NumPy's .npy binary format — the format `extract_cifar10_shards.py`
// (Python baseline) writes shards in: node_NN_train_X.npy, node_NN_train_y.npy,
// test_X.npy, test_y.npy. This is a from-scratch parser of the actual .npy
// spec (no NumPy available in Swift), built against the format as documented
// at https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
// and the original npy-format NEP — verified against official NumPy
// documentation rather than assumed from memory, since a subtly wrong byte
// offset or dtype-string parse here would silently produce garbage training
// data rather than a visible error.
//
// CONFIRMED SHARD FORMAT (read directly from extract_cifar10_shards.py and
// partition.py source, not assumed):
//   node_NN_train_X.npy / test_X.npy:  shape (N, 3, 32, 32), dtype float32,
//                                       values in [0.0, 1.0] (raw uint8 / 255.0,
//                                       no further mean/std standardization),
//                                       channel order matches CIFAR-10's native
//                                       on-disk order (R-plane, G-plane, B-plane,
//                                       each 32x32 row-major) — NOT reordered.
//   node_NN_train_y.npy / test_y.npy:  shape (N,), dtype int64
//
// .npy binary layout (version 1.0, what `numpy.save` writes for header sizes
// this small — this reader also handles version 2.0/3.0's 4-byte header
// length field, since relying on "it'll always be 1.0" would be a fragile
// assumption rather than a verified fact about every NumPy version that
// might produce these files):
//   [0:6]   magic string, exactly 0x93 'N' 'U' 'M' 'P' 'Y'
//   [6]     major version (UInt8)
//   [7]     minor version (UInt8)
//   v1.0:  [8:10]  HEADER_LEN as little-endian UInt16
//   v2.0+: [8:12]  HEADER_LEN as little-endian UInt32
//   header bytes: ASCII (v1.0) or UTF-8 (v2.0+) Python-dict-literal string,
//                 e.g. "{'descr': '<f4', 'fortran_order': False, 'shape': (50000, 3, 32, 32), }"
//                 newline-terminated, space-padded for alignment.
//   remaining bytes: raw array data, row-major (C order) unless
//                    fortran_order is True, in the byte order/type given by
//                    'descr'.

import Foundation

public enum NPYError: Error, CustomStringConvertible {
    case fileNotFound(URL)
    case readFailed(URL, any Error)
    case badMagic(found: [UInt8])
    case unsupportedVersion(major: UInt8, minor: UInt8)
    case truncatedHeader
    case headerParseFailed(String)
    case unsupportedDescr(String)
    case fortranOrderUnsupported
    case dataSizeMismatch(expected: Int, found: Int)
    case shapeMismatch(expected: [Int], found: [Int])

    public var description: String {
        switch self {
        case .fileNotFound(let url):
            return "NPY file not found at \(url.path)"
        case .readFailed(let url, let error):
            return "Failed to read NPY file at \(url.path): \(error)"
        case .badMagic(let found):
            return "Not a valid .npy file — bad magic bytes: \(found)"
        case .unsupportedVersion(let major, let minor):
            return "Unsupported .npy format version \(major).\(minor) — only 1.x, 2.x, 3.x are handled"
        case .truncatedHeader:
            return "NPY file is truncated — header data extends past end of file"
        case .headerParseFailed(let raw):
            return "Could not parse NPY header dict: \(raw)"
        case .unsupportedDescr(let descr):
            return "Unsupported NPY dtype descriptor '\(descr)' — this reader only handles '<f4' (float32) and '<i8' (int64), little-endian, matching what extract_cifar10_shards.py writes"
        case .fortranOrderUnsupported:
            return "NPY file has fortran_order=True — this reader only handles C-order (row-major) arrays, which is what numpy.save produces by default and what extract_cifar10_shards.py writes"
        case .dataSizeMismatch(let expected, let found):
            return "NPY data section size mismatch: header implies \(expected) bytes, file has \(found) remaining"
        case .shapeMismatch(let expected, let found):
            return "NPY array shape \(found) does not match expected shape \(expected)"
        }
    }
}

/// Parsed .npy header metadata, before the raw data bytes are interpreted.
public struct NPYHeader: Sendable {
    public let shape: [Int]
    public let descr: String       // e.g. "<f4", "<i8" — raw dtype string from the header dict
    public let fortranOrder: Bool
    public let dataOffset: Int     // byte offset in the file where raw array data begins

    public var elementCount: Int {
        shape.reduce(1, *)
    }
}

public enum NPYReader {

    // MARK: - Header parsing

    /// Parses just the .npy header (magic, version, shape, dtype) without
    /// reading the full data section — useful for validating a file's shape
    /// before committing to loading potentially-large data into memory.
    public static func readHeader(from url: URL) throws -> NPYHeader {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw NPYError.fileNotFound(url)
        }

        let data: Data
        do {
            // Reading the whole file here even though we only need the
            // header is wasteful for large shards, but Foundation's Data
            // doesn't make "read just the first N bytes, then seek" notably
            // simpler than this for a one-shot header peek, and shard files
            // here are at most ~tens of MB — not worth a streaming read path
            // for this use case. readData(forFullyParsing:) below is the one
            // that actually matters for memory, since that's the path used
            // for full training shards.
            data = try Data(contentsOf: url)
        } catch {
            throw NPYError.readFailed(url, error)
        }

        return try parseHeader(data)
    }

    /// Parses the header from an in-memory buffer that already contains (at
    /// least) the full header section. Shared by `readHeader` and
    /// `readArray` so header-parsing logic exists in exactly one place.
    private static func parseHeader(_ data: Data) throws -> NPYHeader {
        guard data.count >= 8 else { throw NPYError.truncatedHeader }

        let magic = [UInt8](data[0..<6])
        let expectedMagic: [UInt8] = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]  // \x93NUMPY
        guard magic == expectedMagic else {
            throw NPYError.badMagic(found: magic)
        }

        let major = data[6]
        let minor = data[7]
        guard (1...3).contains(major) else {
            throw NPYError.unsupportedVersion(major: major, minor: minor)
        }

        var offset = 8
        let headerLen: Int
        if major == 1 {
            // v1.0: 2-byte little-endian header length
            guard data.count >= offset + 2 else { throw NPYError.truncatedHeader }
            let lenBytes = data[offset..<offset+2]
            headerLen = Int(lenBytes[lenBytes.startIndex]) | (Int(lenBytes[lenBytes.startIndex + 1]) << 8)
            offset += 2
        } else {
            // v2.0/v3.0: 4-byte little-endian header length
            guard data.count >= offset + 4 else { throw NPYError.truncatedHeader }
            let lenBytes = data[offset..<offset+4]
            let b0 = lenBytes[lenBytes.startIndex]
            let b1 = lenBytes[lenBytes.startIndex + 1]
            let b2 = lenBytes[lenBytes.startIndex + 2]
            let b3 = lenBytes[lenBytes.startIndex + 3]
            headerLen = Int(b0) | (Int(b1) << 8) | (Int(b2) << 16) | (Int(b3) << 24)
            offset += 4
        }

        guard data.count >= offset + headerLen else { throw NPYError.truncatedHeader }
        let headerBytes = data[offset..<offset+headerLen]
        offset += headerLen

        guard let headerString = String(data: headerBytes, encoding: major == 1 ? .ascii : .utf8) else {
            throw NPYError.headerParseFailed("<non-decodable header bytes>")
        }

        let (shape, descr, fortranOrder) = try parseHeaderDict(headerString)

        return NPYHeader(shape: shape, descr: descr, fortranOrder: fortranOrder, dataOffset: offset)
    }

    /// Parses the header's Python-dict-literal string, e.g.:
    ///   "{'descr': '<f4', 'fortran_order': False, 'shape': (50000, 3, 32, 32), }"
    /// This is NOT a general Python literal parser — it specifically extracts
    /// the three keys numpy.save always writes ('descr', 'fortran_order',
    /// 'shape') via targeted string scanning, rather than attempting to
    /// handle arbitrary dict syntax. This is sufficient because numpy.save's
    /// own writer always produces exactly this shape of dict for the simple
    /// (non-structured) dtypes used here; it is NOT a substitute for a real
    /// Python literal parser and will not handle structured/record dtypes.
    private static func parseHeaderDict(_ raw: String) throws -> (shape: [Int], descr: String, fortranOrder: Bool) {
        // descr: '<f4'  (or "<f4" — numpy.save uses single quotes, but accept
        // either since this is a small, cheap regex either way and there's
        // no reason to be fragile about quote style specifically)
        guard let descr = firstCapturedGroup(in: raw, pattern: "'descr':\\s*'([^']+)'") else {
            throw NPYError.headerParseFailed(raw)
        }

        guard let fortranOrderStr = firstCapturedGroup(in: raw, pattern: "'fortran_order':\\s*(True|False)") else {
            throw NPYError.headerParseFailed(raw)
        }
        let fortranOrder = (fortranOrderStr == "True")

        guard let shapeStr = firstCapturedGroup(in: raw, pattern: "'shape':\\s*\\(([^)]*)\\)") else {
            throw NPYError.headerParseFailed(raw)
        }
        // shapeStr is like "50000, 3, 32, 32" or "50000," (1-D, note trailing
        // comma) or "" (0-D / scalar, not expected for our shard files but
        // handled rather than crashing on empty split results).
        let shape = shapeStr
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .compactMap { Int($0) }

        return (shape, descr, fortranOrder)
    }

    private static func firstCapturedGroup(in string: String, pattern: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(string.startIndex..., in: string)
        guard let match = regex.firstMatch(in: string, range: range), match.numberOfRanges > 1,
              let groupRange = Range(match.range(at: 1), in: string) else {
            return nil
        }
        return String(string[groupRange])
    }

    // MARK: - Data reading

    /// Controls how the array's data section is loaded.
    public enum LoadStrategy: Sendable {
        /// Request the OS map the file into virtual memory rather than read
        /// it eagerly — pages are faulted in from disk on demand as they're
        /// actually accessed, not all at once at open time. This mirrors
        /// Python's `np.load(path, mmap_mode="r")`, used in the Pi
        /// deployment path specifically for the 512MB RAM constraint (see
        /// trainer.py/_load_cifar10_arrays). Matching that here matters for
        /// a fair memory comparison between the two implementations — an
        /// eager Swift load against Python's mmapped load would make
        /// Swift's RSS numbers incomparable to Python's for reasons that
        /// have nothing to do with the runtimes themselves.
        ///
        /// NOTE: `.mappedIfSafe` is a HINT to Foundation, not a guarantee —
        /// per Apple's own documentation it maps "if possible and safe," and
        /// may silently fall back to a normal read otherwise. Code using
        /// this should not assume mapping definitely happened, only that it
        /// was requested.
        case mapped

        /// Read the full file into memory immediately. Simpler, and the
        /// only option that makes sense for small files (e.g. label arrays,
        /// which are tiny relative to image data) where mmap's per-page
        /// fault overhead isn't worth it.
        case eager
    }

    /// Opens a .npy file and returns its header plus a `NPYArrayData` handle
    /// for zero-copy typed access to the underlying bytes. Does NOT copy the
    /// array's data into a fresh `[Float32]`/`[Int64]` — see `NPYArrayData`
    /// for why that matters when `.mapped` is used.
    public static func open(
        _ url: URL,
        strategy: LoadStrategy,
        expectedDescr: String,
        expectedShape: [Int]? = nil
    ) throws -> (header: NPYHeader, arrayData: NPYArrayData) {
        let data: Data
        do {
            switch strategy {
            case .mapped:
                data = try Data(contentsOf: url, options: .mappedIfSafe)
            case .eager:
                data = try Data(contentsOf: url)
            }
        } catch {
            throw NPYError.readFailed(url, error)
        }

        let header = try parseHeader(data)
        guard !header.fortranOrder else {
            throw NPYError.fortranOrderUnsupported
        }
        if let expectedShape, expectedShape != header.shape {
            throw NPYError.shapeMismatch(expected: expectedShape, found: header.shape)
        }
        guard header.descr == expectedDescr else {
            throw NPYError.unsupportedDescr(header.descr)
        }

        // Validate the file actually contains enough bytes for what the
        // header claims, before handing back a handle to it — catches a
        // truncated/corrupt file immediately rather than letting a later
        // out-of-bounds read inside withFloat32Buffer/withInt64Buffer crash
        // with an unhelpful unsafe-pointer fault.
        let itemSize = itemSize(forDescr: header.descr)
        let expectedByteCount = header.elementCount * itemSize
        let actualByteCount = data.count - header.dataOffset
        guard actualByteCount == expectedByteCount else {
            throw NPYError.dataSizeMismatch(expected: expectedByteCount, found: actualByteCount)
        }

        return (header, NPYArrayData(fileData: data, dataOffset: header.dataOffset))
    }

    private static func itemSize(forDescr descr: String) -> Int {
        switch descr {
        case "<f4": return MemoryLayout<Float32>.size
        case "<i8": return MemoryLayout<Int64>.size
        default: return 1  // unreachable in practice — unsupportedDescr is thrown before this is called for any other descr
        }
    }

    /// Convenience that fully materializes a Float32 array — copies data out
    /// of the underlying buffer into a fresh `[Float32]`. Appropriate for
    /// small arrays (labels) or when the caller genuinely needs an owned,
    /// independent array rather than a view. For image data on a
    /// memory-constrained target, prefer `open(strategy: .mapped, ...)` and
    /// `NPYArrayData.withFloat32Buffer` instead, to avoid the copy.
    public static func readFloat32Array(from url: URL, expectedShape: [Int]? = nil) throws -> (shape: [Int], values: [Float32]) {
        let (header, arrayData) = try open(url, strategy: .eager, expectedDescr: "<f4", expectedShape: expectedShape)
        let values = arrayData.withFloat32Buffer { Array($0) }
        return (header.shape, values)
    }

    /// Convenience that fully materializes an Int64 array. See
    /// `readFloat32Array`'s doc comment — same eager-copy trade-off applies.
    public static func readInt64Array(from url: URL, expectedShape: [Int]? = nil) throws -> (shape: [Int], values: [Int64]) {
        let (header, arrayData) = try open(url, strategy: .eager, expectedDescr: "<i8", expectedShape: expectedShape)
        let values = arrayData.withInt64Buffer { Array($0) }
        return (header.shape, values)
    }
}

/// Zero-copy handle to a .npy file's array data section. Holds the
/// (possibly memory-mapped) `Data` for the WHOLE file — header bytes
/// included — and exposes the data region via `withUnsafeBytes`-style
/// closures that reinterpret bytes in place, without copying them into a
/// new `Array`.
///
/// This distinction matters specifically because of `LoadStrategy.mapped`:
/// if `Data` was created with `.mappedIfSafe`, its backing bytes are pages
/// from the OS's mmap of the file, faulted in on access. Calling
/// `Array(unsafeBufferPointer)` — as the original version of this reader
/// did — copies every byte into freshly-allocated heap memory, which forces
/// every page into RSS immediately and defeats the entire purpose of
/// mapping (matching Python's `mmap_mode="r"` RAM-saving behavior on the Pi
/// deployment path). Keeping access entirely behind `withUnsafeBytes`-style
/// closures means a consumer that only touches a few rows of a large shard
/// (e.g. one mini-batch at a time) only faults in the pages it actually
/// reads, the same property Python's mmap gets.
public struct NPYArrayData: Sendable {
    private let fileData: Data
    private let dataOffset: Int

    init(fileData: Data, dataOffset: Int) {
        self.fileData = fileData
        self.dataOffset = dataOffset
    }

    /// Provides read-only access to the array's data region reinterpreted as
    /// `[Float32]`, without copying. The buffer passed to `body` is only
    /// valid for the duration of the call — do not escape it; if the caller
    /// needs values beyond the closure's scope, copy what's needed inside
    /// `body` (e.g. `Array(buffer[range])` for a specific slice, not the
    /// whole buffer).
    public func withFloat32Buffer<T>(_ body: (UnsafeBufferPointer<Float32>) throws -> T) rethrows -> T {
        try fileData.withUnsafeBytes { raw in
            let dataRegion = UnsafeRawBufferPointer(rebasing: raw[dataOffset..<raw.count])
            // bindMemory(to:) called on an UnsafeRawBufferPointer (immutable)
            // already returns UnsafeBufferPointer<Float32> directly — no
            // further wrapping needed. The previous version wrapped this in
            // an extra `UnsafeBufferPointer(typed)` call, which resolved to
            // a DIFFERENT initializer overload (the one that takes an
            // UnsafeMutableBufferPointer as input, to produce an immutable
            // view of mutable storage) and failed to compile since `typed`
            // was already immutable, not mutable.
            let typed = dataRegion.bindMemory(to: Float32.self)
            return try body(typed)
        }
    }

    /// Same as `withFloat32Buffer`, for Int64 (label) data.
    public func withInt64Buffer<T>(_ body: (UnsafeBufferPointer<Int64>) throws -> T) rethrows -> T {
        try fileData.withUnsafeBytes { raw in
            let dataRegion = UnsafeRawBufferPointer(rebasing: raw[dataOffset..<raw.count])
            let typed = dataRegion.bindMemory(to: Int64.self)
            return try body(typed)
        }
    }
}

