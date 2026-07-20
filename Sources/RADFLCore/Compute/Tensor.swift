// Tensor.swift
// Pure-Swift CNN math — parallelised across all available cores.
//
// Parallelism strategy
// ────────────────────
// Every major operation (conv, pool, FC) loops over N samples. We split that
// N-dimension into `coreCount` contiguous chunks and run each chunk via
// DispatchQueue.concurrentPerform, which maps directly onto all available CPU
// cores with no thread-creation overhead per call.
//
// Results are written into non-overlapping slices of pre-allocated output
// arrays — each chunk owns its rows exclusively, so there are no data races.
//
// For backward passes dW and db require reduction across N: we compute
// per-chunk partial sums stored in a flat [Float] with stride layout, then
// reduce on the calling thread (same pattern as threaded GEMM).
//
// Swift 6 Sendable fix
// ────────────────────
// UnsafeBufferPointer / UnsafeMutableBufferPointer are NOT Sendable, so they
// cannot be captured in the @Sendable closure that concurrentPerform requires.
// Solution: capture the baseAddress as UInt (a plain integer — Sendable), then
// rebind to the typed pointer inside the closure. This is safe because:
//   • the pointed-to storage outlives the concurrentPerform call (it is kept
//     alive by the enclosing withUnsafe*BufferPointer scope), and
//   • each closure iteration writes only to its own non-overlapping slice.
//
// The same trick applies to the flattened partial-gradient buffers: we allocate
// a single [Float] of size nc × stride, take its address once before the
// concurrentPerform, and each chunk writes to offset ci × stride — no nested
// withUnsafeMutableBufferPointer captures inside the closure.
//
// On the Pi Zero 2 W (4 × Cortex-A53) this saturates all 4 cores.
// On macOS it uses GCD's thread pool sized to the physical core count.
//
// Dependencies: Foundation only (DispatchQueue, ProcessInfo).
// No Accelerate, no BLAS, no external packages.
//
// Visibility scheme
// ──────────────────
// Three tiers, deliberate rather than default:
//   • `public`  — the layer-level API (fcForward/fcBackward, convForward/
//     convBackward, maxPool2Forward/Backward, activations, loss functions,
//     CachedMatrix) — this is what a training loop or another module
//     actually builds against.
//   • internal (no modifier) — the low-level math primitives (transpose,
//     dotSIMD4, matmulT, im2col, col2im). These have sharp edges (no
//     bounds checking, implicit layout contracts between caller and
//     callee) that the public layer functions paper over by construction.
//     Kept internal rather than public deliberately: there's no second
//     consumer yet to design a stable public contract against, and once
//     something is public, walking back a signature change later is a
//     breaking change. internal (not private) specifically so they remain
//     callable from other files within this same module if this grows
//     beyond one file.
//   • `private` — coreCount, chunks, addr: pure implementation detail of
//     the parallelism trick used internally by this file. No reason for
//     even same-module code to ever call these directly.

import Foundation

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#elseif canImport(Musl)
import Musl
#endif

// MARK: - Helpers

private var coreCount: Int { ProcessInfo.processInfo.activeProcessorCount }

private func chunks(n: Int, cores: Int) -> [Range<Int>] {
    let c = Swift.min(cores, n)
    let base = n / c, rem = n % c
    var start = 0
    return (0..<c).map { i in
        let len = base + (i < rem ? 1 : 0)
        defer { start += len }
        return start..<(start + len)
    }
}

// Capture a typed pointer as plain UInt so it crosses the @Sendable boundary,
// then rebind with UnsafePointer<T>(bitPattern:)! inside the closure.
@inline(__always)
private func addr<T>(_ p: UnsafePointer<T>) -> UInt { UInt(bitPattern: p) }
@inline(__always)
private func addr<T>(_ p: UnsafeMutablePointer<T>) -> UInt { UInt(bitPattern: p) }

// MARK: - Weighted sum across multiple same-shaped arrays
//
// Added specifically for GossipAggregator's sample-weighted peer-averaging
// step, which was previously implemented as nested scalar Swift loops (no
// SIMD, no parallelism) — a real gap, since the paper's entire claim rests
// on Swift compute efficiency and this sat directly in the per-round
// critical path. NOT a case of "routing through an existing primitive" —
// nothing in this file did element-wise weighted accumulation across N
// same-shaped vectors before this; matmulT/convForward/etc. are all
// matrix-multiply or convolution shaped, a genuinely different access
// pattern from "sum weight[k] * values[k][i] across k peers, for each i."
//
// Parallelised over ELEMENT INDEX, not over peer — there are typically few
// peers (≤10 for this project's full-mesh topology) but potentially many
// thousands of elements per tensor, so chunking by element gives far
// better core utilization than chunking by peer would.
//
// SIMD4 shape here is deliberately different from dotSIMD4's: dotSIMD4
// accumulates along ONE long vector (a dot product contraction); this
// accumulates ACROSS several separate peer arrays at the SAME few element
// positions, then advances to the next 4 positions. Different access
// pattern, so copying dotSIMD4's structure verbatim wouldn't fit.
func weightedSum(arrays: [[Float]], weights: [Float]) -> [Float] {
    precondition(arrays.count == weights.count, "weightedSum: arrays.count (\(arrays.count)) != weights.count (\(weights.count))")
    guard let elementCount = arrays.first?.count else { return [] }
    precondition(arrays.allSatisfy { $0.count == elementCount }, "weightedSum: all arrays must have the same element count")

    var result = [Float](repeating: 0, count: elementCount)
    let k = arrays.count
    let slices = chunks(n: elementCount, cores: coreCount)

    // Build a flat array of base addresses (one per peer array) — captured
    // as plain UInt (Sendable) per this file's established pattern, then
    // rebound to typed pointers inside the closure. The withUnsafeBufferPointer
    // scopes for every peer array must all stay open for the duration of
    // concurrentPerform, hence the nested-closure recursion below rather
    // than a flat loop of separate calls (each `withUnsafeBufferPointer`
    // call's pointer is only valid inside its own closure).
    func withAllPointers<R>(_ remaining: [[Float]], collected: [UInt], _ body: ([UInt]) -> R) -> R {
        guard let first = remaining.first else { return body(collected) }
        return first.withUnsafeBufferPointer { buf in
            withAllPointers(Array(remaining.dropFirst()), collected: collected + [addr(buf.baseAddress!)], body)
        }
    }

    result.withUnsafeMutableBufferPointer { resultBuf in
        let resultAddr = addr(resultBuf.baseAddress!)
        withAllPointers(arrays, collected: []) { peerAddrs in
            DispatchQueue.concurrentPerform(iterations: slices.count) { sliceIdx in
                let resultP = UnsafeMutablePointer<Float>(bitPattern: resultAddr)!
                let peerPs: [UnsafePointer<Float>] = peerAddrs.map { UnsafePointer<Float>(bitPattern: $0)! }
                let range = slices[sliceIdx]

                var i = range.lowerBound
                // SIMD4 over 4 element positions at a time, accumulating
                // across all k peer arrays at each group of 4 positions.
                while i + 3 < range.upperBound {
                    var acc = SIMD4<Float>.zero
                    for peer in 0..<k {
                        let w = weights[peer]
                        let p = peerPs[peer] + i
                        acc += SIMD4(p[0], p[1], p[2], p[3]) * SIMD4(repeating: w)
                    }
                    resultP[i]   = acc.x
                    resultP[i+1] = acc.y
                    resultP[i+2] = acc.z
                    resultP[i+3] = acc.w
                    i += 4
                }
                // Scalar tail for whatever doesn't divide evenly by 4.
                while i < range.upperBound {
                    var sum: Float = 0
                    for peer in 0..<k {
                        sum += peerPs[peer][i] * weights[peer]
                    }
                    resultP[i] = sum
                    i += 1
                }
            }
        }
    }

    return result
}

// MARK: - CachedMatrix
//
// Wraps a weight matrix together with a lazily-computed, auto-invalidating
// cached transpose. Added specifically because `fcForward` was calling
// `transpose(W, ...)` on every single forward pass, even though W only
// actually changes once per optimizer step — every other forward call
// between updates was paying a full O(rows×cols) copy for an identical
// result. This wrapper makes that free: the transpose is computed once,
// reused until `values` is mutated, then recomputed lazily on next access.
//
// Only used for WEIGHTS (data that persists across many forward calls
// between gradient updates) — NOT for per-batch tensors like x/dOut/col,
// which are different every call and so have nothing to cache; those keep
// using the plain `transpose(...)` free function directly. Caching a
// transpose of data that's never reused would just be a more complicated
// way of doing the same single copy.
//
// CLASS, not struct — deliberately, per Swift's own struct-vs-class
// guidance: this needs REFERENCE semantics, not value semantics. A weight
// matrix is shared, mutable, identity-bearing state: the optimizer updates
// it, the forward pass reads it, the backward pass reads it, and all of
// them must see the SAME cache-invalidation state. If this were a struct,
// every function receiving it would get its own independent copy with its
// own independent cache — an update made via one copy would not invalidate
// the cache another copy is still holding, which is exactly the
// stale-gradient bug class this type exists to prevent, just relocated
// rather than solved. This is precisely Apple's stated case for choosing a
// class over a struct (shared mutable identity across multiple owners),
// not a stylistic deviation from the rest of this file's value-type usage.
public final class CachedMatrix {
    public private(set) var values: [Float]
    public let rows: Int
    public let cols: Int

    private var cachedTranspose: [Float]?

    public init(values: [Float], rows: Int, cols: Int) {
        precondition(values.count == rows * cols, "CachedMatrix: values.count (\(values.count)) != rows*cols (\(rows * cols))")
        self.values = values
        self.rows = rows
        self.cols = cols
        self.cachedTranspose = nil
    }

    /// Replaces the matrix's values (e.g. after an optimizer step) and
    /// invalidates the cached transpose — this is the ONLY way `values`
    /// can change, specifically so invalidation can never be forgotten by a
    /// caller. A bare `[Float]` with a separately-maintained cache would
    /// require every call site that mutates weights to remember to also
    /// invalidate the cache; missing one would silently produce a
    /// stale-gradient bug that's much worse than the performance problem
    /// this type exists to fix. Making mutation and invalidation the same
    /// operation removes that failure mode entirely.
    public func update(_ newValues: [Float]) {
        precondition(newValues.count == values.count, "CachedMatrix.update: newValues.count (\(newValues.count)) != existing count (\(values.count))")
        values = newValues
        cachedTranspose = nil
    }

    /// Returns the transpose (cols × rows), computing it on first access
    /// after init or after the most recent `update(_:)` call, and reusing
    /// the cached result on every subsequent access until the next update.
    /// Not `mutating` (this is a class, not a struct) — callers don't need
    /// `inout`/`var` plumbing just to read a cached value, which was the
    /// other concrete cost of the struct-based version this replaced.
    public func transposed() -> [Float] {
        if let cachedTranspose {
            return cachedTranspose
        }
        let t = transpose(values, rows: rows, cols: cols)
        cachedTranspose = t
        return t
    }
}

// MARK: - Transpose  A(rows×cols) → Aᵀ(cols×rows)
//
// Tiled so that both the read and write streams stay in L1 cache.
// Tile size 64×64 = 16KB of floats — fits comfortably in the A53's 32KB L1.
// Paid once per matmul call; eliminates strided column access in the inner loop.
//
// Parallelised over row-tiles, same chunks()/concurrentPerform pattern as
// matmulT's M-tile parallelisation just above — each row-tile (spanning
// rows i..<iEnd) writes only to output columns t[jj*rows + ii] for ii in
// that tile's range, which is disjoint from every other tile's ii range,
// so chunks can run concurrently with no synchronization beyond the
// disjoint writes.
//
// Previously plain sequential, single-core. Called repeatedly per batch
// inside fcBackward (xt, dOutT) and convBackward (dOutR, col, W transposes
// — 3 conv layers × these calls), this was one of the file's most-called
// operations despite the header's "parallelised across all available
// cores" claim not actually covering it.
func transpose(_ a: [Float], rows: Int, cols: Int) -> [Float] {
    var t = [Float](repeating: 0, count: rows * cols)
    let tile = 64
    let rowTiles = (rows + tile - 1) / tile
    let slices = chunks(n: rowTiles, cores: coreCount)

    t.withUnsafeMutableBufferPointer { tBuf in
        a.withUnsafeBufferPointer { aBuf in
            let tAddr = addr(tBuf.baseAddress!)
            let aAddr = addr(aBuf.baseAddress!)
            DispatchQueue.concurrentPerform(iterations: slices.count) { sliceIdx in
                let tP = UnsafeMutablePointer<Float>(bitPattern: tAddr)!
                let aP = UnsafePointer<Float>(bitPattern: aAddr)!
                for rt in slices[sliceIdx] {
                    let i = rt * tile
                    let iEnd = Swift.min(i + tile, rows)
                    var j = 0
                    while j < cols {
                        let jEnd = Swift.min(j + tile, cols)
                        for ii in i..<iEnd {
                            for jj in j..<jEnd {
                                tP[jj * rows + ii] = aP[ii * cols + jj]
                            }
                        }
                        j += tile
                    }
                }
            }
        }
    }
    return t
}

// MARK: - SIMD4 dot product
//
// Both pointers must be contiguous (caller ensures this via transpose).
// Unrolled ×16 for the A53: 4 independent SIMD4 accumulator chains let the
// out-of-order parts of the pipeline overlap multiply and add latencies.
// On Apple Silicon the compiler will widen to SIMD8 automatically at -O.
@inline(__always)
func dotSIMD4(_ ap: UnsafePointer<Float>,
                      _ bp: UnsafePointer<Float>,
                      _ k: Int) -> Float {
    var a0 = SIMD4<Float>.zero
    var a1 = SIMD4<Float>.zero
    var a2 = SIMD4<Float>.zero
    var a3 = SIMD4<Float>.zero
    var kk = 0
    while kk + 15 < k {
        a0 += SIMD4(ap[kk],    ap[kk+1],  ap[kk+2],  ap[kk+3])
             * SIMD4(bp[kk],    bp[kk+1],  bp[kk+2],  bp[kk+3])
        a1 += SIMD4(ap[kk+4],  ap[kk+5],  ap[kk+6],  ap[kk+7])
             * SIMD4(bp[kk+4],  bp[kk+5],  bp[kk+6],  bp[kk+7])
        a2 += SIMD4(ap[kk+8],  ap[kk+9],  ap[kk+10], ap[kk+11])
             * SIMD4(bp[kk+8],  bp[kk+9],  bp[kk+10], bp[kk+11])
        a3 += SIMD4(ap[kk+12], ap[kk+13], ap[kk+14], ap[kk+15])
             * SIMD4(bp[kk+12], bp[kk+13], bp[kk+14], bp[kk+15])
        kk += 16
    }
    var acc = a0 + a1 + a2 + a3
    while kk + 3 < k {
        acc += SIMD4(ap[kk], ap[kk+1], ap[kk+2], ap[kk+3])
             * SIMD4(bp[kk], bp[kk+1], bp[kk+2], bp[kk+3])
        kk += 4
    }
    var sum = acc.x + acc.y + acc.z + acc.w
    while kk < k { sum += ap[kk] * bp[kk]; kk += 1 }
    return sum
}

// MARK: - matmulT  A(M×K) × Bᵀ(P×K) → C(M×P)
//
// Caller passes B already transposed to Bᵀ shape (P×K).
// Both A row i and Bᵀ row j are then contiguous → sequential SIMD4 reads.
//
// TILING: The original row-by-row loop (for each i: for each j: dot(A[i], BT[j]))
// causes BT to be evicted from L1 cache between rows of A when M is large
// (e.g. conv1 dX: M=28,800). The tiled version processes (tileM × tileP) output
// blocks together, keeping the BT tile hot in L1 while streaming through tileM
// rows of A. On Cortex-A53 (32KB L1 data cache):
//   tileM=8, tileP=8 tile: 8×k + 8×k floats in working set
//   For k=16 (conv1 dX): 256 bytes — fits comfortably in L1
//   For k=288 (conv2 dX): 4608 bytes — still fits in L1
// Parallelised over M-tiles via GCD, matching the original parallelism strategy.
func matmulT(a: [Float], bt: [Float], m: Int, k: Int, p: Int) -> [Float] {
    var c = [Float](repeating: 0, count: m * p)

    // Tile sizes tuned for Cortex-A53 32KB L1 data cache
    let tileM = 8
    let tileP = 8

    let mTiles = (m + tileM - 1) / tileM
    let slices = chunks(n: mTiles, cores: coreCount)

    c.withUnsafeMutableBufferPointer { cBuf in
        a.withUnsafeBufferPointer { aBuf in
            bt.withUnsafeBufferPointer { btBuf in
                let cp  = addr(cBuf.baseAddress!)
                let ap  = addr(aBuf.baseAddress!)
                let btp = addr(btBuf.baseAddress!)

                DispatchQueue.concurrentPerform(iterations: slices.count) { sliceIdx in
                    let cp  = UnsafeMutablePointer<Float>(bitPattern: cp)!
                    let ap  = UnsafePointer<Float>(bitPattern: ap)!
                    let btp = UnsafePointer<Float>(bitPattern: btp)!

                    for mt in slices[sliceIdx] {
                        let iStart = mt * tileM
                        let iEnd   = Swift.min(iStart + tileM, m)

                        var jt = 0
                        while jt < p {
                            let jEnd = Swift.min(jt + tileP, p)

                            // Process (iEnd-iStart) × (jEnd-jt) output block.
                            // BT rows [jt..<jEnd] stay in L1 across all i iterations.
                            for i in iStart..<iEnd {
                                let aRow = ap + i * k
                                for j in jt..<jEnd {
                                    cp[i * p + j] = dotSIMD4(aRow, btp + j * k, k)
                                }
                            }
                            jt += tileP
                        }
                    }
                }
            }
        }
    }
    return c
}

// MARK: - Activations

// Parallelised over element index, same pattern as weightedSum (chunks()
// + concurrentPerform, disjoint output slices — no reduction, so no
// partial-sum/stride bookkeeping needed here).
//
// Previously a plain `x.map { ... }` — single-threaded despite this file's
// header claiming every major operation is parallelised across all
// available cores. Applied 3x per forward call (once per conv layer, conv1
// being the largest tensor in the network — 30×30×16 per sample) and
// called once per mini-batch, this ran on 1 core for a real, non-trivial
// fraction of every round's wall-clock time — a concrete, measured
// contributor to effective_cores reading well below coreCount on a whole-
// round average. Below a certain size, chunking overhead isn't worth it
// (dispatch cost can exceed the work itself for tiny arrays, e.g. the FC
// layer's 10-class output) — matmulT/im2col/etc. don't special-case this
// either, so relu doesn't either; the concurrentPerform + chunks(n:cores:)
// combination already degrades gracefully to fewer/no extra chunks when
// element count is small (chunks() clamps to min(cores, n)).
public func relu(_ x: [Float]) -> [Float] {
    var out = [Float](repeating: 0, count: x.count)
    let slices = chunks(n: x.count, cores: coreCount)
    out.withUnsafeMutableBufferPointer { outBuf in
        x.withUnsafeBufferPointer { xBuf in
            let outAddr = addr(outBuf.baseAddress!)
            let xAddr = addr(xBuf.baseAddress!)
            DispatchQueue.concurrentPerform(iterations: slices.count) { sliceIdx in
                let outP = UnsafeMutablePointer<Float>(bitPattern: outAddr)!
                let xP = UnsafePointer<Float>(bitPattern: xAddr)!
                for i in slices[sliceIdx] {
                    let v = xP[i]
                    outP[i] = v > 0 ? v : 0
                }
            }
        }
    }
    return out
}

public func reluBackward(dOut: [Float], preActivation x: [Float]) -> [Float] {
    precondition(dOut.count == x.count, "reluBackward: dOut.count (\(dOut.count)) != preActivation x.count (\(x.count))")
    var out = [Float](repeating: 0, count: x.count)
    let slices = chunks(n: x.count, cores: coreCount)
    out.withUnsafeMutableBufferPointer { outBuf in
        dOut.withUnsafeBufferPointer { dOutBuf in
            x.withUnsafeBufferPointer { xBuf in
                let outAddr = addr(outBuf.baseAddress!)
                let dOutAddr = addr(dOutBuf.baseAddress!)
                let xAddr = addr(xBuf.baseAddress!)
                DispatchQueue.concurrentPerform(iterations: slices.count) { sliceIdx in
                    let outP = UnsafeMutablePointer<Float>(bitPattern: outAddr)!
                    let dOutP = UnsafePointer<Float>(bitPattern: dOutAddr)!
                    let xP = UnsafePointer<Float>(bitPattern: xAddr)!
                    for i in slices[sliceIdx] {
                        outP[i] = xP[i] > 0 ? dOutP[i] : 0
                    }
                }
            }
        }
    }
    return out
}

// MARK: - Softmax (row-wise, numerically stable)

public func softmax(_ logits: [Float], rows: Int, cols: Int) -> [Float] {
    var out = [Float](repeating: 0, count: rows * cols)
    for i in 0..<rows {
        let base = i * cols
        var rowMax = logits[base]
        for j in 1..<cols { if logits[base+j] > rowMax { rowMax = logits[base+j] } }
        var sum: Float = 0
        for j in 0..<cols { let v = expf(logits[base+j] - rowMax); out[base+j] = v; sum += v }
        for j in 0..<cols { out[base+j] /= sum }
    }
    return out
}

// MARK: - Loss & gradient

// `classes` parameter added rather than hardcoding 10 — the original
// version assumed exactly 10 output classes (CIFAR-10/MNIST's class count),
// which works for those two datasets but silently produces wrong gradients
// for anything else: passing data with a different class count wouldn't
// error, it would just index into the wrong column of `probs`. Making this
// explicit is required for the function to be safely reusable beyond the
// two datasets this was originally written against.
public func crossEntropyLoss(probs: [Float], labels: [Int], n: Int, classes: Int) -> Float {
    var loss: Float = 0
    for i in 0..<n { loss += -logf(probs[i * classes + labels[i]] + 1e-9) }
    return loss / Float(n)
}

public func softmaxCrossEntropyGrad(probs: [Float], labels: [Int], n: Int, classes: Int) -> [Float] {
    var grad = probs
    for i in 0..<n { grad[i * classes + labels[i]] -= 1 }
    let scale = 1.0 / Float(n)
    return grad.map { $0 * scale }
}

// MARK: - Argmax

public func argmax(_ v: [Float], row: Int, cols: Int) -> Int {
    let base = row * cols
    var best = base
    for j in 1..<cols { if v[base+j] > v[best] { best = base+j } }
    return best - base
}

// MARK: - FC forward  (N × inF) × (inF × outF) → (N × outF)
//
// W is passed as a CachedMatrix rather than a plain [Float] — W only
// actually changes once per optimizer step, but fcForward is called on
// every batch within that step, so transposing it fresh every call was
// pure wasted work. CachedMatrix.transposed() computes Wᵀ once and reuses
// it until the next CachedMatrix.update(_:) call (made by the optimizer
// step, not by fcForward).
public func fcForward(x: [Float], W: CachedMatrix, b: [Float],
               n: Int, inF: Int, outF: Int) -> [Float] {
    let wt  = W.transposed()   // (outF × inF) — cached after first call
    var out = matmulT(a: x, bt: wt, m: n, k: inF, p: outF)
    for i in 0..<n { for j in 0..<outF { out[i*outF+j] += b[j] } }
    return out
}

// MARK: - FC backward
//
// dW = xᵀ @ dOut  →  (inF × outF)
//    = matmulT(xᵀ, dOutᵀ) where xᵀ is (inF×n), dOutᵀ is (outF×n)
//    Equivalently: transpose x to (inF×n), transpose dOut to (outF×n),
//    then matmulT(xᵀ, dOutᵀ, m=inF, k=n, p=outF)
//
// dX = dOut @ Wᵀ  →  (N × inF)
//    = matmulT(dOut, W, m=n, k=outF, p=inF)
//    W is already (inF×outF) so Wᵀ is (outF×inF) — W itself serves as bt.

// Explicit `public` members AND an explicit `public init` — Swift's
// auto-synthesized memberwise initializer is only ever as visible as
// `internal`, even on a struct marked `public`. Without the explicit init
// below, code outside this module could read an FCGrads handed back from
// fcBackward, but could not construct one itself, which would be a strange
// half-public type. Same applies to PoolResult and ConvGrads further down.
public struct FCGrads {
    public var dW: [Float]
    public var db: [Float]
    public var dX: [Float]

    public init(dW: [Float], db: [Float], dX: [Float]) {
        self.dW = dW
        self.db = db
        self.dX = dX
    }
}

// Takes W as CachedMatrix purely for type consistency with fcForward (so
// callers pass the same weight object to both, rather than juggling a
// CachedMatrix for forward and a raw [Float] for backward) — this function
// itself doesn't use the cached transpose, since W already happens to be
// in the right (inF×outF) layout to serve directly as matmulT's bt:
// argument for the dX computation below.
public func fcBackward(dOut: [Float], x: [Float], W: CachedMatrix,
                n: Int, inF: Int, outF: Int) -> FCGrads {
    // db = column sums of dOut  (cheap scalar loop)
    var db = [Float](repeating: 0, count: outF)
    for i in 0..<n { for j in 0..<outF { db[j] += dOut[i*outF+j] } }

    // dW = xᵀ @ dOut  →  matmulT(xᵀ, dOutᵀ, m=inF, k=n, p=outF)
    let xt    = transpose(x,    rows: n,    cols: inF)   // (inF × n)
    let dOutT = transpose(dOut, rows: n,    cols: outF)  // (outF × n)
    let dW    = matmulT(a: xt, bt: dOutT, m: inF, k: n, p: outF)

    // dX = dOut @ W  →  matmulT(dOut, W, m=n, k=outF, p=inF)
    // W is (inF×outF); treat as Bᵀ with shape (inF×outF) → p=inF, k=outF
    let dX = matmulT(a: dOut, bt: W.values, m: n, k: outF, p: inF)

    return FCGrads(dW: dW, db: db, dX: dX)
}

// MARK: - MaxPool 2×2 stride 2  (N, C, H, W)

public struct PoolResult {
    public var out: [Float]
    public var mask: [Float]   // Float, not Bool: tied maxima get 1/count each, matching Python's _maxpool_forward

    public init(out: [Float], mask: [Float]) {
        self.out = out
        self.mask = mask
    }
}

// Parallelised over n*c, EXACTLY mirroring maxPool2Backward's existing
// pattern below (direct DispatchQueue.concurrentPerform(iterations: n*c),
// not the chunks()-based pre-slicing used elsewhere in this file) — same
// access shape (one (ni, ci) combo per iteration, disjoint output/mask
// regions per iteration: `out` gets ho*wo elements per t, `mask` gets h*w
// elements per t, both non-overlapping across different t values, so no
// synchronization needed beyond the disjoint writes themselves).
//
// This was previously the one sequential (non-parallel) forward/backward
// pair in the file: maxPool2Backward already had exactly this
// parallelisation, maxPool2Forward did not, with no comment explaining why
// forward specifically was left out — an oversight, not a deliberate
// choice, given this file's header claims every major operation is
// parallelised. Called once per conv layer (3x) per mini-batch, on conv1's
// activations in particular (the largest tensor in the network), this ran
// single-threaded for a real, measurable share of every round's wall-clock
// time.
public func maxPool2Forward(x: [Float], n: Int, c: Int, h: Int, w: Int) -> PoolResult {
    let ho = h/2, wo = w/2
    var out  = [Float](repeating: -Float.infinity, count: n*c*ho*wo)
    var mask = [Float](repeating: 0, count: n*c*h*w)

    // Two-pass per window: first find max, then distribute gradient equally
    // among tied positions — matching Python's _maxpool_forward exactly:
    //   counts = mask_r.sum(axis=(3,5), keepdims=True)
    //   mask_r = mask_r / counts.clip(min=1)
    // This matters because ReLU outputs frequently tie at 0.0 early in
    // training, and biasing gradients to an arbitrary single position
    // (the Swift original) causes systematic gradient errors that compound
    // through the backward pass across all three pooling layers.
    out.withUnsafeMutableBufferPointer { outBuf in
        mask.withUnsafeMutableBufferPointer { maskBuf in
            x.withUnsafeBufferPointer { xBuf in
                let outAddr  = addr(outBuf.baseAddress!)
                let maskAddr = addr(maskBuf.baseAddress!)
                let xAddr    = addr(xBuf.baseAddress!)
                DispatchQueue.concurrentPerform(iterations: n*c) { t in
                    let outP  = UnsafeMutablePointer<Float>(bitPattern: outAddr)!
                    let maskP = UnsafeMutablePointer<Float>(bitPattern: maskAddr)!
                    let xP    = UnsafePointer<Float>(bitPattern: xAddr)!
                    let ni = t/c, ci = t%c
                    for oh in 0..<ho {
                        for ow in 0..<wo {
                            // Pass 1: find max value in 2×2 window
                            var best = -Float.infinity
                            for kh in 0..<2 { for kw in 0..<2 {
                                let idx = ((ni*c+ci)*h + oh*2+kh)*w + ow*2+kw
                                if xP[idx] > best { best = xP[idx] }
                            }}
                            let outIdx = ((ni*c+ci)*ho+oh)*wo+ow
                            outP[outIdx] = best

                            // Pass 2: count ties, distribute 1/count to each tied position
                            var count: Float = 0
                            for kh in 0..<2 { for kw in 0..<2 {
                                let idx = ((ni*c+ci)*h + oh*2+kh)*w + ow*2+kw
                                if xP[idx] == best { count += 1 }
                            }}
                            let weight = 1.0 / count
                            for kh in 0..<2 { for kw in 0..<2 {
                                let idx = ((ni*c+ci)*h + oh*2+kh)*w + ow*2+kw
                                if xP[idx] == best { maskP[idx] = weight }
                            }}
                        }
                    }
                }
            }
        }
    }
    return PoolResult(out: out, mask: mask)
}

public func maxPool2Backward(dOut: [Float], mask: [Float],
                      n: Int, c: Int, h: Int, w: Int) -> [Float] {
    let ho = h/2, wo = w/2
    var dX = [Float](repeating: 0, count: n*c*h*w)

    dX.withUnsafeMutableBufferPointer { dxBuf in
        dOut.withUnsafeBufferPointer { doBuf in
            mask.withUnsafeBufferPointer { mBuf in
                let dxp = addr(dxBuf.baseAddress!)
                let dop = addr(doBuf.baseAddress!)
                let mp  = addr(mBuf.baseAddress!)
                DispatchQueue.concurrentPerform(iterations: n*c) { t in
                    let dxp = UnsafeMutablePointer<Float>(bitPattern: dxp)!
                    let dop = UnsafePointer<Float>(bitPattern: dop)!
                    let mp  = UnsafePointer<Float>(bitPattern: mp)!
                    let ni = t/c, ci = t%c
                    for oh in 0..<ho {
                        for ow in 0..<wo {
                            let grad = dop[((ni*c+ci)*ho+oh)*wo+ow]
                            for kh in 0..<2 { for kw in 0..<2 {
                                let idx = ((ni*c+ci)*h + oh*2+kh)*w + ow*2+kw
                                dxp[idx] += grad * mp[idx]  // mp[idx] = 1/count or 0
                            }}
                        }
                    }
                }
            }
        }
    }
    return dX
}

// MARK: - Conv forward/backward via im2col
//
// im2col transforms the convolution into a single matrix multiply:
//
//   col  = im2col(x)          shape: (N, H*W, cIn*kH*kW)   — patch matrix
//   W_2d = W reshaped         shape: (cOut,   cIn*kH*kW)   — filter matrix
//
//   forward:   out = col @ W_2dᵀ + b   →  (N, H*W, cOut) → (N, cOut, H, W)
//   backward:  dW  = dOut_r @ col      →  (cOut, cIn*kH*kW), summed over N
//              dcol = dOut_r @ W_2d    →  (N, H*W, cIn*kH*kW)
//              dX  = col2im(dcol)      →  (N, cIn, H, W)
//
// convBackward is now two matmuls + col2im — no 6-deep nested loops,
// same parallel gemm pattern as fcForward/fcBackward.

public struct ConvGrads {
    public var dW: [Float]
    public var db: [Float]
    public var dX: [Float]

    public init(dW: [Float], db: [Float], dX: [Float]) {
        self.dW = dW
        self.db = db
        self.dX = dX
    }
}

// MARK: im2col
// For each sample ni, writes every (cIn*kH*kW)-element patch for every output
// position (oh, ow) into a row of `col`.
// col shape per sample: (outH*outW) rows × (cIn*kH*kW) cols, row-major.
// Zero-padding is handled by writing 0 for out-of-bounds input positions.
// Parallelised over N samples.
//
// outH/outW are now DERIVED from `pad` via the standard convolution output
// formula — outH = h + 2*pad - kH + 1 (stride fixed at 1, matching every
// layer in both this project's models and Python's trainer.py) — NOT
// hardcoded to h/w as an earlier version of this function did. That earlier
// version could ONLY express same-padding convolution (output size always
// equals input size); it could not produce Python's SimpleCNNCIFAR
// architecture, which uses valid (pad=0) convolution and therefore shrinks
// spatial size at every conv layer (32→30→15→13→6→4→2, confirmed against
// trainer.py's own _im2col: out_h = (H - kh) // stride + 1 with stride
// always 1 in this model). pad=kH/2 still produces the original
// same-padding behavior for any EXISTING caller that wants it — this is a
// strict generalization, not a behavior change for same-padding callers,
// as long as they continue passing pad=kH/2 explicitly (verified below in
// convForward/convBackward, which now expose pad as a real parameter
// instead of hardcoding it internally).
func im2col(x: [Float],
                    n: Int, cIn: Int, h: Int, w: Int,
                    kH: Int, kW: Int, pad: Int) -> [Float] {
    let outH   = h + 2 * pad - kH + 1
    let outW   = w + 2 * pad - kW + 1
    let colCols = cIn * kH * kW   // length of one patch vector
    let colRows = outH * outW     // number of patches per sample
    var col = [Float](repeating: 0, count: n * colRows * colCols)

    col.withUnsafeMutableBufferPointer { colBuf in
        x.withUnsafeBufferPointer { xBuf in
            let cp = addr(colBuf.baseAddress!)
            let xp = addr(xBuf.baseAddress!)
            DispatchQueue.concurrentPerform(iterations: n) { ni in
                let cp = UnsafeMutablePointer<Float>(bitPattern: cp)!
                let xp = UnsafePointer<Float>(bitPattern: xp)!
                // Base of this sample's patch block in col
                let colBase = ni * colRows * colCols
                for oh in 0..<outH {
                    for ow in 0..<outW {
                        let rowBase = colBase + (oh * outW + ow) * colCols
                        var patchIdx = 0
                        for ci in 0..<cIn {
                            for kh in 0..<kH {
                                let ih = oh - pad + kh
                                for kw in 0..<kW {
                                    let iw = ow - pad + kw
                                    if ih >= 0, ih < h, iw >= 0, iw < w {
                                        cp[rowBase + patchIdx] =
                                            xp[((ni*cIn + ci)*h + ih)*w + iw]
                                    }
                                    // else stays 0 (zero-pad)
                                    patchIdx += 1
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return col
}

// MARK: col2im
// Inverse of im2col: scatter-add patch columns back into (N, cIn, H, W).
// Parallelised over N — each sample's dX slice is independent.
//
// outH/outW derived from the SAME formula as im2col (h + 2*pad - kH + 1) —
// guaranteed consistent with whatever im2col call produced `col` in the
// first place, since both functions compute it identically from the same
// (h, pad, kH) inputs. An earlier version hardcoded outH=h/outW=w here too
// (same bug as im2col's original version) — col2im's own scatter-addressing
// logic (ih = oh - pad + kh) was ALREADY correct for any pad value; only
// the output-size computation was wrong, the same narrow bug in both
// functions. dX's own shape (n, cIn, h, w) is unaffected by this fix — dX
// is always the ORIGINAL (pre-conv) input shape regardless of padding mode,
// only `col`'s row count (which depends on outH*outW) was wrong.
func col2im(col: [Float],
                    n: Int, cIn: Int, h: Int, w: Int,
                    kH: Int, kW: Int, pad: Int) -> [Float] {
    let outH    = h + 2 * pad - kH + 1
    let outW    = w + 2 * pad - kW + 1
    let colCols = cIn * kH * kW
    let colRows = outH * outW
    var dX = [Float](repeating: 0, count: n * cIn * h * w)

    dX.withUnsafeMutableBufferPointer { dxBuf in
        col.withUnsafeBufferPointer { colBuf in
            let dxp = addr(dxBuf.baseAddress!)
            let cp  = addr(colBuf.baseAddress!)
            DispatchQueue.concurrentPerform(iterations: n) { ni in
                let dxp = UnsafeMutablePointer<Float>(bitPattern: dxp)!
                let cp  = UnsafePointer<Float>(bitPattern: cp)!
                let colBase = ni * colRows * colCols
                for oh in 0..<outH {
                    for ow in 0..<outW {
                        let rowBase = colBase + (oh * outW + ow) * colCols
                        var patchIdx = 0
                        for ci in 0..<cIn {
                            for kh in 0..<kH {
                                let ih = oh - pad + kh
                                for kw in 0..<kW {
                                    let iw = ow - pad + kw
                                    if ih >= 0, ih < h, iw >= 0, iw < w {
                                        dxp[((ni*cIn + ci)*h + ih)*w + iw] +=
                                            cp[rowBase + patchIdx]
                                    }
                                    patchIdx += 1
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return dX
}

// MARK: convForward
//
// 1. im2col(x)      → col: (N·outH·outW, cIn·kH·kW)
// 2. matmulT(col, W)  W is (cOut × colCols) = already Bᵀ layout
//                   → raw: (N·outH·outW, cOut)
// 3. add bias, reshape → (N, cOut, outH, outW)
//
// W is stored (cOut × colCols) — each filter is a contiguous row.
// col row i is a contiguous patch vector.
// matmulT(col, W, m=colRows, k=colCols, p=cOut):
//   bt = W (already in Bᵀ layout: each of the p=cOut rows has k=colCols elements)
//
// NOTE: still plain [Float], not CachedMatrix, unlike fcForward/fcBackward.
// This is deliberate, not an inconsistency — conv's W is ALREADY stored in
// the exact layout matmulT needs as bt: (cOut × colCols), so there was
// never a repeated-transpose cost here to fix. CachedMatrix exists
// specifically to eliminate redundant transpose() calls; wrapping
// something that never called transpose() in the first place would add
// API friction (forcing every caller to wrap conv filters too) for zero
// performance benefit.
//
// `pad` is now an explicit caller-supplied parameter, NOT hardcoded to
// kH/2 internally as an earlier version did. That earlier version could
// only ever express same-padding convolution (output spatial size always
// equal to input) — it had no way to express valid (no-padding)
// convolution, which Python's SimpleCNNCIFAR uses for every conv layer
// (out_h = (H - kh) // stride + 1, confirmed against trainer.py's
// _im2col). Existing same-padding callers must now pass pad: kH/2
// explicitly to preserve their previous behavior — this is a real,
// deliberate signature change, not purely additive; every call site in
// this project was checked and updated (see SimpleCNN.swift).
public func convForward(x: [Float], W: [Float], b: [Float],
                 n: Int, cIn: Int, cOut: Int, h: Int, w: Int,
                 kH: Int, kW: Int, pad: Int) -> [Float] {
    let outH = h + 2 * pad - kH + 1
    let outW = w + 2 * pad - kW + 1
    let colCols = cIn * kH * kW
    let colRows = n * outH * outW

    let col = im2col(x: x, n: n, cIn: cIn, h: h, w: w, kH: kH, kW: kW, pad: pad)
    // W is (cOut × colCols) — already in Bᵀ shape for matmulT
    var raw = matmulT(a: col, bt: W, m: colRows, k: colCols, p: cOut)
    // Add bias
    for i in 0..<colRows { for j in 0..<cOut { raw[i*cOut+j] += b[j] } }

    // Reshape (N·outH·outW, cOut) → (N, cOut, outH, outW)
    var out = [Float](repeating: 0, count: n * cOut * outH * outW)
    for ni in 0..<n {
        for oh in 0..<outH {
            for ow in 0..<outW {
                let rawRow = ni * outH * outW + oh * outW + ow
                for co in 0..<cOut {
                    out[((ni*cOut + co)*outH + oh)*outW + ow] = raw[rawRow*cOut + co]
                }
            }
        }
    }
    return out
}

// MARK: convBackward
//
// dOut_r = reshape dOut (N,cOut,H,W) → (N·H·W, cOut)
// col    = im2col(x)                    recomputed
//
// dW   = dOut_rᵀ @ col
//      = matmulT(dOut_rᵀ, colᵀ, m=cOut, k=colRows, p=colCols)
//      transpose both: dOut_rᵀ is (cOut×colRows), colᵀ is (colCols×colRows)
//      → matmulT(dOut_rᵀ, colᵀ, m=cOut, k=colRows, p=colCols)
//
// dcol = dOut_r @ W
//      W is (cOut×colCols) = already Bᵀ layout for matmulT
//      → matmulT(dOut_r, W, m=colRows, k=cOut, p=colCols)
//
// dX   = col2im(dcol)
// MARK: convBackward
//
// dOut_r = reshape dOut (N,cOut,outH,outW) → (N·outH·outW, cOut)
// col    = im2col(x)                    recomputed
//
// dW   = dOut_rᵀ @ col
//      = matmulT(dOut_rᵀ, colᵀ, m=cOut, k=colRows, p=colCols)
//      transpose both: dOut_rᵀ is (cOut×colRows), colᵀ is (colCols×colRows)
//      → matmulT(dOut_rᵀ, colᵀ, m=cOut, k=colRows, p=colCols)
//
// dcol = dOut_r @ W
//      W is (cOut×colCols) = already Bᵀ layout for matmulT
//      → matmulT(dOut_r, W, m=colRows, k=cOut, p=colCols)
//
// dX   = col2im(dcol)   — always (N, cIn, h, w), the ORIGINAL input shape,
//                          regardless of pad/outH/outW
//
// `pad` is now explicit (see convForward's doc comment for the full
// rationale — same signature-change reasoning applies here).
public func convBackward(dOut: [Float], x: [Float], W: [Float],
                  n: Int, cIn: Int, cOut: Int, h: Int, w: Int,
                  kH: Int, kW: Int, pad: Int) -> ConvGrads {
    let outH = h + 2 * pad - kH + 1
    let outW = w + 2 * pad - kW + 1
    let colCols = cIn * kH * kW
    let colRows = n * outH * outW

    let col = im2col(x: x, n: n, cIn: cIn, h: h, w: w, kH: kH, kW: kW, pad: pad)

    // Reshape dOut: (N, cOut, outH, outW) → dOut_r: (N·outH·outW, cOut)
    var dOutR = [Float](repeating: 0, count: colRows * cOut)
    for ni in 0..<n {
        for oh in 0..<outH {
            for ow in 0..<outW {
                let row = ni * outH * outW + oh * outW + ow
                for co in 0..<cOut {
                    dOutR[row*cOut + co] = dOut[((ni*cOut + co)*outH + oh)*outW + ow]
                }
            }
        }
    }

    // db = column sums of dOut_r
    var db = [Float](repeating: 0, count: cOut)
    for i in 0..<colRows { for j in 0..<cOut { db[j] += dOutR[i*cOut+j] } }

    // dW = dOut_rᵀ @ col  →  (cOut × colCols)
    // = matmulT(dOut_rᵀ, colᵀ, m=cOut, k=colRows, p=colCols)
    let dOutRT = transpose(dOutR, rows: colRows, cols: cOut)  // (cOut × colRows)
    let colT   = transpose(col,   rows: colRows, cols: colCols) // (colCols × colRows)
    let dW     = matmulT(a: dOutRT, bt: colT, m: cOut, k: colRows, p: colCols)

    // dcol = dOut_r @ W  →  (colRows × colCols)
    // W is (cOut × colCols) — need to transpose to (colCols × cOut) for matmulT's bt argument.
    // matmulT(a, bt, m, k, p) computes a @ bt^T where bt is (p, k).
    // We want dOutR (colRows, cOut) @ W (cOut, colCols) = (colRows, colCols).
    // Rewrite as matmulT(dOutR, W^T, m=colRows, k=cOut, p=colCols)
    // where W^T has shape (colCols, cOut), satisfying bt's (p=colCols, k=cOut) requirement.
    let WT   = transpose(W, rows: cOut, cols: colCols)  // (colCols × cOut)
    let dcol = matmulT(a: dOutR, bt: WT, m: colRows, k: cOut, p: colCols)

    let dX = col2im(col: dcol, n: n, cIn: cIn, h: h, w: w, kH: kH, kW: kW, pad: pad)

    return ConvGrads(dW: dW, db: db, dX: dX)
}


