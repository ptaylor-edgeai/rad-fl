// SimpleCNN.swift
//
// CNN matching Python's trainer.py SimpleCNNCIFAR architecture EXACTLY —
// same layer depths, same channel counts, same valid (no-padding)
// convolution at every conv layer, same spatial dimensions at every stage,
// same He weight initialization. This is a deliberate, verified
// architectural match, not an approximation — see this project's RQ1: the
// paper's entire claim rests on Swift vs. Python being a controlled
// comparison where the ONLY difference is the runtime/language, not the
// model. An earlier version of this file had a single 4-channel
// same-padding conv layer — a much weaker, structurally different model
// that produced a real, visible divergence (flat/noisy training curves vs.
// Python's clean convergence) when run side-by-side on real 10-node Pi
// hardware. This rebuild closes that gap.
//
// ARCHITECTURE (matches trainer.py's SimpleCNNCIFAR docstring exactly):
//   Input:  (N, 3, 32, 32)
//   Conv1:  3  → 16 filters, 3×3, VALID (no padding) → (N, 16, 30, 30)
//   ReLU, MaxPool 2×2                                 → (N, 16, 15, 15)
//   Conv2: 16  → 32 filters, 3×3, VALID               → (N, 32, 13, 13)
//   ReLU, MaxPool 2×2 (floor: 13/2=6)                 → (N, 32,  6,  6)
//   Conv3: 32  → 64 filters, 3×3, VALID                → (N, 64,  4,  4)
//   ReLU, MaxPool 2×2                                 → (N, 64,  2,  2)
//   Flatten                                            → (N, 256)
//   FC:    256 → 10, Softmax
//
// VALID (no-padding) convolution at every layer — NOT same-padding —
// confirmed against trainer.py's _im2col: out_h = (H - kh) // stride + 1,
// stride always 1 in this model. This required real changes to
// Tensor.swift's im2col/col2im/convForward/convBackward, which were
// previously hardcoded to same-padding only (outH/outW always equal to
// input H/W) — see those functions' own doc comments in Tensor.swift for
// the full rationale.
//
// He initialization (rng.standard_normal(shape) * sqrt(2/fan_in) in
// trainer.py) — NOT the flat ±0.1 uniform scale an earlier version of this
// file used. He init is the standard, principled choice for ReLU networks
// specifically; a flat small-uniform scale has no such grounding and was
// itself a real contributor to Swift's weaker convergence before this fix
// (alongside the architecture-depth gap, which was the larger factor).

import Foundation

/// `Codable`: lets this config be saved alongside trained weights (see
/// RoundOrchestrator.saveFinalWeights / NPYWriter), so a future experiment
/// can reconstruct the exact SimpleCNN architecture that produced a given
/// set of saved .npy weight files.
///
/// Most fields are now FIXED, matching trainer.py's SimpleCNNCIFAR class
/// constants exactly (C1/C2/C3, K, IN_CHANNELS, NUM_CLASSES) — an earlier
/// version of this struct made cOut/kH/kW/classes freely configurable,
/// appropriate for a single-conv-layer wiring-test model but wrong for
/// this rebuild, where the entire point is architectural equivalence with
/// a SPECIFIC Python model, not a general-purpose configurable CNN. Only
/// `n` (batch size) is a real per-instance parameter from the caller's
/// perspective — the rest are set via `init()`'s own internal defaults,
/// not freely passed in.
///
/// IMPORTANT: these are declared as genuine `let` properties assigned in
/// `init`, NOT `let cIn: Int = 3` inline-default-value declarations. This
/// distinction is real and was deliberately checked, not stylistic:
/// Swift's synthesized Codable conformance SILENTLY DOES NOT DECODE `let`
/// properties that have an inline default value at their declaration —
/// confirmed via Swift's own issue tracker (swiftlang/swift#50689): such a
/// property is still ENCODED with its real value, but on DECODE the
/// synthesized init(from:) always falls back to the declared default,
/// discarding whatever was actually in the JSON, with no warning or
/// error. For a config struct whose entire purpose is "load this back
/// later to know exactly what architecture produced these weights," that
/// would be a serious, silent correctness bug — it would currently
/// "work" only by coincidence (because the hardcoded defaults happen to
/// match what's used), and would become a real landmine the moment these
/// constants ever changed and someone tried to reload an older
/// config.json. Using a real `init` instead avoids this trap entirely.
public struct SimpleCNNConfig: Sendable, Codable {
    public let n: Int          // batch size — the one real per-instance parameter

    // Fixed architecture constants, matching trainer.py's SimpleCNNCIFAR exactly:
    public let cIn: Int            // IN_CHANNELS
    public let h: Int              // CIFAR-10 native spatial size
    public let w: Int
    public let c1: Int             // conv1 output channels
    public let c2: Int             // conv2 output channels
    public let c3: Int             // conv3 output channels
    public let kH: Int             // K
    public let kW: Int
    public let classes: Int        // NUM_CLASSES

    public init(n: Int) {
        self.n = n
        self.cIn = 3
        self.h = 32
        self.w = 32
        self.c1 = 16
        self.c2 = 32
        self.c3 = 64
        self.kH = 3
        self.kW = 3
        self.classes = 10
    }

    // Spatial sizes at every stage, derived from the SAME valid-convolution
    // formula now used by Tensor.swift's im2col/convForward (outH = h - kH + 1,
    // i.e. pad=0): confirmed against trainer.py's _im2col exactly
    // (out_h = (H - kh) // stride + 1, stride=1).
    public var conv1OutH: Int { h - kH + 1 }              // 32 - 3 + 1 = 30
    public var conv1OutW: Int { w - kW + 1 }
    public var pool1OutH: Int { conv1OutH / 2 }           // 30 / 2 = 15
    public var pool1OutW: Int { conv1OutW / 2 }

    public var conv2OutH: Int { pool1OutH - kH + 1 }      // 15 - 3 + 1 = 13
    public var conv2OutW: Int { pool1OutW - kW + 1 }
    public var pool2OutH: Int { conv2OutH / 2 }           // 13 / 2 = 6 (floor)
    public var pool2OutW: Int { conv2OutW / 2 }

    public var conv3OutH: Int { pool2OutH - kH + 1 }      // 6 - 3 + 1 = 4
    public var conv3OutW: Int { pool2OutW - kW + 1 }
    public var pool3OutH: Int { conv3OutH / 2 }           // 4 / 2 = 2
    public var pool3OutW: Int { conv3OutW / 2 }

    /// Flattened feature count entering the FC layer — 64 * 2 * 2 = 256,
    /// matching trainer.py's FC_IN = 64 * 2 * 2 (= 256) exactly.
    public var fcInputFeatures: Int { c3 * pool3OutH * pool3OutW }
}

/// Holds every learnable parameter for the 3-conv-layer CNN above. Conv
/// weights stay plain [Float] (per Tensor.swift's own design — conv's W is
/// already in the layout convForward/convBackward need, so CachedMatrix's
/// transpose-caching wouldn't help there); FC weights use CachedMatrix
/// specifically because fcForward benefits from not re-transposing W on
/// every batch within the same optimizer step.
///
/// `@unchecked Sendable`: SimpleCNN is a class with mutable state — Swift
/// cannot automatically verify this is safe to share across concurrency
/// domains, hence `@unchecked`. The actual safety argument: a SimpleCNN
/// instance is owned and mutated by exactly one DFL node's local-training
/// step at a time (forward → backward → SGD update, sequentially, on
/// whichever thread/task is running that node's round) — there is no
/// concurrent access to the SAME instance from multiple threads anywhere
/// in this codebase. Different nodes each have their OWN SimpleCNN
/// instance; nothing here shares one instance across nodes.
public final class SimpleCNN: @unchecked Sendable {
    public let config: SimpleCNNConfig

    var conv1W: [Float]; var conv1B: [Float]    // (c1, cIn*kH*kW), (c1,)
    var conv2W: [Float]; var conv2B: [Float]    // (c2, c1*kH*kW),  (c2,)
    var conv3W: [Float]; var conv3B: [Float]    // (c3, c2*kH*kW),  (c3,)
    let fcW: CachedMatrix                       // (fcInputFeatures, classes)
    var fcB: [Float]                            // (classes,)

    // Momentum velocity state — one zero-initialized array per parameter
    // tensor, matching Python's SGD._velocity dict exactly (trainer.py:
    // self._velocity: Dict[str, np.ndarray] = {}, initialized to zeros_like
    // on first use). Reset to zero after each aggregation step, matching
    // Python's optimizer.reset() call at the start of train_round() —
    // "Aggregated weights may be very different from the previous local
    // weights; keeping stale momentum would apply a misleading gradient
    // step" (trainer.py docstring). This is the missing piece that caused
    // Swift's per-round training to be less effective than Python's:
    // momentum=0.9 makes a substantial difference on the small, noisy
    // per-node batches produced by Dirichlet non-IID partitioning,
    // confirmed directly from Python's SGD class docstring.
    var vConv1W: [Float]; var vConv1B: [Float]
    var vConv2W: [Float]; var vConv2B: [Float]
    var vConv3W: [Float]; var vConv3B: [Float]
    var vFCW: [Float]
    var vFCB: [Float]

    public init(config: SimpleCNNConfig, seed: UInt64 = 42) {
        self.config = config

        // He initialization: N(0, sqrt(2/fan_in)) via Box-Muller over SplitMix64.
        // Statistically equivalent to Python's trainer.py _he() which uses
        // NumPy's default_rng(seed).standard_normal(shape) * sqrt(2/fan_in).
        // Different specific values than NumPy's PCG64+Ziggurat RNG, but
        // same distribution — confirmed by 25-round Pi run matching Python's
        // test_acc, local_acc, and train_loss within noise, so bit-identical
        // initialization is not required for correct federated convergence.
        var rng = SplitMix64(seed: seed)
        let conv1FanIn = config.cIn * config.kH * config.kW
        self.conv1W = SimpleCNN.heWeights(count: config.c1 * conv1FanIn, fanIn: conv1FanIn, rng: &rng)
        self.conv1B = [Float](repeating: 0, count: config.c1)

        let conv2FanIn = config.c1 * config.kH * config.kW
        self.conv2W = SimpleCNN.heWeights(count: config.c2 * conv2FanIn, fanIn: conv2FanIn, rng: &rng)
        self.conv2B = [Float](repeating: 0, count: config.c2)

        let conv3FanIn = config.c2 * config.kH * config.kW
        self.conv3W = SimpleCNN.heWeights(count: config.c3 * conv3FanIn, fanIn: conv3FanIn, rng: &rng)
        self.conv3B = [Float](repeating: 0, count: config.c3)

        let fcFanIn = config.fcInputFeatures
        // fc_W layout fix: Python stores fc_W as (outF=classes, inF=fcInputFeatures)
        // row-major, so the flat weight sequence generated by He-init represents
        // a (classes × fcInputFeatures) matrix. CachedMatrix stores (inF × outF)
        // for fcForward/fcBackward's transposed-access pattern, so transpose
        // from (classes, fcInputFeatures) to (fcInputFeatures, classes) before
        // storing. fcForward then calls W.transposed() to get (classes, fcInputFeatures),
        // matching Python's x @ fc_W.T computation exactly.
        let fcWRaw = SimpleCNN.heWeights(count: config.classes * fcFanIn, fanIn: fcFanIn, rng: &rng)
        self.fcW = CachedMatrix(values: SimpleCNN.transposeMatrix(fcWRaw, rows: config.classes, cols: fcFanIn),
                                rows: fcFanIn, cols: config.classes)
        self.fcB = [Float](repeating: 0, count: config.classes)

        // Initialize all velocity arrays to zero — same shape as corresponding weights
        self.vConv1W = [Float](repeating: 0, count: config.c1 * config.cIn * config.kH * config.kW)
        self.vConv1B = [Float](repeating: 0, count: config.c1)
        self.vConv2W = [Float](repeating: 0, count: config.c2 * config.c1 * config.kH * config.kW)
        self.vConv2B = [Float](repeating: 0, count: config.c2)
        self.vConv3W = [Float](repeating: 0, count: config.c3 * config.c2 * config.kH * config.kW)
        self.vConv3B = [Float](repeating: 0, count: config.c3)
        self.vFCW    = [Float](repeating: 0, count: config.fcInputFeatures * config.classes)
        self.vFCB    = [Float](repeating: 0, count: config.classes)
    }


    /// He initialization: each weight ~ N(0, sqrt(2/fanIn)) — matching
    /// trainer.py's _he() exactly: `rng.standard_normal(shape) *
    /// sqrt(2.0 / fan_in)`. SplitMix64 only generates uniform [0,1)
    /// samples (nextUniform()); standard-normal samples are derived via the
    /// Box-Muller transform, a standard, well-known way to turn two
    /// independent uniform samples into two independent standard-normal
    /// samples. NOT expected to reproduce NumPy's bit-for-bit normal
    /// samples for a given seed (NumPy's default Generator uses a
    /// different underlying algorithm — Ziggurat over PCG64 — not
    /// Box-Muller over SplitMix64), same caveat already documented on
    /// SplitMix64.shuffledIndices for the shuffle case: what matters for a
    /// fair comparison is that BOTH implementations sample from a genuine
    /// N(0,1) distribution scaled by sqrt(2/fan_in), not that they produce
    /// identical numbers for the same seed.
    /// Transpose a flat (rows × cols) matrix to (cols × rows) row-major.
    /// Used to correct the fc_W layout: PCG64/SplitMix64 generates values
    /// in (outF × inF) = (classes × fcInputFeatures) order matching Python,
    /// but CachedMatrix stores (inF × outF) for fcForward's transposed access.
    private static func transposeMatrix(_ flat: [Float], rows: Int, cols: Int) -> [Float] {
        var out = [Float](repeating: 0, count: rows * cols)
        for r in 0..<rows {
            for c in 0..<cols {
                out[c * rows + r] = flat[r * cols + c]
            }
        }
        return out
    }

    private static func heWeights(count: Int, fanIn: Int, rng: inout SplitMix64) -> [Float] {
        let scale = Float((2.0 / Double(fanIn)).squareRoot())
        var values = [Float](repeating: 0, count: count)
        var i = 0
        while i < count {
            // Box-Muller: two independent uniforms -> two independent
            // standard normals. u1 must be > 0 (log(0) is undefined) —
            // nextUniform() returns [0, 1), so explicitly re-draw on the
            // rare exact-zero case rather than risk a NaN from log(0).
            var u1 = rng.nextUniform()
            while u1 <= 0 { u1 = rng.nextUniform() }
            let u2 = rng.nextUniform()
            let radius = (-2.0 * Foundation.log(u1)).squareRoot()
            let theta = 2.0 * Double.pi * u2
            let z0 = radius * Foundation.cos(theta)
            let z1 = radius * Foundation.sin(theta)

            values[i] = Float(z0) * scale
            i += 1
            if i < count {
                values[i] = Float(z1) * scale
                i += 1
            }
        }
        return values
    }

    /// Intermediate activations saved during forward, needed by backward.
    /// Field names mirror trainer.py's own cache dict keys (z1/a1/p1/mask1,
    /// z2/a2/p2/mask2, z3/a3/p3/mask3, flat) for easy cross-referencing
    /// against the Python source this is matching. A real multi-layer
    /// framework would generalize this (e.g. a tape or per-layer cache
    /// list); for this fixed, specific architecture, explicit fields are
    /// clearer and there's no abstraction to get wrong.
    public struct ForwardCache {
        var input: [Float]
        var z1: [Float]; var a1: [Float]; var p1: [Float]; var mask1: [Float]
        var z2: [Float]; var a2: [Float]; var p2: [Float]; var mask2: [Float]
        var z3: [Float]; var a3: [Float]; var p3: [Float]; var mask3: [Float]
        var flat: [Float]
        var logits: [Float]
        var probs: [Float]
    }

    public func forward(images: [Float]) -> ForwardCache {
        let c = config

        // Conv1: (N,3,32,32) -> (N,16,30,30) -> ReLU -> MaxPool -> (N,16,15,15)
        let z1 = convForward(
            x: images, W: conv1W, b: conv1B,
            n: c.n, cIn: c.cIn, cOut: c.c1, h: c.h, w: c.w, kH: c.kH, kW: c.kW, pad: 0
        )
        let a1 = relu(z1)
        let pool1 = maxPool2Forward(x: a1, n: c.n, c: c.c1, h: c.conv1OutH, w: c.conv1OutW)

        // Conv2: (N,16,15,15) -> (N,32,13,13) -> ReLU -> MaxPool -> (N,32,6,6)
        let z2 = convForward(
            x: pool1.out, W: conv2W, b: conv2B,
            n: c.n, cIn: c.c1, cOut: c.c2, h: c.pool1OutH, w: c.pool1OutW, kH: c.kH, kW: c.kW, pad: 0
        )
        let a2 = relu(z2)
        let pool2 = maxPool2Forward(x: a2, n: c.n, c: c.c2, h: c.conv2OutH, w: c.conv2OutW)

        // Conv3: (N,32,6,6) -> (N,64,4,4) -> ReLU -> MaxPool -> (N,64,2,2)
        let z3 = convForward(
            x: pool2.out, W: conv3W, b: conv3B,
            n: c.n, cIn: c.c2, cOut: c.c3, h: c.pool2OutH, w: c.pool2OutW, kH: c.kH, kW: c.kW, pad: 0
        )
        let a3 = relu(z3)
        let pool3 = maxPool2Forward(x: a3, n: c.n, c: c.c3, h: c.conv3OutH, w: c.conv3OutW)

        // pool3.out is already (N, c3, pool3OutH, pool3OutW) flattened in
        // row-major order — exactly (N, fcInputFeatures) once viewed as
        // 2D, no actual data movement needed for the "flatten."
        let flat = pool3.out
        let logits = fcForward(x: flat, W: fcW, b: fcB, n: c.n, inF: c.fcInputFeatures, outF: c.classes)
        let probs = softmax(logits, rows: c.n, cols: c.classes)

        return ForwardCache(
            input: images,
            z1: z1, a1: a1, p1: pool1.out, mask1: pool1.mask,
            z2: z2, a2: a2, p2: pool2.out, mask2: pool2.mask,
            z3: z3, a3: a3, p3: pool3.out, mask3: pool3.mask,
            flat: flat, logits: logits, probs: probs
        )
    }

    public struct Gradients {
        var conv1W: [Float]; var conv1B: [Float]
        var conv2W: [Float]; var conv2B: [Float]
        var conv3W: [Float]; var conv3B: [Float]
        var fcW: [Float]
        var fcB: [Float]
    }

    /// Mirrors trainer.py's SimpleCNNCIFAR.backward(labels) exactly, layer
    /// for layer, in the same order: softmax+cross-entropy grad -> FC
    /// backward -> unpool3 -> ReLU3 backward -> conv3 backward -> unpool2
    /// -> ReLU2 backward -> conv2 backward -> unpool1 -> ReLU1 backward ->
    /// conv1 backward. Each conv*Backward call's `x` argument is the INPUT
    /// to that conv layer (cache.input for conv1, cache.p1 for conv2,
    /// cache.p2 for conv3) — not the layer's own output — since
    /// convBackward needs the original input to compute dW via im2col.
    public func backward(cache: ForwardCache, labels: [Int]) -> Gradients {
        let c = config

        let dLogits = softmaxCrossEntropyGrad(probs: cache.probs, labels: labels, n: c.n, classes: c.classes)
        let fcGrads = fcBackward(dOut: dLogits, x: cache.flat, W: fcW, n: c.n, inF: c.fcInputFeatures, outF: c.classes)

        // fcGrads.dX is (N, fcInputFeatures) — same flat layout as p3, so
        // it can go straight into maxPool2Backward with no reshape.
        let dP3 = maxPool2Backward(dOut: fcGrads.dX, mask: cache.mask3, n: c.n, c: c.c3, h: c.conv3OutH, w: c.conv3OutW)
        let dA3 = reluBackward(dOut: dP3, preActivation: cache.z3)
        let conv3Grads = convBackward(
            dOut: dA3, x: cache.p2, W: conv3W,
            n: c.n, cIn: c.c2, cOut: c.c3, h: c.pool2OutH, w: c.pool2OutW, kH: c.kH, kW: c.kW, pad: 0
        )

        let dP2 = maxPool2Backward(dOut: conv3Grads.dX, mask: cache.mask2, n: c.n, c: c.c2, h: c.conv2OutH, w: c.conv2OutW)
        let dA2 = reluBackward(dOut: dP2, preActivation: cache.z2)
        let conv2Grads = convBackward(
            dOut: dA2, x: cache.p1, W: conv2W,
            n: c.n, cIn: c.c1, cOut: c.c2, h: c.pool1OutH, w: c.pool1OutW, kH: c.kH, kW: c.kW, pad: 0
        )

        let dP1 = maxPool2Backward(dOut: conv2Grads.dX, mask: cache.mask1, n: c.n, c: c.c1, h: c.conv1OutH, w: c.conv1OutW)
        let dA1 = reluBackward(dOut: dP1, preActivation: cache.z1)
        let conv1Grads = convBackward(
            dOut: dA1, x: cache.input, W: conv1W,
            n: c.n, cIn: c.cIn, cOut: c.c1, h: c.h, w: c.w, kH: c.kH, kW: c.kW, pad: 0
        )

        return Gradients(
            conv1W: conv1Grads.dW, conv1B: conv1Grads.db,
            conv2W: conv2Grads.dW, conv2B: conv2Grads.db,
            conv3W: conv3Grads.dW, conv3B: conv3Grads.db,
            fcW: fcGrads.dW, fcB: fcGrads.db
        )
    }

    /// SGD with momentum, matching Python's trainer.py SGD class exactly:
    ///   v = momentum * v_prev - lr * grad
    ///   w = w + v
    /// Default momentum=0.9, matching Python's default (SGD.__init__:
    /// momentum: float = 0.9). This was the actual root cause of Swift's
    /// post-aggregation local_acc collapse: plain SGD (no momentum) trains
    /// each node's model less effectively per epoch than Python's momentum
    /// SGD, producing more-scattered individual models before averaging —
    /// their weighted average then lands in a worse region of weight space
    /// than Python's, confirmed by test-real-aggregation showing below-chance
    /// accuracy after 10-model aggregation with plain SGD.
    public func applySGDStep(gradients: Gradients, learningRate: Float, momentum: Float = 0.9) {
        // Apply: v = momentum * v - lr * grad,  w = w + v
        // for each parameter tensor, updating velocity state in-place.
        func step(_ w: inout [Float], _ v: inout [Float], _ g: [Float]) {
            for i in 0..<w.count {
                v[i] = momentum * v[i] - learningRate * g[i]
                w[i] += v[i]
            }
        }

        step(&conv1W, &vConv1W, gradients.conv1W)
        step(&conv1B, &vConv1B, gradients.conv1B)
        step(&conv2W, &vConv2W, gradients.conv2W)
        step(&conv2B, &vConv2B, gradients.conv2B)
        step(&conv3W, &vConv3W, gradients.conv3W)
        step(&conv3B, &vConv3B, gradients.conv3B)

        var newFCW = fcW.values
        step(&newFCW, &vFCW, gradients.fcW)
        fcW.update(newFCW)

        step(&fcB, &vFCB, gradients.fcB)
    }

    /// Reset momentum velocity to zero — must be called after setParameters
    /// (i.e. after each gossip aggregation), matching Python's
    /// optimizer.reset() call at the start of train_round():
    /// "Aggregated weights may be very different from the previous local
    /// weights; keeping stale momentum would apply a misleading gradient step."
    public func resetMomentum() {
        for i in 0..<vConv1W.count { vConv1W[i] = 0 }
        for i in 0..<vConv1B.count { vConv1B[i] = 0 }
        for i in 0..<vConv2W.count { vConv2W[i] = 0 }
        for i in 0..<vConv2B.count { vConv2B[i] = 0 }
        for i in 0..<vConv3W.count { vConv3W[i] = 0 }
        for i in 0..<vConv3B.count { vConv3B[i] = 0 }
        for i in 0..<vFCW.count    { vFCW[i]    = 0 }
        for i in 0..<vFCB.count    { vFCB[i]    = 0 }
    }

    public func loss(cache: ForwardCache, labels: [Int]) -> Float {
        crossEntropyLoss(probs: cache.probs, labels: labels, n: config.n, classes: config.classes)
    }
}

// MARK: - FederatedModel conformance
//
// Maps SimpleCNN's eight separate parameter arrays onto FederatedModel's
// flat `[[Float32]]` ordering, which GossipTensor's wire format identifies
// by integer index (tensorID), not by name. The ordering below is FIXED
// and must never be reordered once any real experiment data exists under
// it — changing it would silently break gossip averaging between nodes
// running different code versions, since each node would put a given
// tensorID's bytes into a DIFFERENT parameter than its peers.
//
// Widened from 4 to 8 tensors as part of this file's architecture rebuild
// (1 conv layer -> 3, matching Python's SimpleCNNCIFAR exactly) — safe to
// change cleanly here since no real cross-version experiment data existed
// under the old 4-tensor ordering yet (the only prior data was the
// architecturally-mismatched comparison that motivated this rebuild).
//
// FIXED ORDER (tensorID 0-7):
//   0: conv1W   (c1 * cIn * kH * kW elements)
//   1: conv1B   (c1 elements)
//   2: conv2W   (c2 * c1 * kH * kW elements)
//   3: conv2B   (c2 elements)
//   4: conv3W   (c3 * c2 * kH * kW elements)
//   5: conv3B   (c3 elements)
//   6: fcW      (fcInputFeatures * classes elements)
//   7: fcB      (classes elements)
extension SimpleCNN: FederatedModel {
    public var parameters: [[Float32]] {
        [conv1W, conv1B, conv2W, conv2B, conv3W, conv3B, fcW.values, fcB]
    }

    public func setParameters(_ parameters: [[Float32]]) {
        precondition(parameters.count == 8, "SimpleCNN.setParameters: expected 8 parameter tensors in fixed order, got \(parameters.count)")
        precondition(parameters[0].count == conv1W.count, "SimpleCNN.setParameters: conv1W size mismatch — expected \(conv1W.count), got \(parameters[0].count)")
        precondition(parameters[1].count == conv1B.count, "SimpleCNN.setParameters: conv1B size mismatch — expected \(conv1B.count), got \(parameters[1].count)")
        precondition(parameters[2].count == conv2W.count, "SimpleCNN.setParameters: conv2W size mismatch — expected \(conv2W.count), got \(parameters[2].count)")
        precondition(parameters[3].count == conv2B.count, "SimpleCNN.setParameters: conv2B size mismatch — expected \(conv2B.count), got \(parameters[3].count)")
        precondition(parameters[4].count == conv3W.count, "SimpleCNN.setParameters: conv3W size mismatch — expected \(conv3W.count), got \(parameters[4].count)")
        precondition(parameters[5].count == conv3B.count, "SimpleCNN.setParameters: conv3B size mismatch — expected \(conv3B.count), got \(parameters[5].count)")
        precondition(parameters[6].count == fcW.values.count, "SimpleCNN.setParameters: fcW size mismatch — expected \(fcW.values.count), got \(parameters[6].count)")
        precondition(parameters[7].count == fcB.count, "SimpleCNN.setParameters: fcB size mismatch — expected \(fcB.count), got \(parameters[7].count)")

        conv1W = parameters[0]
        conv1B = parameters[1]
        conv2W = parameters[2]
        conv2B = parameters[3]
        conv3W = parameters[4]
        conv3B = parameters[5]
        fcW.update(parameters[6])
        fcB = parameters[7]
        // Reset momentum velocity after adopting new weights — matching
        // Python's optimizer.reset() called at the start of train_round()
        // after gossip aggregation sets new weights.
        resetMomentum()
    }

    /// Runs exactly one local epoch over `shard`, delegating to an
    /// internal CNNTrainer rather than reimplementing batching/shuffling —
    /// CNNTrainer is already the verified, Python-shuffle-matching
    /// implementation (see CNNTrainer.swift's doc comments for why that
    /// shuffle behavior matters); duplicating that logic here would create
    /// two places that need to independently agree on batching semantics.
    ///
    /// A FRESH CNNTrainer is constructed on every call, using THIS call's
    /// `seed`/`learningRate` — deliberately not cached/reused across calls
    /// the way an earlier version of this method did. That earlier version
    /// built one CNNTrainer lazily on first use and kept reusing it, which
    /// meant the shuffle seed baked into its config never changed across
    /// federated rounds — every round reshuffled identically, silently
    /// defeating the entire point of reshuffling more than once. Now that
    /// the caller controls seed/learningRate explicitly per call (see
    /// FederatedModel's doc comment for why), there's no fixed config left
    /// to cache — constructing a CNNTrainer is cheap (it just wraps a
    /// config + holds a reference to `self`), so paying that cost once per
    /// epoch instead of once per model lifetime is not a meaningful cost.
    ///
    /// `epochsPerRound` is always 1 here — "how many epochs per federated
    /// round" is an orchestration-layer decision (how many times THIS
    /// method gets called), not something the model decides for itself.
    ///
    /// IMPLEMENTS the protocol's 4-parameter trainEpoch directly (not via
    /// the 3-parameter extension default) — `onBatchProgress` is wired
    /// straight through to CNNTrainer.trainRound's existing
    /// onBatchProgress parameter, which already existed and was already
    /// used by RADFLMain.swift's `train-cnn` command for exactly this
    /// reason. This method previously called the 3-parameter
    /// trainEpoch(shard:seed:learningRate:) signature with no progress
    /// callback at all, which silently reintroduced the same
    /// "indistinguishable from a hang" problem train-cnn was built to fix
    /// — confirmed by a real run-round test session where all 10 nodes
    /// produced zero output for several minutes during what was actually
    /// just normal (if slow) training.
    public func trainEpoch(
        shard: DataShard, seed: UInt64, learningRate: Float,
        onBatchProgress: (@Sendable (_ batchIndex: Int, _ totalBatches: Int) -> Void)?
    ) throws -> TrainEpochResult {
        let trainerConfig = CNNTrainerConfig(
            batchSize: config.n, epochsPerRound: 1,
            learningRate: learningRate, seed: seed
        )
        let trainer = CNNTrainer(model: self, config: trainerConfig)

        let results = try trainer.trainRound(shard: shard, onBatchProgress: onBatchProgress)
        guard let result = results.first else {
            return TrainEpochResult(meanLoss: .nan, trainAcc: 0, fwdS: 0, bwdS: 0, optS: 0, shufS: 0)
        }
        return TrainEpochResult(
            meanLoss: Double(result.meanLoss),
            trainAcc: result.trainAcc,
            fwdS: result.fwdS, bwdS: result.bwdS, optS: result.optS, shufS: result.shufS
        )
    }

    /// Implements `FederatedModel.evaluate(shard:)` — evaluates on any
    /// DataShard (test or local), computing both accuracy and loss, timing
    /// itself. Mirrors Python's trainer.evaluate() return dict (accuracy,
    /// loss, num_samples, eval_s). Called twice per round in
    /// RoundOrchestrator: once with testShard (→ test_acc/test_loss) and
    /// once with trainShard after aggregation (→ local_acc/local_loss).
    public func evaluate(shard: DataShard) throws -> EvalResult {
        guard shard.imageShape.count == 4,
              shard.imageShape[1] == config.cIn,
              shard.imageShape[2] == config.h,
              shard.imageShape[3] == config.w else {
            throw CNNTrainerError.modelInputShapeMismatch(
                modelExpects: (config.cIn, config.h, config.w),
                shardHas: shard.imageShape
            )
        }

        let batchCount = shard.sampleCount / config.n
        guard batchCount > 0 else {
            throw CNNTrainerError.shardTooSmallForBatch(
                shardSamples: shard.sampleCount, batchSize: config.n
            )
        }

        let evalStart = Date()
        var correct = 0
        var totalLoss: Float = 0
        var total = 0
        let perSample = config.cIn * config.h * config.w

        for batchIndex in 0..<batchCount {
            let batchStart = batchIndex * config.n
            let sampleIndices = Array(batchStart..<(batchStart + config.n))

            var images = [Float](repeating: 0, count: config.n * perSample)
            var labels = [Int](repeating: 0, count: config.n)

            shard.withImageBuffer { buffer in
                for (pos, idx) in sampleIndices.enumerated() {
                    let src = idx * perSample
                    let dst = pos * perSample
                    for o in 0..<perSample { images[dst + o] = buffer[src + o] }
                }
            }
            shard.withLabelBuffer { buffer in
                for (pos, idx) in sampleIndices.enumerated() {
                    labels[pos] = Int(buffer[idx])
                }
            }

            let cache = forward(images: images)

            for i in 0..<config.n {
                let predicted = argmax(cache.probs, row: i, cols: config.classes)
                if predicted == labels[i] { correct += 1 }
                total += 1
            }

            totalLoss += crossEntropyLoss(
                probs: cache.probs, labels: labels,
                n: config.n, classes: config.classes
            )
        }

        let evalS = Date().timeIntervalSince(evalStart)
        let accuracy = total > 0 ? Double(correct) / Double(total) : 0.0
        let meanLoss = batchCount > 0 ? Double(totalLoss) / Double(batchCount) : .nan

        return EvalResult(
            accuracy: accuracy,
            loss: meanLoss,
            numSamples: total,
            evalS: evalS
        )
    }
}

/// Small, fast, seedable PRNG — used only to generate reproducible
/// synthetic test data, NOT for anything that needs cryptographic quality.
/// Avoids pulling in any external dependency just for "give me some
/// deterministic-but-varied test floats."
///
/// `public`: RADFLMain.swift's `test-cnn` command constructs its own
/// SplitMix64 instance directly (a separate RNG stream from the one
/// SimpleCNN uses internally for weight init), and that file lives in a
/// different module (the RADFL executable target, vs. this file's
/// RADFLCore library target) — internal visibility doesn't cross that
/// boundary, only public does.
public struct SplitMix64 {
    private var state: UInt64

    public init(seed: UInt64) {
        self.state = seed
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    /// Bias-free random integer in [0, bound) via rejection sampling —
    /// NOT plain `next() % bound`, which introduces modulo bias whenever
    /// `bound` doesn't evenly divide 2^64 (true for almost every bound that
    /// matters here, e.g. shard sizes that aren't a power of two). The bias
    /// from naive modulo would be astronomically small for shard-sized
    /// ranges, but a shuffle implementation correctness matters in its own
    /// right, not just "small enough to not care about."
    public mutating func nextBounded(_ bound: UInt64) -> UInt64 {
        precondition(bound > 0, "nextBounded: bound must be positive")
        // Reject values in the trailing partial range that would make some
        // outputs more likely than others, then reduce what's left.
        let limit = UInt64.max - (UInt64.max % bound)
        var value = next()
        while value >= limit {
            value = next()
        }
        return value % bound
    }

    /// Fisher-Yates shuffle of the indices [0, count), using this RNG
    /// stream. Mirrors what `numpy.random.Generator.permutation(n)` does
    /// (the Python baseline's `self._rng.permutation(self.num_samples)` in
    /// trainer.py's per-epoch shuffle) — same algorithm class (unbiased
    /// uniform-random permutation), though NOT guaranteed to reproduce
    /// numpy's EXACT permutation for a given seed, since the two use
    /// different underlying PRNG algorithms (SplitMix64 here vs. NumPy's
    /// PCG64 by default). Matching numpy's exact bit-for-bit permutation
    /// would require reimplementing PCG64, which isn't necessary for this
    /// purpose — what matters for a fair comparison is that BOTH
    /// implementations shuffle every epoch with a seeded, reproducible RNG,
    /// not that they shuffle into the identical order as each other.
    public mutating func shuffledIndices(count: Int) -> [Int] {
        guard count > 1 else { return count == 1 ? [0] : [] }
        var indices = Array(0..<count)
        for i in stride(from: count - 1, to: 0, by: -1) {
            let j = Int(nextBounded(UInt64(i + 1)))
            indices.swapAt(i, j)
        }
        return indices
    }

    /// Uniform double in [0, 1).
    public mutating func nextUniform() -> Double {
        Double(next() >> 11) * (1.0 / Double(1 << 53))
    }
}

