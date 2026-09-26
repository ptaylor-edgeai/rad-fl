// FederatedModel.swift
//
// Protocol seam for the model/training layer. NOT implemented in this session —
// this defines the interface the orchestration and aggregation layers need so
// they can be built now against a stable contract, with the actual CNN
// (built on the existing Tensor.swift compute kernels) plugged in separately.
//
// Implementer's responsibility:
//   - `parameters` / `setParameters` expose/consume the flat per-tensor weight
//     arrays that GossipTensor wraps for transmission — tensor ordering must
//     be stable across all nodes (e.g. fixed layer-iteration order) since the
//     wire format identifies tensors by integer ID, not name.
//   - `trainEpoch` runs one local epoch over this node's shard and returns the
//     training loss for that epoch (for the train_loss telemetry field).
//   - `evaluate` runs the held-out test set and returns accuracy (test_acc).
//
// trainEpoch's `seed` and `learningRate` parameters: the CALLER (the
// federated round orchestration loop, not the model) decides these per
// call, rather than a model hiding its own internal randomness/learning
// rate policy. This matters for reproducibility specifically — a research
// codebase should be able to name and reconstruct "what seed did round N
// use" externally, not have it be an emergent property of whatever
// internal RNG state a model implementation happens to carry between
// calls. The straightforward alternative (a model keeps one persistent
// internal RNG stream across calls, advancing it each time on its own) was
// considered and rejected for exactly this reason — it would have fixed
// the "every round reshuffles identically" bug without giving the caller
// any actual visibility or control over the result.

import Foundation

public protocol FederatedModel: Sendable {
    /// Flat weight tensors in stable, fixed order. Index in this array == tensorID
    /// used on the wire (GossipTensor.tensorID). All nodes must agree on this
    /// ordering for gossip averaging to be meaningful.
    var parameters: [[Float32]] { get }

    /// Replace this model's parameters in place (e.g. after gossip averaging).
    mutating func setParameters(_ parameters: [[Float32]])

    /// Train one local epoch over the node's current shard.
    ///
    /// - Parameters:
    ///   - shard: this node's local training data for this round.
    ///   - seed: drives this epoch's data shuffle order. The caller should
    ///     vary this across rounds (e.g. derived from a base experiment
    ///     seed mixed with the round number) — reusing the same seed every
    ///     round would reshuffle identically every time, defeating the
    ///     purpose of reshuffling across multiple rounds.
    ///   - learningRate: this epoch's SGD learning rate. Passed per-call
    ///     rather than fixed at model-construction time so the caller can
    ///     implement a decay schedule across rounds if desired; a model
    ///     that doesn't want to support schedules can simply ignore
    ///     variation and have its caller always pass the same value.
    ///   - onBatchProgress: OPTIONAL. If provided, a conformer SHOULD call
    ///     this periodically during the epoch with (batchIndex,
    ///     totalBatches), purely for caller-side progress reporting. This
    ///     exists specifically because a large shard's single epoch can
    ///     take long enough to be indistinguishable from a hang with zero
    ///     intermediate output (this was a real, confusing problem
    ///     encountered with this exact training path before this
    ///     parameter was added) — a conformer that ignores this parameter
    ///     entirely is still protocol-conformant (default is a no-op, via
    ///     the extension below), just less observable mid-epoch.
    ///
    /// Returns a `TrainEpochResult`, not a plain Double — this was
    /// deliberately widened from an earlier plain-loss-only return type
    /// specifically to carry the same per-phase timing granularity as the
    /// Python baseline (fwd_s/bwd_s/opt_s/shuf_s — see run_experiment.py's
    /// real CSV_COLUMNS), so a Swift-vs-Python timing comparison can be
    /// made at matching phase boundaries rather than Swift only reporting
    /// one undifferentiated "training took N seconds" figure.
    mutating func trainEpoch(
        shard: DataShard, seed: UInt64, learningRate: Float,
        onBatchProgress: (@Sendable (_ batchIndex: Int, _ totalBatches: Int) -> Void)?
    ) throws -> TrainEpochResult

    /// Evaluate on a DataShard — either the shared test set or this node's
    /// local training shard. Returns an EvalResult carrying accuracy, loss,
    /// sample count, and eval timing. Mirrors Python's trainer.evaluate()
    /// which takes a split parameter ("test" or "local"); here the caller
    /// simply passes the appropriate DataShard rather than a string, letting
    /// Swift's type system express this more naturally without a stringly-
    /// typed split parameter. Called twice per round in RoundOrchestrator:
    /// once with testShard (→ test_acc/test_loss) and once with trainShard
    /// (→ local_acc/local_loss in Python's terminology, meaning "how well
    /// does the post-aggregation model fit this node's own local data").
    func evaluate(shard: DataShard) throws -> EvalResult
}

/// Result of one `FederatedModel.trainEpoch` call — mean loss plus
/// per-phase timing, matching the Python baseline's fwd_s/bwd_s/opt_s/
/// Result of one `FederatedModel.trainEpoch` call — mean loss, train
/// accuracy, and per-phase timing. Matches Python's train_round() return
/// dict fields exactly (see trainer.py: loss, train_acc, fwd_s, bwd_s,
/// opt_s, shuf_s). `trainAcc` is computed after all batches complete on
/// the fully-trained (pre-aggregation) weights — Python's trainer.py
/// computes it the same way ("Quick training accuracy (full shard, no
/// shuffle)" at the end of train_round, before returning to run_experiment
/// which then does gossip aggregation).
public struct TrainEpochResult: Sendable {
    public let meanLoss: Double

    /// Accuracy on this node's own shard using PRE-aggregation weights, or nil
    /// when the pass was skipped.
    ///
    /// Optional rather than 0, matching how the evaluation fields already
    /// behave: a model scoring zero and a model that was never scored are
    /// different facts, and 0.0 is a value a genuinely broken model produces.
    /// The CSV writes nil as an empty field, which the analysis reads as
    /// missing rather than as a real accuracy.
    public let trainAcc: Double?

    /// Seconds spent computing `trainAcc`; 0 when it was skipped.
    ///
    /// Zero rather than nil, because no pass ran and zero seconds is what was
    /// measured — the same distinction the -1 sentinel draws elsewhere between
    /// "not instrumented" and "measured as zero".
    ///
    /// This closes an unattributed remainder present since the project began.
    /// The pass is a full sequential forward pass over the whole local shard —
    /// the same work as an evaluation — and was never timed, so it appeared
    /// only as an ~11s gap between round wall-clock and the sum of the logged
    /// phases, about 12% of an 88-second round. F-001 identified it by dividing
    /// that gap by the per-sample evaluation rate, which implied 4,963 samples
    /// against a shard of 5,000; this measures it directly.
    public let trainAccS: Double

    public let fwdS: Double
    public let bwdS: Double
    public let optS: Double
    public let shufS: Double

    public init(meanLoss: Double, trainAcc: Double?, trainAccS: Double = 0,
                fwdS: Double, bwdS: Double, optS: Double, shufS: Double) {
        self.meanLoss = meanLoss
        self.trainAcc = trainAcc
        self.trainAccS = trainAccS
        self.fwdS = fwdS
        self.bwdS = bwdS
        self.optS = optS
        self.shufS = shufS
    }
}

/// Result of one `FederatedModel.evaluate(shard:)` call — mirrors Python's
/// trainer.evaluate() return dict exactly (accuracy, loss, num_samples,
/// eval_s). Used for both test-set evaluation AND local-shard evaluation
/// (matching Python's split="test" and split="local" respectively) — same
/// return type for both, caller chooses which DataShard to pass.
public struct EvalResult: Sendable {
    public let accuracy: Double
    public let loss: Double
    public let numSamples: Int
    public let evalS: Double   // wall-clock time for this evaluate() call, matching Python's eval_s

    public init(accuracy: Double, loss: Double, numSamples: Int, evalS: Double) {
        self.accuracy = accuracy
        self.loss = loss
        self.numSamples = numSamples
        self.evalS = evalS
    }
}

public extension FederatedModel {
    /// Default overload omitting `onBatchProgress` — purely additive
    /// convenience so existing call sites written against the earlier
    /// trainEpoch(shard:seed:learningRate:) signature (no progress
    /// callback) still compile without modification.
    mutating func trainEpoch(shard: DataShard, seed: UInt64, learningRate: Float) throws -> TrainEpochResult {
        try trainEpoch(shard: shard, seed: seed, learningRate: learningRate, onBatchProgress: nil)
    }
}

/// A single node's training shard or the shared test set: images + labels,
/// loaded from the same per-condition .npy files Python's
/// extract_cifar10_shards.py produces (node_NN_train_X.npy /
/// node_NN_train_y.npy / test_X.npy / test_y.npy).
///
/// DESIGN — contiguous storage with explicit shape, not [[Float32]]:
/// images are stored as ONE flat Float32 buffer plus a `shape` array
/// ([N, C, H, W]), mirroring how NumPy itself represents the array — a
/// single contiguous block, not N separate per-image allocations. This
/// isn't an optimization aimed at making Swift look better than Python: it
/// matches what Python's baseline ALREADY does (a NumPy array IS one
/// contiguous buffer with a shape attribute), so this is the fair
/// like-for-like representation rather than a structural handicap on one
/// side. The shape being a plain [Int] (not hardcoded "channels/height/width"
/// fields) also means this isn't CIFAR-10-specific — MNIST's (N, 1, 28, 28)
/// shards fit the same struct with no changes.
///
/// DESIGN — backing storage and the mmap decision: `images` is backed by
/// `NPYArrayData`, which can be either eagerly read or memory-mapped
/// depending on how it was loaded (see `CIFAR10Shard.load`). Memory-mapping
/// matters specifically because Python's real Pi deployment path
/// (`trainer.py`'s `_load_cifar10_arrays`, strategy 1) uses
/// `np.load(path, mmap_mode="r")` for exactly this data, specifically to
/// stay within the Pi Zero 2W's 512MB RAM budget. An eager Swift load while
/// Python mmaps would make any RSS/memory comparison between the two
/// implementations apples-to-oranges for reasons that have nothing to do
/// with Swift or Python themselves — just a different loading strategy on
/// one side. Matching Python's approach here keeps the comparison fair.
public struct DataShard: Sendable {
    /// [N, C, H, W] for image data (e.g. [8758, 3, 32, 32] for a CIFAR-10
    /// shard) — read directly from the .npy header, not assumed.
    public let imageShape: [Int]
    private let imageData: NPYArrayData

    public let labelCount: Int
    private let labelData: NPYArrayData

    init(imageShape: [Int], imageData: NPYArrayData, labelCount: Int, labelData: NPYArrayData) {
        self.imageShape = imageShape
        self.imageData = imageData
        self.labelCount = labelCount
        self.labelData = labelData
    }

    /// Number of samples (first dimension of imageShape) — the batching/
    /// training-loop-facing sample count, distinct from `imageShape`'s full
    /// dimension list.
    public var sampleCount: Int {
        imageShape.first ?? 0
    }

    /// Per-sample element count (C*H*W) — how many Float32 values make up
    /// one image, for slicing a flat sample index into a byte range.
    public var perSampleElementCount: Int {
        imageShape.dropFirst().reduce(1, *)
    }

    /// Read-only access to the full image buffer, without copying. See
    /// `NPYArrayData.withFloat32Buffer`'s doc comment for why this matters
    /// when the shard was loaded with `.mapped` — only the pages actually
    /// touched inside `body` get faulted into RSS.
    public func withImageBuffer<T>(_ body: (UnsafeBufferPointer<Float32>) throws -> T) rethrows -> T {
        try imageData.withFloat32Buffer(body)
    }

    /// Read-only access to the full label buffer, without copying.
    public func withLabelBuffer<T>(_ body: (UnsafeBufferPointer<Int64>) throws -> T) rethrows -> T {
        try labelData.withInt64Buffer(body)
    }
}

public enum CIFAR10ShardError: Error, CustomStringConvertible {
    case sampleCountMismatch(imagesN: Int, labelsN: Int)

    public var description: String {
        switch self {
        case .sampleCountMismatch(let imagesN, let labelsN):
            return "Image shard has \(imagesN) samples but label shard has \(labelsN) — these must match"
        }
    }
}

/// Loads a DataShard from the per-condition shard files
/// extract_cifar10_shards.py produces. Not CIFAR-10-specific in any deep
/// way despite the name (the shape comes from the file's own header) — named
/// for what it's actually used for today, per project scope.
public enum CIFAR10Shard {
    /// Loads one node's training shard: `node_<NN>_train_X.npy` +
    /// `node_<NN>_train_y.npy` from `directory`, where `<NN>` is
    /// `nodeIndex` zero-padded to 2 digits (matching Python's
    /// `node_{i:02d}_train_X.npy` naming exactly).
    public static func loadTrainShard(
        directory: URL,
        nodeIndex: Int,
        strategy: NPYReader.LoadStrategy = .mapped
    ) throws -> DataShard {
        let suffix = String(format: "%02d", nodeIndex)
        let imagesURL = directory.appendingPathComponent("node_\(suffix)_train_X.npy")
        let labelsURL = directory.appendingPathComponent("node_\(suffix)_train_y.npy")
        return try load(imagesURL: imagesURL, labelsURL: labelsURL, strategy: strategy)
    }

    /// Loads the shared test set: `test_X.npy` + `test_y.npy` from
    /// `directory` — identical content across every node and every alpha
    /// condition's subdirectory (extract_cifar10_shards.py writes the same
    /// test set into each condition's output directory; see that script's
    /// comments on the deliberate ~118MB-per-condition duplication
    /// trade-off).
    public static func loadTestSet(
        directory: URL,
        strategy: NPYReader.LoadStrategy = .mapped
    ) throws -> DataShard {
        let imagesURL = directory.appendingPathComponent("test_X.npy")
        let labelsURL = directory.appendingPathComponent("test_y.npy")
        return try load(imagesURL: imagesURL, labelsURL: labelsURL, strategy: strategy)
    }

    private static func load(imagesURL: URL, labelsURL: URL, strategy: NPYReader.LoadStrategy) throws -> DataShard {
        let (imageHeader, imageData) = try NPYReader.open(imagesURL, strategy: strategy, expectedDescr: "<f4")
        // Labels are tiny relative to images (N int64s vs N*C*H*W float32s) —
        // always eager-load them regardless of `strategy`. mmap's per-page
        // fault overhead isn't worth it for an array this small, and the
        // memory-fairness argument for mmap is specifically about the large
        // image data, not labels.
        let (labelHeader, labelData) = try NPYReader.open(labelsURL, strategy: .eager, expectedDescr: "<i8")

        let imagesN = imageHeader.shape.first ?? 0
        let labelsN = labelHeader.shape.first ?? 0
        guard imagesN == labelsN else {
            throw CIFAR10ShardError.sampleCountMismatch(imagesN: imagesN, labelsN: labelsN)
        }

        return DataShard(
            imageShape: imageHeader.shape,
            imageData: imageData,
            labelCount: labelsN,
            labelData: labelData
        )
    }
}


