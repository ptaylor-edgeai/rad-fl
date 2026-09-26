// CNNTrainer.swift
//
// Trains a SimpleCNN on a real DataShard (CIFAR-10 shards loaded via
// NPYReader/CIFAR10Shard), as opposed to SimpleCNN's own synthetic-data
// test harness (see RADFLMain.swift's `test-cnn` command, and
// SimpleCNN.swift's doc comments) which exists purely to verify
// Tensor.swift's wiring in isolation from the data pipeline.
//
// SHUFFLING: matches the Python baseline's actual behavior — trainer.py's
// train_round() reshuffles the ENTIRE local shard every epoch via a
// seeded RNG (`self._rng.permutation(self.num_samples)`), not a sequential
// walk through stored order. This matters beyond just style-matching:
//   1. Dirichlet-partitioned shards are NOT randomly ordered in storage —
//      we directly confirmed this earlier (node 0's labels cluster by
//      class at the start of the array). Sequential-order training over a
//      class-clustered shard would be a real methodological difference
//      from Python, not just a cosmetic one.
//   2. Python's project history shows this shuffle step has a measured,
//      non-trivial performance cost on the Pi (the `shuf_s` instrumentation
//      column exists specifically because of this). If Swift skipped
//      shuffling, the resource comparison between the two runtimes would
//      be comparing different workloads, not just different language
//      implementations of the same workload.
// NOT reproduced: Python's exact shuffle ORDER for a given seed (NumPy's
// PCG64 vs. this file's SplitMix64 are different algorithms) — see
// SplitMix64.shuffledIndices' doc comment. What's preserved is the
// methodological property (seeded, reproducible, per-epoch reshuffle), not
// bit-for-bit cross-language identical output.

import Foundation

public struct CNNTrainerConfig: Sendable {
    public let batchSize: Int
    public let epochsPerRound: Int
    public let learningRate: Float
    public let seed: UInt64

    /// Whether to run the post-training accuracy pass. Default true, which is
    /// the original behaviour.
    ///
    /// That pass is a full sequential forward pass over the entire local shard
    /// — the same work as an evaluation, and measured at ~11s on a 5,000-sample
    /// shard, about 12% of an 88-second round. It was never timed, so it
    /// appeared only as an unattributed remainder between round wall-clock and
    /// the sum of the logged phases; F-001 identified it by dividing that
    /// remainder by the per-sample evaluation rate, which implied 4,963 samples
    /// against a shard of 5,000.
    ///
    /// It is a diagnostic, not a result. `local_acc` measures nearly the same
    /// quantity — accuracy on this node's own shard — differing only in using
    /// post-aggregation rather than pre-aggregation weights. Skipping it costs
    /// little analytically and is the largest single item left in the round
    /// once evaluation cadence has been reduced.
    public let computeTrainAcc: Bool

    public init(batchSize: Int, epochsPerRound: Int, learningRate: Float, seed: UInt64,
                computeTrainAcc: Bool = true) {
        self.batchSize = batchSize
        self.epochsPerRound = epochsPerRound
        self.learningRate = learningRate
        self.seed = seed
        self.computeTrainAcc = computeTrainAcc
    }
}

public struct EpochResult: Sendable {
    public let epoch: Int
    public let meanLoss: Float
    /// nil when the post-training accuracy pass was skipped.
    ///
    /// Optional rather than 0, for the same reason the evaluation fields are:
    /// a model scoring zero and a model that was never scored are different
    /// facts, and 0.0 is a value a genuinely broken model could produce. The
    /// CSV writes nil as an empty field, which the analysis reads as missing.
    public let trainAcc: Double?
    /// Seconds spent on the post-training accuracy pass; 0 when skipped.
    ///
    /// Zero rather than nil here, because no pass ran and zero seconds is what
    /// was measured — the same distinction the -1 sentinel draws elsewhere
    /// between "not instrumented" and "measured as zero". Timing it also closes
    /// the unattributed remainder that has appeared in every phase breakdown
    /// since the project began.
    public let trainAccS: Double
    public let batchCount: Int
    public let fwdS: Double
    public let bwdS: Double
    public let optS: Double
    public let shufS: Double
}

public enum CNNTrainerError: Error, CustomStringConvertible {
    case shardTooSmallForBatch(shardSamples: Int, batchSize: Int)
    case modelInputShapeMismatch(modelExpects: (cIn: Int, h: Int, w: Int), shardHas: [Int])

    public var description: String {
        switch self {
        case .shardTooSmallForBatch(let shardSamples, let batchSize):
            return "CNNTrainer: shard has \(shardSamples) samples, smaller than batch size \(batchSize) — cannot form even one full batch"
        case .modelInputShapeMismatch(let expects, let shardHas):
            return "CNNTrainer: model expects input (C,H,W) = (\(expects.cIn),\(expects.h),\(expects.w)) but shard's imageShape is \(shardHas) — these must agree (after the leading N dimension) for batches to be sliced correctly"
        }
    }
}

/// Trains a SimpleCNN on a DataShard for one or more local epochs — this is
/// the per-round local-training step a federated learning node performs
/// before gossiping its updated weights. Mirrors Python trainer.py's
/// train_round(), one epoch loop with per-epoch reshuffling, though the
/// gossip/round-orchestration layer itself is built separately (this file
/// is the local-training piece only).
public final class CNNTrainer {
    private let model: SimpleCNN
    private let config: CNNTrainerConfig
    private var rng: SplitMix64

    public init(model: SimpleCNN, config: CNNTrainerConfig) {
        self.model = model
        self.config = config
        self.rng = SplitMix64(seed: config.seed)
    }

    /// Runs `config.epochsPerRound` epochs of training over `shard`,
    /// reshuffling the shard's sample order fresh each epoch. Returns one
    /// EpochResult per epoch (mean loss across that epoch's batches) so a
    /// caller can log per-epoch progress, matching Python's per-epoch
    /// logging granularity.
    ///
    /// Any trailing samples that don't fill a complete batch are dropped
    /// for that epoch (matches the common "drop last partial batch"
    /// convention) — this means a shard whose sample count isn't an exact
    /// multiple of batchSize trains on slightly fewer than its full sample
    /// count per epoch, recovered across epochs since the shuffle changes
    /// which samples fall in the dropped remainder each time.
    /// `onEpochComplete`, if provided, fires immediately after each epoch
    /// finishes (so a caller can print/log progress as training runs,
    /// rather than waiting for ALL epochs to finish before seeing
    /// anything — with a large shard and many batches per epoch, that wait
    /// can be long enough to look indistinguishable from a hang).
    /// `onBatchProgress`, if provided, fires periodically WITHIN an epoch
    /// (every `batchProgressInterval` batches) with (batchIndex,
    /// totalBatches) — for visible progress even mid-epoch on a slow run.
    public func trainRound(
        shard: DataShard,
        onEpochComplete: ((EpochResult) -> Void)? = nil,
        onBatchProgress: ((_ batchIndex: Int, _ totalBatches: Int) -> Void)? = nil,
        batchProgressInterval: Int = 50
    ) throws -> [EpochResult] {
        let cfg = model.config
        guard shard.imageShape.count == 4,
              shard.imageShape[1] == cfg.cIn, shard.imageShape[2] == cfg.h, shard.imageShape[3] == cfg.w else {
            throw CNNTrainerError.modelInputShapeMismatch(
                modelExpects: (cfg.cIn, cfg.h, cfg.w),
                shardHas: shard.imageShape
            )
        }
        guard shard.sampleCount >= config.batchSize else {
            throw CNNTrainerError.shardTooSmallForBatch(shardSamples: shard.sampleCount, batchSize: config.batchSize)
        }

        var results: [EpochResult] = []
        for epoch in 1...config.epochsPerRound {
            let result = try trainOneEpoch(
                shard: shard, epoch: epoch,
                onBatchProgress: onBatchProgress, batchProgressInterval: batchProgressInterval
            )
            results.append(result)
            onEpochComplete?(result)
        }
        return results
    }

    private func trainOneEpoch(
        shard: DataShard, epoch: Int,
        onBatchProgress: ((_ batchIndex: Int, _ totalBatches: Int) -> Void)?,
        batchProgressInterval: Int
    ) throws -> EpochResult {
        let perm = rng.shuffledIndices(count: shard.sampleCount)
        let batchCount = shard.sampleCount / config.batchSize  // drop trailing partial batch

        var totalLoss: Float = 0
        var totalShufS: Double = 0
        var totalFwdS: Double = 0
        var totalBwdS: Double = 0
        var totalOptS: Double = 0

        for batchIndex in 0..<batchCount {
            let batchStart = batchIndex * config.batchSize
            let batchSampleIndices = Array(perm[batchStart..<(batchStart + config.batchSize)])

            let shufStart = Date()
            let (images, labels) = extractBatch(shard: shard, sampleIndices: batchSampleIndices)
            totalShufS += Date().timeIntervalSince(shufStart)

            let fwdStart = Date()
            let cache = model.forward(images: images)
            totalFwdS += Date().timeIntervalSince(fwdStart)

            let lossValue = model.loss(cache: cache, labels: labels)
            totalLoss += lossValue

            let bwdStart = Date()
            let gradients = model.backward(cache: cache, labels: labels)
            totalBwdS += Date().timeIntervalSince(bwdStart)

            let optStart = Date()
            model.applySGDStep(gradients: gradients, learningRate: config.learningRate)
            totalOptS += Date().timeIntervalSince(optStart)

            if let onBatchProgress, batchIndex % batchProgressInterval == 0 {
                onBatchProgress(batchIndex, batchCount)
            }
        }

        let meanLoss = batchCount > 0 ? totalLoss / Float(batchCount) : Float.nan

        // Post-training accuracy pass: full shard, sequential, no shuffle —
        // matching Python's trainer.py exactly: "Quick training accuracy
        // (full shard, no shuffle)" computed at the end of train_round,
        // before returning to run_experiment.py (which then does gossip
        // aggregation). This uses the pre-aggregation weights — the model
        // as it is after this node's own local training, before any
        // cross-peer averaging. The per-batch inference here IS NOT timed
        // as part of fwdS/bwdS/optS — it's a pure evaluation pass with no
        // gradient computation, analogous to Python's _accuracy_batched()
        // being separate from training timing. It's also not included in
        // trainTimeS (that clock stopped above), which is correct.
        //
        // NOW TIMED, and skippable. Previously neither: the pass ran
        // unconditionally and its cost showed up only as the gap between round
        // wall-clock and the sum of the logged phases.
        var trainAcc: Double? = nil
        var trainAccS: Double = 0
        if config.computeTrainAcc {
            let trainAccStart = Date()
            let trainAccBatchCount = shard.sampleCount / config.batchSize
            var correct = 0
            var total = 0
            for batchIndex in 0..<trainAccBatchCount {
                let batchStart = batchIndex * config.batchSize
                let sequentialIndices = Array(batchStart..<(batchStart + config.batchSize))
                let (images, labels) = extractBatch(shard: shard, sampleIndices: sequentialIndices)
                let cache = model.forward(images: images)
                for i in 0..<config.batchSize {
                    let predicted = argmax(cache.probs, row: i, cols: model.config.classes)
                    if predicted == labels[i] { correct += 1 }
                    total += 1
                }
            }
            trainAcc = total > 0 ? Double(correct) / Double(total) : 0.0
            trainAccS = Date().timeIntervalSince(trainAccStart)
        }

        return EpochResult(
            epoch: epoch, meanLoss: meanLoss, trainAcc: trainAcc,
            trainAccS: trainAccS, batchCount: batchCount,
            fwdS: totalFwdS, bwdS: totalBwdS, optS: totalOptS, shufS: totalShufS
        )
    }

    /// Copies out exactly the samples named by `sampleIndices` (in that
    /// order) from `shard`'s underlying buffers into fresh, batch-sized
    /// [Float]/[Int] arrays. This IS a real copy, deliberately — unlike
    /// DataShard's own mmap-preserving read access (see NPYArrayData's doc
    /// comments), a training batch is genuinely short-lived working data
    /// that SimpleCNN.forward needs as a plain contiguous buffer; there's
    /// no mmap benefit being defeated here, since the whole point of
    /// shuffled batching is non-contiguous access into the shard anyway —
    /// gathering scattered samples into one contiguous batch is exactly
    /// what has to happen for that access pattern, copy or no copy.
    private func extractBatch(shard: DataShard, sampleIndices: [Int]) -> (images: [Float], labels: [Int]) {
        let perSample = shard.perSampleElementCount
        var images = [Float](repeating: 0, count: sampleIndices.count * perSample)
        var labels = [Int](repeating: 0, count: sampleIndices.count)

        shard.withImageBuffer { imageBuffer in
            for (batchPos, sampleIdx) in sampleIndices.enumerated() {
                let srcStart = sampleIdx * perSample
                let dstStart = batchPos * perSample
                for offset in 0..<perSample {
                    images[dstStart + offset] = imageBuffer[srcStart + offset]
                }
            }
        }

        shard.withLabelBuffer { labelBuffer in
            for (batchPos, sampleIdx) in sampleIndices.enumerated() {
                labels[batchPos] = Int(labelBuffer[sampleIdx])
            }
        }

        return (images, labels)
    }
}


