# RAD-FL Swift Client

Native Swift implementation of a gossip-based decentralized federated
learning (DFL) client, benchmarked against a Python+NumPy+BLAS baseline on
a 10-node Raspberry Pi Zero 2W cluster (Cortex-A53, ARMv8.0-A). Companion
code for *"Swift as a serverless DFL runtime on ARM Linux."*

## Status

Working, buildable client with a real CLI, a from-scratch pure-Swift CNN
compute layer, and a gossip round loop verified end-to-end on the real
10-node Pi cluster — not a scaffold. Experiment results themselves live in
a separate repo, not here.

## Layout

- **`Networking/`** — NIO-based gossip transport. Custom length-prefixed
  binary wire format (magic `RAFD`, host byte order — every node is the same
  architecture, so no endian normalization), per-connection read timeouts,
  and `Topology.swift`, which parses the same `topology.json` schema as the
  Python baseline and resolves a node's peers under `full-mesh` (used for
  real experiments) or `file`/ring (Python's smoke-test mode). Connect-per-push,
  not persistent connections — a dead peer just fails to connect, no
  stale-connection bookkeeping. `sparseCOO` encoding is reserved on the wire
  but unimplemented; everything today is dense.
- **`Compute/` (`Tensor.swift`, `SimpleCNN.swift`, `CNNTrainer.swift`)** —
  the actual CNN, pure Swift, no Accelerate/BLAS. `Tensor.swift` implements
  conv (via im2col + tiled GEMM), maxpool, FC, and activations, parallelized
  across all cores with `DispatchQueue.concurrentPerform` and SIMD4 inner
  loops, tuned for the Cortex-A53's 32KB L1 (8×8 GEMM tiles, 64×64 transpose
  tiles). `SimpleCNN` matches the Python baseline's `SimpleCNNCIFAR`
  architecture exactly — 3 conv layers (3→16→32→64, valid/no-padding 3×3),
  He init, momentum SGD (0.9) reset after each gossip aggregation.
  `CNNTrainer` reshuffles the shard every epoch via a seeded RNG, matching
  Python's per-epoch shuffle (shards are Dirichlet-partitioned and *not*
  randomly ordered on disk, so this isn't cosmetic).
- **`Aggregation/`** — `GossipAggregator.aggregate` does sample-weighted
  averaging across a node's local update and its peers' updates, routed
  through `Tensor.swift`'s `weightedSum` (SIMD4 + parallel across cores) —
  the aggregation step sits in the per-round critical path, so it's held to
  the same compute-efficiency bar as the rest of the model. Empty peer list
  returns the local update unchanged rather than dividing by zero.
- **`Model/`** — `FederatedModel` protocol (the seam `SimpleCNN` conforms
  to: `parameters`/`setParameters` for gossip exchange, `trainEpoch`,
  `evaluate`), `DataShard`/`CIFAR10Shard` (loads the same per-node
  `node_NN_train_X/y.npy` shards `extract_cifar10_shards.py` produces, with
  optional mmap so RSS is comparable to Python's `mmap_mode="r"` deployment
  path), and `NPYWriter` for saving trained weights back out as real,
  NumPy-readable `.npy` files.
- **`Orchestration/`** — `NPYReader` (from-scratch `.npy` v1.0–v3.0 parser,
  mmap-backed zero-copy reads) and `RoundOrchestrator`, which drives the
  actual round loop: train locally → push parameters to every peer → wait
  for every peer's update *for this round* → aggregate → adopt → log →
  repeat. Currently waits **indefinitely** for peers each round (no
  timeout) — a deliberate, for-now policy: a stalled round should be
  visible and investigated, not silently completed with a subset of peers.
- **`Telemetry/`** — `MetricsLogger` writes `training_log.csv` and
  `peer_push_log_*.jsonl` in the same schema Python's `metrics.py`/
  `power_analysis.py` already consume, and `ResourceUsage` wraps
  `getrusage` for CPU-seconds (`effective_cores`) and peak RSS. Nearly every
  `RoundMetrics` field is real; `bytes_received` is still a `-1` placeholder
  (the receive path isn't instrumented yet).

## CLI

```
radfl serve --port <port>                    # bare listener, transport smoke test
radfl push --host <h> --port <p>              # send one test message to a peer
radfl gossip --topology <path> --node-id <id> # resolve peers, push a test message to each
radfl test-connectivity --topology <path> --node-id <id>
                                               # live up/down status for every peer, forever
radfl inspect-npy --file <path> [--dtype float32|int64]
                                               # parse and sanity-check one .npy file
radfl inspect-shard --dir <path> [--node-index n] [--test]
                                               # load a full DataShard, print stats
radfl test-cnn [--fixed-batch]                # trains SimpleCNN on synthetic data —
                                               # verifies Tensor.swift's wiring in isolation
radfl train-cnn --dir <path> [--node-index n] # trains SimpleCNN on one real CIFAR-10 shard,
                                               # no gossip — verifies data + compute together
radfl test-aggregator                         # GossipAggregator vs hand-computed expected values
radfl run-round --topology <path> --node-id <id> --data-dir <path> --condition <alpha_0p5|iid|...>
                                               # the real thing: full gossip FL loop on real
                                               # hardware, N rounds, real shards, real peers
```

`run-round` is the actual experiment driver. It starts this node's server,
loads its shard (index derived from position in the topology file, matching
Python's `run_experiment.py`), waits for every peer to come up, then runs
`RoundOrchestrator` for `--rounds` rounds. All output — `training_log.csv`,
`peer_push_log_*.jsonl`, final weights (`conv1W.npy` … `fcB.npy`), and
`config.json` (the exact `SimpleCNNConfig` used) — goes to
`<output-dir>/<node-id>/<condition>/` (default `<output-dir>`: `results`,
overridable with `--output-dir`).

## Toolchain: Debian Trixie aarch64 SDK

This project cross-compiles with **`6.2.4-RELEASE_debian_trixie_aarch64`**
([swift-embedded-linux/swift-sdks](https://github.com/swift-embedded-linux/swift-sdks)),
not Apple's musl-based Static Linux SDK. The musl SDK
(`aarch64-swift-linux-musl`) has an open upstream issue
([swiftlang/swift#88351](https://github.com/swiftlang/swift/issues/88351))
where its prebuilt runtime uses ARMv8.1+ LSE atomics, which `SIGILL`s on the
Pi Zero 2W's ARMv8.0-A Cortex-A53 — the same failure class that excluded
PyTorch from the Python baseline. The Trixie SDK avoids this (a different
build pipeline, a real Debian sysroot) and has been confirmed working on the
actual hardware.

Trade-off: the resulting binary is dynamically linked against the target's
glibc (not fully static), even with `--static-swift-stdlib` linking in the
Swift runtime itself. Confirmed match at time of writing: Pi nodes run
glibc `2.41-12+rpt1` (Raspberry Pi OS Trixie), SDK is built against glibc
`2.41` — same ABI, `+rpt1` is just Raspberry Pi Foundation's packaging
revision. Re-verify on every new/re-imaged node with `ssh <pi-node> ldd --version`;
`build-linux.sh` prints this reminder after every build but can't check it
for you (no SSH access to your nodes from the build script).

## Build

```bash
swift build                 # macOS native — local dev/testing only
./build-linux.sh            # cross-compile aarch64 Linux binary for the Pi cluster
```

`build-linux.sh` auto-detects the installed Trixie aarch64 SDK from `swift sdk list`
rather than hardcoding a triple (these vary by exactly how the SDK bundle was
generated). If detection fails it prints the full listing so you can pass one
explicitly: `SWIFT_SDK_TARGET=<triple> ./build-linux.sh`. Debug info is kept
by default — pass `STRIP=1` for a smaller deployment artifact once you don't
need symbolicated crashes for whatever you're currently debugging.

## Known gaps

- `bytes_received` in `RoundMetrics` is an unimplemented placeholder (`-1`) —
  the transport receive path doesn't track cumulative bytes yet.
- No round-level dead-peer timeout: a genuinely down peer stalls the whole
  node indefinitely, by design, for this development phase.
- `sparseCOO` tensor encoding is reserved on the wire but not implemented —
  every gossip payload today is dense.
