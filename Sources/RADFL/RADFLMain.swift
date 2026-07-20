// RADFLMain.swift
//
// CLI entrypoint. Named RADFLMain.swift rather than main.swift deliberately —
// a file literally named main.swift gets implicit top-level-code execution
// rights in Swift, which conflicts with the @main attribute used below. Using
// any other filename avoids that conflict.
//
// Currently a stub — wires together GossipServer/GossipClient for a smoke
// test. The actual experiment orchestration (alpha sweep, round loop, model
// training, calling into FederatedModel/GossipAggregator) is not implemented
// in this session; this is here so `swift build` / `swift run` produce a
// working binary you can deploy and sanity-check the networking layer
// against, ahead of the harness being built out.

import Foundation
import NIOPosix
import RADFLCore

@main
struct RADFLMain {
    static func main() async throws {
        let args = CommandLine.arguments

        guard args.count >= 2 else {
            print("""
            Usage:
              radfl serve  --port <port>
                           [or: --topology <path> --node-id <id>, to look up the port from the file]
                           (bare listener, smoke-test only — prints whatever it receives. For
                            actually running a node as part of a mesh, use test-connectivity
                            instead, which listens AND polls peers — this exists mainly for
                            quick one-off transport-layer checks.)
              radfl push   --host <host> --port <port>   (sends a tiny test message to one peer)
              radfl gossip --topology <path> --topology-mode <full-mesh|file> --node-id <id>
                           (loads topology.json, resolves this node's peers, pushes a test
                            message to each)
              radfl test-connectivity --topology <path> --topology-mode <full-mesh|file> --node-id <id>
                           [--poll-interval-seconds <n>] [--heartbeat-seconds <n>] [--no-push]
                           (simulates what a real gossip round needs: starts THIS node's own
                            listener (so peers can reach it) AND continuously monitors this
                            node's peers (so it can reach them) — both at once, forever, not
                            sequential phases with a finish line. Prints a line whenever either
                            direction's status CHANGES, e.g. "pi-3 is now reachable" or "pi-3 is
                            now UNREACHABLE" if it later goes down — this keeps checking every
                            peer for as long as the process runs, so a peer that was up and then
                            crashed or was stopped will correctly show as unreachable again, not
                            stay marked reachable forever from one past success. Plus a periodic
                            heartbeat (default 15s, --heartbeat-seconds) showing current N/total
                            reachable. By default also pushes a one-off test message to each
                            peer the moment it comes up (--no-push to skip). Runs until Ctrl-C,
                            same as `serve` — there's no "done" state for a node meant to stay
                            part of an ongoing mesh. Start this on every node — it replaces
                            needing a separate `serve` process — to see the whole mesh's live
                            status as you start, stop, and restart terminals.)
              radfl inspect-npy --file <path> [--dtype float32|int64] [--expect-shape N,C,H,W]
                           (reads a single .npy file via NPYReader and prints its parsed
                            shape, dtype descriptor, a handful of sample values, and basic
                            statistics (min/max/mean) — for verifying NPYReader against a
                            real CIFAR-10 shard file produced by extract_cifar10_shards.py.
                            --dtype defaults to float32 for *_X.npy-style files; pass
                            --dtype int64 for label files (*_y.npy). --expect-shape is
                            optional and just adds a pass/fail check against a shape you
                            already know from the Python side, e.g. --expect-shape 5000,3,32,32.)
              radfl inspect-shard --dir <path> [--node-index <n>] [--test] [--mapped|--eager]
                           (loads a full DataShard via CIFAR10Shard — node_<NN>_train_X/y.npy
                            with --node-index, or test_X/y.npy with --test — and prints
                            sample count, image shape, label sample values, and image
                            min/max/mean, the same way inspect-npy does for a single file.
                            This is the layer ABOVE NPYReader: verifies the DataShard
                            wrapper, the train/test sample-count cross-check, and (with
                            --mapped, the default) that mmap-backed access actually works
                            correctly end to end, not just that NPYReader's lower-level
                            byte parsing is correct.)
              radfl test-cnn [--steps <n>] [--batch-size <n>] [--learning-rate <f>] [--seed <n>] [--fixed-batch]
                           (trains SimpleCNN — conv→relu→maxpool→fc→softmax — on SYNTHETIC
                            random data for --steps SGD steps, printing loss each step.
                            This exercises every Tensor.swift primitive end-to-end: forward/
                            backward shape-matching across conv/pool/fc, CachedMatrix's
                            transpose caching staying correct across optimizer updates, and
                            gradient flow through the whole stack. Synthetic data
                            deliberately — this verifies the MATH/WIRING is correct,
                            separately from whether the real CIFAR-10 data pipeline is (that
                            part is already verified via inspect-shard). Loss should
                            decrease over the run; if it doesn't move, increases, or goes
                            NaN, something in Tensor.swift's wiring is wrong. By default a
                            FRESH random batch (and fresh random labels) is drawn every
                            step — with --fixed-batch, ONE batch is generated once and
                            reused every step instead. This matters as a diagnostic: fresh
                            random labels every step give the model nothing consistent to
                            learn even with correct wiring, so flat loss there is
                            ambiguous. A fixed batch DOES have a learnable pattern (same
                            examples every time) — if loss won't drop even with
                            --fixed-batch, that's real evidence of a wiring bug rather than
                            an artifact of the test's own data generation. Defaults:
                            50 steps, batch size 32, learning rate 0.01.)
              radfl train-cnn --dir <path> [--node-index <n>] [--epochs <n>] [--batch-size <n>]
                           [--learning-rate <f>] [--seed <n>]
                           (trains SimpleCNN on a REAL CIFAR-10 shard loaded via
                            CIFAR10Shard/NPYReader — node_<NN>_train_X/y.npy from --dir.
                            This is the real thing test-cnn's synthetic-data version was
                            built to verify in isolation: now the data pipeline and the
                            compute pipeline run together. Reshuffles the shard every
                            epoch via a seeded RNG, matching the Python baseline's actual
                            per-epoch shuffle behavior in trainer.py (NOT sequential order
                            — Dirichlet-partitioned shards are NOT randomly ordered in
                            storage, confirmed earlier via inspect-shard on node 0, so
                            sequential-order training would be a real methodological
                            difference from Python, not just cosmetic). Prints mean loss
                            per epoch. Defaults: 5 epochs, batch size 32, learning rate 0.01.)
              radfl test-aggregator
                           (verifies GossipAggregator.aggregate / Tensor.swift's new
                            weightedSum primitive against hand-computed expected values —
                            several peers with known sample counts and known parameter
                            values, checked against the sample-weighted average computed
                            independently in this test, not just "did it run without
                            crashing." weightedSum is newly-added code (SIMD4 + parallel
                            across cores, replacing GossipAggregator's previous nested
                            scalar loops) that hasn't been verified against a real
                            compiler — this is the first real check of its correctness,
                            the same way test-cnn was for Tensor.swift's other primitives.)
                           [--rounds <n>] [--learning-rate <f>] [--seed <n>] [--output-dir <path>]
                           (runs the FULL federated learning loop for real, on real
                            hardware: starts this node's GossipServer, loads its real
                            train/test shards via CIFAR10Shard, builds a SimpleCNN, and
                            runs RoundOrchestrator for --rounds rounds — train locally,
                            push parameters to every peer, WAIT INDEFINITELY for every
                            peer's update this round (no timeout — a genuinely down peer
                            stalls this node, by design, for this development phase; a
                            periodic heartbeat shows which peer(s) are still missing while
                            blocked), aggregate, adopt, log a real RoundMetrics row. Start
                            this on EVERY node in the topology (same as test-connectivity)
                            before expecting any round to complete — every node is both
                            pushing to and waiting on every other node; this command also
                            waits for every peer's port to be reachable before round 1,
                            printing progress as each comes up. --data-dir is the FLAT
                            base shard directory (e.g. data/cifar10) — NOT including the
                            condition subdirectory; that's appended automatically from
                            --condition, matching extract_cifar10_shards.py's own
                            convention exactly (shard_out_dir = data_dir/condition_dir).
                            An earlier version required typing the condition string twice
                            (once inside --data-dir's path, once in --condition) with
                            nothing enforcing they matched — fixed, same foot-gun class as
                            the --node-index/--node-id issue resolved earlier. Which
                            node_<NN>_train_X/y.npy shard to load is derived AUTOMATICALLY
                            from this node's 0-based position in the topology file's node
                            list — matching Python's run_experiment.py exactly ("the
                            topology file is the single source of truth for node
                            ordering"). --condition is REQUIRED (e.g. alpha_0p5, iid) —
                            used both to find the shard directory (--data-dir/<condition>/)
                            and as part of the output path. ALL output for this run
                            (RoundMetrics CSV, peer_push_log.jsonl, saved final weights as
                            conv1W/conv1B/conv2W/conv2B/conv3W/conv3B/fcW/fcB.npy, and config.json recording the exact
                            SimpleCNNConfig used) is written under
                            --output-dir/<node-id>/<condition>/ (default --output-dir:
                            "results"), e.g. results/pi-1/alpha_0p5/. Some RoundMetrics
                            fields (effective_cores, peak_rss_bytes, bytes_sent,
                            bytes_received) are PLACEHOLDER VALUES (-1) in this version —
                            see RoundOrchestrator.swift for why. Defaults: 5 rounds,
                            learning rate 0.01.)
            """)
            return
        }

        // inspect-npy is handled before the event loop group is created —
        // it's pure file I/O with no networking, so spinning up a
        // MultiThreadedEventLoopGroup for it would be wasted setup (and a
        // wasted teardown call) for a command that will exit in milliseconds.
        if args[1] == "inspect-npy" {
            runInspectNPY(args)
            return
        }

        if args[1] == "inspect-shard" {
            runInspectShard(args)
            return
        }

        if args[1] == "test-cnn" {
            runTestCNN(args)
            return
        }

        if args[1] == "train-cnn" {
            runTrainCNN(args)
            return
        }

        if args[1] == "test-aggregator" {
            runTestAggregator()
            return
        }

        // Using ProcessInfo (Foundation) rather than NIOPosix's System.coreCount here —
        // functionally equivalent for this purpose (thread count for the event loop
        // group), and avoids relying on a NIOPosix symbol that wasn't resolving in
        // at least one Xcode/toolchain configuration during development. Foundation
        // is already imported above and ProcessInfo.activeProcessorCount is available
        // on both macOS and Linux.
        let group = MultiThreadedEventLoopGroup(numberOfThreads: ProcessInfo.processInfo.activeProcessorCount)
        // NOTE: no `defer { try? group.syncShutdownGracefully() }` here — that's a
        // blocking call and `defer` blocks can't `await`, so mixing them inside an
        // `async` function is flagged as unsafe under Swift 6's strict concurrency
        // checking (it could block the calling thread). MultiThreadedEventLoopGroup
        // has an async-safe `shutdownGracefully()` instead; it's called explicitly
        // at each exit path below rather than via `defer`.

        switch args[1] {
        case "serve":
            // Two ways to specify the port:
            //   --port <n>                              — direct, original smoke-test usage
            //   --topology <path> --node-id <id>        — look up this node's own port from
            //                                              the topology file, so you don't have
            //                                              to manually cross-reference the JSON
            // If both are given, --port wins (explicit beats derived).
            let explicitPort = arg("--port", in: args).flatMap(Int.init)
            var port = explicitPort ?? 9090
            var selfDescription = "port \(port)"

            if explicitPort == nil, let topologyPathStr = arg("--topology", in: args), let nodeID = arg("--node-id", in: args) {
                let topologyURL = URL(fileURLWithPath: topologyPathStr)
                do {
                    // mode doesn't affect self-lookup (we only need this
                    // node's own entry, not its peer set), so "full-mesh" is
                    // just a harmless default here.
                    let topology = try Topology.load(path: topologyURL, localNodeID: nodeID, mode: .fullMesh)
                    port = topology.localNode.port
                    selfDescription = "\(topology.localNode.id) (port \(port))"
                } catch {
                    print("[serve] could not resolve --node-id '\(nodeID)' from \(topologyPathStr): \(error)")
                    try await group.shutdownGracefully()
                    return
                }
            }

            let server = GossipServer(port: port, group: group) { message, peer in
                print("[serve] received \(message.messageType) round=\(message.round) from=\(peer.host):\(peer.port) tensors=\(message.tensors.count)")
            }
            try await server.start()
            print("[serve] listening as \(selfDescription). This process IS the listener — leave it running.")
            print("[serve] Ctrl-C to stop.")
            // Keep running. This path normally exits via the process being killed
            // (Ctrl-C), not by returning from main() — so there's no graceful
            // group shutdown here; the OS reclaims everything on process exit.
            // If this ever needs a clean in-process shutdown path (e.g. a signal
            // handler that calls server.shutdown() and group.shutdownGracefully()
            // before exiting), that's a real gap to revisit, not handled here.
            try await Task.sleep(for: .seconds(365 * 24 * 60 * 60))

        case "push":
            let host = arg("--host", in: args) ?? "127.0.0.1"
            let port = Int(arg("--port", in: args) ?? "9090") ?? 9090
            let client = GossipClient(group: group)
            let testMessage = GossipMessage(
                messageType: .weightsFull,
                senderNodeID: 1,
                round: 0,
                sampleCount: 0,  // synthetic smoke-test message, no real training happened
                tensors: [GossipTensor(tensorID: 0, values: [1.0, 2.0, 3.0, 4.0])]
            )
            do {
                try await client.push(testMessage, to: GossipNodeAddress(host: host, port: port))
                print("[push] sent test message to \(host):\(port)")
                try await group.shutdownGracefully()
            } catch {
                try? await group.shutdownGracefully()
                throw error
            }

        case "gossip":
            guard let topology = try await loadTopologyFromArgs(args, commandName: "gossip", group: group) else {
                return
            }

            do {
                let senderID = try topology.localNode.numericID()
                let client = GossipClient(group: group)

                // Push to every resolved peer in parallel via a task group,
                // rather than sequentially — matches what an actual gossip
                // round will need to do (push to all peers without waiting
                // on each one before starting the next), and surfaces
                // per-peer failures individually instead of one failure
                // aborting the rest.
                try await withThrowingTaskGroup(of: (TopologyNode, Result<Void, Error>).self) { taskGroup in
                    for peer in topology.peers {
                        taskGroup.addTask {
                            let message = GossipMessage(
                                messageType: .weightsFull,
                                senderNodeID: senderID,
                                round: 0,
                                sampleCount: 0,  // synthetic smoke-test message, no real training happened
                                tensors: [GossipTensor(tensorID: 0, values: [1.0, 2.0, 3.0, 4.0])]
                            )
                            do {
                                try await client.push(message, to: GossipNodeAddress(host: peer.host, port: peer.port))
                                return (peer, .success(()))
                            } catch {
                                return (peer, .failure(error))
                            }
                        }
                    }

                    var succeeded = 0
                    var failed = 0
                    for try await (peer, result) in taskGroup {
                        switch result {
                        case .success:
                            succeeded += 1
                            print("[gossip] pushed to \(peer.id) (\(peer.host):\(peer.port))")
                        case .failure(let error):
                            failed += 1
                            print("[gossip] FAILED to push to \(peer.id) (\(peer.host):\(peer.port)): \(error)")
                        }
                    }
                    print("[gossip] done — \(succeeded) succeeded, \(failed) failed out of \(topology.peers.count) peer(s)")
                }

                try await group.shutdownGracefully()
            } catch {
                try? await group.shutdownGracefully()
                throw error
            }

        case "test-connectivity":
            guard let topology = try await loadTopologyFromArgs(args, commandName: "test-connectivity", group: group) else {
                return
            }

            do {
                let client = GossipClient(group: group)
                let senderID = try topology.localNode.numericID()

                let pollIntervalSeconds = Int64(arg("--poll-interval-seconds", in: args) ?? "2") ?? 2
                let heartbeatSeconds = Int64(arg("--heartbeat-seconds", in: args) ?? "15") ?? 15
                let pushOnUp = !args.contains("--no-push")

                // This command now does what a real gossip round needs to do:
                // listen for inbound peers AND continuously check outbound
                // peers, at the same time, reporting status as either side
                // changes — not two separate sequential phases that exit
                // once done. Runs until Ctrl-C, same lifecycle as `serve`,
                // since there's no natural "finished" state for a node that's
                // meant to stay part of an ongoing mesh.
                //
                // Uses `monitorReachability`, NOT `waitForAllReachable` —
                // the latter treats reachability as a one-shot terminal
                // state and stops checking a peer once reached, so it can
                // never notice that peer later going away (e.g. its process
                // exits mid-run). `monitorReachability` never stops
                // checking anyone, and reports both up->down and down->up
                // transitions for as long as this process runs.

                // Inbound: a real GossipServer, so peers running their own
                // `radfl serve` (or their own `test-connectivity`) can reach
                // THIS node, not just the other way around.
                //
                // Same fix as run-round's identical issue: resolve the
                // peer's name from message.senderNodeID, not from the
                // connection's ephemeral source port (GossipClient.push
                // connects fresh per push, so the source port is
                // OS-assigned and different every time — never the peer's
                // real listening port).
                let peerNameByNumericID: [UInt32: String] = Dictionary(
                    uniqueKeysWithValues: topology.peers.compactMap { peer in
                        (try? peer.numericID()).map { ($0, peer.id) }
                    }
                )
                let server = GossipServer(port: topology.localNode.port, group: group) { message, peer in
                    let senderName = peerNameByNumericID[message.senderNodeID] ?? "node#\(message.senderNodeID) (\(peer.host):\(peer.port))"
                    print("[\(topology.localNode.id)] <- received \(message.messageType) round=\(message.round) from=\(senderName)")
                }
                try await server.start()
                print("[test-connectivity] \(topology.localNode.id) listening on port \(topology.localNode.port) — this node can now receive from peers.")
                print("[test-connectivity] \(topology.peers.count) peer(s) to reach: \(topology.peers.map(\.id).joined(separator: ", "))")
                print("[test-connectivity] Ctrl-C to stop. Reporting outbound status as peers come up or go down...")

                let addressToNode = Dictionary(
                    uniqueKeysWithValues: topology.peers.map {
                        (GossipNodeAddress(host: $0.host, port: $0.port), $0)
                    }
                )
                let addresses = topology.peers.map { GossipNodeAddress(host: $0.host, port: $0.port) }

                // Outbound: continuous monitoring, no deadline, runs until
                // this Task is cancelled (process exit via Ctrl-C). Does not
                // return a final results dictionary the way the old one-shot
                // call did — there is no "final" state for an ongoing
                // monitor, only "current" state, which is what the
                // onStatusChange callback reports as it changes.
                await monitorReachability(
                    client: client,
                    addresses: addresses,
                    pollInterval: .seconds(pollIntervalSeconds),
                    heartbeatInterval: .seconds(heartbeatSeconds),
                    onStatusChange: { address, liveness in
                        guard let peer = addressToNode[address] else { return }
                        switch liveness {
                        case .up:
                            print("[\(topology.localNode.id)] -> \(peer.id) (\(peer.host):\(peer.port)) is now reachable")
                            guard pushOnUp else { return }
                            Task {
                                let message = GossipMessage(
                                    messageType: .weightsFull,
                                    senderNodeID: senderID,
                                    round: 0,
                                    sampleCount: 0,  // synthetic smoke-test message, no real training happened
                                    tensors: [GossipTensor(tensorID: 0, values: [1.0, 2.0, 3.0, 4.0])]
                                )
                                do {
                                    try await client.push(message, to: address)
                                    print("[\(topology.localNode.id)] -> pushed test message to \(peer.id)")
                                } catch {
                                    print("[\(topology.localNode.id)] -> push to \(peer.id) FAILED: \(error)")
                                }
                            }
                        case .down:
                            print("[\(topology.localNode.id)] -> \(peer.id) (\(peer.host):\(peer.port)) is now UNREACHABLE")
                        }
                    },
                    onHeartbeat: { upCount, totalCount in
                        print("[\(topology.localNode.id)] [status] \(upCount)/\(totalCount) peers currently reachable")
                    }
                )

                // monitorReachability only returns once this Task is
                // cancelled (Ctrl-C) — this line runs during shutdown, not
                // as a normal "finished" summary.
                try await group.shutdownGracefully()
            } catch {
                try? await group.shutdownGracefully()
                throw error
            }

        case "run-round":
            guard let topology = try await loadTopologyFromArgs(args, commandName: "run-round", group: group) else {
                return
            }

            do {
                guard let dataDirStr = arg("--data-dir", in: args) else {
                    print("[run-round] --data-dir is required, e.g. --data-dir data/cifar10 (the FLAT base directory — NOT including the condition subdirectory, that's appended automatically from --condition)")
                    try await group.shutdownGracefully()
                    return
                }
                // --condition is REQUIRED, not optional with a fallback —
                // it's now load-bearing for BOTH the shard directory lookup
                // AND the output directory structure, not just a metadata
                // tag recorded inside the CSV. Silently defaulting to
                // "unspecified" would mean an easy-to-forget flag produces
                // a real, possibly-confusing directory named literally
                // "unspecified" rather than a clear upfront error.
                guard let condition = arg("--condition", in: args) else {
                    print("[run-round] --condition is required, e.g. --condition alpha_0p5 — used both to find the shard directory (--data-dir/<condition>/) and as part of the output path (results/<node-id>/<condition>/)")
                    try await group.shutdownGracefully()
                    return
                }
                // --data-dir is the FLAT base directory (e.g. data/cifar10),
                // matching extract_cifar10_shards.py's own convention
                // exactly: shard_out_dir = os.path.join(args.data_dir,
                // condition_dir) — the condition subdirectory is appended
                // here automatically from --condition, NOT typed separately
                // into --data-dir. An earlier version of this command
                // required typing the same condition string twice (once
                // inside --data-dir's path, once in --condition), with
                // nothing enforcing they matched — the same class of
                // foot-gun as the --node-index/--node-id issue fixed
                // earlier in this project.
                let dataDir = URL(fileURLWithPath: dataDirStr).appendingPathComponent(condition)
                let rounds = Int(arg("--rounds", in: args) ?? "5") ?? 5
                // Default 0.01, matching Python's ACTUAL default exactly
                // (run_experiment.py: "--lr", type=float, default=0.01) —
                // an earlier version of this default was 0.1, a real,
                // unflagged 10x mismatch against Python that was never
                // caught when the architecture was rebuilt from 1 conv
                // layer to 3. Confirmed as the actual root cause of a real
                // training-instability bug: test-cnn --fixed-batch showed
                // loss decreasing cleanly for ~50 steps then spiking
                // catastrophically (2.96 -> 0.88 -> 9.79 -> plateau ~2.0),
                // and a live 9-round Pi run showed local_acc collapsing to
                // near-zero after aggregation every round — both
                // consistent with a learning rate too high for this
                // deeper, 3-layer network (gradients compounding through
                // more layers than the single-conv-layer model this
                // default was originally tuned against).
                let learningRate = Float(arg("--learning-rate", in: args) ?? "0.01") ?? 0.01
                let seed = UInt64(arg("--seed", in: args) ?? "42") ?? 42
                let outputBaseStr = arg("--output-dir", in: args) ?? "results"

                // Single shared output directory for EVERYTHING this run
                // produces — MetricsLogger's CSV/JSONL, the saved weights
                // (.npy), and config.json — structured as
                // <output-base>/<node-id>/<condition>/, e.g.
                // results/pi-1/alpha_0p5/, using the topology's actual
                // string node-id (not the numeric wire-protocol ID) since
                // that's the more legible identifier for browsing a
                // results tree on disk. Replaces the earlier design where
                // these three things lived in three different places with
                // three different naming conventions (a flat outputDir
                // with condition/nodeID baked into filenames for metrics,
                // a separate "weights_<condition>_node<N>" subdirectory
                // for weights) — this is a real, deliberate consolidation,
                // not an additive change.
                let outputDir = URL(fileURLWithPath: outputBaseStr)
                    .appendingPathComponent(topology.localNode.id)
                    .appendingPathComponent(condition)
                try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
                print("[run-round] all output for this run will be written under \(outputDir.path)")

                // Shard index derived from this node's position in
                // topology.allNodes — NOT a separate --node-index flag.
                // Matches Python's run_experiment.py exactly: "the node
                // index (0-based position in topology nodes list) is
                // derived automatically from --node-id, so the topology
                // file is the single source of truth for node ordering."
                // An earlier version of this command took --node-index as
                // its own flag, independent of --node-id — that was a real
                // mistake, not just a stylistic difference from Python:
                // it created exactly the foot-gun Python's design
                // deliberately avoids (two values that are SUPPOSED to
                // always agree, with nothing enforcing that they do — easy
                // to run 10 processes with 10 different --node-id values
                // but accidentally the same --node-index for all of them).
                guard let nodeIndex = topology.allNodes.firstIndex(where: { $0.id == topology.localNode.id }) else {
                    print("[run-round] could not find \(topology.localNode.id) in topology.allNodes — this should be impossible given Topology.load already resolved localNode from this same file")
                    try await group.shutdownGracefully()
                    return
                }

                print("[run-round] loading shards from \(dataDir.path) (node index \(nodeIndex), derived from \(topology.localNode.id)'s position in the topology file) — this can take a moment for large shards...")
                let shardLoadStart = Date()
                let trainShard = try CIFAR10Shard.loadTrainShard(directory: dataDir, nodeIndex: nodeIndex)
                let testShard = try CIFAR10Shard.loadTestSet(directory: dataDir)
                let shardLoadDuration = Date().timeIntervalSince(shardLoadStart)
                print("[run-round] shards loaded in \(String(format: "%.2f", shardLoadDuration))s")
                print("[run-round] train shard: \(trainShard.sampleCount) samples, imageShape \(trainShard.imageShape)")
                print("[run-round] test shard: \(testShard.sampleCount) samples")

                guard trainShard.imageShape.count == 4 else {
                    print("[run-round] unexpected train shard imageShape \(trainShard.imageShape)")
                    try await group.shutdownGracefully()
                    return
                }
                // SimpleCNN's architecture is now FIXED (matching Python's
                // SimpleCNNCIFAR exactly: 3 conv layers, 32x32 CIFAR-10
                // input) — cIn/h/w are no longer configurable per-shard the
                // way an earlier version of this command derived them.
                // Still validate the shard's actual shape against what the
                // fixed architecture expects, though, so a wrong/unexpected
                // dataset produces a clear error HERE rather than a
                // confusing shape-mismatch failure deep inside convForward.
                let expectedShape = (cIn: 3, h: 32, w: 32)
                guard trainShard.imageShape[1] == expectedShape.cIn,
                      trainShard.imageShape[2] == expectedShape.h,
                      trainShard.imageShape[3] == expectedShape.w else {
                    print("[run-round] train shard imageShape \(trainShard.imageShape) doesn't match SimpleCNN's fixed architecture (expects N,\(expectedShape.cIn),\(expectedShape.h),\(expectedShape.w) — i.e. real CIFAR-10)")
                    try await group.shutdownGracefully()
                    return
                }
                // batch size for SimpleCNN's config — reuses the same default
                // batch size CNNTrainer/train-cnn use elsewhere in this file,
                // not exposed as a separate run-round flag in this version.
                let modelConfig = SimpleCNNConfig(n: 32)
                let model = SimpleCNN(config: modelConfig, seed: seed)

                let collector = InboundUpdateCollector()
                let client = GossipClient(group: group)

                let timestamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
                let nodeNumericID = try topology.localNode.numericID()
                let metricsLogger = try MetricsLogger(
                    outputDirectory: outputDir, condition: condition, nodeID: Int(nodeNumericID), timestamp: timestamp
                )

                let orchestratorConfig = RoundOrchestratorConfig(
                    totalRounds: rounds, condition: condition, baseSeed: seed, learningRate: learningRate
                )
                let orchestrator = try RoundOrchestrator(
                    model: model, topology: topology, config: orchestratorConfig,
                    client: client, collector: collector, metricsLogger: metricsLogger,
                    onTrainingProgress: { round, batchIndex, totalBatches in
                        print("[run-round] round \(round): training batch \(batchIndex)/\(totalBatches)")
                    }
                )

                // Inbound: GossipServer's onMessage is a plain (non-async)
                // @Sendable closure, but feeding a received message into
                // the orchestrator's actor-backed collector is async —
                // bridged via Task { } here, the same fire-and-forget
                // pattern used elsewhere in this file for sync-callback ->
                // async-actor handoffs. Messages arrive in whatever order
                // the network delivers them; InboundUpdateCollector itself
                // (an actor) serializes concurrent record(_:) calls
                // correctly regardless of which order these Tasks run in.
                //
                // Peer NAME resolved from message.senderNodeID, NOT from
                // the connection's `peer` (host:port) argument — GossipClient.push
                // connects fresh per push (no persistent connection, by
                // design), so the inbound connection's SOURCE port is an
                // OS-assigned ephemeral port, different every single push,
                // never the peer's actual listening port from topology.json.
                // Matching on host:port would almost always fail to find a
                // name (host matches, port never will), which is exactly
                // why this was falling back to printing the raw IP:port —
                // not a missing lookup, a lookup against the wrong key.
                // senderNodeID is the message's real, stable identity,
                // tagged by the sender itself, and is exactly what
                // peerNameByNumericID below is built to resolve.
                let peerNameByNumericID: [UInt32: String] = Dictionary(
                    uniqueKeysWithValues: topology.peers.compactMap { peer in
                        (try? peer.numericID()).map { ($0, peer.id) }
                    }
                )
                let server = GossipServer(port: topology.localNode.port, group: group) { message, peer in
                    let senderName = peerNameByNumericID[message.senderNodeID] ?? "node#\(message.senderNodeID) (\(peer.host):\(peer.port))"
                    print("[run-round] <- received \(message.messageType) round=\(message.round) from=\(senderName)")
                    Task {
                        await orchestrator.handleInboundMessage(message)
                    }
                }
                try await server.start()
                print("[run-round] \(topology.localNode.id) listening on port \(topology.localNode.port)")
                print("[run-round] \(topology.peers.count) peer(s): \(topology.peers.map(\.id).joined(separator: ", "))")

                // Readiness gate: wait until every peer's port is reachable
                // before starting round 1 — a simple, reused-code proxy for
                // "that node has finished its own startup (shard loading,
                // server start) and is up," per project decision (a real
                // readiness handshake message was considered and explicitly
                // deferred as unnecessary complexity for now). Uses the
                // same waitForAllReachable/heartbeat machinery already
                // built and verified for `test-connectivity` — effectively
                // unbounded timeout (1 year) rather than a true "wait
                // forever" since waitForAllReachable's signature requires
                // a concrete TimeAmount; this matches the project's
                // explicit "no timeout" policy for this development phase
                // in spirit, just not literally infinite.
                print("[run-round] waiting for all \(topology.peers.count) peer(s) to be reachable before starting...")
                let readinessClient = GossipClient(group: group)
                let peerAddresses = topology.peers.map { GossipNodeAddress(host: $0.host, port: $0.port) }
                let peerNameByAddress = Dictionary(uniqueKeysWithValues: topology.peers.map {
                    (GossipNodeAddress(host: $0.host, port: $0.port), $0.id)
                })
                _ = await waitForAllReachable(
                    client: readinessClient,
                    addresses: peerAddresses,
                    timeout: .seconds(Int64(365 * 24 * 60 * 60)),
                    onStatusChange: { address, status in
                        if status == .reachable {
                            let name = peerNameByAddress[address] ?? "\(address.host):\(address.port)"
                            print("[run-round] \(name) is up")
                        }
                    },
                    onHeartbeat: { reachableCount, totalCount in
                        print("[run-round] waiting for peers to come up: \(reachableCount)/\(totalCount) ready so far")
                    }
                )
                print("[run-round] all peers reachable — starting \(rounds) round(s)...")

                // Save modelConfig as config.json INTO outputDir, BEFORE
                // training starts (not just at the end alongside the
                // weights) — done here rather than inside RoundOrchestrator
                // deliberately: RoundOrchestrator is built against the
                // abstract FederatedModel protocol and doesn't know about
                // SimpleCNNConfig (or SimpleCNN) specifically, so it has no
                // way to save this itself; this is the one place that
                // already has the concrete config in scope. Saving it now
                // rather than only after all rounds finish means the
                // architecture is recorded even if the run gets
                // interrupted partway through — useful for knowing what
                // was being trained, independent of whether weights ever
                // got saved. Closes the gap flagged when NPYWriter/
                // saveFinalWeights were built: a flat .npy file has no
                // shape/architecture metadata of its own, so without this
                // file, reloading saved weights to seed a future
                // experiment would require already knowing the exact
                // SimpleCNNConfig from some other, unenforced source.
                // outputDir already exists (created right after it was
                // computed, above), so no separate createDirectory call
                // is needed here the way the old weightsDir-specific
                // version required.
                let configURL = outputDir.appendingPathComponent("config.json")
                let configData = try JSONEncoder().encode(modelConfig)
                try configData.write(to: configURL, options: .atomic)
                print("[run-round] model config saved to \(configURL.path)")

                try await orchestrator.run(trainShard: trainShard, testShard: testShard, outputDirectory: outputDir)

                print("[run-round] all \(rounds) round(s) complete.")
                try metricsLogger.close()
                try await group.shutdownGracefully()
            } catch let error as RoundOrchestratorError {
                print("[run-round] \(error.description)")
                try? await group.shutdownGracefully()
            } catch let error as NPYError {
                print("[run-round] \(error.description)")
                try? await group.shutdownGracefully()
            } catch let error as CIFAR10ShardError {
                print("[run-round] \(error.description)")
                try? await group.shutdownGracefully()
            } catch {
                print("[run-round] unexpected error: \(error)")
                try? await group.shutdownGracefully()
                throw error
            }

        default:
            print("Unknown command: \(args[1])")
            try await group.shutdownGracefully()
        }
    }

    /// Reads one .npy file via NPYReader and prints enough information to
    /// manually verify it was parsed correctly against a real file produced
    /// by extract_cifar10_shards.py — shape, dtype, a handful of sample
    /// values (to eyeball against `np.load(...).flatten()[:5]` on the Python
    /// side), and basic min/max/mean statistics (a wrong byte order or wrong
    /// dtype read tends to produce wildly implausible values here, e.g.
    /// "min/max" in the billions for what should be normalized [0,1] image
    /// data, which is a fast, obvious sanity signal even without a
    /// side-by-side Python comparison).
    private static func runInspectNPY(_ args: [String]) {
        guard let filePath = arg("--file", in: args) else {
            print("[inspect-npy] --file is required, e.g. --file data/cifar10/alpha_0p5/node_00_train_X.npy")
            return
        }
        let url = URL(fileURLWithPath: filePath)

        let dtypeArg = arg("--dtype", in: args) ?? "float32"
        let expectedShape: [Int]? = arg("--expect-shape", in: args).map { str in
            str.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        }

        print("[inspect-npy] reading \(url.path)")

        do {
            switch dtypeArg {
            case "float32":
                // Deliberately NOT passing expectedShape into NPYReader here —
                // that would throw before returning anything if the guess is
                // wrong, hiding the actual data this command exists to show.
                // The shape comparison for display happens separately below,
                // after a successful unconditional read.
                let (shape, values) = try NPYReader.readFloat32Array(from: url)
                printFloatSummary(shape: shape, values: values, expectedShape: expectedShape)

            case "int64":
                let (shape, values) = try NPYReader.readInt64Array(from: url)
                printInt64Summary(shape: shape, values: values, expectedShape: expectedShape)

            default:
                print("[inspect-npy] unknown --dtype '\(dtypeArg)', expected 'float32' or 'int64'")
            }
        } catch let error as NPYError {
            print("[inspect-npy] \(error.description)")
        } catch {
            print("[inspect-npy] unexpected error: \(error)")
        }
    }

    private static func printFloatSummary(shape: [Int], values: [Float32], expectedShape: [Int]?) {
        print("  shape: \(shape)  (\(values.count) elements)")
        if let expectedShape {
            print("  expected shape: \(expectedShape)  →  \(shape == expectedShape ? "MATCH" : "MISMATCH")")
        }

        let sampleCount = min(10, values.count)
        print("  first \(sampleCount) values: \(values.prefix(sampleCount).map { String(format: "%.6f", $0) }.joined(separator: ", "))")

        guard !values.isEmpty else {
            print("  (empty array — no statistics to report)")
            return
        }
        let minVal = values.min()!
        let maxVal = values.max()!
        // Mean computed via Double accumulation, not Float32, to avoid
        // precision loss summing potentially millions of float32 values —
        // this is a diagnostic statistic, not training code, so correctness
        // of the sanity check itself matters more than raw speed here.
        let mean = values.reduce(0.0) { $0 + Double($1) } / Double(values.count)
        print("  min: \(minVal)   max: \(maxVal)   mean: \(String(format: "%.6f", mean))")
        print("  (sanity check: CIFAR-10 image data should be in [0.0, 1.0] — values far outside that range likely indicate a dtype/byte-order misread)")
    }

    private static func printInt64Summary(shape: [Int], values: [Int64], expectedShape: [Int]?) {
        print("  shape: \(shape)  (\(values.count) elements)")
        if let expectedShape {
            print("  expected shape: \(expectedShape)  →  \(shape == expectedShape ? "MATCH" : "MISMATCH")")
        }

        let sampleCount = min(20, values.count)
        print("  first \(sampleCount) values: \(values.prefix(sampleCount).map(String.init).joined(separator: ", "))")

        guard !values.isEmpty else {
            print("  (empty array — no statistics to report)")
            return
        }
        let minVal = values.min()!
        let maxVal = values.max()!
        let uniqueCount = Set(values).count
        print("  min: \(minVal)   max: \(maxVal)   unique values: \(uniqueCount)")
        print("  (sanity check: CIFAR-10 labels should be integers in [0, 9] with up to 10 unique values — anything else likely indicates a dtype/byte-order misread)")
    }

    private static func runInspectShard(_ args: [String]) {
        guard let dirPath = arg("--dir", in: args) else {
            print("[inspect-shard] --dir is required, e.g. --dir data/cifar10/alpha_0p5")
            return
        }
        let directory = URL(fileURLWithPath: dirPath)

        let strategy: NPYReader.LoadStrategy = args.contains("--eager") ? .eager : .mapped
        let isTestSet = args.contains("--test")
        let nodeIndex = Int(arg("--node-index", in: args) ?? "0") ?? 0

        print("[inspect-shard] loading \(isTestSet ? "test set" : "node \(nodeIndex) train shard") from \(directory.path)")
        print("[inspect-shard] strategy: \(strategy)")

        do {
            let shard: DataShard
            if isTestSet {
                shard = try CIFAR10Shard.loadTestSet(directory: directory, strategy: strategy)
            } else {
                shard = try CIFAR10Shard.loadTrainShard(directory: directory, nodeIndex: nodeIndex, strategy: strategy)
            }

            print("  imageShape: \(shard.imageShape)")
            print("  sampleCount: \(shard.sampleCount)")
            print("  perSampleElementCount: \(shard.perSampleElementCount)")
            print("  labelCount: \(shard.labelCount)  →  \(shard.sampleCount == shard.labelCount ? "MATCHES sampleCount" : "MISMATCH with sampleCount")")

            // Exercise withImageBuffer/withLabelBuffer for real — this is
            // the part that specifically verifies the mmap-backed path
            // works correctly end to end (NPYArrayData's bindMemory +
            // UnsafeBufferPointer plumbing), not just that NPYReader's
            // header/byte parsing is correct in isolation.
            let sampleValueCount = min(10, shard.perSampleElementCount)
            shard.withImageBuffer { buffer in
                let firstSample = buffer.prefix(sampleValueCount)
                print("  first \(sampleValueCount) values of sample 0: \(firstSample.map { String(format: "%.6f", $0) }.joined(separator: ", "))")

                guard !buffer.isEmpty else {
                    print("  (empty image buffer — no statistics to report)")
                    return
                }
                let minVal = buffer.min()!
                let maxVal = buffer.max()!
                let mean = buffer.reduce(0.0) { $0 + Double($1) } / Double(buffer.count)
                print("  image min: \(minVal)   max: \(maxVal)   mean: \(String(format: "%.6f", mean))")
            }

            shard.withLabelBuffer { buffer in
                let sampleCount = min(20, buffer.count)
                print("  first \(sampleCount) labels: \(buffer.prefix(sampleCount).map(String.init).joined(separator: ", "))")
                guard !buffer.isEmpty else { return }
                let uniqueCount = Set(buffer).count
                print("  label unique values: \(uniqueCount)  (CIFAR-10 expects ≤10)")
            }
        } catch let error as CIFAR10ShardError {
            print("[inspect-shard] \(error.description)")
        } catch let error as NPYError {
            print("[inspect-shard] \(error.description)")
        } catch {
            print("[inspect-shard] unexpected error: \(error)")
        }
    }

    private static func runTrainCNN(_ args: [String]) {
        guard let dirPath = arg("--dir", in: args) else {
            print("[train-cnn] --dir is required, e.g. --dir data/cifar10/alpha_0p5")
            return
        }
        let directory = URL(fileURLWithPath: dirPath)
        let nodeIndex = Int(arg("--node-index", in: args) ?? "0") ?? 0
        let epochs = Int(arg("--epochs", in: args) ?? "5") ?? 5
        let batchSize = Int(arg("--batch-size", in: args) ?? "32") ?? 32
        // Default 0.01, matching Python's actual default exactly
        // (run_experiment.py: "--lr", default=0.01) — see run-round's
        // identical fix for the full explanation of why the earlier 0.1
        // default was wrong and what it caused.
        let learningRate = Float(arg("--learning-rate", in: args) ?? "0.01") ?? 0.01
        let seed = UInt64(arg("--seed", in: args) ?? "42") ?? 42

        print("[train-cnn] loading node \(nodeIndex) train shard from \(directory.path)")

        let shard: DataShard
        do {
            shard = try CIFAR10Shard.loadTrainShard(directory: directory, nodeIndex: nodeIndex)
        } catch let error as NPYError {
            print("[train-cnn] \(error.description)")
            return
        } catch let error as CIFAR10ShardError {
            print("[train-cnn] \(error.description)")
            return
        } catch {
            print("[train-cnn] unexpected error loading shard: \(error)")
            return
        }

        print("[train-cnn] shard loaded: \(shard.sampleCount) samples, imageShape \(shard.imageShape)")

        // SimpleCNN's architecture is now FIXED (matching Python's
        // SimpleCNNCIFAR exactly: 3 conv layers, 32x32 CIFAR-10 input) —
        // an earlier version of this command derived cIn/h/w from
        // whatever shard it was pointed at, working for any shape; that's
        // no longer true after the architecture rebuild. Still validate
        // the shard's shape against what's now fixed, so a wrong dataset
        // produces a clear error here rather than a confusing
        // shape-mismatch failure inside convForward.
        guard shard.imageShape.count == 4 else {
            print("[train-cnn] unexpected imageShape \(shard.imageShape) — expected 4 dimensions (N,C,H,W)")
            return
        }
        let expectedShape = (cIn: 3, h: 32, w: 32)
        guard shard.imageShape[1] == expectedShape.cIn,
              shard.imageShape[2] == expectedShape.h,
              shard.imageShape[3] == expectedShape.w else {
            print("[train-cnn] shard imageShape \(shard.imageShape) doesn't match SimpleCNN's fixed architecture (expects N,\(expectedShape.cIn),\(expectedShape.h),\(expectedShape.w) — i.e. real CIFAR-10)")
            return
        }
        let modelConfig = SimpleCNNConfig(n: batchSize)
        let model = SimpleCNN(config: modelConfig, seed: seed)

        let trainerConfig = CNNTrainerConfig(batchSize: batchSize, epochsPerRound: epochs, learningRate: learningRate, seed: seed)
        let trainer = CNNTrainer(model: model, config: trainerConfig)

        print("[train-cnn] training: \(epochs) epoch(s), batch size \(batchSize), lr \(learningRate), seed \(seed)")
        print("[train-cnn] architecture: (N,3,32,32) -> conv1(3->16,3x3) -> relu -> pool -> conv2(16->32,3x3) -> relu -> pool -> conv3(32->64,3x3) -> relu -> pool -> fc(\(modelConfig.fcInputFeatures)->10) -> softmax")
        print("[train-cnn] reshuffling shard each epoch via seeded RNG, matching Python's per-epoch shuffle (see CNNTrainer.swift for why this matters)")

        do {
            let results = try trainer.trainRound(
                shard: shard,
                onEpochComplete: { result in
                    print("  epoch \(result.epoch)/\(epochs): mean loss = \(String(format: "%.4f", result.meanLoss))  (\(result.batchCount) batches)")
                },
                onBatchProgress: { batchIndex, totalBatches in
                    // Mid-epoch progress — without this, a large shard with
                    // many batches per epoch gives zero output for however
                    // long that epoch takes, which is indistinguishable
                    // from a hang. Printed with \r... no, plain print is
                    // fine here (simpler, and avoids terminal-control-code
                    // portability concerns) — slightly more output lines,
                    // not worth the complexity trade for this.
                    print("    batch \(batchIndex)/\(totalBatches)")
                },
                batchProgressInterval: 100
            )

            guard let first = results.first, let last = results.last else { return }
            print("""

            [train-cnn] Summary:
              first epoch mean loss: \(String(format: "%.4f", first.meanLoss))
              last epoch mean loss:  \(String(format: "%.4f", last.meanLoss))
            """)
        } catch let error as CNNTrainerError {
            print("[train-cnn] \(error.description)")
        } catch {
            print("[train-cnn] unexpected error during training: \(error)")
        }
    }

    /// Numerical parity diagnostic — prints initial weight values and runs a
    /// fixed forward+backward pass so outputs can be compared directly against
    /// Python's trainer.py SimpleCNNCIFAR on the same inputs.
    /// Run with same seed=42 as Python. Compare against Python output from:
    ///   model = SimpleCNNCIFAR(seed=42)
    ///   x = np.random.default_rng(0).uniform(0,1,(1,3,32,32)).astype(np.float32)

    /// real SimpleCNN instances on two different node shards independently,
    /// evaluates each pre-aggregation, aggregates via the REAL
    /// GossipAggregator.aggregate, then evaluates the aggregated result —
    /// isolating model/aggregation correctness from gossip/networking/
    private static func runTestAggregator() {
        print("[test-aggregator] verifying GossipAggregator.aggregate / Tensor.swift's weightedSum against hand-computed expected values")
        print("[test-aggregator] this is real, newly-added code (SIMD4 + parallel across cores) not yet verified against a compiler — same purpose as test-cnn was for the other Tensor.swift primitives")

        var allPassed = true

        // Test 1: 6-element tensor (4 in the SIMD4 main loop + 2 in the
        // scalar tail), 3 peers with sample counts [10, 20, 30]. Expected
        // values computed independently in Python, not by trusting this
        // implementation to grade itself:
        //   weights = [10/60, 20/60, 30/60] = [0.16667, 0.33333, 0.5]
        //   expected = [53.5, 107.0, 160.5, 214.0, 267.5, 321.0]
        do {
            let local = PeerUpdate(nodeID: 0, sampleCount: 10, parameters: [[1, 2, 3, 4, 5, 6]])
            let peerB = PeerUpdate(nodeID: 1, sampleCount: 20, parameters: [[10, 20, 30, 40, 50, 60]])
            let peerC = PeerUpdate(nodeID: 2, sampleCount: 30, parameters: [[100, 200, 300, 400, 500, 600]])

            let result = GossipAggregator.aggregate(localUpdate: local, peerUpdates: [peerB, peerC])
            let expected: [Float32] = [53.5, 107.0, 160.5, 214.0, 267.5, 321.0]

            let passed = result.count == 1 && closeEnough(result[0], expected)
            print("  test 1 (6-element, 3 peers, SIMD4 main loop + scalar tail): \(passed ? "PASS" : "FAIL")")
            if !passed {
                print("    expected: \(expected)")
                print("    got:      \(result.first ?? [])")
                allPassed = false
            }
        }

        // Test 2: 4-element tensor (exactly one SIMD4 group, no scalar
        // tail at all) — exercises the main-loop path with nothing left
        // over, a real edge case the first test doesn't isolate cleanly
        // since it mixes both paths.
        do {
            let local = PeerUpdate(nodeID: 0, sampleCount: 1, parameters: [[2, 4, 6, 8]])
            let peer = PeerUpdate(nodeID: 1, sampleCount: 1, parameters: [[10, 10, 10, 10]])
            // equal weights (1/2 each): expected = [(2+10)/2, (4+10)/2, (6+10)/2, (8+10)/2] = [6, 7, 8, 9]
            let result = GossipAggregator.aggregate(localUpdate: local, peerUpdates: [peer])
            let expected: [Float32] = [6, 7, 8, 9]
            let passed = result.count == 1 && closeEnough(result[0], expected)
            print("  test 2 (4-element exact SIMD4 group, no scalar tail, 2 equal-weight peers): \(passed ? "PASS" : "FAIL")")
            if !passed {
                print("    expected: \(expected)")
                print("    got:      \(result.first ?? [])")
                allPassed = false
            }
        }

        // Test 3: empty peerUpdates — must return localUpdate.parameters
        // UNCHANGED rather than dividing by zero (this is existing
        // documented behavior in GossipAggregator's doc comment, not new
        // for this change, but worth re-confirming it still holds after
        // routing through weightedSum).
        do {
            let local = PeerUpdate(nodeID: 0, sampleCount: 5, parameters: [[1, 2, 3]])
            let result = GossipAggregator.aggregate(localUpdate: local, peerUpdates: [])
            let passed = result.count == 1 && closeEnough(result[0], [1, 2, 3])
            print("  test 3 (no peers — must return local parameters unchanged): \(passed ? "PASS" : "FAIL")")
            if !passed {
                print("    expected: [1, 2, 3]")
                print("    got:      \(result.first ?? [])")
                allPassed = false
            }
        }

        // Test 4: multiple tensors in one update (a generic 2-tensor
        // example here, illustrative only — not tied to SimpleCNN's
        // actual current tensor count, which is 8) — confirms
        // aggregation is applied independently per tensor index, not
        // accidentally flattened/mixed across tensors.
        do {
            let local = PeerUpdate(nodeID: 0, sampleCount: 1, parameters: [[1, 1], [100, 100]])
            let peer = PeerUpdate(nodeID: 1, sampleCount: 1, parameters: [[3, 3], [300, 300]])
            let result = GossipAggregator.aggregate(localUpdate: local, peerUpdates: [peer])
            // equal weights: tensor 0 -> [2,2], tensor 1 -> [200,200]
            let passed = result.count == 2 && closeEnough(result[0], [2, 2]) && closeEnough(result[1], [200, 200])
            print("  test 4 (multiple tensors aggregated independently): \(passed ? "PASS" : "FAIL")")
            if !passed {
                print("    expected: [[2, 2], [200, 200]]")
                print("    got:      \(result)")
                allPassed = false
            }
        }

        print("")
        print(allPassed ? "[test-aggregator] ALL TESTS PASSED" : "[test-aggregator] SOME TESTS FAILED — see above")
    }

    /// Float32 equality check with a small absolute tolerance, since
    /// expected values were computed independently (in Python, at Double
    /// precision) and SIMD4 accumulation order can differ slightly from a
    /// naive left-to-right sum — exact bit-for-bit equality isn't the right
    /// check here, "matches to within float precision" is.
    private static func closeEnough(_ actual: [Float32], _ expected: [Float32], tolerance: Float32 = 0.001) -> Bool {
        guard actual.count == expected.count else { return false }
        return zip(actual, expected).allSatisfy { abs($0 - $1) < tolerance }
    }

    private static func runTestCNN(_ args: [String]) {
        let steps = Int(arg("--steps", in: args) ?? "50") ?? 50
        let batchSize = Int(arg("--batch-size", in: args) ?? "32") ?? 32
        // Default 0.01, matching Python's actual default exactly
        // (run_experiment.py: "--lr", default=0.01) — see run-round's
        // identical fix for the full explanation. This is the same
        // command whose --fixed-batch run (loss dropping cleanly 2.96 ->
        // 0.88 over ~50 steps, then spiking to 9.79 and plateauing ~2.0)
        // is what surfaced this bug in the first place — the old 0.1
        // default was directly responsible for that instability.
        let learningRate = Float(arg("--learning-rate", in: args) ?? "0.01") ?? 0.01
        let seed = UInt64(arg("--seed", in: args) ?? "42") ?? 42
        let fixedBatch = args.contains("--fixed-batch")

        // SimpleCNN's architecture is now FIXED (matching Python's
        // SimpleCNNCIFAR exactly: 3 conv layers, 32x32 CIFAR-10 input) —
        // an earlier version of this command used a deliberately tiny 8x8
        // synthetic shape for a fast wiring smoke test; that's no longer
        // possible since cIn/h/w aren't configurable anymore. Still a
        // valid wiring test, just against the real (larger, slower-but-
        // still-fast-enough) architecture rather than a toy one — the
        // random-data generation logic below already reads
        // config.cIn/h/w/classes dynamically, so it needs no changes
        // beyond this construction call and the print statements.
        let config = SimpleCNNConfig(n: batchSize)
        let model = SimpleCNN(config: config, seed: seed)

        print("[test-cnn] training SimpleCNN on SYNTHETIC data — \(steps) steps, batch size \(batchSize), lr \(learningRate), seed \(seed)")
        print("[test-cnn] architecture: (N,3,32,32) -> conv1(3->16,3x3) -> relu -> pool -> conv2(16->32,3x3) -> relu -> pool -> conv3(32->64,3x3) -> relu -> pool -> fc(\(config.fcInputFeatures)->10) -> softmax")
        print("[test-cnn] this verifies Tensor.swift's wiring, NOT model quality — loss should decrease, that's the whole test")
        if fixedBatch {
            print("[test-cnn] --fixed-batch: training on ONE fixed batch repeated every step (not fresh random data each step).")
            print("[test-cnn] This is the right diagnostic if a normal run shows flat loss: a fixed batch with fixed labels")
            print("[test-cnn] DOES have a learnable pattern (it's the same N examples every time) — if loss still won't drop")
            print("[test-cnn] here, that's real evidence of a wiring bug, not just 'random labels have nothing to learn.'")
        }

        var rng = SplitMix64(seed: seed &+ 1)  // separate stream from the model's own weight-init RNG
        var lossHistory: [Float] = []

        let imageCount = batchSize * config.cIn * config.h * config.w
        let fixedImages = fixedBatch ? (0..<imageCount).map { _ in Float(rng.nextUniform()) } : nil
        let fixedLabels = fixedBatch ? (0..<batchSize).map { _ in Int(rng.next() % UInt64(config.classes)) } : nil

        for step in 1...steps {
            let images = fixedImages ?? (0..<imageCount).map { _ in Float(rng.nextUniform()) }
            let labels = fixedLabels ?? (0..<batchSize).map { _ in Int(rng.next() % UInt64(config.classes)) }

            let cache = model.forward(images: images)
            let lossValue = model.loss(cache: cache, labels: labels)
            lossHistory.append(lossValue)

            guard lossValue.isFinite else {
                print("[test-cnn] step \(step): loss is \(lossValue) — NOT FINITE. Something in the forward/backward wiring is broken (NaN/Inf), stopping early.")
                return
            }

            let gradients = model.backward(cache: cache, labels: labels)
            model.applySGDStep(gradients: gradients, learningRate: learningRate)

            if step == 1 || step % max(1, steps / 10) == 0 || step == steps {
                print("  step \(step)/\(steps): loss = \(String(format: "%.4f", lossValue))")
            }
        }

        guard lossHistory.count >= 2 else { return }
        let firstLoss = lossHistory[0]
        let lastLoss = lossHistory[lossHistory.count - 1]
        // Random 10-class data has an expected cross-entropy loss around
        // ln(10) ≈ 2.303 at initialization (uniform-ish predictions before
        // any training signal). This isn't checked as a hard assertion —
        // just printed as a sanity reference alongside the actual numbers.
        print("""

        [test-cnn] Summary:
          first loss: \(String(format: "%.4f", firstLoss))
          last loss:  \(String(format: "%.4f", lastLoss))
          change:     \(String(format: "%.4f", lastLoss - firstLoss))  (negative = decreased, expected if wiring is correct)
          reference:  ln(10) ≈ 2.303 is the expected loss for uniform random predictions over 10 classes
        """)
        if lastLoss < firstLoss {
            print("  Loss decreased — forward/backward/update wiring appears correct.")
        } else {
            print("  Loss did NOT decrease — this points at a real bug in Tensor.swift's forward/backward wiring, not just bad luck (synthetic data with a consistent seed should show SOME learning signal even if the architecture itself is a poor fit for the task).")
        }
    }

    private static func arg(_ name: String, in args: [String]) -> String? {
        guard let idx = args.firstIndex(of: name), idx + 1 < args.count else { return nil }
        return args[idx + 1]
    }

    /// Shared topology-loading path for `gossip` and `test-connectivity`,
    /// which both need: parse `--topology`/`--topology-mode`/`--node-id`,
    /// load and resolve the file, print the resolved peer set. Returns nil
    /// (after printing an error and shutting down the group) rather than
    /// throwing, so each call site can just `guard let topology = ... else
    /// { return }` instead of repeating error-handling boilerplate.
    private static func loadTopologyFromArgs(
        _ args: [String],
        commandName: String,
        group: MultiThreadedEventLoopGroup
    ) async throws -> Topology? {
        let topologyPathStr = arg("--topology", in: args) ?? "topology.json"
        let modeStr = arg("--topology-mode", in: args) ?? "full-mesh"

        guard let nodeID = arg("--node-id", in: args) else {
            print("[\(commandName)] --node-id is required (e.g. --node-id pi-1)")
            try await group.shutdownGracefully()
            return nil
        }
        guard let mode = TopologyMode(rawValue: modeStr) else {
            print("[\(commandName)] unknown --topology-mode '\(modeStr)', expected 'full-mesh' or 'file'")
            try await group.shutdownGracefully()
            return nil
        }

        let topologyURL = URL(fileURLWithPath: topologyPathStr)
        let topology = try Topology.load(path: topologyURL, localNodeID: nodeID, mode: mode)
        print("[\(commandName)] node \(topology.localNode.id) (\(topology.mode.rawValue)) — \(topology.peers.count) peer(s): \(topology.peers.map(\.id).joined(separator: ", "))")
        return topology
    }
}

