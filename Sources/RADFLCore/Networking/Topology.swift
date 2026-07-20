// Topology.swift
//
// Parses topology.json (the same file format used by the Python baseline's
// gossipunlearn/gossip.py Topology class) and computes each node's peer set.
//
// File shape (matches topology.json / topology_mac.json as used by the
// Python harness):
//   {
//     "nodes": [
//       {"id": "pi-1", "host": "127.0.0.1", "port": 9001},
//       ...
//     ],
//     "links": [
//       {"source": "pi-1", "target": "pi-2"},
//       ...
//     ]
//   }
//
// Mode semantics (mirrors Python's `--topology-mode` flag):
//   - "full-mesh": every node's peer set is every OTHER node in `nodes`,
//     regardless of what `links` says. This is the mode used for the actual
//     paper experiments (RQ1), per project decisions.
//   - "file": peer set is derived from `links` (ring topology in the
//     existing topology_mac.json) — implemented for completeness/parity
//     with the Python side's smoke-testing mode, but not the mode used for
//     real experiments.
//
// Deliberately NOT implemented: any topology mode beyond these two. RQ2
// (topology comparison) is out of scope for the current paper per project
// decisions, so there's no need to build out arbitrary topology shapes here
// yet — full-mesh and file/ring cover what the Python side actually uses.

import Foundation

public struct TopologyNode: Codable, Sendable, Hashable {
    public let id: String
    public let host: String
    public let port: Int

    /// Derives a numeric ID from the `id` string for use as `GossipMessage.senderNodeID`,
    /// which is a fixed `UInt32` on the wire (see GossipFrame.swift) — changing that to a
    /// variable-length string would be a real wire-protocol change, not just a type swap,
    /// so this bridges the topology file's string IDs to the existing protocol instead of
    /// changing it. Assumes the convention already used throughout this project: IDs of the
    /// form "pi-<N>" (e.g. "pi-1", "pi-10"). Throws rather than silently defaulting if `id`
    /// doesn't match that pattern, since a silent fallback (e.g. hashing the string) would
    /// make wire-level node identification non-obvious and hard to debug.
    public func numericID() throws -> UInt32 {
        guard id.hasPrefix("pi-"), let n = UInt32(id.dropFirst(3)) else {
            throw TopologyError.nonNumericNodeID(id)
        }
        return n
    }
}

public struct TopologyLink: Codable, Sendable, Hashable {
    public let source: String
    public let target: String
}

/// Raw on-disk shape of topology.json — kept separate from the resolved
/// `Topology` type below so parsing and peer-resolution are distinct steps.
struct TopologyFile: Codable {
    let nodes: [TopologyNode]
    let links: [TopologyLink]
}

public enum TopologyMode: String, Sendable {
    case fullMesh = "full-mesh"
    case file = "file"
}

public enum TopologyError: Error, CustomStringConvertible {
    case fileNotFound(URL)
    case decodingFailed(URL, Error)
    case localNodeNotInTopology(nodeID: String, knownIDs: [String])
    case duplicateNodeID(String)
    case nonNumericNodeID(String)

    public var description: String {
        switch self {
        case .fileNotFound(let url):
            return "Topology file not found at \(url.path)"
        case .decodingFailed(let url, let error):
            return "Failed to decode topology file at \(url.path): \(error)"
        case .localNodeNotInTopology(let nodeID, let knownIDs):
            return "Local node ID '\(nodeID)' not found in topology file. Known IDs: \(knownIDs.joined(separator: ", "))"
        case .duplicateNodeID(let id):
            return "Topology file contains duplicate node ID '\(id)' — node IDs must be unique"
        case .nonNumericNodeID(let id):
            return "Node ID '\(id)' doesn't match the expected \"pi-<N>\" pattern, so it can't be mapped to the wire protocol's numeric senderNodeID. Rename it to match the existing convention, or extend GossipMessage's wire format to carry string IDs directly."
        }
    }
}

/// Resolved topology for a specific local node: who it is, and who its
/// peers are, given a topology file and a mode.
public struct Topology: Sendable {
    public let localNode: TopologyNode
    public let peers: [TopologyNode]
    public let mode: TopologyMode
    public let allNodes: [TopologyNode]

    /// Loads and parses `path`, then resolves the peer set for `localNodeID`
    /// under the given `mode`.
    public static func load(path: URL, localNodeID: String, mode: TopologyMode) throws -> Topology {
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw TopologyError.fileNotFound(path)
        }

        let data: Data
        do {
            data = try Data(contentsOf: path)
        } catch {
            throw TopologyError.decodingFailed(path, error)
        }

        let file: TopologyFile
        do {
            file = try JSONDecoder().decode(TopologyFile.self, from: data)
        } catch {
            throw TopologyError.decodingFailed(path, error)
        }

        // Validate uniqueness up front — a duplicate ID would silently break
        // peer resolution (lookups would be ambiguous) rather than raise a
        // clear error, so check this before anything else.
        var seenIDs = Set<String>()
        for node in file.nodes {
            guard seenIDs.insert(node.id).inserted else {
                throw TopologyError.duplicateNodeID(node.id)
            }
        }

        guard let localNode = file.nodes.first(where: { $0.id == localNodeID }) else {
            throw TopologyError.localNodeNotInTopology(
                nodeID: localNodeID,
                knownIDs: file.nodes.map(\.id)
            )
        }

        let peers: [TopologyNode]
        switch mode {
        case .fullMesh:
            // Every other node, regardless of `links`. This matches the
            // Python side's full-mesh mode, which is what's actually used
            // for the paper's real experiments.
            peers = file.nodes.filter { $0.id != localNodeID }

        case .file:
            // Peer set derived from `links`: any node connected to
            // `localNodeID` in either direction (links aren't assumed to be
            // declared symmetrically in the file).
            let nodesByID = Dictionary(uniqueKeysWithValues: file.nodes.map { ($0.id, $0) })
            var peerIDs = Set<String>()
            for link in file.links {
                if link.source == localNodeID {
                    peerIDs.insert(link.target)
                } else if link.target == localNodeID {
                    peerIDs.insert(link.source)
                }
            }
            peers = peerIDs.compactMap { nodesByID[$0] }
        }

        return Topology(localNode: localNode, peers: peers, mode: mode, allNodes: file.nodes)
    }
}

