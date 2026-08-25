#!/usr/bin/env bash
# build-linux.sh
#
# Cross-compiles RADFL from macOS to an aarch64 Linux binary for deployment to
# Raspberry Pi Zero 2W nodes (Cortex-A53, ARMv8.0-A, aarch64), running
# Raspberry Pi OS Trixie (Debian 13, glibc 2.41).
#
# TOOLCHAIN: 6.2.4-RELEASE_debian_trixie_aarch64
#   This is a swift-sdk-generator-produced cross-compilation SDK built from a
#   real Debian Trixie Docker image (glibc-based), NOT the official musl-based
#   Static Linux SDK. This distinction matters and was arrived at empirically:
#
#   - The official musl static-linux SDK (aarch64-swift-linux-musl) has an
#     OPEN, UNRESOLVED upstream issue (swiftlang/swift#88351, filed Apr 2026)
#     where its prebuilt runtime uses ARMv8.1+ LSE atomics, causing SIGILL on
#     real ARMv8.0-A Cortex-A53 hardware — i.e. this exact chip. This is the
#     same failure class that excluded PyTorch from the Python baseline.
#   - The debian_trixie SDK used here is a different build pipeline (real
#     Debian sysroot via swift-sdk-generator, not the bespoke musl runtime)
#     and has been CONFIRMED WORKING on the actual Pi Zero 2W hardware for
#     this project (manually verified, not just QEMU).
#
# IMPLICATION — this is NOT a fully static binary:
#   Unlike the musl SDK, this produces a binary dynamically linked against the
#   target's glibc. This is fine here because it's been verified the Pi nodes'
#   installed glibc matches:
#
#     Pi node:  glibc 2.41-12+rpt1 (Raspberry Pi OS Trixie)
#     SDK:      glibc 2.41 (Debian Trixie base image)
#
#   These match (the `+rpt1` suffix is just Raspberry Pi Foundation's own
#   packaging revision on stock Debian glibc — same ABI). If you ever deploy
#   to a node still on Bookworm (older glibc) or a future Pi OS release
#   (newer glibc than what this SDK was built against), re-verify with:
#     ssh <pi-node> ldd --version
#   A binary built against newer glibc symbols than the target has will fail
#   to load (a different failure mode than SIGILL — "version GLIBC_2.4x not
#   found" — but equally fatal). This is a real, recurring check to make
#   whenever you add nodes or re-image any Pi, not a one-time concern.
#
# `--static-swift-stdlib` below statically links the *Swift* runtime/stdlib
# into the binary, so individual Pi nodes don't need a separate Swift install
# — but it does NOT make the binary independent of the target's glibc, which
# is why the glibc-match check above still matters.
#
# One-time setup (run once per Mac toolchain), if not already installed:
#   swift sdk install <url-to-6.2.4-RELEASE_debian_trixie_aarch64.artifactbundle.tar.gz> \
#     --checksum <checksum>
#   (source: https://github.com/swift-embedded-linux/swift-sdks/releases)
#
# NOTE ON THE SDK TRIPLE NAME: this script does NOT hardcode a guessed triple
# (e.g. "aarch64-swift-linux-gnu" / "aarch64-unknown-linux-gnu") because the
# exact string a swift-sdk-generator bundle registers under can vary by how
# it was generated, and guessing wrong here would just produce a confusing
# "SDK not found" error instead of a working build. Instead it discovers the
# triple from `swift sdk list` automatically.
#
# ─────────────────────────────────────────────────────────────────────────────
# CHANGELOG — artifact selection bug, fixed here
#
# The previous version located the built binary with:
#
#     find .build -path "*/release/radfl" -type f | head -n1
#
# `find` returns results in FILESYSTEM order, not build order, and `head -n1`
# takes whichever came first. When .build contained more than one
# `*/release/radfl` — e.g. a macOS native build alongside the cross build, or
# artifacts from two different SDK triples after re-registering a bundle — the
# script could select a STALE binary, report "Build succeeded", and print a
# deploy command pointing at it.
#
# This was not hypothetical. It silently deployed a pre-change binary to the
# cluster: the Pi's binary had a current mtime and the correct size, the source
# tree was correct, the build genuinely succeeded — and the running binary
# still lacked the newest changes. Diagnosing it cost a full round of
# elimination across source, build, and deployment, because every individual
# check looked fine.
#
# Three fixes, all cheap:
#   1. Select the NEWEST match (`ls -t`), not the first found.
#   2. HARD-FAIL if the selected file is not an aarch64 Linux ELF. `file` was
#      already being run purely for display; making its result load-bearing
#      costs nothing and catches a macOS binary being selected.
#   3. WARN when more than one candidate exists at all, since that ambiguity
#      is the actual precondition for the bug and is worth surfacing rather
#      than silently resolving.
#
# There is also a --clean flag now, because `rm -rf .build` is the definitive
# answer whenever anything about artifact identity looks wrong.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

CONFIGURATION="release"
EXPECTED_GLIBC="2.41"
CLEAN=0
VERIFY_MARKER="${VERIFY_MARKER:-batch-size}"

usage() {
    cat <<'USAGE'
Usage: ./build-linux.sh [--clean] [--help]

  --clean     Remove .build entirely before building. Slower, but it is the
              definitive fix whenever artifact identity is in doubt — a stale
              cross-compile output cannot be selected if it does not exist.

Environment:
  SWIFT_SDK_TARGET   Override SDK auto-detection with an exact triple.
  STRIP=1            Strip debug info from the resulting binary.
  VERIFY_MARKER      String expected to appear in the built binary, used as a
                     freshness check (default: "batch-size"). Set to a token
                     from whatever you most recently changed to confirm the
                     build actually contains it, or empty to skip the check.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --clean) CLEAN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

if [ "${CLEAN}" = "1" ]; then
    echo "==> --clean: removing .build/"
    rm -rf .build
fi

# ── Generate BuildInfo.swift ────────────────────────────────────────────────
# Compiles the source revision into the binary so run_config.json can record
# which commit produced a given set of results.
#
# Generated rather than computed at runtime because reading git state from
# Swift would need a subprocess spawn, which this project avoids in the
# training path on principle — the same reasoning that moved ResourceUsage off
# sched_getaffinity to procfs after the Trixie SDK's Glibc modulemap failed to
# expose it. Metadata collection must not be able to break a cross-compile.
#
# binarySHA256 stays "pending" here for the obvious reason: the binary does not
# exist yet at compile time. It is written to build_info.json post-link below,
# where it can be computed correctly, rather than embedding a value the binary
# cannot know about itself.
BUILDINFO_PATH="Sources/RADFLCore/Telemetry/BuildInfo.swift"
if [ -d "$(dirname "${BUILDINFO_PATH}")" ]; then
    GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    # Only meaningful if a commit was actually resolved. Outside a git tree
    # `git diff` also fails, which would otherwise append "-dirty" to "unknown"
    # and produce the nonsensical "unknown-dirty".
    if [ "${GIT_COMMIT}" != "unknown" ]; then
        if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
            GIT_COMMIT="${GIT_COMMIT}-dirty"
        fi
    fi
    BUILD_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    cat > "${BUILDINFO_PATH}" <<EOF
// GENERATED by build-linux.sh — do not edit by hand.
// Regenerated on every build; see build-linux.sh for why this is a generated
// file rather than a runtime lookup.

public enum BuildInfo {
    /// Short git commit of the source this binary was built from, with a
    /// "-dirty" suffix if the working tree had uncommitted changes.
    public static let gitCommit = "${GIT_COMMIT}"

    /// UTC timestamp of the build, ISO 8601.
    public static let buildTimestampUTC = "${BUILD_TS}"

    /// SHA-256 of the linked binary — written to build_info.json post-link,
    /// since the binary cannot contain its own hash.
    public static let binarySHA256 = "pending"
}
EOF
    echo "==> BuildInfo.swift generated (commit ${GIT_COMMIT}, ${BUILD_TS})"
    case "${GIT_COMMIT}" in
        *-dirty)
            echo "    NOTE: working tree has uncommitted changes. Results from this"
            echo "    binary trace to a commit that does not fully describe them."
            ;;
        unknown)
            echo "    NOTE: not a git tree (or git unavailable) — commit recorded as"
            echo "    'unknown'. Honest, but results from this binary cannot be traced"
            echo "    to a revision."
            ;;
    esac
    echo ""
else
    echo "==> WARNING: ${BUILDINFO_PATH%/*} not found — skipping BuildInfo generation."
    echo "    run_config.json will record gitCommit as whatever is checked in."
    echo ""
fi

# ── Locate the SDK ──────────────────────────────────────────────────────────
echo "==> Looking for an installed SDK matching 'trixie' or 'aarch64' + 'linux' (excluding musl)"
SDK_LIST="$(swift sdk list)"
echo "${SDK_LIST}"
echo ""

CANDIDATE_BUNDLE="$(echo "${SDK_LIST}" | grep -i 'trixie' | grep -i 'aarch64' | head -n1 || true)"

if [ -z "${CANDIDATE_BUNDLE}" ]; then
    echo "ERROR: Could not auto-detect the debian_trixie_aarch64 SDK bundle from 'swift sdk list' output above."
    echo "If it IS installed, find its exact bundle/triple name in the listing above and set"
    echo "SWIFT_SDK_TARGET manually, then re-run:"
    echo ""
    echo "    SWIFT_SDK_TARGET=<exact-triple-from-swift-sdk-list> ./build-linux.sh"
    echo ""
    exit 1
fi

echo "==> Found candidate bundle: ${CANDIDATE_BUNDLE}"

SWIFT_SDK_TARGET="${SWIFT_SDK_TARGET:-${CANDIDATE_BUNDLE}}"

echo "==> Building for SDK target: ${SWIFT_SDK_TARGET} (${CONFIGURATION})"
echo "    (glibc-based SDK — binary will dynamically link against target glibc;"
echo "     confirmed compatible with Pi nodes running glibc ${EXPECTED_GLIBC} as of last check)"

swift build \
    --configuration "${CONFIGURATION}" \
    --swift-sdk "${SWIFT_SDK_TARGET}" \
    --static-swift-stdlib \
    --product radfl

# ── Locate the built binary ─────────────────────────────────────────────────
# See the CHANGELOG at the top of this file. The previous `| head -n1` on an
# unordered `find` is what silently deployed a stale binary to the cluster.
CANDIDATES="$(find .build -path "*/${CONFIGURATION}/radfl" -type f 2>/dev/null || true)"
CANDIDATE_COUNT="$(printf '%s\n' "${CANDIDATES}" | grep -c . || true)"

if [ "${CANDIDATE_COUNT}" -eq 0 ]; then
    echo "ERROR: could not locate a built 'radfl' binary under .build/."
    echo "The build command above may have failed, or used a different SDK triple"
    echo "than expected — check its output, and inspect .build/ directly:"
    find .build -maxdepth 2 -type d 2>/dev/null || true
    exit 1
fi

if [ "${CANDIDATE_COUNT}" -gt 1 ]; then
    echo ""
    echo "==> WARNING: ${CANDIDATE_COUNT} candidate 'radfl' binaries found under .build/:"
    printf '%s\n' "${CANDIDATES}" | while read -r c; do
        [ -n "$c" ] && echo "      $(ls -l "$c" | awk '{print $6, $7, $8}')  $c"
    done
    echo "    Selecting the NEWEST. This ambiguity is exactly the condition that"
    echo "    previously caused a stale binary to be deployed — if anything looks"
    echo "    wrong, re-run with --clean to remove it entirely."
    echo ""
fi

# Newest by mtime, not first by filesystem order.
BIN_PATH="$(printf '%s\n' "${CANDIDATES}" | grep . | xargs ls -t 2>/dev/null | head -n1)"

if [ -z "${BIN_PATH}" ] || [ ! -f "${BIN_PATH}" ]; then
    echo "ERROR: candidate selection failed unexpectedly."
    exit 1
fi

# ── Verify it is what we think it is ────────────────────────────────────────
FILE_OUT="$(file "${BIN_PATH}")"
echo "==> Build succeeded: ${BIN_PATH}"
echo "    ${FILE_OUT}"

if ! echo "${FILE_OUT}" | grep -q "ELF 64-bit.*aarch64"; then
    echo ""
    echo "ERROR: ${BIN_PATH} is NOT an aarch64 Linux ELF binary."
    echo "Most likely a macOS native build was selected instead of the cross build."
    echo "Re-run with --clean, or remove the stray artifact."
    exit 1
fi

# Freshness check. A build can succeed, produce a valid aarch64 ELF, and still
# not contain the change just made — if it compiled a different source tree, or
# an incremental build did not pick something up. Grepping the binary for a
# token from the most recent change is a direct check that the artifact
# actually contains it.
if [ -n "${VERIFY_MARKER}" ]; then
    if strings "${BIN_PATH}" 2>/dev/null | grep -q -- "${VERIFY_MARKER}"; then
        echo "    freshness: marker '${VERIFY_MARKER}' present in binary"
    else
        echo ""
        echo "==> WARNING: marker '${VERIFY_MARKER}' NOT found in the built binary."
        echo "    Either the marker is stale (update VERIFY_MARKER to a token from"
        echo "    your most recent change), or this binary does not contain that"
        echo "    change. Do NOT deploy until this is understood — a binary that"
        echo "    runs correctly while missing recent changes is difficult to"
        echo "    diagnose from the cluster side."
        echo ""
    fi
fi

# ── Strip (opt-in) ──────────────────────────────────────────────────────────
#
# DEBUG INFO: kept by default, NOT stripped — a deliberate choice, not an
# accidental side effect of -c release. "release" controls optimization level
# (-O); whether DWARF debug symbols are generated/embedded is a SEPARATE,
# independently-controlled thing — this is why a release build can still
# legitimately show "with debug_info, not stripped" via `file`.
#
# Opt in explicitly once an artifact is considered done:
#   STRIP=1 ./build-linux.sh
#
if [ "${STRIP:-0}" = "1" ]; then
    echo "==> STRIP=1: stripping debug info from ${BIN_PATH}"
    # llvm-strip (not the host's native `strip`) is the safer choice here: the
    # Mac's own `strip` is built for Mach-O and may not correctly handle this
    # aarch64/Linux ELF binary when cross-compiling from macOS.
    if command -v llvm-strip >/dev/null 2>&1; then
        llvm-strip "${BIN_PATH}"
    else
        echo "    WARNING: llvm-strip not found on PATH — falling back to host 'strip',"
        echo "    which may not correctly handle this aarch64/Linux ELF binary when"
        echo "    cross-compiling from macOS. If this fails or produces a broken binary,"
        echo "    install/locate llvm-strip (usually bundled with the Swift toolchain)."
        strip "${BIN_PATH}"
    fi
    echo "==> Stripped binary:"
    file "${BIN_PATH}" || true
else
    echo "    debug info KEPT (default) — set STRIP=1 for a stripped artifact."
fi

# ── build_info.json ─────────────────────────────────────────────────────────
# Written next to the binary, post-link, so the SHA is of the artifact that
# actually exists. Collect this alongside results (experiments/runs/<id>/env/)
# so a set of results can be tied to the exact binary that produced it.
BIN_DIR="$(dirname "${BIN_PATH}")"
BIN_SHA="$( { shasum -a 256 "${BIN_PATH}" 2>/dev/null || sha256sum "${BIN_PATH}"; } | awk '{print $1}')"
cat > "${BIN_DIR}/build_info.json" <<EOF
{
  "binary": "$(basename "${BIN_PATH}")",
  "binary_sha256": "${BIN_SHA}",
  "git_commit": "${GIT_COMMIT:-unknown}",
  "build_timestamp_utc": "${BUILD_TS:-unknown}",
  "sdk_target": "${SWIFT_SDK_TARGET}",
  "configuration": "${CONFIGURATION}",
  "stripped": $([ "${STRIP:-0}" = "1" ] && echo true || echo false),
  "expected_target_glibc": "${EXPECTED_GLIBC}"
}
EOF
echo "==> build_info.json written to ${BIN_DIR}/build_info.json"
echo "    sha256: ${BIN_SHA}"

# ── Deployment ──────────────────────────────────────────────────────────────
echo ""
echo "==> Before deploying, verify glibc compatibility on the TARGET node(s):"
echo "    ssh <pi-node> ldd --version"
echo "    Expect glibc ${EXPECTED_GLIBC} or newer (this SDK was built against glibc ${EXPECTED_GLIBC})."
echo "    If a node reports an OLDER glibc, this binary will fail to load there —"
echo "    re-image that node to Trixie, or rebuild against a matching older SDK."
echo ""
echo "==> Deploy to all 10 nodes:"
echo ""
echo "    for n in \$(seq 1 10); do scp -q ${BIN_PATH} pi-\$n.local:~/swift-dfl/radfl; done"
echo ""
echo "==> Then VERIFY every node got the same binary. A partial deployment"
echo "    completes a run normally while silently losing whatever the stale"
echo "    nodes were missing — check, do not assume:"
echo ""
echo "    for n in \$(seq 1 10); do"
echo "        echo -n \"pi-\$n: \"; ssh pi-\$n.local 'shasum -a 256 ~/swift-dfl/radfl | cut -c1-12'"
echo "    done"
echo ""
echo "    All ten should print: ${BIN_SHA:0:12}"
echo ""

