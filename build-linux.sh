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

set -euo pipefail

CONFIGURATION="release"
EXPECTED_GLIBC="2.41"

echo "==> Looking for an installed SDK matching 'trixie' or 'aarch64' + 'linux' (excluding musl)"
SDK_LIST="$(swift sdk list)"
echo "${SDK_LIST}"
echo ""

# Try to find a line that looks like the Trixie aarch64 bundle and isn't musl.
CANDIDATE_BUNDLE="$(echo "${SDK_LIST}" | grep -i 'trixie' | grep -i 'aarch64' | head -n1 || true)"

if [ -z "${CANDIDATE_BUNDLE}" ]; then
    echo "ERROR: Could not auto-detect the debian_trixie_aarch64 SDK bundle from 'swift sdk list' output above."
    echo "If it IS installed, find its exact bundle/triple name in the listing above and set"
    echo "SWIFT_SDK_TARGET manually below this line, then re-run:"
    echo ""
    echo "    SWIFT_SDK_TARGET=<exact-triple-from-swift-sdk-list> ./build-linux.sh"
    echo ""
    exit 1
fi

echo "==> Found candidate bundle: ${CANDIDATE_BUNDLE}"

# Allow override via env var; otherwise try the bundle name directly as the
# --swift-sdk argument (SwiftPM accepts either the bundle id or the target
# triple depending on version — if this fails, run `swift sdk list` yourself
# to get the exact triple swift build expects).
SWIFT_SDK_TARGET="${SWIFT_SDK_TARGET:-${CANDIDATE_BUNDLE}}"

echo "==> Building for SDK target: ${SWIFT_SDK_TARGET} (${CONFIGURATION})"
echo "    (glibc-based SDK — binary will dynamically link against target glibc;"
echo "     confirmed compatible with Pi nodes running glibc ${EXPECTED_GLIBC} as of last check)"

swift build \
    --configuration "${CONFIGURATION}" \
    --swift-sdk "${SWIFT_SDK_TARGET}" \
    --static-swift-stdlib \
    --product radfl

# Output path varies by exact triple SwiftPM resolves to — search for it
# rather than assuming the bundle name and the .build/ subdirectory name match.
BIN_PATH="$(find .build -path "*/${CONFIGURATION}/radfl" -type f 2>/dev/null | head -n1 || true)"

if [ -z "${BIN_PATH}" ] || [ ! -f "${BIN_PATH}" ]; then
    echo "ERROR: could not locate the built 'radfl' binary under .build/."
    echo "The build command above may have failed, or used a different SDK triple"
    echo "than expected — check its output, and inspect .build/ directly:"
    find .build -maxdepth 2 -type d 2>/dev/null || true
    exit 1
fi

echo "==> Build succeeded: ${BIN_PATH}"
file "${BIN_PATH}" || true

# DEBUG INFO: kept by default, NOT stripped — a deliberate choice, not an
# accidental side effect of -c release. "release" controls optimization
# level (-O); whether DWARF debug symbols are generated/embedded is a
# SEPARATE, independently-controlled thing — this is why a release build
# can still legitimately show "with debug_info, not stripped" via `file`.
#
# Keeping debug info matters RIGHT NOW specifically because the SimpleCNN
# architecture (3 conv layers, valid convolution, He init — see
# SimpleCNN.swift/Tensor.swift) has never been compiled or run on real Pi
# hardware. If something crashes there (a real possibility — array-bounds
# trap, integer overflow trap, segfault — none of this has been
# compiler-checked yet, only hand-traced), debug info is what turns a bare
# hex-address crash into a symbolicated backtrace pointing at an actual
# line of Swift source. Stripping now, before that's been verified, would
# trade away exactly the diagnostic capability most needed at this stage.
#
# Once the new architecture is confirmed working correctly on real
# hardware (clean runs, sane loss/accuracy curves matching the Python
# comparison), stripping becomes a reasonable, standard step for a "done"
# deployment artifact — meaningfully smaller binaries are a real benefit
# given the Pi Zero 2W's constrained storage, and faster to `scp` to all
# 10 nodes. Opt in explicitly with STRIP=1 when that point is reached:
#
#   STRIP=1 ./build-linux.sh
#
if [ "${STRIP:-0}" = "1" ]; then
    echo "==> STRIP=1: stripping debug info from ${BIN_PATH}"
    # llvm-strip (not the host's native `strip`) is the safer choice here:
    # the Mac's own `strip` is built for Mach-O and may not correctly
    # handle this aarch64/Linux ELF binary when cross-compiling from
    # macOS. llvm-strip is COMMONLY available alongside a Swift toolchain
    # installation (LLVM is a build dependency of Swift itself), but this
    # wasn't directly confirmed for your specific toolchain/install while
    # writing this script — hence checking for it explicitly with
    # command -v rather than assuming it's always there, and falling back
    # to host `strip` with a clear warning if it's missing, rather than
    # silently failing.
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
    echo "==> Debug info KEPT (default) — set STRIP=1 to produce a stripped binary once the"
    echo "    new architecture is verified working on real Pi hardware (see comment above)."
fi

echo ""
echo "==> Before deploying, verify glibc compatibility on the TARGET node(s):"
echo "    ssh <pi-node> ldd --version"
echo "    Expect glibc ${EXPECTED_GLIBC} or newer (this SDK was built against glibc ${EXPECTED_GLIBC})."
echo "    If a node reports an OLDER glibc, this binary will fail to load there —"
echo "    re-image that node to Trixie, or rebuild against a matching older SDK."
echo ""
echo "==> Copy the binary AND topology.json to your first test node (just pi-1 for now,"
echo "    per the plan to manually verify one node before rolling out further):"
echo "    scp ${BIN_PATH} pi-1:/home/pi/radfl"
echo "    scp topology.json pi-1:/home/pi/topology.json"
echo ""
echo "==> First real-hardware run — test-connectivity on pi-1 alone:"
echo "    ssh pi-1"
echo "    /home/pi/radfl test-connectivity --topology /home/pi/topology.json --node-id pi-1"
echo ""
echo "    With no other nodes running yet, expect:"
echo "      - The listener starts cleanly (this is the main thing being verified —"
echo "        does the binary even RUN on real Pi Zero 2W hardware without SIGILL"
echo "        or a missing-glibc-symbol error? Neither has been confirmed yet for"
echo "        THIS binary specifically — only the toolchain itself, via a trivial"
echo "        hello-world, was previously verified on real hardware.)"
echo "      - 0/9 peers reachable (heartbeat ticking) — correct, since nothing else is running"
echo "      - Ctrl-C should exit cleanly, no hang"
echo ""
echo "==> Once pi-1 alone runs cleanly, the natural next step is the same binary on a"
echo "    second node to confirm two real Pis can actually see each other — then scale"
echo "    out to the rest. (Not done automatically here — that's your call once pi-1"
echo "    is confirmed working.)"

