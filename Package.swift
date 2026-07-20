// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "RADFL",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "radfl", targets: ["RADFL"]),
        .library(name: "RADFLCore", targets: ["RADFLCore"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-nio.git", from: "2.65.0"),
    ],
    targets: [
        .target(
            name: "RADFLCore",
            dependencies: [
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOConcurrencyHelpers", package: "swift-nio"),
            ],
            swiftSettings: [
                .unsafeFlags(["-Ounchecked"], .when(configuration: .release))
            ]
        ),
        .executableTarget(
            name: "RADFL",
            dependencies: ["RADFLCore"],
            swiftSettings: [
                .unsafeFlags(["-Ounchecked"], .when(configuration: .release))
            ]
            // No hardcoded static-stdlib linker flag here — pass
            // `--static-swift-stdlib` on the `swift build` command line instead
            // (see build-linux.sh), so it's applied once via the CLI rather than
            // duplicated/possibly conflicting with a baked-in unsafeFlags entry.
            //
            // Toolchain: 6.2.4-RELEASE_debian_trixie_aarch64 (glibc-based,
            // confirmed working on real Pi Zero 2W hardware — see README for
            // why this is used instead of the official musl static-linux SDK,
            // which has an open SIGILL issue on Cortex-A53 as of writing).
        ),
        .testTarget(
            name: "RADFLCoreTests",
            dependencies: ["RADFLCore"]
        )
    ]
)
