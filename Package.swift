// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MessageDifferentiationKit",
    platforms: [
        .macOS("15.0")
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "MessageDifferentiationKit",
            targets: ["MessageDifferentiationKit"]
        ),
        .library(
            name: "MPISwift",
            targets: ["MPISwift"]
        ),
        // Executable examples
        .executable(
            name: "HelloMPI",
            targets: ["HelloMPI"]
        ),
        .executable(
            name: "CollectivesDemo",
            targets: ["CollectivesDemo"]
        ),
        .executable(
            name: "PointToPointDemo",
            targets: ["PointToPointDemo"]
        ),
        .executable(
            name: "TopologyDemo",
            targets: ["TopologyDemo"]
        ),
        .executable(
            name: "ScalarOperationsExample",
            targets: ["ScalarOperationsExample"]
        ),
        .executable(
            name: "DLPackExample",
            targets: ["DLPackExample"]
        ),
        .executable(
            name: "DistributedGradientDescentExample",
            targets: ["DistributedGradientDescentExample"]
        ),
        .executable(
            name: "AsyncOperationsExample",
            targets: ["AsyncOperationsExample"]
        ),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0"),
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.3"),
    ],
    targets: [
        // C module for MPI 5.0 headers
        .systemLibrary(
            name: "CMPIBindings",
            pkgConfig: "ompi",
            providers: [
                .brew(["open-mpi"]),
                .apt(["libopenmpi-dev"])
            ]
        ),

        // C module for DLPack headers
        .target(
            name: "CDLPack",
            dependencies: [],
            path: "Sources/CDLPack"
        ),

        // Low-level MPI Swift bindings
        .target(
            name: "MPISwift",
            dependencies: ["CMPIBindings"],
            swiftSettings: [
                .enableExperimentalFeature("Differentiable"),
            ]
        ),

        // High-level differentiable MPI API
        .target(
            name: "MessageDifferentiationKit",
            dependencies: [
                "MPISwift",
                "CDLPack",
                .product(name: "Numerics", package: "swift-numerics"),
            ],
            swiftSettings: [
                .enableExperimentalFeature("Differentiable"),
            ]
        ),

        // Tests
        .testTarget(
            name: "MPISwiftTests",
            dependencies: ["MPISwift"]
        ),
        .testTarget(
            name: "MessageDifferentiationKitTests",
            dependencies: ["MessageDifferentiationKit"]
        ),

        // Example executables
        .executableTarget(
            name: "HelloMPI",
            dependencies: [
                "MessageDifferentiationKit",
                "MPISwift",
                "CMPIBindings",
            ],
            path: "Examples",
            exclude: [
                "README.md",
                "CollectivesDemo.swift",
                "TopologyDemo.swift",
                "PointToPointDemo.swift",
                "ScalarOperationsExample.swift",
                "DLPackExample.swift",
                "DistributedGradientDescentExample.swift",
                "AsyncOperationsExample.swift",
            ],
            sources: ["HelloMPI.swift"]
        ),
        .executableTarget(
            name: "CollectivesDemo",
            dependencies: [
                "MessageDifferentiationKit",
                "MPISwift",
                "CMPIBindings",
            ],
            path: "Examples",
            exclude: [
                "README.md",
                "HelloMPI.swift",
                "TopologyDemo.swift",
                "PointToPointDemo.swift",
                "ScalarOperationsExample.swift",
                "DLPackExample.swift",
                "DistributedGradientDescentExample.swift",
                "AsyncOperationsExample.swift",
            ],
            sources: ["CollectivesDemo.swift"]
        ),
        .executableTarget(
            name: "PointToPointDemo",
            dependencies: [
                "MessageDifferentiationKit",
                "MPISwift",
                "CMPIBindings",
            ],
            path: "Examples",
            exclude: [
                "README.md",
                "HelloMPI.swift",
                "CollectivesDemo.swift",
                "TopologyDemo.swift",
                "ScalarOperationsExample.swift",
                "DLPackExample.swift",
                "DistributedGradientDescentExample.swift",
                "AsyncOperationsExample.swift",
            ],
            sources: ["PointToPointDemo.swift"]
        ),
        .executableTarget(
            name: "TopologyDemo",
            dependencies: [
                "MessageDifferentiationKit",
                "MPISwift",
                "CMPIBindings",
            ],
            path: "Examples",
            exclude: [
                "README.md",
                "HelloMPI.swift",
                "CollectivesDemo.swift",
                "PointToPointDemo.swift",
                "ScalarOperationsExample.swift",
                "DLPackExample.swift",
                "DistributedGradientDescentExample.swift",
                "AsyncOperationsExample.swift",
            ],
            sources: ["TopologyDemo.swift"]
        ),
        .executableTarget(
            name: "ScalarOperationsExample",
            dependencies: [
                "MessageDifferentiationKit",
                "MPISwift",
            ],
            path: "Examples",
            exclude: [
                "README.md",
                "CollectivesDemo.swift",
                "HelloMPI.swift",
                "TopologyDemo.swift",
                "PointToPointDemo.swift",
                "DLPackExample.swift",
                "DistributedGradientDescentExample.swift",
                "AsyncOperationsExample.swift",
            ],
            sources: ["ScalarOperationsExample.swift"],
            swiftSettings: [
                .enableExperimentalFeature("Differentiable"),
            ]
        ),
        .executableTarget(
            name: "DLPackExample",
            dependencies: [
                "MessageDifferentiationKit",
                "MPISwift",
                "CDLPack",
            ],
            path: "Examples",
            exclude: [
                "README.md",
                "CollectivesDemo.swift",
                "HelloMPI.swift",
                "TopologyDemo.swift",
                "PointToPointDemo.swift",
                "ScalarOperationsExample.swift",
                "DistributedGradientDescentExample.swift",
                "AsyncOperationsExample.swift",
            ],
            sources: ["DLPackExample.swift"],
            swiftSettings: [
                .enableExperimentalFeature("Differentiable"),
            ]
        ),
        .executableTarget(
            name: "DistributedGradientDescentExample",
            dependencies: [
                "MessageDifferentiationKit",
                "MPISwift",
            ],
            path: "Examples",
            exclude: [
                "README.md",
                "CollectivesDemo.swift",
                "HelloMPI.swift",
                "TopologyDemo.swift",
                "PointToPointDemo.swift",
                "ScalarOperationsExample.swift",
                "DLPackExample.swift",
                "AsyncOperationsExample.swift",
            ],
            sources: ["DistributedGradientDescentExample.swift"],
            swiftSettings: [
                .enableExperimentalFeature("Differentiable"),
            ]
        ),
        .executableTarget(
            name: "AsyncOperationsExample",
            dependencies: [
                "MessageDifferentiationKit",
                "MPISwift",
                "CMPIBindings",
            ],
            path: "Examples",
            exclude: [
                "README.md",
                "CollectivesDemo.swift",
                "HelloMPI.swift",
                "TopologyDemo.swift",
                "PointToPointDemo.swift",
                "ScalarOperationsExample.swift",
                "DLPackExample.swift",
                "DistributedGradientDescentExample.swift",
            ],
            sources: ["AsyncOperationsExample.swift"],
            swiftSettings: [
                .enableExperimentalFeature("Differentiable"),
            ]
        ),
    ]
)
// Note: SimpleDistributedTest added manually for debugging
