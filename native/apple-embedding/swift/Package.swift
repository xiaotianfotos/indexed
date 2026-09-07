// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "IndexedAppleEmbedding",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "AppleEmbeddingCore", targets: ["AppleEmbeddingCore"]),
        .executable(
            name: "indexed-apple-embedding",
            targets: ["IndexedAppleEmbeddingService"]
        ),
        .executable(name: "indexed-private-ane-check", targets: ["PrivateANECheck"]),
    ],
    dependencies: [
        .package(path: "Vendor/mlx-swift-lm"),
        .package(
            url: "https://github.com/ml-explore/mlx-swift.git",
            exact: "0.31.4"
        ),
        .package(
            url: "https://github.com/huggingface/swift-transformers.git",
            exact: "1.3.3"
        ),
    ],
    targets: [
        .target(
            name: "PrivateANEBridge",
            publicHeadersPath: "include",
            cSettings: [.unsafeFlags(["-fobjc-arc"])],
            linkerSettings: [.linkedFramework("Foundation"), .linkedFramework("IOSurface")]
        ),
        .target(
            name: "AppleEmbeddingCore",
            dependencies: [
                "PrivateANEBridge",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            linkerSettings: [
                .linkedFramework("AVFoundation"),
                .linkedFramework("CoreML"),
                .linkedFramework("CryptoKit"),
            ]
        ),
        .executableTarget(
            name: "IndexedAppleEmbeddingService",
            dependencies: ["AppleEmbeddingCore"]
        ),
        .executableTarget(
            name: "PrivateANECheck",
            dependencies: ["PrivateANEBridge", "AppleEmbeddingCore"]
        ),
    ]
)
