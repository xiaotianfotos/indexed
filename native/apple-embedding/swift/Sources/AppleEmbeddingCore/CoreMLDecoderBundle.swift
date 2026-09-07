import CoreML
import Foundation
import MLX

/// Complete 24-layer WeMM/Qwen3.5 decoder split into fixed-shape Core ML graphs.
/// Segment outputs stay as MLMultiArray values between graphs, avoiding
/// unnecessary Core ML -> MLX -> Core ML copies.
final class CoreMLDecoderBundle: @unchecked Sendable {
    static let hiddenSize = 2048
    static let rotarySize = 64
    static let nextLayer = 24

    struct Prediction {
        let hiddenStates: MLXArray
        let milliseconds: Double
        let segmentMilliseconds: [String: Double]
    }

    private struct LoadedSegment {
        let compiledURL: URL
        let residentModel: MLModel?
        let firstLayer: Int
        let lastLayer: Int
    }

    private struct BundleManifest: Decodable {
        struct Segment: Decodable {
            let path: String
            let segmentSHA256Tree: String
            let firstLayer: Int
            let lastLayer: Int

            enum CodingKeys: String, CodingKey {
                case path
                case segmentSHA256Tree = "segment_sha256_tree"
                case firstLayer = "first_layer"
                case lastLayer = "last_layer"
            }
        }

        let schemaVersion: Int
        let runtimeSemantics: String
        let modelPackageFingerprint: String
        let sequenceLength: Int
        let hiddenSize: Int
        let rotarySize: Int
        let firstLayer: Int
        let lastLayer: Int
        let nextLayer: Int
        let accuracyProfile: String
        let segments: [Segment]

        enum CodingKeys: String, CodingKey {
            case segments
            case schemaVersion = "schema_version"
            case runtimeSemantics = "runtime_semantics"
            case modelPackageFingerprint = "model_package_fingerprint"
            case sequenceLength = "sequence_length"
            case hiddenSize = "hidden_size"
            case rotarySize = "rotary_size"
            case firstLayer = "first_layer"
            case lastLayer = "last_layer"
            case nextLayer = "next_layer"
            case accuracyProfile = "accuracy_profile"
        }
    }

    private let segments: [LoadedSegment]
    let loadingStrategy: CoreMLDecoderLoadingStrategy
    let fingerprint: String
    let sequenceLength: Int

    private init(
        segments: [LoadedSegment],
        fingerprint: String,
        sequenceLength: Int,
        loadingStrategy: CoreMLDecoderLoadingStrategy
    ) {
        self.segments = segments
        self.fingerprint = fingerprint
        self.sequenceLength = sequenceLength
        self.loadingStrategy = loadingStrategy
    }

    static func load(
        from bundleURL: URL,
        modelPackageFingerprint: String,
        cacheDirectory: URL? = nil,
        loadingStrategy: CoreMLDecoderLoadingStrategy = .resident
    ) async throws -> CoreMLDecoderBundle {
        let root = bundleURL.standardizedFileURL.resolvingSymlinksInPath()
        let values = try root.resourceValues(forKeys: [.isDirectoryKey])
        guard values.isDirectory == true else {
            throw ServiceError.model("Core ML decoder bundle 路径不是目录")
        }
        let manifestURL = root.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ServiceError.model("Core ML decoder bundle 缺少 manifest.json")
        }
        let manifest = try JSONDecoder().decode(
            BundleManifest.self,
            from: Data(contentsOf: manifestURL, options: .mappedIfSafe)
        )
        guard manifest.schemaVersion == 1,
              manifest.runtimeSemantics == "wemm-coreml-decoder-bundle-v1",
              (1 ... 8192).contains(manifest.sequenceLength),
              manifest.hiddenSize == hiddenSize,
              manifest.rotarySize == rotarySize,
              manifest.firstLayer == 0,
              manifest.lastLayer == nextLayer - 1,
              manifest.nextLayer == nextLayer,
              ["high", "production"].contains(manifest.accuracyProfile),
              manifest.modelPackageFingerprint == modelPackageFingerprint else {
            throw ServiceError.model("Core ML decoder bundle manifest 与当前模型/运行时不兼容")
        }

        var expectedFirstLayer = 0
        var loaded: [LoadedSegment] = []
        for segment in manifest.segments {
            guard segment.firstLayer == expectedFirstLayer,
                  segment.lastLayer >= segment.firstLayer else {
                throw ServiceError.model("Core ML decoder bundle 的 layer 范围不连续")
            }
            let sourceURL = try containedURL(root: root, relativePath: segment.path)
            guard sourceURL.pathExtension == "mlpackage" else {
                throw ServiceError.model("decoder bundle segment 必须是 .mlpackage")
            }
            let digest = try ModelPackageValidator.sha256Tree(of: sourceURL)
            guard digest == segment.segmentSHA256Tree.lowercased() else {
                throw ServiceError.model(
                    "decoder L\(segment.firstLayer)-L\(segment.lastLayer) SHA-256 tree 校验失败"
                )
            }
            let compiledURL = try await CoreMLCompilationCache.compiledModel(
                from: sourceURL,
                prefix: "decoder-l\(segment.firstLayer)-l\(segment.lastLayer)",
                expectedDigest: digest,
                cacheDirectory: cacheDirectory
            )
            let model = loadingStrategy == .resident
                ? try loadModel(
                    compiledURL: compiledURL,
                    firstLayer: segment.firstLayer,
                    lastLayer: segment.lastLayer,
                    loadingStrategy: loadingStrategy
                ) : nil
            loaded.append(.init(
                compiledURL: compiledURL,
                residentModel: model,
                firstLayer: segment.firstLayer,
                lastLayer: segment.lastLayer
            ))
            expectedFirstLayer = segment.lastLayer + 1
        }
        guard expectedFirstLayer == nextLayer, !loaded.isEmpty else {
            throw ServiceError.model("Core ML decoder bundle 未覆盖完整 L0-L23")
        }
        return CoreMLDecoderBundle(
            segments: loaded,
            fingerprint: try ModelPackageValidator.sha256(of: manifestURL),
            sequenceLength: manifest.sequenceLength,
            loadingStrategy: loadingStrategy
        )
    }

    func predict(
        hiddenStates: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        tokens: Int,
        cancellation: CancellationToken
    ) throws -> Prediction {
        guard tokens > 0, tokens <= sequenceLength,
              hiddenStates.shape == [1, tokens, Self.hiddenSize],
              cos.shape == [1, tokens, Self.rotarySize],
              sin.shape == [1, tokens, Self.rotarySize] else {
            throw ServiceError.internalFailure(
                "decoder bundle 收到不兼容 shape: hidden=\(hiddenStates.shape), "
                    + "cos=\(cos.shape), sin=\(sin.shape)"
            )
        }

        var current = try makePaddedInput(
            hiddenStates,
            channels: Self.hiddenSize,
            tokens: tokens,
            padding: 0
        )
        let cosInput = try makePaddedInput(
            cos,
            channels: Self.rotarySize,
            tokens: tokens,
            padding: 1
        )
        let sinInput = try makePaddedInput(
            sin,
            channels: Self.rotarySize,
            tokens: tokens,
            padding: 0
        )

        var totalMS = 0.0
        var segmentMS: [String: Double] = [:]
        for segment in segments {
            try cancellation.check()
            let segmentResult: (output: MLMultiArray, milliseconds: Double) = try autoreleasepool {
                let started = ContinuousClock.now
                let model = try segment.residentModel ?? Self.loadModel(
                    compiledURL: segment.compiledURL,
                    firstLayer: segment.firstLayer,
                    lastLayer: segment.lastLayer,
                    loadingStrategy: loadingStrategy
                )
                let provider = try MLDictionaryFeatureProvider(dictionary: [
                    "hidden_states": current,
                    "position_cos": cosInput,
                    "position_sin": sinInput,
                ])
                let prediction = try model.prediction(from: provider)
                let duration = started.duration(to: .now).components
                let elapsedMS = (
                    Double(duration.seconds) + Double(duration.attoseconds) / 1e18
                ) * 1000
                guard let output = prediction.featureValue(for: "output")?.multiArrayValue,
                      output.dataType == .float16,
                      output.shape.map(\.intValue)
                        == [1, Self.hiddenSize, 1, sequenceLength] else {
                    throw ServiceError.internalFailure(
                        "decoder L\(segment.firstLayer)-L\(segment.lastLayer) 返回不兼容 output"
                    )
                }
                return (output, elapsedMS)
            }
            current = segmentResult.output
            let elapsedMS = segmentResult.milliseconds
            totalMS += elapsedMS
            segmentMS["coreml_decoder_l\(segment.firstLayer)_l\(segment.lastLayer)"] = elapsedMS
        }
        try cancellation.check()
        return Prediction(
            hiddenStates: mlxArray(from: current, tokens: tokens, dtype: hiddenStates.dtype),
            milliseconds: totalMS,
            segmentMilliseconds: segmentMS
        )
    }

    private static func loadModel(
        compiledURL: URL,
        firstLayer: Int,
        lastLayer: Int,
        loadingStrategy: CoreMLDecoderLoadingStrategy
    ) throws -> MLModel {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        // FastPrediction pays a large specialization cost at model load time.
        // It is profitable for resident models, but sequential mode reloads
        // each segment for every request and must keep loads cheap.
        if loadingStrategy == .resident, #available(macOS 15.0, *) {
            var hints = MLOptimizationHints()
            hints.specializationStrategy = .fastPrediction
            configuration.optimizationHints = hints
        }
        let model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        let inputs = Set(model.modelDescription.inputDescriptionsByName.keys)
        guard inputs.isSuperset(of: ["hidden_states", "position_cos", "position_sin"]),
              model.modelDescription.outputDescriptionsByName["output"] != nil else {
            throw ServiceError.model(
                "decoder L\(firstLayer)-L\(lastLayer) 输入输出契约不兼容"
            )
        }
        return model
    }

    private static func containedURL(root: URL, relativePath: String) throws -> URL {
        guard !relativePath.isEmpty, !relativePath.hasPrefix("/") else {
            throw ServiceError.model("decoder bundle manifest 包含非法路径")
        }
        let candidate = root.appendingPathComponent(relativePath)
            .standardizedFileURL.resolvingSymlinksInPath()
        let prefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
        guard candidate.path.hasPrefix(prefix) else {
            throw ServiceError.model("decoder bundle segment 路径越出 bundle")
        }
        return candidate
    }

    private func makePaddedInput(
        _ source: MLXArray,
        channels: Int,
        tokens: Int,
        padding: Float16
    ) throws -> MLMultiArray {
        let result = try MLMultiArray(
            shape: [1, channels, 1, sequenceLength] as [NSNumber],
            dataType: .float16
        )
        let values = source.asType(.float16)
        eval(values)
        let sourceValues = values.asArray(Float16.self)
        let target = result.dataPointer.assumingMemoryBound(to: Float16.self)
        target.initialize(repeating: padding, count: result.count)
        for channel in 0 ..< channels {
            let destination = channel * sequenceLength
            for token in 0 ..< tokens {
                target[destination + token] = sourceValues[token * channels + channel]
            }
        }
        return result
    }

    private func mlxArray(
        from source: MLMultiArray,
        tokens: Int,
        dtype: DType
    ) -> MLXArray {
        let values = Array(
            UnsafeBufferPointer(
                start: source.dataPointer.assumingMemoryBound(to: Float16.self),
                count: source.count
            )
        )
        return MLXArray(
            Data(bytes: values, count: values.count * MemoryLayout<Float16>.size),
            [1, Self.hiddenSize, 1, sequenceLength],
            type: Float16.self
        )[0..., 0..., 0, 0 ..< tokens].transposed(0, 2, 1).asType(dtype)
    }
}
