import CoreML
import Foundation
import MLX

final class CoreMLDecoderSegment: @unchecked Sendable {
    static let sequenceLength = 256
    static let hiddenSize = 2048
    static let rotarySize = 64
    static let nextLayer = 4

    private let model: MLModel
    let fingerprint: String

    private init(model: MLModel, fingerprint: String) {
        self.model = model
        self.fingerprint = fingerprint
    }

    static func load(
        from sourceURL: URL,
        modelPackageFingerprint: String,
        cacheDirectory: URL? = nil
    ) async throws -> CoreMLDecoderSegment {
        let values = try sourceURL.resourceValues(forKeys: [.isDirectoryKey])
        guard values.isDirectory == true else {
            throw ServiceError.model("Core ML decoder segment 路径不是目录")
        }
        let digest = try ModelPackageValidator.sha256Tree(of: sourceURL)
        try validateManifest(
            for: sourceURL,
            digest: digest,
            modelPackageFingerprint: modelPackageFingerprint
        )
        let compiledURL = try await CoreMLCompilationCache.compiledModel(
            from: sourceURL,
            prefix: "decoder",
            expectedDigest: digest,
            cacheDirectory: cacheDirectory
        )

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        if #available(macOS 15.0, *) {
            var hints = MLOptimizationHints()
            hints.specializationStrategy = .fastPrediction
            configuration.optimizationHints = hints
        }
        let model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        let inputs = Set(model.modelDescription.inputDescriptionsByName.keys)
        guard inputs.isSuperset(of: ["hidden_states", "position_cos", "position_sin"]),
              model.modelDescription.outputDescriptionsByName["output"] != nil else {
            throw ServiceError.model("Core ML decoder segment 的输入输出契约不兼容")
        }
        return CoreMLDecoderSegment(model: model, fingerprint: digest)
    }

    private struct SegmentManifest: Decodable {
        let schemaVersion: Int
        let runtimeSemantics: String
        let modelPackageFingerprint: String
        let segmentSHA256Tree: String
        let sequenceLength: Int
        let hiddenSize: Int
        let rotarySize: Int
        let firstLayer: Int
        let lastLayer: Int
        let nextLayer: Int

        enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version"
            case runtimeSemantics = "runtime_semantics"
            case modelPackageFingerprint = "model_package_fingerprint"
            case segmentSHA256Tree = "segment_sha256_tree"
            case sequenceLength = "sequence_length"
            case hiddenSize = "hidden_size"
            case rotarySize = "rotary_size"
            case firstLayer = "first_layer"
            case lastLayer = "last_layer"
            case nextLayer = "next_layer"
        }
    }

    private static func validateManifest(
        for sourceURL: URL,
        digest: String,
        modelPackageFingerprint: String
    ) throws {
        let manifestURL = sourceURL.appendingPathExtension("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ServiceError.model("Core ML decoder segment 缺少绑定 manifest：\(manifestURL.path)")
        }
        let manifest = try JSONDecoder().decode(
            SegmentManifest.self,
            from: Data(contentsOf: manifestURL, options: .mappedIfSafe)
        )
        guard manifest.schemaVersion == 1,
              manifest.runtimeSemantics == "wemm-coreml-decoder-segment-v1",
              manifest.sequenceLength == sequenceLength,
              manifest.hiddenSize == hiddenSize,
              manifest.rotarySize == rotarySize,
              manifest.firstLayer == 0,
              manifest.lastLayer == nextLayer - 1,
              manifest.nextLayer == nextLayer else {
            throw ServiceError.model("Core ML decoder segment manifest 与运行时契约不兼容")
        }
        guard manifest.modelPackageFingerprint == modelPackageFingerprint else {
            throw ServiceError.model("Core ML decoder segment 不属于当前模型包")
        }
        guard manifest.segmentSHA256Tree.lowercased() == digest else {
            throw ServiceError.model("Core ML decoder segment SHA-256 tree 校验失败")
        }
    }

    func predict(
        hiddenStates: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        tokens: Int
    ) throws -> (hiddenStates: MLXArray, milliseconds: Double) {
        guard tokens > 0, tokens <= Self.sequenceLength,
              hiddenStates.shape == [1, tokens, Self.hiddenSize],
              cos.shape == [1, tokens, Self.rotarySize],
              sin.shape == [1, tokens, Self.rotarySize] else {
            throw ServiceError.internalFailure(
                "decoder segment 收到不兼容 shape: hidden=\(hiddenStates.shape), "
                    + "cos=\(cos.shape), sin=\(sin.shape)"
            )
        }

        let hiddenInput = try MLMultiArray(
            shape: [1, Self.hiddenSize, 1, Self.sequenceLength] as [NSNumber],
            dataType: .float16
        )
        let cosInput = try MLMultiArray(
            shape: [1, Self.rotarySize, 1, Self.sequenceLength] as [NSNumber],
            dataType: .float16
        )
        let sinInput = try MLMultiArray(
            shape: [1, Self.rotarySize, 1, Self.sequenceLength] as [NSNumber],
            dataType: .float16
        )

        let hidden = hiddenStates.asType(.float16)
        let cosValues = cos.asType(.float16)
        let sinValues = sin.asType(.float16)
        eval(hidden, cosValues, sinValues)
        let hiddenSource = hidden.asArray(Float16.self)
        let cosSource = cosValues.asArray(Float16.self)
        let sinSource = sinValues.asArray(Float16.self)
        let hiddenTarget = hiddenInput.dataPointer.assumingMemoryBound(to: Float16.self)
        let cosTarget = cosInput.dataPointer.assumingMemoryBound(to: Float16.self)
        let sinTarget = sinInput.dataPointer.assumingMemoryBound(to: Float16.self)
        hiddenTarget.initialize(repeating: 0, count: hiddenInput.count)
        cosTarget.initialize(repeating: 1, count: cosInput.count)
        sinTarget.initialize(repeating: 0, count: sinInput.count)
        for channel in 0 ..< Self.hiddenSize {
            let destination = channel * Self.sequenceLength
            for token in 0 ..< tokens {
                hiddenTarget[destination + token] = hiddenSource[token * Self.hiddenSize + channel]
            }
        }
        for channel in 0 ..< Self.rotarySize {
            let destination = channel * Self.sequenceLength
            for token in 0 ..< tokens {
                let source = token * Self.rotarySize + channel
                cosTarget[destination + token] = cosSource[source]
                sinTarget[destination + token] = sinSource[source]
            }
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": hiddenInput,
            "position_cos": cosInput,
            "position_sin": sinInput,
        ])
        let started = ContinuousClock.now
        let prediction = try model.prediction(from: provider)
        let elapsed = started.duration(to: .now).components
        let predictMS = (
            Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
        ) * 1000
        guard let output = prediction.featureValue(for: "output")?.multiArrayValue,
              output.dataType == .float16,
              output.shape.map(\.intValue) == [1, Self.hiddenSize, 1, Self.sequenceLength] else {
            throw ServiceError.internalFailure("Core ML decoder segment 返回不兼容的 output")
        }
        let outputValues = Array(
            UnsafeBufferPointer(
                start: output.dataPointer.assumingMemoryBound(to: Float16.self),
                count: output.count
            )
        )
        let result = MLXArray(
            Data(bytes: outputValues, count: outputValues.count * MemoryLayout<Float16>.size),
            [1, Self.hiddenSize, 1, Self.sequenceLength],
            type: Float16.self
        )[0..., 0..., 0, 0 ..< tokens].transposed(0, 2, 1).asType(hiddenStates.dtype)
        return (result, predictMS)
    }
}
