import Foundation
import AVFoundation
import CoreImage
import CoreML
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN
import MLXVLM
import Tokenizers

public enum VisionComputeTarget: String, Sendable {
    case ane
    case gpu

    var computeUnits: MLComputeUnits {
        switch self {
        case .ane: .cpuAndNeuralEngine
        case .gpu: .cpuAndGPU
        }
    }

    var backendLabel: String {
        switch self {
        case .ane: "coreml-ane-vision"
        case .gpu: "coreml-gpu-vision"
        }
    }
}

public enum LanguageComputeTarget: String, Sendable {
    /// Use Core ML bundles when a matching bucket exists, otherwise MLX GPU.
    case auto
    /// Always use the complete MLX decoder on GPU.
    case gpu
    /// Require a complete Core ML bundle and keep the small MLX frontend on CPU.
    case ane
}

public enum CoreMLDecoderLoadingStrategy: String, Sendable {
    /// Keep every decoder segment resident for the lowest request latency.
    case resident
    /// Keep compiled assets on disk and load one segment at a time.
    case sequential
}

public final class MLXWeMMEngine: EmbeddingEngine, @unchecked Sendable {
    private let modelPackage: ValidatedModelPackage
    private let languageModel: Qwen35Language.LanguageModel
    private let tokenizer: any MLXLMCommon.Tokenizer
    private let visionModel: MLModel
    private let visionComputeTarget: VisionComputeTarget
    private let languageComputeTarget: LanguageComputeTarget
    private let decoderSegment: CoreMLDecoderSegment?
    private let decoderBundles: [CoreMLDecoderBundle]
    private let decoderLoadingStrategy: CoreMLDecoderLoadingStrategy
    private let decoderSegmentMinimumTokens: Int
    private let imageContext = CIContext(options: [
        .cacheIntermediates: false,
        .workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
        .outputColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
    ])
    private let loadSeconds: Double
    private let warmupSeconds: Double
    private let privateMLPContext: PrivateMLPContext?
    private let pipeline: InferencePipeline
    private let normalizationTable: [Float]
    public var maxConcurrentRequests: Int { pipeline.admission.capacity }

    public var metadata: EngineMetadata {
        let languageBackend = if languageComputeTarget == .ane {
            "decoder-l0-l23-coreml-cpu-ane-only"
        } else if !decoderBundles.isEmpty {
            "decoder-l0-l23-coreml-cpu-ane"
        } else if decoderSegment != nil {
            "decoder-l0-l3-coreml-cpu-ane+mlx-swift-gpu"
        } else if privateMLPContext != nil {
            privateMLPContext?.recurrence == nil
                ? "mlx-swift-gpu+private-ane-mlp-candidate"
                : "mlx-swift-gpu+private-ane-mlp-gdn-candidate"
        } else {
            "mlx-swift-gpu"
        }
        return EngineMetadata(
            backend: "\(visionComputeTarget.backendLabel)+\(languageBackend)",
            loadSeconds: loadSeconds,
            warmupSeconds: warmupSeconds,
            allocatedBytes: Memory.activeMemory,
            packageFingerprint: modelPackage.manifest.packageFingerprint,
            embeddingSpaceTemplate: modelPackage.manifest.embeddingSpaceTemplate,
            computeDevices: MLComputeDevice.allComputeDevices.map(String.init(describing:)),
            decoderSegmentLoaded: decoderSegment != nil,
            decoderBundleLoaded: !decoderBundles.isEmpty,
            decoderMinimumTokens: (decoderSegment != nil || !decoderBundles.isEmpty)
                ? decoderSegmentMinimumTokens : nil,
            decoderMaximumTokens: !decoderBundles.isEmpty
                ? decoderBundles.map(\.sequenceLength).max()
                : decoderSegment != nil ? CoreMLDecoderSegment.sequenceLength : nil,
            decoderBucketTokenLimits: decoderBundles.map(\.sequenceLength),
            decoderSegmentFingerprint: decoderSegment?.fingerprint,
            decoderBundleFingerprint: decoderBundles.isEmpty ? nil : decoderBundles.map {
                "\($0.sequenceLength):\($0.fingerprint)"
            }.joined(separator: ","),
            decoderLoadingStrategy: decoderBundles.isEmpty ? nil : decoderLoadingStrategy.rawValue,
            languageCompute: languageComputeTarget.rawValue,
            mlxDecoderLoaded: languageComputeTarget != .ane,
            mlxDevice: languageComputeTarget == .ane ? "cpu" : "gpu",
            maxModelLength: languageComputeTarget == .ane
                ? (decoderBundles.map(\.sequenceLength).max() ?? 0) : 8192,
            privateANE: privateMLPContext?.diagnostics,
            scheduling: pipeline.diagnostics
        )
    }

    private init(
        modelPackage: ValidatedModelPackage,
        languageModel: Qwen35Language.LanguageModel,
        tokenizer: any MLXLMCommon.Tokenizer,
        visionModel: MLModel,
        visionComputeTarget: VisionComputeTarget,
        languageComputeTarget: LanguageComputeTarget,
        decoderSegment: CoreMLDecoderSegment?,
        decoderBundles: [CoreMLDecoderBundle],
        decoderLoadingStrategy: CoreMLDecoderLoadingStrategy,
        decoderSegmentMinimumTokens: Int,
        loadSeconds: Double,
        warmupSeconds: Double,
        privateMLPContext: PrivateMLPContext? = nil,
        pipeline: InferencePipeline, normalizationTable: [Float]
    ) {
        self.modelPackage = modelPackage
        self.languageModel = languageModel
        self.tokenizer = tokenizer
        self.visionModel = visionModel
        self.visionComputeTarget = visionComputeTarget
        self.languageComputeTarget = languageComputeTarget
        self.decoderSegment = decoderSegment
        self.decoderBundles = decoderBundles
        self.decoderLoadingStrategy = decoderLoadingStrategy
        self.decoderSegmentMinimumTokens = decoderSegmentMinimumTokens
        self.loadSeconds = loadSeconds
        self.warmupSeconds = warmupSeconds
        self.privateMLPContext = privateMLPContext
        self.pipeline = pipeline
        self.normalizationTable = normalizationTable
    }

    public static func load(
        packageURL: URL,
        decoderSegmentURL: URL? = nil,
        decoderBundleURLs: [URL] = [],
        decoderSegmentMinimumTokens: Int = 224,
        coreMLCacheDirectory: URL? = nil,
        visionComputeTarget: VisionComputeTarget = .ane,
        languageComputeTarget: LanguageComputeTarget = .auto,
        decoderLoadingStrategy: CoreMLDecoderLoadingStrategy = .resident,
        warmup: Bool = true,
        cacheLimitBytes: Int = 128 * 1024 * 1024,
        developmentPrivateMLP: PrivateMLPConfiguration? = nil,
        developmentGDNProfile: Data? = nil,
        developmentPipelineDepth: Int = 1
    ) async throws -> MLXWeMMEngine {
        guard (1...2).contains(developmentPipelineDepth),
              developmentPipelineDepth == 1 || developmentPrivateMLP != nil else {
            throw ServiceError.invalidRequest("Pipeline depth 2 requires an explicit private candidate")
        }
        let pipeline = InferencePipeline(depth: developmentPipelineDepth)
        if developmentPrivateMLP != nil,
           languageComputeTarget != .gpu || decoderSegmentURL != nil || !decoderBundleURLs.isEmpty {
            throw ServiceError.invalidRequest("Private MLP candidate requires the complete MLX GPU decoder")
        }
        if developmentGDNProfile != nil,
           developmentPrivateMLP?.sequenceLength != 2112 || developmentPrivateMLP?.fraction != 0.75 || developmentPrivateMLP?.maxLayers != 24 {
            throw ServiceError.invalidRequest("Private D candidate requires the frozen complete C MLP configuration")
        }
        guard decoderSegmentURL == nil || decoderBundleURLs.isEmpty else {
            throw ServiceError.invalidRequest(
                "--decoder-segment 与 --decoder-bundle 不能同时使用"
            )
        }
        if languageComputeTarget == .gpu,
           decoderSegmentURL != nil || !decoderBundleURLs.isEmpty {
            throw ServiceError.invalidRequest(
                "--language-compute gpu 不能同时加载 decoder Core ML 资产"
            )
        }
        if languageComputeTarget == .ane {
            guard decoderSegmentURL == nil, !decoderBundleURLs.isEmpty else {
                throw ServiceError.invalidRequest(
                    "--language-compute ane 必须提供至少一个完整 --decoder-bundle"
                )
            }
        }
        let started = ContinuousClock.now
        let modelPackage = try ModelPackageValidator.validate(at: packageURL, depth: .quick)
        if let developmentGDNProfile {
            _ = try PrivateGDNProfile.validate(developmentGDNProfile, packageFingerprint: modelPackage.manifest.packageFingerprint)
        }
        let configURL = modelPackage.languageURL.deletingLastPathComponent()
            .appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL, options: .mappedIfSafe)
        let config = try JSONDecoder().decode(Qwen35Configuration.self, from: configData)
        let baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: configData)

        Memory.cacheLimit = cacheLimitBytes
        let languageModel: Qwen35Language.LanguageModel
        if languageComputeTarget == .ane {
            languageModel = try Device.withDefaultDevice(.cpu) {
                let model = Qwen35Language.LanguageModel(config, embeddingOnly: true)
                try loadCoreMLFrontendWeights(
                    modelDirectory: modelPackage.languageURL.deletingLastPathComponent(),
                    model: model,
                    perLayerQuantization: baseConfig.perLayerQuantization
                )
                return model
            }
        } else {
            languageModel = Qwen35Language.LanguageModel(config, embeddingOnly: true)
            try loadWeights(
                modelDirectory: modelPackage.languageURL.deletingLastPathComponent(),
                model: languageModel,
                perLayerQuantization: baseConfig.perLayerQuantization
            )
        }
        let privateMLPContext: PrivateMLPContext?
        if let developmentPrivateMLP {
            let context = PrivateMLPContext()
            if developmentGDNProfile == nil { context.aneLane = pipeline.ane }
            _ = try PrivateHybridMLP.install(in: languageModel, configuration: developmentPrivateMLP, context: context)
            try FastQ8Projection.installGDN(in: languageModel, context: context)
            if developmentGDNProfile != nil {
                let recurrence = try PrivateGDNRecurrence()
                context.recurrence = recurrence
                languageModel.setEmbeddingPrefillBackend { layer, q, k, v, g, beta, state, mask in
                    // Python uses its unpatched fast text model below 1056
                    // tokens, while images/videos always use the patched model.
                    guard context.recurrenceAllowed else { return nil }
                    do {
                        let slot = (layer / 4) * 3 + layer % 4
                        let evaluate = {
                            try recurrence.evaluate(q: q, k: k, v: v, g: g, beta: beta,
                                state: state, mask: mask, layerSlot: slot,
                                cancellation: context.requestCancellation)
                        }
                        // Preserve the frozen D policy: a whole selected GDN
                        // burst has priority over the next Core ML vision call.
                        if slot == 0 {
                            return try pipeline.ane.withPermit(priority: true,
                                cancellation: context.requestCancellation, evaluate)
                        }
                        return try evaluate()
                    } catch ServiceError.cancelled { return (.zeros(v.shape, dtype: v.dtype), state) }
                    catch {
                        context.fallback("recurrence_error", layer: "decoder-\(layer)", error: error)
                        return nil
                    }
                }
            }
            privateMLPContext = context
        } else {
            privateMLPContext = nil
        }
        let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: modelPackage.root)
        let tokenizer: any MLXLMCommon.Tokenizer = #adaptHuggingFaceTokenizer(upstream)
        let coreMLConfiguration = MLModelConfiguration()
        coreMLConfiguration.computeUnits = visionComputeTarget.computeUnits
        let compiledVisionURL = try await CoreMLCompilationCache.compiledModel(
            from: modelPackage.visionURL,
            prefix: "vision",
            expectedDigest: modelPackage.manifest.vision.packagedSHA256Tree,
            cacheDirectory: coreMLCacheDirectory
        )
        let visionModel = try MLModel(
            contentsOf: compiledVisionURL,
            configuration: coreMLConfiguration
        )
        let decoderSegment: CoreMLDecoderSegment?
        if let decoderSegmentURL {
            decoderSegment = try await CoreMLDecoderSegment.load(
                from: decoderSegmentURL,
                modelPackageFingerprint: modelPackage.manifest.packageFingerprint,
                cacheDirectory: coreMLCacheDirectory
            )
        } else {
            decoderSegment = nil
        }
        var decoderBundles: [CoreMLDecoderBundle] = []
        for decoderBundleURL in decoderBundleURLs {
            decoderBundles.append(try await CoreMLDecoderBundle.load(
                from: decoderBundleURL,
                modelPackageFingerprint: modelPackage.manifest.packageFingerprint,
                cacheDirectory: coreMLCacheDirectory,
                loadingStrategy: decoderLoadingStrategy
            ))
        }
        decoderBundles.sort { $0.sequenceLength < $1.sequenceLength }
        guard Set(decoderBundles.map(\.sequenceLength)).count == decoderBundles.count else {
            throw ServiceError.invalidRequest("decoder bundle 的 sequence_length 不能重复")
        }
        // Materialize exact legacy normalization once. Vision staging below
        // uses CPU values only, so the language stage owns every MLX operation.
        let normalized = MLXArray((0...255).map(UInt8.init)).asType(.float32)
            / MLXArray(127.5) - MLXArray(1.0)
        let normalizationTable = normalized.asArray(Float.self)
        let loadedAt = ContinuousClock.now

        var engine = MLXWeMMEngine(
            modelPackage: modelPackage,
            languageModel: languageModel,
            tokenizer: tokenizer,
            visionModel: visionModel,
            visionComputeTarget: visionComputeTarget,
            languageComputeTarget: languageComputeTarget,
            decoderSegment: decoderSegment,
            decoderBundles: decoderBundles,
            decoderLoadingStrategy: decoderLoadingStrategy,
            decoderSegmentMinimumTokens: languageComputeTarget == .ane
                ? 1 : decoderSegmentMinimumTokens,
            loadSeconds: seconds(started.duration(to: loadedAt)),
            warmupSeconds: 0,
            privateMLPContext: privateMLPContext,
            pipeline: pipeline, normalizationTable: normalizationTable
        )
        if warmup {
            let warmupStarted = ContinuousClock.now
            _ = try engine.embed(
                messages: [.object([
                    "role": .string("user"),
                    "content": .array([.object([
                        "type": .string("text"), "text": .string("warmup"),
                    ])]),
                ])],
                dimension: 64,
                cancellation: CancellationToken()
            )
            let warmupSeconds = seconds(warmupStarted.duration(to: .now))
            engine = MLXWeMMEngine(
                modelPackage: modelPackage,
                languageModel: languageModel,
                tokenizer: tokenizer,
                visionModel: visionModel,
                visionComputeTarget: visionComputeTarget,
                languageComputeTarget: languageComputeTarget,
                decoderSegment: decoderSegment,
                decoderBundles: decoderBundles,
                decoderLoadingStrategy: decoderLoadingStrategy,
                decoderSegmentMinimumTokens: languageComputeTarget == .ane
                    ? 1 : decoderSegmentMinimumTokens,
                loadSeconds: engine.loadSeconds,
                warmupSeconds: warmupSeconds,
                privateMLPContext: privateMLPContext,
                pipeline: pipeline, normalizationTable: normalizationTable
            )
        }
        return engine
    }

    public func embed(
        messages: [JSONValue], dimension: Int, cancellation: CancellationToken
    ) throws -> EmbeddingResult {
        let totalStarted = ContinuousClock.now
        return try pipeline.admission.withPermit(cancellation: cancellation) {
            let visionWaitStarted = ContinuousClock.now
            var visionWaitMS = 0.0
            let prepared = try pipeline.withStage(.vision, cancellation: cancellation) {
                visionWaitMS = milliseconds(visionWaitStarted.duration(to: .now))
                return try prepareVision(messages: messages, cancellation: cancellation)
            }
            let languageWaitStarted = ContinuousClock.now
            return try pipeline.withStage(.language, cancellation: cancellation) {
                let languageWaitMS = milliseconds(languageWaitStarted.duration(to: .now))
                let finish = {
                    try self.embedPrepared(prepared, dimension: dimension, cancellation: cancellation,
                        totalStarted: totalStarted, visionWaitMS: visionWaitMS, languageWaitMS: languageWaitMS)
                }
                if let privateMLPContext { return try privateMLPContext.withRequest(cancellation, finish) }
                if languageComputeTarget == .ane { return try Device.withDefaultDevice(.cpu, finish) }
                return try finish()
            }
        }
    }

    private struct PreparedVision {
        let tokenIDs: [Int]
        let features: [VisionFeatures]
        let grid: THW?
        let isVideo: Bool
        let frameCount: Int
        let modality: String
        let preprocessMS: Double
        let visionMS: Double
    }

    private func prepareVision(messages: [JSONValue], cancellation: CancellationToken) throws -> PreparedVision {
        try cancellation.check()
        let preprocessStarted = ContinuousClock.now
        let parsed = try WeMMPrompt.parse(messages)
        var visualPatches: [VisionPatches] = []
        var visualGrid: THW?
        var isVideo = false
        var videoFrameCount = 0
        var prompt = parsed.prompt
        if let imageData = parsed.imageData {
            let processed = try preprocessImage(imageData)
            visualPatches = [processed.patches]
            visualGrid = processed.grid
            let imageTokenCount = processed.grid.product / 4
            prompt = prompt.replacingFirstOccurrence(
                of: "<|image_pad|>",
                with: String(repeating: "<|image_pad|>", count: imageTokenCount)
            )
        } else if let videoFrames = parsed.videoFrames {
            let processed = try preprocessVideoFrames(videoFrames, cancellation: cancellation)
            visualPatches = processed.patches
            visualGrid = processed.grid
            isVideo = true
            videoFrameCount = processed.frameCount
            let videoTokensPerTemporalPair = processed.grid.h * processed.grid.w / 4
            let videoPads = String(
                repeating: "<|video_pad|>", count: videoTokensPerTemporalPair)
            let videoPrompt = processed.timestamps.map { timestamp in
                let timestampToken = String(
                    format: "<%.1f seconds>",
                    locale: Locale(identifier: "en_US_POSIX"),
                    timestamp
                )
                return "\(timestampToken)<|vision_start|>\(videoPads)<|vision_end|>"
            }.joined()
            prompt = prompt.replacingFirstOccurrence(
                of: "<|video_pad|>",
                with: videoPrompt
            )
        } else if let videoData = parsed.videoData {
            let processed = try preprocessVideo(videoData, cancellation: cancellation)
            visualPatches = processed.patches
            visualGrid = processed.grid
            isVideo = true
            videoFrameCount = processed.frameCount
            let videoTokensPerTemporalPair = processed.grid.h * processed.grid.w / 4
            let videoPads = String(
                repeating: "<|video_pad|>", count: videoTokensPerTemporalPair)
            let videoPrompt = processed.timestamps.map { timestamp in
                let timestampToken = String(
                    format: "<%.1f seconds>",
                    locale: Locale(identifier: "en_US_POSIX"),
                    timestamp
                )
                return "\(timestampToken)<|vision_start|>\(videoPads)<|vision_end|>"
            }.joined()
            prompt = prompt.replacingFirstOccurrence(
                of: "<|video_pad|>",
                with: videoPrompt
            )
        }
        let tokenIDs = tokenizer.encode(text: prompt, addSpecialTokens: false)
        guard !tokenIDs.isEmpty else { throw ServiceError.invalidRequest("tokenizer 返回空输入") }
        guard tokenIDs.count <= 8192 else {
            throw ServiceError.invalidRequest("输入超过 WeMM 8192 token 限制", status: 413)
        }
        guard tokenIDs.last == modelPackage.manifest.embeddingTokenID else {
            throw ServiceError.model("WeMM prompt 最后一个 token 不是 <embedding>")
        }
        let preprocessMS = milliseconds(preprocessStarted.duration(to: .now))

        try cancellation.check()
        var visionMS = 0.0
        var featureParts: [VisionFeatures] = []
        if !visualPatches.isEmpty {
            let visionStarted = ContinuousClock.now
            featureParts.reserveCapacity(visualPatches.count)
            for patches in visualPatches {
                try cancellation.check()
                featureParts.append(try predictVision(patches, cancellation: cancellation))
            }
            visionMS = milliseconds(visionStarted.duration(to: .now))
        }

        return PreparedVision(tokenIDs: tokenIDs, features: featureParts, grid: visualGrid,
            isVideo: isVideo, frameCount: videoFrameCount,
            modality: isVideo ? "video" : parsed.imageData == nil ? "text" : "image",
            preprocessMS: preprocessMS, visionMS: visionMS)
    }

    private func embedPrepared(_ vision: PreparedVision, dimension: Int, cancellation: CancellationToken,
                               totalStarted: ContinuousClock.Instant, visionWaitMS: Double,
                               languageWaitMS: Double) throws -> EmbeddingResult {
        try cancellation.check()
        let languageStarted = ContinuousClock.now
        let tokenIDs = vision.tokenIDs, isVideo = vision.isVideo, visualGrid = vision.grid
        let inputIDs = MLXArray(tokenIDs.map(Int32.init)).reshaped(1, tokenIDs.count)
        privateMLPContext?.recurrenceAllowed = !vision.features.isEmpty || tokenIDs.count >= 1056
        let featureParts = vision.features.map { MLXArray($0.values).reshaped($0.shape).asType(.bfloat16) }
        let imageFeatures: MLXArray? = featureParts.isEmpty ? nil
            : featureParts.count == 1 ? featureParts[0] : concatenated(featureParts, axis: 0)
        let prepared = try languageModel.prepareEmbeddingInputs(
            inputIds: inputIDs,
            imageFeatures: imageFeatures,
            imageGridTHW: isVideo ? nil : visualGrid.map { [$0] },
            videoGridTHW: isVideo ? visualGrid.map { [$0] } : nil
        )
        var decoderMS = 0.0
        var decoderSegmentTimings: [String: Double] = [:]
        let decoderBundle: CoreMLDecoderBundle? = switch languageComputeTarget {
        case .ane:
            decoderBundles.first(where: { tokenIDs.count <= $0.sequenceLength })
        case .auto:
            tokenIDs.count >= decoderSegmentMinimumTokens
                ? decoderBundles.first(where: { tokenIDs.count <= $0.sequenceLength })
                : nil
        case .gpu:
            nil
        }
        if languageComputeTarget == .ane, decoderBundle == nil {
            throw ServiceError.invalidRequest(
                "输入的 \(tokenIDs.count) token 超过已安装 ANE bucket 容量",
                status: 413
            )
        }
        let routeToCoreML = languageComputeTarget == .auto && decoderSegment != nil
            && tokenIDs.count >= decoderSegmentMinimumTokens
            && tokenIDs.count <= CoreMLDecoderSegment.sequenceLength
        let hidden: MLXArray
        if let decoderBundle {
            try cancellation.check()
            let rotary = languageModel.embeddingRotaryValues(
                hiddenStates: prepared.embeddings,
                positionIds: prepared.positionIds
            )
            let decoded = try decoderBundle.predict(
                hiddenStates: prepared.embeddings,
                cos: rotary.cos,
                sin: rotary.sin,
                tokens: tokenIDs.count,
                cancellation: cancellation
            )
            decoderMS = decoded.milliseconds
            decoderSegmentTimings = decoded.segmentMilliseconds
            hidden = languageModel.finalizeEmbeddingHiddenStates(decoded.hiddenStates)
        } else if routeToCoreML, let decoderSegment {
            try cancellation.check()
            let rotary = languageModel.embeddingRotaryValues(
                hiddenStates: prepared.embeddings,
                positionIds: prepared.positionIds
            )
            let decoded = try decoderSegment.predict(
                hiddenStates: prepared.embeddings,
                cos: rotary.cos,
                sin: rotary.sin,
                tokens: tokenIDs.count
            )
            decoderMS = decoded.milliseconds
            try cancellation.check()
            hidden = languageModel.embeddingHiddenStates(
                inputIds: inputIDs,
                hiddenStates: decoded.hiddenStates,
                positionIds: prepared.positionIds,
                startLayer: CoreMLDecoderSegment.nextLayer
            )
        } else {
            hidden = languageModel.embeddingHiddenStates(
                inputIds: inputIDs,
                hiddenStates: prepared.embeddings,
                positionIds: prepared.positionIds,
                startLayer: 0
            )
        }
        try cancellation.check()
        var vector = hidden[0, tokenIDs.count - 1, 0 ..< dimension].asType(.float32)
        let magnitude = sqrt((vector * vector).sum())
        vector = vector / magnitude
        eval(vector)
        let values = vector.asArray(Float.self)
        let languageMS = milliseconds(languageStarted.duration(to: .now))
        try cancellation.check()

        var timings: [String: Double] = [
            "preprocess": vision.preprocessMS,
            "vision_queue": visionWaitMS, "language_queue": languageWaitMS,
            "vision": vision.visionMS,
            "coreml_decoder": decoderMS,
            "language": languageMS,
            "total": milliseconds(totalStarted.duration(to: .now)),
            "video_frames": Double(vision.frameCount),
        ]
        timings.merge(decoderSegmentTimings) { _, new in new }
        return EmbeddingResult(
            vector: values,
            timingsMS: timings,
            modality: vision.modality,
            promptTokens: tokenIDs.count
        )
    }

    private func preprocessImage(_ data: Data) throws -> (patches: VisionPatches, grid: THW) {
        let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let image = CIImage(
            data: data,
            options: [.applyOrientationProperty: true, .colorSpace: sRGB]
        ),
              image.extent.width.isFinite, image.extent.height.isFinite,
              image.extent.width > 0, image.extent.height > 0 else {
            throw ServiceError.invalidRequest("无法解码图片")
        }
        let rgb = try normalizedRGB(image, colorSpace: sRGB)
        let patches = temporalPairPatches(first: rgb, second: rgb)
        debugPatchesIfRequested(patches)
        return (patches, THW(1, 28, 28))
    }

    private func preprocessVideo(
        _ data: Data, cancellation: CancellationToken
    ) throws -> (patches: [VisionPatches], grid: THW, frameCount: Int, timestamps: [Double]) {
        let temporaryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("indexed-wemm-\(UUID().uuidString)")
            .appendingPathExtension("mp4")
        try data.write(to: temporaryURL, options: .atomic)
        defer { try? FileManager.default.removeItem(at: temporaryURL) }

        let asset = AVURLAsset(url: temporaryURL)
        let durationSeconds = CMTimeGetSeconds(asset.duration)
        guard durationSeconds.isFinite, durationSeconds > 0 else {
            throw ServiceError.invalidRequest("无法解码视频或视频时长无效")
        }
        guard durationSeconds <= 600 else {
            throw ServiceError.invalidRequest("视频超过 10 分钟安全限制", status: 413)
        }

        // Match Qwen's default 2 fps sampling. Qwen3.5 groups every two sampled
        // frames into one temporal patch, so keep the count even.
        let requestedFrames = max(2, Int(floor(durationSeconds * 2.0)))
        var frameCount = min(64, requestedFrames)
        if frameCount % 2 != 0 { frameCount -= 1 }
        frameCount = max(2, frameCount)

        let sourceFPS = Double(
            asset.tracks(withMediaType: .video).first?.nominalFrameRate ?? 0)
        let safeSourceFPS = sourceFPS.isFinite && sourceFPS > 0 ? sourceFPS : 30.0
        let estimatedTotalFrames = max(2, Int((durationSeconds * safeSourceFPS).rounded()))
        let frameIndices: [Int] = (0 ..< frameCount).map { index in
            guard frameCount > 1 else { return 0 }
            return Int(
                (Double(index) * Double(estimatedTotalFrames - 1) / Double(frameCount - 1))
                    .rounded()
            )
        }

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = CMTime(seconds: 1.0 / 60.0, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter = CMTime(seconds: 1.0 / 60.0, preferredTimescale: 600)
        let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!
        var frames: [[Float]] = []
        frames.reserveCapacity(frameCount)
        for (index, frameIndex) in frameIndices.enumerated() {
            try cancellation.check()
            let requestedSecond = min(
                Double(frameIndex) / safeSourceFPS,
                max(0, durationSeconds - 1.0 / 600.0))
            let requestedTime = CMTime(seconds: requestedSecond, preferredTimescale: 600)
            var actualTime = CMTime.invalid
            let cgImage: CGImage
            do {
                cgImage = try generator.copyCGImage(at: requestedTime, actualTime: &actualTime)
            } catch {
                throw ServiceError.invalidRequest(
                    "视频第 \(index + 1) 帧解码失败：\(error.localizedDescription)"
                )
            }
            frames.append(try normalizedRGB(CIImage(cgImage: cgImage), colorSpace: sRGB))
        }

        var patchPairs: [VisionPatches] = []
        var timestamps: [Double] = []
        patchPairs.reserveCapacity(frameCount / 2)
        timestamps.reserveCapacity(frameCount / 2)
        for index in stride(from: 0, to: frameCount, by: 2) {
            try cancellation.check()
            patchPairs.append(temporalPairPatches(first: frames[index], second: frames[index + 1]))
            timestamps.append(
                (Double(frameIndices[index]) + Double(frameIndices[index + 1]))
                    / (2.0 * safeSourceFPS)
            )
        }
        if let first = patchPairs.first { debugPatchesIfRequested(first) }
        return (patchPairs, THW(frameCount / 2, 28, 28), frameCount, timestamps)
    }

    private func preprocessVideoFrames(
        _ inputs: [WeMMVideoFrame], cancellation: CancellationToken
    ) throws -> (patches: [VisionPatches], grid: THW, frameCount: Int, timestamps: [Double]) {
        guard !inputs.isEmpty else { throw ServiceError.invalidRequest("视频帧包不能为空") }
        guard inputs.count <= 64 else {
            throw ServiceError.invalidRequest("视频帧包最多支持 64 帧", status: 413)
        }
        let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!
        var frames: [[Float]] = []
        var frameTimes: [Double] = []
        frames.reserveCapacity(inputs.count + 1)
        frameTimes.reserveCapacity(inputs.count + 1)
        for (index, input) in inputs.enumerated() {
            try cancellation.check()
            guard input.timestamp.isFinite, input.timestamp >= 0,
                  index == 0 || input.timestamp >= inputs[index - 1].timestamp else {
                throw ServiceError.invalidRequest("视频帧时间戳必须递增且不能为负数")
            }
            guard let image = CIImage(
                data: input.data,
                options: [.applyOrientationProperty: true, .colorSpace: sRGB]
            ), image.extent.width.isFinite, image.extent.height.isFinite,
               image.extent.width > 0, image.extent.height > 0 else {
                throw ServiceError.invalidRequest("视频第 \(index + 1) 帧无法解码")
            }
            frames.append(try normalizedRGB(image, colorSpace: sRGB))
            frameTimes.append(input.timestamp)
        }
        if frames.count == 1 {
            frames.append(frames[0])
            frameTimes.append(frameTimes[0])
        } else if frames.count % 2 != 0 {
            frames.removeLast()
            frameTimes.removeLast()
        }
        var patches: [VisionPatches] = []
        var timestamps: [Double] = []
        patches.reserveCapacity(frames.count / 2)
        timestamps.reserveCapacity(frames.count / 2)
        for index in stride(from: 0, to: frames.count, by: 2) {
            try cancellation.check()
            patches.append(temporalPairPatches(first: frames[index], second: frames[index + 1]))
            timestamps.append((frameTimes[index] + frameTimes[index + 1]) / 2.0)
        }
        if let first = patches.first { debugPatchesIfRequested(first) }
        return (patches, THW(frames.count / 2, 28, 28), frames.count, timestamps)
    }

    private func normalizedRGB(_ image: CIImage, colorSpace sRGB: CGColorSpace) throws -> [Float] {
        let side = modelPackage.manifest.vision.imageSize
        let sourceWidth = Int(image.extent.width.rounded())
        let sourceHeight = Int(image.extent.height.rounded())
        guard sourceWidth > 0, sourceHeight > 0,
              sourceWidth <= 16_384, sourceHeight <= 16_384,
              sourceWidth * sourceHeight <= 100_000_000 else {
            throw ServiceError.invalidRequest("图片像素尺寸超出安全限制", status: 413)
        }
        var sourceRGBA = [UInt8](repeating: 0, count: sourceWidth * sourceHeight * 4)
        sourceRGBA.withUnsafeMutableBytes { buffer in
            imageContext.render(
                image,
                toBitmap: buffer.baseAddress!,
                rowBytes: sourceWidth * 4,
                bounds: image.extent,
                format: .RGBA8,
                colorSpace: sRGB
            )
        }
        imageContext.clearCaches()
        let rgba = PillowBicubic.resizeRGBA(
            sourceRGBA,
            width: sourceWidth,
            height: sourceHeight,
            to: side,
            side
        )

        return VisionPatches.normalizedCHW(rgba, side: side, table: normalizationTable)
    }

    private func temporalPairPatches(first: [Float], second: [Float]) -> VisionPatches {
        VisionPatches.pair(first, second, side: modelPackage.manifest.vision.imageSize)
    }

    private func debugPatchesIfRequested(_ patches: VisionPatches) {
        if ProcessInfo.processInfo.environment["INDEXED_EMBEDDING_DEBUG_PATCHES"] == "1" {
            fputs("[apple-embedding] patches rows=\(patches.rows) first=\(Array(patches.values.prefix(32)))\n", stderr)
        }
    }

    private func predictVision(_ patches: VisionPatches, cancellation: CancellationToken) throws -> VisionFeatures {
        let values = patches.values
        let input = try MLMultiArray(
            shape: [patches.rows, VisionPatches.columns].map(NSNumber.init),
            dataType: .float32
        )
        values.withUnsafeBufferPointer { source in
            input.dataPointer.copyMemory(
                from: source.baseAddress!,
                byteCount: values.count * MemoryLayout<Float>.size
            )
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: ["patches": input])
        let prediction: any MLFeatureProvider
        if privateMLPContext != nil {
            prediction = try pipeline.ane.withPermit(cancellation: cancellation) {
                try visionModel.prediction(from: provider)
            }
        } else { prediction = try visionModel.prediction(from: provider) }
        try cancellation.check()
        guard let output = prediction.featureValue(for: "image_embeds")?.multiArrayValue else {
            throw ServiceError.internalFailure("Core ML 视觉模型没有返回 image_embeds")
        }
        guard output.dataType == .float32 else {
            throw ServiceError.internalFailure("Core ML image_embeds 不是 Float32")
        }
        let outputValues = Array(
            UnsafeBufferPointer(
                start: output.dataPointer.assumingMemoryBound(to: Float.self),
                count: output.count
            )
        )
        let shape = output.shape.map(\.intValue)
        return VisionFeatures(values: outputValues, shape: shape)
    }
}

/// Load only the token embedding table and final RMSNorm required around a
/// complete Core ML L0-L23 decoder.  The decoder layer objects remain lazy and
/// their random initializer arrays are never evaluated or copied to the GPU.
private func loadCoreMLFrontendWeights(
    modelDirectory: URL,
    model: Qwen35Language.LanguageModel,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization?
) throws {
    let acceptedPrefixes = ["model.embed_tokens.", "model.norm."]
    var selected: [String: MLXArray] = [:]
    guard let enumerator = FileManager.default.enumerator(
        at: modelDirectory,
        includingPropertiesForKeys: nil
    ) else {
        throw ServiceError.model("无法读取 MLX language 权重目录")
    }
    for case let url as URL in enumerator where url.pathExtension == "safetensors" {
        let arrays = try loadArrays(url: url, stream: .cpu)
        for (key, value) in arrays where acceptedPrefixes.contains(where: key.hasPrefix) {
            selected[key] = value
        }
    }
    let required = [
        "model.embed_tokens.weight",
        "model.embed_tokens.scales",
        "model.norm.weight",
    ]
    let missing = required.filter { selected[$0] == nil }
    guard missing.isEmpty else {
        throw ServiceError.model("ANE-only frontend 缺少权重：\(missing.joined(separator: ", "))")
    }
    guard let quantization = perLayerQuantization?.quantization(layer: "model.embed_tokens") else {
        throw ServiceError.model("ANE-only frontend 需要量化 token embedding 配置")
    }
    MLXNN.quantize(model: model) { path, _ in
        path == "model.embed_tokens" ? quantization.asTuple : nil
    }
    let parameters = ModuleParameters.unflattened(selected)
    try model.update(
        parameters: parameters,
        verify: .noUnusedKeys.union(.shapeMismatch)
    )
    eval(selected.values)
}

private struct WeMMVideoFrame {
    let data: Data
    let timestamp: Double
}

private struct WeMMPrompt {
    let prompt: String
    let imageData: Data?
    let videoData: Data?
    let videoFrames: [WeMMVideoFrame]?

    static func parse(_ messages: [JSONValue]) throws -> WeMMPrompt {
        guard !messages.isEmpty else { throw ServiceError.invalidRequest("messages 必须是非空数组") }
        var prompt = ""
        var imageData: Data?
        var videoData: Data?
        var videoFrames: [WeMMVideoFrame]?

        for (messageIndex, messageValue) in messages.enumerated() {
            guard let message = messageValue.objectValue else {
                throw ServiceError.invalidRequest("messages 中的每一项必须是对象")
            }
            let role = (message["role"]?.stringValue ?? "user").lowercased()
            guard role == "system" || role == "user" else {
                throw ServiceError.invalidRequest("不支持的 message role：\(role)")
            }
            prompt += "<|im_start|>\(role)"
            guard let content = message["content"] else {
                throw ServiceError.invalidRequest("message.content 不能为空")
            }
            if let text = content.stringValue {
                prompt += "\n\(text)"
            } else if let items = content.arrayValue, !items.isEmpty {
                for (itemIndex, itemValue) in items.enumerated() {
                    guard let item = itemValue.objectValue else {
                        throw ServiceError.invalidRequest("message.content 中的每一项必须是对象")
                    }
                    let type = (item["type"]?.stringValue ?? "text").lowercased()
                    switch type {
                    case "text":
                        if itemIndex == 0 { prompt += "\n" }
                        prompt += item["text"]?.stringValue ?? ""
                    case "image", "image_url":
                        guard imageData == nil, videoData == nil, videoFrames == nil else {
                            throw ServiceError.invalidRequest("Apple helper 每次只支持一个视觉输入", status: 422)
                        }
                        let source: String?
                        if type == "image" {
                            source = item["image"]?.stringValue
                        } else if let object = item["image_url"]?.objectValue {
                            source = object["url"]?.stringValue
                        } else {
                            source = item["image_url"]?.stringValue
                        }
                        imageData = try decodeMediaDataURI(source, mediaType: "image", displayName: "图片")
                        prompt += "<|vision_start|><|image_pad|><|vision_end|>"
                    case "video", "video_url":
                        guard imageData == nil, videoData == nil, videoFrames == nil else {
                            throw ServiceError.invalidRequest("Apple helper 每次只支持一个视觉输入", status: 422)
                        }
                        let source: String?
                        if type == "video" {
                            source = item["video"]?.stringValue
                        } else if let object = item["video_url"]?.objectValue {
                            source = object["url"]?.stringValue
                        } else {
                            source = item["video_url"]?.stringValue
                        }
                        videoData = try decodeMediaDataURI(
                            source, mediaType: "video", displayName: "视频")
                        prompt += "\n<|video_pad|>"
                    case "video_frames":
                        guard imageData == nil, videoData == nil, videoFrames == nil else {
                            throw ServiceError.invalidRequest("Apple helper 每次只支持一个视觉输入", status: 422)
                        }
                        guard let frameValues = item["frames"]?.arrayValue,
                              !frameValues.isEmpty else {
                            throw ServiceError.invalidRequest("video_frames.frames 必须是非空数组")
                        }
                        guard frameValues.count <= 64 else {
                            throw ServiceError.invalidRequest("视频帧包最多支持 64 帧", status: 413)
                        }
                        var decoded: [WeMMVideoFrame] = []
                        decoded.reserveCapacity(frameValues.count)
                        for (frameIndex, frameValue) in frameValues.enumerated() {
                            guard let frame = frameValue.objectValue else {
                                throw ServiceError.invalidRequest("video_frames 中的每一帧必须是对象")
                            }
                            let source: String?
                            if let object = frame["image_url"]?.objectValue {
                                source = object["url"]?.stringValue
                            } else {
                                source = frame["url"]?.stringValue
                                    ?? frame["image_url"]?.stringValue
                            }
                            let data = try decodeMediaDataURI(
                                source, mediaType: "image", displayName: "视频帧")
                            let timestamp = frame["timestamp"]?.numberValue
                                ?? Double(frameIndex) / 2.0
                            decoded.append(WeMMVideoFrame(data: data, timestamp: timestamp))
                        }
                        videoFrames = decoded
                        prompt += "\n<|video_pad|>"
                    default:
                        throw ServiceError.invalidRequest("不支持的 content type：\(type)")
                    }
                }
            } else {
                throw ServiceError.invalidRequest("message.content 必须是字符串或非空数组")
            }
            prompt += "<|im_end|>"
            if messageIndex + 1 < messages.count { prompt += "\n" }
        }
        prompt += "<embedding>"
        return WeMMPrompt(
            prompt: prompt,
            imageData: imageData,
            videoData: videoData,
            videoFrames: videoFrames
        )
    }

    private static func decodeMediaDataURI(
        _ source: String?, mediaType: String, displayName: String
    ) throws -> Data {
        guard let source, source.hasPrefix("data:\(mediaType)/"),
              let marker = source.range(of: ";base64,") else {
            throw ServiceError.invalidRequest("Apple helper 只接受 base64 data URI \(displayName)")
        }
        guard let data = Data(base64Encoded: String(source[marker.upperBound...]), options: []) else {
            throw ServiceError.invalidRequest("\(displayName) base64 无效")
        }
        guard !data.isEmpty else { throw ServiceError.invalidRequest("\(displayName)内容为空") }
        return data
    }
}

private func seconds(_ duration: Duration) -> Double {
    let components = duration.components
    return Double(components.seconds) + Double(components.attoseconds) / 1e18
}

private func milliseconds(_ duration: Duration) -> Double { seconds(duration) * 1000 }

private extension String {
    func replacingFirstOccurrence(of target: String, with replacement: String) -> String {
        guard let range = range(of: target) else { return self }
        return replacingCharacters(in: range, with: replacement)
    }
}
