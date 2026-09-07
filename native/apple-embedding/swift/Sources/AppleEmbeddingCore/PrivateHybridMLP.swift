import Foundation
import MLX
import MLXNN

/// First fixed contract being ported, matching the frozen Q8/G64 C reference.
/// Reject unsupported settings instead of silently claiming a different kernel.
public struct PrivateMLPConfiguration: Sendable {
    public let sequenceLength: Int
    public let fraction: Double
    public let maxLayers: Int
    public var tailMinimumTokens: Int { sequenceLength / 2 }

    public init(sequenceLength: Int = 2112, fraction: Double = 0.75, maxLayers: Int = 24) throws {
        guard sequenceLength >= 64, sequenceLength <= 8192, sequenceLength.isMultiple(of: 64),
              fraction > 0, fraction < 1, fraction.isFinite, maxLayers > 0, maxLayers <= 24 else {
            throw ServiceError.invalidRequest("Invalid private MLP configuration")
        }
        self.sequenceLength = sequenceLength; self.fraction = fraction; self.maxLayers = maxLayers
    }
}

/// Request scope plus counters are shared by all substituted layers. The model
/// remains single-writer; device overlap happens inside a layer, not by running
/// two complete language models concurrently.
public final class PrivateMLPContext: @unchecked Sendable {
    private let requestLock = NSLock()
    private let counterLock = NSLock()
    fileprivate var cancellation = CancellationToken()
    var requestCancellation: CancellationToken { cancellation }
    var recurrenceAllowed = false
    var recurrence: PrivateGDNRecurrence?
    var aneLane: InferenceGate?
    // All MLP layers execute under the same language owner. Their compiled
    // weights differ, but only one layer needs the fixed-shape I/O at a time.
    fileprivate var sharedProgram: PrivateANEProgram?
    private var operationCount = 0
    private var projectionCount = 0
    private var stageMilliseconds: [String: Double] = [:]
    private var fallbackCounts: [String: Int] = [:]
    private var failures: [String: String] = [:]
    public var operations: Int { counterLock.withLock { operationCount } }
    public var fallbacks: [String: Int] { counterLock.withLock { fallbackCounts } }

    public init() {}

    public func withRequest<T>(_ token: CancellationToken, _ body: () throws -> T) throws -> T {
        try requestLock.withLock {
            cancellation = token
            recurrenceAllowed = false
            defer { cancellation = CancellationToken(); recurrenceAllowed = false }
            try token.check()
            let result = try body()
            try token.check()
            return result
        }
    }

    fileprivate func recordOperations(_ count: Int) { counterLock.withLock { operationCount += count } }
    func recordGPUProjection() { counterLock.withLock { projectionCount += 1 } }
    fileprivate func recordStage(_ name: String, since start: ContinuousClock.Instant) {
        let duration = start.duration(to: .now).components
        let milliseconds = Double(duration.seconds) * 1_000 + Double(duration.attoseconds) / 1e15
        counterLock.withLock { stageMilliseconds[name, default: 0] += milliseconds }
    }
    fileprivate func recordStages(_ stages: [String: Double]) {
        counterLock.withLock { for (name, time) in stages { stageMilliseconds[name, default: 0] += time } }
    }
    func fallback(_ reason: String, layer: String, error: Error? = nil) {
        counterLock.withLock {
            fallbackCounts[reason, default: 0] += 1
            if let error { failures[layer] = String(describing: error) }
        }
    }

    public var diagnostics: JSONValue {
        counterLock.withLock {
            .object([
                "backend": .string("swift-private-ane-mlp-candidate"),
                "gpu_kernel": .string("classic-q8-g64-bm64-bk32-bn64"),
                "long_projection_kernel": .string(FastQ8Projection.isSupportedDevice ? "classic-q8-g64-bm64-bk32-bn64" : "stock-mlx"),
                "native_profile": .object(["mlp": .object([
                    "operations": .number(Double(operationCount)),
                    "gpu_projection_operations": .number(Double(projectionCount)),
                    "stage_ms": .object(stageMilliseconds.mapValues { .number($0) }),
                ])]),
                "fallbacks": .object(fallbackCounts.mapValues { .number(Double($0)) }),
                "failed_layers": .object(failures.mapValues { .string($0) }),
                "recurrence": recurrence?.diagnostics ?? .null,
            ])
        }
    }
}

/// Fixed ANE gate/up prefix + asynchronous Q8 GPU suffix + Q8 down projection.
/// Original weights are retained for short/tail/failed-kernel fallback.
public final class PrivateHybridMLP: Module, UnaryLayer {
    private let original: any UnaryLayer
    private let down: QuantizedLinear
    private let program: PrivateANEProgram
    private let gpuWeight: MLXArray
    private let gpuScales: MLXArray
    private let gpuBiases: MLXArray
    private let configuration: PrivateMLPConfiguration
    private let context: PrivateMLPContext
    private let layer: String
    private let inputDimension: Int
    private let aneHidden: Int
    private let gpuHidden: Int
    private let dtype: DType
    private var failed = false

    public init(original module: Module, configuration: PrivateMLPConfiguration,
                context: PrivateMLPContext, layer: String) throws {
        let children = Dictionary(uniqueKeysWithValues: module.children().flattened())
        guard let original = module as? any UnaryLayer,
              let gate = children["gate_proj"] as? QuantizedLinear,
              let up = children["up_proj"] as? QuantizedLinear,
              let down = children["down_proj"] as? QuantizedLinear,
              gate.bits == 8, up.bits == 8, down.bits == 8,
              gate.groupSize == 64, up.groupSize == 64, down.groupSize == 64,
              gate.mode == .affine, up.mode == .affine, down.mode == .affine,
              gate.bias == nil, up.bias == nil, down.bias == nil,
              gate.shape == up.shape, down.shape == (gate.shape.1, gate.shape.0),
              let gateBias = gate.biases, let upBias = up.biases,
              [DType.float16, .bfloat16].contains(gate.scales.dtype),
              gate.scales.dtype == up.scales.dtype, gate.scales.dtype == down.scales.dtype else {
            throw ServiceError.model("Private MLP requires matching dense Q8/G64 affine gate/up/down layers")
        }
        let aneHidden = (Int(Double(gate.shape.0) * configuration.fraction) / 64) * 64
        let gpuHidden = gate.shape.0 - aneHidden
        guard aneHidden > 0, gpuHidden > 0, gpuHidden.isMultiple(of: 64),
              gate.shape.1.isMultiple(of: 64) else {
            throw ServiceError.model("Private MLP channel partition is unsupported")
        }
        let dense = concatenated([gate, up].map { linear in
            dequantized(linear.weight[0..<aneHidden], scales: linear.scales[0..<aneHidden],
                        biases: linear.biases![0..<aneHidden], groupSize: 64, bits: 8).asType(.float32)
        }, axis: 0)
        let quantized = try PrivateANELinearWeights(dense: dense.asArray(Float.self),
            inputDimension: gate.shape.1, outputDimension: 2 * aneHidden)
        self.program = try PrivateANEProgram(
            mil: PrivateANELinearWeights.mil(inputDimension: gate.shape.1, outputDimension: 2 * aneHidden,
                                            sequenceLength: configuration.sequenceLength),
            weights: quantized.data, scales: quantized.scales,
            inputElements: [gate.shape.1 * configuration.sequenceLength],
            outputElements: [2 * aneHidden * configuration.sequenceLength], sharing: context.sharedProgram)
        if context.sharedProgram == nil { context.sharedProgram = self.program }
        self.gpuWeight = concatenated([gate.weight[aneHidden...], up.weight[aneHidden...]], axis: 0)
        self.gpuScales = concatenated([gate.scales[aneHidden...], up.scales[aneHidden...]], axis: 0)
        self.gpuBiases = concatenated([gateBias[aneHidden...], upBias[aneHidden...]], axis: 0)
        eval(gpuWeight, gpuScales, gpuBiases)
        self.original = original
        self.down = FastQ8Projection.isSupportedDevice ? FastQ8Projection(down, context: context) : down
        self.configuration = configuration
        self.context = context; self.layer = layer; self.inputDimension = gate.shape.1
        self.aneHidden = aneHidden; self.gpuHidden = gpuHidden; self.dtype = gate.scales.dtype
        super.init()
    }

    private func exact(_ x: MLXArray, project: Bool = true) throws -> MLXArray {
        try context.cancellation.check()
        let producerStarted = ContinuousClock.now
        let packed = PrivateMLPKernels.packedANEInput(x)
        // Complete the input producer before starting either consumer. ANE's
        // evaluation can overlap the already submitted GPU suffix.
        eval(packed)
        context.recordStage("input_producer", since: producerStarted)
        let copyStarted = ContinuousClock.now
        let input = packed.asArray(Float16.self)
        context.recordStage("input_copy", since: copyStarted)
        try context.cancellation.check()
        let gpu = PrivateMLPKernels.affine(x, weight: gpuWeight, scales: gpuScales, biases: gpuBiases)
        asyncEval(gpu)
        // Always join submitted GPU work, including ANE error/cancellation paths,
        // before releasing request-local arrays or reusing buffers.
        defer { eval(gpu) }
        let before = program.evaluations
        defer { context.recordOperations(program.evaluations - before) }
        let aneStarted = ContinuousClock.now
        let activation = try program.evaluateActivation(input, gpu: gpu, rows: configuration.sequenceLength,
            aneHidden: aneHidden, gpuHidden: gpuHidden, cancellation: context.cancellation, lane: context.aneLane)
        context.recordStage("ane_evaluate_activation", since: aneStarted)
        context.recordStages(program.lastEvaluationStages)
        try context.cancellation.check()
        return project ? down(activation) : activation
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if context.cancellation.isCancelled { return .zeros(x.shape, dtype: x.dtype) }
        guard !failed else {
            context.fallback("latched_failure", layer: layer)
            return original(x)
        }
        guard x.ndim == 3, x.dim(0) == 1, x.dim(-1) == inputDimension, x.dtype == dtype else {
            context.fallback("unsupported_shape_or_dtype", layer: layer)
            return original(x)
        }
        let rows = x.dim(1)
        guard rows >= configuration.tailMinimumTokens else {
            context.fallback("short_input", layer: layer)
            return original(x)
        }
        do {
            var parts: [MLXArray] = []
            let length = configuration.sequenceLength
            let tail = rows % length
            // Keep the fixed ANE blocks intact. On measured long M4 inputs,
            // trim their padding before one Q8 down projection so GPU work is
            // proportional to real tokens. A short GPU-fallback tail retains
            // the original per-block path and its rounding behavior.
            let joinedDown = FastQ8Projection.isSupportedDevice && rows >= 2112
                && (tail == 0 || tail >= configuration.tailMinimumTokens)
            for start in stride(from: 0, to: rows, by: length) {
                let count = min(length, rows - start)
                let part = x[0..., start..<(start + count), 0...]
                if count == length { parts.append(try exact(part, project: !joinedDown)) }
                else if count >= configuration.tailMinimumTokens {
                    let padded = concatenated([part, .zeros([1, length - count, inputDimension], dtype: dtype)], axis: 1)
                    parts.append(try exact(padded, project: !joinedDown)[0..., 0..<count, 0...])
                } else {
                    context.fallback("short_tail", layer: layer)
                    parts.append(original(part))
                }
            }
            let result = parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
            return joinedDown ? down(result) : result
        } catch ServiceError.cancelled {
            return .zeros(x.shape, dtype: x.dtype)
        } catch {
            failed = true
            context.fallback("runtime_failure", layer: layer, error: error)
            return original(x)
        }
    }

    /// Compile all replacements before changing the model. No partially patched
    /// model escapes if a layer fails to compile or the expected layout differs.
    public static func install(in model: Module, configuration: PrivateMLPConfiguration,
                               context: PrivateMLPContext) throws -> Int {
        let modules = model.namedModules().filter { $0.0.hasSuffix(".mlp") }
        guard modules.count == 24 else { throw ServiceError.model("Private C candidate requires the 24-layer WeMM model") }
        var replacements: [(String, Module)] = []
        for (name, module) in modules.prefix(configuration.maxLayers) {
            replacements.append((name, try PrivateHybridMLP(original: module,
                configuration: configuration, context: context, layer: name)))
        }
        try model.update(modules: .unflattened(replacements), verify: .all)
        return replacements.count
    }
}
