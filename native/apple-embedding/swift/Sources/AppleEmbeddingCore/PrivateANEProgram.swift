import Foundation
import MLX
import Cmlx
import PrivateANEBridge

private final class PrivateANEWorkspace {
    let lock = NSLock()
    var shared = false
    var outputs: [[Float16]]
    var outputArray: MLXArray?
    init(outputElements: [Int]) {
        outputs = outputElements.map { [Float16](repeating: 0, count: $0) }
    }
}

/// Fixed-shape experimental ANE program. Callers retain GPU fallback policy.
/// The C bridge owns IOSurfaces; this wrapper serializes their use and bounds copies.
public final class PrivateANEProgram: @unchecked Sendable {
    private let handle: OpaquePointer
    private let workspace: PrivateANEWorkspace
    private var lock: NSLock { workspace.lock }
    private let inputElements: [Int]
    private let outputElements: [Int]
    private var initializedInputs: Set<Int> = []
    private var evaluationCount = 0
    private var lastStages: [String: Double] = [:]
    public var evaluations: Int { lock.withLock { evaluationCount } }
    var lastEvaluationStages: [String: Double] { lock.withLock { lastStages } }

    private static func message(_ buffer: [CChar]) -> String {
        String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    public init(mil: String, weights: [Int8] = [], scales: [Float16] = [], inputElements: [Int], outputElements: [Int], sharing: PrivateANEProgram? = nil) throws {
        guard !inputElements.isEmpty, !outputElements.isEmpty,
              (inputElements + outputElements).allSatisfy({ $0 > 0 && $0 <= 128 * 1024 * 1024 }) else {
            throw ServiceError.invalidRequest("Private ANE surface dimensions are invalid")
        }
        var error = [CChar](repeating: 0, count: 4096)
        guard sharing == nil || (sharing!.inputElements == inputElements && sharing!.outputElements == outputElements) else {
            throw ServiceError.invalidRequest("Shared private ANE workspace shapes differ")
        }
        let selectedWorkspace = sharing?.workspace ?? PrivateANEWorkspace(outputElements: outputElements)
        workspace = selectedWorkspace
        selectedWorkspace.lock.lock()
        defer { selectedWorkspace.lock.unlock() }
        let inputSizes = inputElements.map { $0 * MemoryLayout<Float16>.stride }
        let outputSizes = outputElements.map { $0 * MemoryLayout<Float16>.stride }
        let program = Array(mil.utf8).withUnsafeBufferPointer { text in
            weights.withUnsafeBytes { weightBytes in
                scales.withUnsafeBytes { scaleBytes in
                    indexed_ane_create_sharing(text.baseAddress, text.count,
                        weightBytes.baseAddress?.assumingMemoryBound(to: UInt8.self), weightBytes.count,
                        scaleBytes.baseAddress?.assumingMemoryBound(to: UInt8.self), scaleBytes.count,
                        inputSizes, inputSizes.count, outputSizes, outputSizes.count, sharing?.handle, &error, error.count)
                }
            }
        }
        guard let program else { throw ServiceError.model(Self.message(error)) }
        handle = program
        self.inputElements = inputElements
        self.outputElements = outputElements
        if sharing != nil { workspace.shared = true }
    }

    deinit { indexed_ane_free(handle) }

    public func evaluate(_ inputs: [[Float16]], cancellation: CancellationToken) throws -> [[Float16]] {
        guard inputs.count == inputElements.count else { throw ServiceError.invalidRequest("Private ANE input count does not match its compiled shape") }
        return try evaluate(updating: Dictionary(uniqueKeysWithValues: inputs.enumerated().map { ($0.offset, $0.element) }), cancellation: cancellation)
    }

    /// Write immutable shared masks once. Later evaluations must supply every
    /// other input; a missing, never-initialized surface is always rejected.
    public func prepareInput(_ input: [Float16], at index: Int) throws {
        try lock.withLock {
            guard !workspace.shared else { throw ServiceError.invalidRequest("Shared ANE workspaces require complete inputs per evaluation") }
            try write(input, at: index)
        }
    }

    private func write(_ input: [Float16], at index: Int) throws {
        guard inputElements.indices.contains(index), input.count == inputElements[index] else {
            throw ServiceError.invalidRequest("Private ANE input does not match its compiled shape")
        }
        var error = [CChar](repeating: 0, count: 4096)
        let written = input.withUnsafeBytes { indexed_ane_write(handle, index, $0.baseAddress, $0.count, &error, error.count) }
        guard written else { throw ServiceError.model(Self.message(error)) }
        initializedInputs.insert(index)
    }

    public func evaluate(updating inputs: [Int: [Float16]], cancellation: CancellationToken) throws -> [[Float16]] {
        try withEvaluation(updating: inputs, cancellation: cancellation, outputStage: "surface_read") {
            var error = [CChar](repeating: 0, count: 4096)
            for index in workspace.outputs.indices {
                let read = workspace.outputs[index].withUnsafeMutableBytes { indexed_ane_read(handle, index, $0.baseAddress, $0.count, &error, error.count) }
                guard read else { throw ServiceError.model(Self.message(error)) }
            }
            return workspace.outputs
        }
    }

    /// Consume ANE's single output directly on the language owner's GPU. The
    /// returned activation must own its storage; evaluating it here completes
    /// every surface read before a later layer can reuse the shared workspace.
    /// The input alias is internal and cannot escape the supplied transform.
    func evaluateActivation(_ input: [Float16], gpu: MLXArray, rows: Int,
                            aneHidden: Int, gpuHidden: Int, cancellation: CancellationToken,
                            lane: InferenceGate?) throws -> MLXArray {
        guard inputElements.count == 1, outputElements == [2 * aneHidden * rows] else {
            throw ServiceError.invalidRequest("Private ANE activation shape differs from its output")
        }
        return try withEvaluation(updating: [0: input], cancellation: cancellation, outputStage: "surface_activation", lane: lane) {
            var error = [CChar](repeating: 0, count: 4096)
            guard let pointer = indexed_ane_lock_output(handle, 0, outputElements[0] * MemoryLayout<Float16>.stride, &error, error.count) else {
                throw ServiceError.model(Self.message(error))
            }
            defer { indexed_ane_unlock_output(handle, 0) }
            if workspace.outputArray == nil {
                guard let surface = indexed_ane_retain_output_surface(handle, 0) else {
                    throw ServiceError.model("Private ANE output storage unavailable")
                }
                // Retain only storage, not this program: caching an array whose
                // destructor owns the program would form a workspace cycle.
                // Cmlx also avoids 0.31.4's retained Swift finalizer-box leak.
                workspace.outputArray = MLXArray(mlx_array_new_data_managed_payload(
                    pointer, [Int32(outputElements[0])], 1, MLX_FLOAT16, surface,
                    { payload in indexed_ane_release_surface(payload) }))
                // Cmlx can fall back to copying an unwrappable allocation.
                // A cached copy would become stale on the next ANE evaluation.
                let aliasesSurface = workspace.outputArray!.asData(access: .noCopy).data.withUnsafeBytes {
                    $0.baseAddress == UnsafeRawPointer(pointer)
                }
                guard aliasesSurface else {
                    workspace.outputArray = nil
                    throw ServiceError.model("Private ANE output cannot be mapped without copying")
                }
            }
            let borrowed = workspace.outputArray!
            let activation = PrivateMLPKernels.activation(borrowed, gpu: gpu,
                rows: rows, aneHidden: aneHidden, gpuHidden: gpuHidden)
            eval(activation)
            try cancellation.check()
            return activation
        }
    }

    private func withEvaluation<T>(updating inputs: [Int: [Float16]], cancellation: CancellationToken,
                                   outputStage: String, lane: InferenceGate? = nil, consume: () throws -> T) throws -> T {
        try lock.withLock {
            try cancellation.check()
            guard inputs.allSatisfy({ inputElements.indices.contains($0.key) && $0.value.count == inputElements[$0.key] }),
                  inputElements.indices.allSatisfy({ inputs[$0] != nil || (!workspace.shared && initializedInputs.contains($0)) }) else {
                throw ServiceError.invalidRequest("Private ANE input does not match its compiled shape")
            }
            var error = [CChar](repeating: 0, count: 4096)
            func milliseconds(_ start: ContinuousClock.Instant, _ end: ContinuousClock.Instant) -> Double {
                let duration = start.duration(to: end).components
                return Double(duration.seconds) * 1_000 + Double(duration.attoseconds) / 1e15
            }
            let writeStarted = ContinuousClock.now
            for (index, input) in inputs { try write(input, at: index) }
            let evaluateStarted = ContinuousClock.now
            var nativeStarted = evaluateStarted
            let evaluate = {
                nativeStarted = .now
                guard indexed_ane_evaluate(self.handle, &error, error.count) else { throw ServiceError.model(Self.message(error)) }
            }
            // Vision uses different surfaces. Release the device lane as soon
            // as ANE finishes; the sharing-group lock still protects GPU reads.
            if let lane { try lane.withPermit(priority: true, cancellation: cancellation, evaluate) }
            else { try evaluate() }
            let readStarted = ContinuousClock.now
            evaluationCount += 1
            try cancellation.check()
            let result = try consume()
            lastStages = ["surface_write": milliseconds(writeStarted, evaluateStarted),
                          "lane_wait": milliseconds(evaluateStarted, nativeStarted),
                          "ane_call": milliseconds(nativeStarted, readStarted),
                          outputStage: milliseconds(readStarted, .now)]
            return result
        }
    }
}

/// Matches oMLX v0.6.4's FP32 row quantization, including ties-to-even rounding
/// and FP16 row scales. This is separate from the source Q8 affine quantization.
public struct PrivateANELinearWeights {
    public let data: [Int8]
    public let scales: [Float16]

    public init(dense: [Float], inputDimension: Int, outputDimension: Int) throws {
        guard inputDimension > 0, outputDimension > 0,
              inputDimension <= 32768, outputDimension <= 32768,
              dense.count == inputDimension * outputDimension,
              dense.allSatisfy(\.isFinite) else {
            throw ServiceError.invalidRequest("Invalid private ANE linear weights")
        }
        var data = [Int8](repeating: 0, count: dense.count)
        var scales = [Float16](repeating: 0, count: outputDimension)
        for row in 0..<outputDimension {
            let start = row * inputDimension
            let maximum = dense[start..<(start + inputDimension)].reduce(Float.zero) { max($0, abs($1)) }
            let scale = max(maximum / 127, 1e-8)
            scales[row] = Float16(scale)
            for column in 0..<inputDimension {
                data[start + column] = Int8(max(-127, min(127, (dense[start + column] / scale).rounded(.toNearestOrEven))))
            }
        }
        self.data = data; self.scales = scales
    }

    public static func mil(inputDimension: Int, outputDimension: Int, sequenceLength: Int) throws -> String {
        guard inputDimension > 0, inputDimension <= 32768, outputDimension > 0, outputDimension <= 32768,
              sequenceLength >= 64, sequenceLength <= 8192, sequenceLength.isMultiple(of: 64) else {
            throw ServiceError.invalidRequest("Invalid private ANE linear shape")
        }
        return """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
          func main<ios18>(tensor<fp16, [1, \(inputDimension), 1, \(sequenceLength)]> x) {
            tensor<int8, [\(outputDimension), \(inputDimension), 1, 1]> wd = const()[name=string("wd"), val=tensor<int8, [\(outputDimension), \(inputDimension), 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_data.bin"), offset=uint64(64)))];
            tensor<fp16, [\(outputDimension), 1, 1, 1]> ws = const()[name=string("ws"), val=tensor<fp16, [\(outputDimension), 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_scale.bin"), offset=uint64(64)))];
            tensor<fp16, [\(outputDimension), \(inputDimension), 1, 1]> w = constexpr_blockwise_shift_scale(data=wd, scale=ws)[name=string("dequant")];
            string pt = const()[name=string("pt"), val=string("valid")];
            tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
            tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
            tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
            int32 gr = const()[name=string("gr"), val=int32(1)];
            tensor<fp16, [1, \(outputDimension), 1, \(sequenceLength)]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string("conv")];
          } -> (y);
        }
        """
    }
}
