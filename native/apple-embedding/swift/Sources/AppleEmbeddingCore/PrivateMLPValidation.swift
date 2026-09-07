import Foundation
import MLX
import MLXNN

private func syntheticValues(_ count: Int, seed: UInt32, scale: Float) -> [Float] {
    var state = seed
    return (0..<count).map { _ in
        state = state &* 1664525 &+ 1013904223
        return (Float(state >> 8) / Float(1 << 24) * 2 - 1) * scale
    }
}

private final class SyntheticMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: QuantizedLinear
    @ModuleInfo(key: "up_proj") var up: QuantizedLinear
    @ModuleInfo(key: "down_proj") var down: QuantizedLinear

    init(dtype: DType) {
        func linear(_ input: Int, _ output: Int, _ seed: UInt32) -> QuantizedLinear {
            let values = syntheticValues(input * output, seed: seed, scale: 0.05)
            return QuantizedLinear(weight: MLXArray(values, [output, input]).asType(dtype),
                                   bias: nil, groupSize: 64, bits: 8)
        }
        _gate.wrappedValue = linear(128, 256, 101)
        _up.wrappedValue = linear(128, 256, 107)
        _down.wrappedValue = linear(256, 128, 113)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

/// Maintainer hardware regression entry point, called only by the check binary.
/// Synthetic weights exercise routing/rounding; they do not replace model parity.
public enum PrivateMLPValidation {
    public static func benchmarkProjections() throws {
        Memory.cacheLimit = 128 * 1024 * 1024
        var results: [[String: Any]] = []
        for (rows, input, output) in [(2112, 6144, 2048), (3689, 2048, 6144), (3689, 2048, 2048)] {
            func signal(_ shape: [Int], frequency: Float) -> MLXArray {
                let position = MLXArray(0..<shape.reduce(1, *)).asType(.float32)
                return (sin(position * frequency) * 0.05).reshaped(shape).asType(.bfloat16)
            }
            let linear = QuantizedLinear(weight: signal([output, input], frequency: 0.031),
                bias: nil, groupSize: 64, bits: 8)
            let x = signal([1, rows, input], frequency: 0.037)
            eval(x, linear)
            func project(_ variant: Bool) -> MLXArray {
                variant ? PrivateMLPKernels.affine(x, weight: linear.weight, scales: linear.scales, biases: linear.biases!) : linear(x)
            }
            let reference = project(false), candidate = project(true)
            eval(reference, candidate)
            let difference = reference.asType(.float32) - candidate.asType(.float32)
            let relativeL2 = sqrt(sum(difference * difference) / sum(square(reference.asType(.float32)))).item(Float.self)
            guard relativeL2.isFinite, relativeL2 < 0.01 else { throw ServiceError.model("Projection tile numerical regression") }
            var stock: [Double] = [], variant: [Double] = []
            for useVariant in [false, true, true, false, false, true, true, false, false, true] {
                let started = ContinuousClock.now
                eval(project(useVariant))
                let duration = started.duration(to: .now).components
                let milliseconds = Double(duration.seconds) * 1000 + Double(duration.attoseconds) / 1e15
                if useVariant { variant.append(milliseconds) } else { stock.append(milliseconds) }
            }
            results.append(["M": rows, "K": input, "N": output, "relativeL2": relativeL2,
                "stockMS": stock, "variant8MS": variant])
        }
        let data = try JSONSerialization.data(withJSONObject: ["schema": 1, "results": results], options: [.sortedKeys])
        print(String(decoding: data, as: UTF8.self))
    }

    public static func check() throws -> Int {
        var valuesChecked = 0
        // A borrowed Metal output must release its program owner after its GPU
        // consumer retires. Reopening the same graph also exercises unloading
        // and the private bridge's exclusive staging-directory ownership.
        for _ in 0..<2 {
            weak var retired: PrivateANEProgram?
            try autoreleasepool {
                var program: PrivateANEProgram? = try PrivateANEProgram(
                    mil: PrivateANELinearWeights.mil(inputDimension: 64, outputDimension: 128, sequenceLength: 64),
                    weights: [Int8](repeating: 1, count: 64 * 128), scales: [Float16](repeating: 0.125, count: 128),
                    inputElements: [64 * 64], outputElements: [128 * 64])
                retired = program
                let activation = try program!.evaluateActivation([Float16](repeating: 0, count: 64 * 64),
                    gpu: .zeros([1, 64, 128], dtype: .float16), rows: 64, aneHidden: 64, gpuHidden: 64,
                    cancellation: CancellationToken(), lane: nil)
                guard activation.asArray(Float16.self).allSatisfy({ $0 == 0 }) else {
                    throw ServiceError.model("Borrowed output activation changed")
                }
                program = nil
            }
            Stream.gpu.synchronize()
            for _ in 0..<100 where retired != nil { Thread.sleep(forTimeInterval: 0.001) }
            guard retired == nil else { throw ServiceError.model("Borrowed output retained its ANE program after GPU completion") }
        }
        for dtype in [DType.float16, .bfloat16] {
            let original = SyntheticMLP(dtype: dtype)
            // Check the GPU suffix independently so the 75% ANE prefix cannot
            // conceal a broken dispatch, row boundary or quantized suffix.
            let qmmInput = MLXArray(syntheticValues(65 * 128, seed: 127, scale: 1), [1, 65, 128]).asType(dtype)
            // Check the actual row layout as well as logical values: a returned
            // transposed view would be numerically right but still copy slowly
            // on the CPU. Include an offset slice used by multi-block prefill.
            for input in [qmmInput, qmmInput[0..., 1..<64, 0...]] {
                let packed = PrivateMLPKernels.packedANEInput(input)
                let bytes = packed.asData(access: .noCopy)
                let rows = input.dim(1)
                guard bytes.strides == [rows, 1],
                      packed.asArray(Float16.self) == input.reshaped(rows, 128).transposed().asType(.float16).asArray(Float16.self) else {
                    throw ServiceError.model("ANE input packing changed FP16 values or retained strided storage")
                }
            }
            let qmmActual = PrivateMLPKernels.affine(qmmInput, weight: original.gate.weight,
                scales: original.gate.scales, biases: original.gate.biases!).asType(.float32).asArray(Float.self)
            let qmmExpected = original.gate(qmmInput).asType(.float32).asArray(Float.self)
            var qmmDifference: Double = 0, qmmNorm: Double = 0
            for (a, e) in zip(qmmActual, qmmExpected) {
                qmmDifference += pow(Double(a - e), 2); qmmNorm += pow(Double(e), 2)
            }
            guard sqrt(qmmDifference / qmmNorm) < 0.01 else { throw ServiceError.model("Variant 8 GPU suffix differs from Q8 affine reference") }
            valuesChecked += qmmActual.count
            let context = PrivateMLPContext()
            let configuration = try PrivateMLPConfiguration(sequenceLength: 64)
            let hybrid = try PrivateHybridMLP(original: original, configuration: configuration,
                                             context: context, layer: "synthetic")
            for rows in [13, 31, 32, 63, 64, 65, 96, 128, 2112, 2113, 2144] {
                let input = MLXArray(syntheticValues(rows * 128, seed: 131, scale: 1), [1, rows, 128]).asType(dtype)
                let before = context.operations
                let actual = try context.withRequest(CancellationToken()) { hybrid(input).asType(.float32).asArray(Float.self) }
                let expected = original(input).asType(.float32).asArray(Float.self)
                let expectedOperations = rows / 64 + (rows % 64 >= 32 ? 1 : 0)
                guard context.operations - before == expectedOperations else {
                    throw ServiceError.model("Hybrid MLP did not execute the expected ANE workload at \(rows) rows")
                }
                var difference: Double = 0, norm: Double = 0
                for (a, e) in zip(actual, expected) {
                    guard a.isFinite else { throw ServiceError.model("Nonfinite hybrid MLP value") }
                    difference += pow(Double(a - e), 2); norm += pow(Double(e), 2)
                }
                guard sqrt(difference / max(norm, 1e-20)) < 0.05 else {
                    throw ServiceError.model("Hybrid MLP numerical regression at \(rows) rows (\(dtype)): relative L2 \(sqrt(difference / norm))")
                }
                if rows < 32, actual != expected { throw ServiceError.model("Short-input GPU fallback changed") }
                valuesChecked += actual.count
            }
            let before = context.operations
            let cancelled = CancellationToken(); cancelled.cancel()
            do {
                _ = try context.withRequest(cancelled) { hybrid(.zeros([1, 64, 128], dtype: dtype)) }
                throw ServiceError.model("Cancelled hybrid MLP request was accepted")
            } catch ServiceError.cancelled { }
            guard context.operations == before else { throw ServiceError.model("Cancelled MLP submitted ANE work") }
            guard context.fallbacks["runtime_failure"] == nil else { throw ServiceError.model("Private MLP silently fell back after kernel failure") }
        }
        return valuesChecked
    }
}
