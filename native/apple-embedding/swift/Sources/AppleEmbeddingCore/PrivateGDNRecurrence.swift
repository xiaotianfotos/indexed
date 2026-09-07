import Darwin
import Foundation
import MLX

/// D's fixed 16-head, 64-token, block-8 recurrence with FP16 IOSurfaces.
/// The state is transposed at the MLX boundary and returned as FP32, matching
/// the frozen Python implementation. Only tokenwise recurrence is chunked.
public final class PrivateGDNRecurrence: @unchecked Sendable {
    private let lock = NSLock()
    private let program: PrivateANEProgram
    private var dynamic = [Float16](repeating: 0, count: 7 * 16 * 128 * 128)
    private var calls = 0
    private var evaluations = 0
    private var fallbacks: [String: Int] = [:]
    private var runtimeMS: Double = 0
    private var failed = false
    private var lastError: String?
    private let slots: Set<Int>
    private let compileSeconds: Double
    public var aneEvaluations: Int { lock.withLock { evaluations } }

    public init(layerSlots: [Int] = [0]) throws {
        guard !layerSlots.isEmpty, Set(layerSlots).count == layerSlots.count,
              layerSlots.allSatisfy({ (0..<18).contains($0) }) else {
            throw ServiceError.invalidRequest("Invalid GDN layer slots")
        }
        slots = Set(layerSlots)
        try PrivateGDNGraph.validateFrozenIdentity()
        let started = Date.timeIntervalSinceReferenceDate
        program = try PrivateANEProgram(mil: PrivateGDNGraph.mil,
            inputElements: [7 * 16 * 128 * 128, 18 * 128 * 128],
            outputElements: [2 * 16 * 128 * 128])
        try program.prepareInput(PrivateGDNGraph.shared, at: 1)
        compileSeconds = Date.timeIntervalSinceReferenceDate - started
    }

    public func evaluate(q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray,
                         beta: MLXArray, state: MLXArray, mask: MLXArray?,
                         layerSlot: Int, cancellation: CancellationToken) throws -> (MLXArray, MLXArray)? {
        try lock.withLock {
            calls += 1
            func fallback(_ reason: String) -> (MLXArray, MLXArray)? {
                fallbacks[reason, default: 0] += 1
                return nil
            }
            try cancellation.check()
            guard slots.contains(layerSlot) else { return fallback("layer_filter") }
            guard !failed else { return fallback("latched_failure") }
            guard q.ndim == 4, k.ndim == 4, v.ndim == 4, g.ndim == 3 else { return fallback("rank") }
            let tokens = q.dim(1)
            guard q.shape == [1, tokens, 16, 128], k.shape == q.shape, v.shape == q.shape,
                  g.shape == [1, tokens, 16], tokens > 1, tokens <= 8192 else { return fallback("shape") }
            guard state.shape == [1, 16, 128, 128] else { return fallback("state_shape") }
            guard beta.shape == [1, tokens, 16] else { return fallback("beta_shape") }
            if let mask, mask.size != tokens || !mask.asType(.bool).asArray(Bool.self).allSatisfy({ $0 }) {
                return fallback("mask")
            }
            let started = Date.timeIntervalSinceReferenceDate
            defer { runtimeMS += (Date.timeIntervalSinceReferenceDate - started) * 1000 }
            let query = q.asType(.float16).asArray(Float16.self)
            let key = k.asType(.float16).asArray(Float16.self)
            let value = v.asType(.float16).asArray(Float16.self)
            let decay = g.asType(.float32).asArray(Float.self)
            let betaValues = beta.asType(.float16).asArray(Float16.self)
            var currentState = state.transposed(0, 1, 3, 2).asType(.float16).asArray(Float16.self)
            guard decay.allSatisfy(\.isFinite), query.allSatisfy(\.isFinite), key.allSatisfy(\.isFinite),
                  value.allSatisfy(\.isFinite), betaValues.allSatisfy(\.isFinite), currentState.allSatisfy(\.isFinite) else {
                return fallback("nonfinite")
            }
            let logDecay = decay.map { logf(min(1, max(1e-6, $0))) }
            let matrix = 128 * 128, plane = 16 * matrix
            var output = [Float16](repeating: 0, count: tokens * 16 * 128)
            do {
                for start in stride(from: 0, to: tokens, by: 64) {
                    try cancellation.check()
                    let valid = min(64, tokens - start)
                    // Keep the staging allocation and clear padding, especially
                    // the last partial chunk after a previous longer request.
                    for index in dynamic.indices { dynamic[index] = 0 }
                    for head in 0..<16 {
                        let headBase = head * matrix
                        for index in 0..<matrix { dynamic[4 * plane + headBase + index] = currentState[headBase + index] }
                        for row in 0..<valid {
                            let source = ((start + row) * 16 + head) * 128
                            let scalar = (start + row) * 16 + head
                            dynamic[headBase + row * 128 + row] = Float16(logDecay[scalar])
                            for column in 0..<128 {
                                let target = headBase + row * 128 + column
                                dynamic[plane + target] = key[source + column]
                                dynamic[2 * plane + headBase + column * 128 + row] = key[source + column]
                                dynamic[3 * plane + target] = betaValues[scalar]
                                dynamic[5 * plane + target] = value[source + column]
                                dynamic[6 * plane + target] = Float16(Float(query[source + column]) * 4096)
                            }
                        }
                    }
                    let before = program.evaluations
                    let raw: [Float16]
                    do { raw = try program.evaluate(updating: [0: dynamic], cancellation: cancellation)[0] }
                    catch { evaluations += program.evaluations - before; throw error }
                    evaluations += program.evaluations - before
                    currentState = Array(raw[0..<plane])
                    for head in 0..<16 {
                        for row in 0..<valid {
                            let source = plane + head * matrix + row * 128
                            let target = ((start + row) * 16 + head) * 128
                            for column in 0..<128 { output[target + column] = raw[source + column] }
                        }
                    }
                }
            } catch ServiceError.cancelled { throw ServiceError.cancelled }
            catch {
                failed = true
                lastError = String(String(describing: error).prefix(1024))
                return fallback("runtime_failure")
            }
            try cancellation.check()
            return (MLXArray(output, [1, tokens, 16, 128]).asType(v.dtype),
                    MLXArray(currentState, [1, 16, 128, 128]).transposed(0, 1, 3, 2).asType(.float32))
        }
    }

    public var diagnostics: JSONValue {
        lock.withLock { .object([
            "backend": .string("swift-private-inmemory-ane-gdn-recurrence"),
            "io_dtype": .string("fp16"), "calls": .number(Double(calls)),
            "ane_evaluations": .number(Double(evaluations)),
            "fallbacks": .object(fallbacks.mapValues { .number(Double($0)) }),
            "last_error": lastError.map(JSONValue.string) ?? .null,
            "runtime_ms": .number(runtimeMS), "compile_seconds": .number(compileSeconds),
            "max_tokens": .number(8192), "enabled_layer_slots": .array(slots.sorted().map { .number(Double($0)) }),
            "total_layer_slots": .number(18), "query_scale": .number(4096),
            "algorithm": .string("block-forward-substitution-v1"), "active_chunk_size": .number(64),
            "solve_block_size": .number(8), "blocks_per_chunk": .number(8),
            "ane_operation_count": .number(Double(PrivateGDNGraph.operations.count)),
            "dynamic_surface_bytes_per_chunk": .number(Double(7 * 16 * 128 * 128 * 2)),
            "shared_surface_bytes_written_once": .number(Double(18 * 128 * 128 * 2)),
            "output_surface_bytes_per_chunk": .number(Double(2 * 16 * 128 * 128 * 2)),
            "staging_buffer_reuse": .bool(true), "mlx_transfer_dtype": .string("fp16"),
        ]) }
    }
}
