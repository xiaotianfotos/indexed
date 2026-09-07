import Foundation
import MLX

public enum PrivateGDNValidation {
    public static func checkGraphAndProfile() throws {
        try PrivateGDNGraph.validateFrozenIdentity()
        let valid = """
        {"package_fingerprint":"synthetic-fingerprint","settings":{"mlp":true,"recurrence_ane":true,"sequence_length":2112,"fraction":0.75,"max_layers":24,"recurrence_layer_slots":[0]},"quality_gate":{"passed":true,"minimum_vector_cosine":0.999,"reference":{"passed":true}},"comparison":{"vector_cosine":0.9998},"model":{"recurrence":{"algorithm":"block-forward-substitution-v1","solve_block_size":8,"query_scale":4096,"max_tokens":8192,"io_dtype":"fp16","enabled_layer_slots":[0]}}}
        """
        _ = try PrivateGDNProfile.validate(Data(valid.utf8), packageFingerprint: "synthetic-fingerprint")
        for invalid in ["{}", valid.replacingOccurrences(of: "true", with: "false"),
                        valid.replacingOccurrences(of: "0.9998", with: "0.5"),
                        valid.replacingOccurrences(of: "4096", with: "2048"),
                        valid.replacingOccurrences(of: "[0]", with: "[1]"),
                        valid.replacingOccurrences(of: "synthetic-fingerprint", with: "wrong-model")] {
            do {
                _ = try PrivateGDNProfile.validate(Data(invalid.utf8), packageFingerprint: "synthetic-fingerprint")
                throw ServiceError.internalFailure("Invalid D quality profile was accepted")
            } catch ServiceError.model { }
        }
    }

    /// Compare actual ANE recurrence with a direct token-by-token CPU oracle.
    /// The oracle does not use the MIL graph or its block solve.
    public static func checkHardware() throws -> Int {
        try checkGraphAndProfile()
        let backend = try PrivateGDNRecurrence()
        var valuesChecked = 0
        for tokens in [65, 2] {
            var seed: UInt32 = 173
            func random(_ count: Int, scale: Float) -> [Float16] {
                (0..<count).map { _ in
                    seed = seed &* 1664525 &+ 1013904223
                    return Float16((Float(seed >> 8) / Float(1 << 24) * 2 - 1) * scale)
                }
            }
            let count = tokens * 16 * 128, stateCount = 16 * 128 * 128
            let q = random(count, scale: 0.01), k = random(count, scale: 0.08), v = random(count, scale: 0.2)
            let initial = random(stateCount, scale: 0.002)
            let beta = [Float16](repeating: 0.3, count: tokens * 16)
            let decay = [Float](repeating: 0.97, count: tokens * 16)
            let query = MLXArray(q, [1, tokens, 16, 128])
            let key = MLXArray(k, query.shape), value = MLXArray(v, query.shape)
            let state = MLXArray(initial, [1, 16, 128, 128]).asType(.float32)
            let b = MLXArray(beta, [1, tokens, 16]), g = MLXArray(decay, [1, tokens, 16])
            let before = backend.aneEvaluations
            guard let result = try backend.evaluate(q: query, k: key, v: value, g: g, beta: b,
                state: state, mask: nil, layerSlot: 0, cancellation: CancellationToken()) else {
                throw ServiceError.model("Synthetic GDN incorrectly fell back")
            }
            guard backend.aneEvaluations - before == (tokens + 63) / 64 else { throw ServiceError.model("GDN chunk count changed") }
            var expectedState = initial.map(Float.init)
            var expectedOutput = [Float](repeating: 0, count: count)
            let decayStep = exp(Float(Float16(log(Float(0.97)))))
            for token in 0..<tokens {
                for head in 0..<16 {
                    let inputBase = (token * 16 + head) * 128
                    for row in 0..<128 {
                        let stateBase = (head * 128 + row) * 128
                        var read: Float = 0
                        for column in 0..<128 {
                            expectedState[stateBase + column] *= decayStep
                            read += expectedState[stateBase + column] * Float(k[inputBase + column])
                        }
                        let update = (Float(v[inputBase + row]) - read) * Float(beta[token * 16 + head])
                        var output: Float = 0
                        for column in 0..<128 {
                            expectedState[stateBase + column] += update * Float(k[inputBase + column])
                            output += expectedState[stateBase + column] * Float(q[inputBase + column])
                        }
                        expectedOutput[inputBase + row] = output
                    }
                }
                if (token + 1).isMultiple(of: 64) {
                    expectedState = expectedState.map { Float(Float16($0)) }
                }
            }
            for (actual, expected) in [(result.0.asType(.float32).asArray(Float.self), expectedOutput),
                                       (result.1.asArray(Float.self), expectedState)] {
                var difference: Double = 0, norm: Double = 0
                for (a, e) in zip(actual, expected) {
                    guard a.isFinite else { throw ServiceError.model("Nonfinite GDN output") }
                    difference += pow(Double(a - e), 2); norm += pow(Double(e), 2)
                }
                guard sqrt(difference / norm) < 0.05 else {
                    throw ServiceError.model("Private GDN differs from sequential CPU oracle: relative L2 \(sqrt(difference / norm))")
                }
                valuesChecked += actual.count
            }
            let after = backend.aneEvaluations
            let excluded = try backend.evaluate(q: query, k: key, v: value, g: g, beta: b, state: state,
                mask: nil, layerSlot: 1, cancellation: CancellationToken())
            let masked = try backend.evaluate(q: query, k: key, v: value, g: g, beta: b, state: state,
                mask: .zeros([1, tokens], dtype: .bool), layerSlot: 0, cancellation: CancellationToken())
            let cancelled = CancellationToken(); cancelled.cancel()
            do {
                _ = try backend.evaluate(q: query, k: key, v: value, g: g, beta: b, state: state,
                    mask: nil, layerSlot: 0, cancellation: cancelled)
                throw ServiceError.internalFailure("Cancelled GDN was accepted")
            } catch ServiceError.cancelled { }
            guard excluded == nil, masked == nil, backend.aneEvaluations == after else {
                throw ServiceError.model("GDN mask/layer/cancellation fallback still submitted ANE work")
            }
        }
        return valuesChecked
    }
}
