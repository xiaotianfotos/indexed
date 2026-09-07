import Foundation

/// Validates the actual frozen quality record before enabling the D candidate.
/// Source paths and historical timing fields are deliberately not runtime inputs.
struct PrivateGDNProfile: Decodable {
    struct Settings: Decodable {
        let mlp: Bool
        let recurrence_ane: Bool
        let sequence_length: Int
        let fraction: Double
        let max_layers: Int
        let recurrence_layer_slots: [Int]?
    }
    struct Quality: Decodable {
        struct Reference: Decodable { let passed: Bool }
        let passed: Bool
        let minimum_vector_cosine: Double
        let reference: Reference
    }
    struct Comparison: Decodable { let vector_cosine: Double }
    struct Model: Decodable {
        struct Recurrence: Decodable {
            let algorithm: String
            let solve_block_size: Int
            let query_scale: Double
            let max_tokens: Int
            let io_dtype: String
            let enabled_layer_slots: [Int]
        }
        let recurrence: Recurrence
    }
    let package_fingerprint: String
    let settings: Settings
    let quality_gate: Quality
    let comparison: Comparison
    let model: Model

    static func validate(_ data: Data, packageFingerprint: String) throws -> PrivateGDNProfile {
        let profile: PrivateGDNProfile
        do { profile = try JSONDecoder().decode(Self.self, from: data) }
        catch { throw ServiceError.model("Invalid private D quality profile schema") }
        let recurrence = profile.model.recurrence
        let slots = profile.settings.recurrence_layer_slots ?? recurrence.enabled_layer_slots
        guard profile.package_fingerprint == packageFingerprint,
              profile.settings.mlp, profile.settings.recurrence_ane,
              profile.settings.sequence_length == 2112, profile.settings.fraction == 0.75,
              profile.settings.max_layers == 24, slots == [0], recurrence.enabled_layer_slots == [0],
              profile.quality_gate.passed, profile.quality_gate.reference.passed,
              profile.quality_gate.minimum_vector_cosine >= 0.999,
              profile.comparison.vector_cosine >= profile.quality_gate.minimum_vector_cosine,
              profile.comparison.vector_cosine <= 1,
              recurrence.algorithm == "block-forward-substitution-v1", recurrence.solve_block_size == 8,
              recurrence.query_scale == 4096, recurrence.max_tokens == 8192, recurrence.io_dtype == "fp16" else {
            throw ServiceError.model("Private D quality profile does not match the frozen model/kernel/slot contract")
        }
        return profile
    }
}
