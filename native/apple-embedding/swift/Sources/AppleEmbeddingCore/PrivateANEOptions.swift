import Foundation

/// Product C/D use the same native implementations exercised by the reference
/// runner. Reject unsupported kernel choices before loading or patching a model.
public struct PrivateANEOptions {
    public let mlp: PrivateMLPConfiguration?
    public let recurrenceProfile: Data?
    public let pipelineDepth: Int

    public init(arguments: [String], mode: String) throws {
        func value(_ name: String, _ fallback: String) throws -> String {
            guard let index = arguments.firstIndex(of: name) else { return fallback }
            guard index + 1 < arguments.count, !arguments[index + 1].hasPrefix("--") else {
                throw ServiceError.invalidRequest("Missing value for \(name)")
            }
            return arguments[index + 1]
        }
        func integer(_ name: String, _ fallback: Int) throws -> Int {
            guard let number = Int(try value(name, String(fallback))) else {
                throw ServiceError.invalidRequest("Invalid integer for \(name)")
            }
            return number
        }
        let known: Set<String> = ["--private-ane-sequence-length", "--private-ane-mlp-fraction",
            "--private-ane-mlp-variant", "--private-ane-mlp-max-layers", "--private-ane-recurrence-profile",
            "--private-ane-recurrence-block-size", "--private-ane-recurrence-layer-slots",
            "--private-ane-recurrence-query-scale", "--private-ane-recurrence-max-tokens",
            "--private-ane-recurrence-io-dtype", "--video-down-projection", "--video-pipeline"]
        let supplied = arguments.filter { $0.hasPrefix("--private-ane-") || $0.hasPrefix("--video-") || $0.hasPrefix("--research-") }
        guard supplied.allSatisfy(known.contains) else {
            throw ServiceError.invalidRequest("Unsupported native C/D option; Python scripts and external bridges are no longer runtime inputs")
        }
        guard ["c", "d"].contains(mode) else {
            guard supplied.isEmpty else { throw ServiceError.invalidRequest("Private ANE options require explicit C or D mode") }
            mlp = nil; recurrenceProfile = nil; pipelineDepth = 1
            return
        }
        guard try integer("--private-ane-mlp-variant", 8) == 8,
              try value("--video-down-projection", "q8") == "q8" else {
            throw ServiceError.invalidRequest("Native C/D support variant 8 and Q8 down projection")
        }
        let depth = try integer("--video-pipeline", 2)
        guard [1, 2].contains(depth), let fraction = Double(try value("--private-ane-mlp-fraction", "0.75")) else {
            throw ServiceError.invalidRequest("Invalid native MLP fraction or pipeline depth")
        }
        mlp = try PrivateMLPConfiguration(sequenceLength: integer("--private-ane-sequence-length", 2112),
            fraction: fraction, maxLayers: integer("--private-ane-mlp-max-layers", 24))
        pipelineDepth = depth
        if mode == "d" {
            guard mlp?.sequenceLength == 2112, fraction == 0.75, mlp?.maxLayers == 24,
                  try integer("--private-ane-recurrence-block-size", 8) == 8,
                  try value("--private-ane-recurrence-layer-slots", "0") == "0",
                  try integer("--private-ane-recurrence-query-scale", 4096) == 4096,
                  try integer("--private-ane-recurrence-max-tokens", 8192) == 8192,
                  try value("--private-ane-recurrence-io-dtype", "fp16") == "fp16" else {
                throw ServiceError.invalidRequest("Native D requires seq2112/MLP24/75%/block8/slot0/FP16/scale4096/max8192")
            }
            let profile = try value("--private-ane-recurrence-profile", "")
            recurrenceProfile = profile.isEmpty ? FrozenPrivateDProfile.data : try Data(contentsOf: URL(fileURLWithPath: profile))
        } else { recurrenceProfile = nil }
    }
}
