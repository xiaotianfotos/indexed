import Foundation

/// Protocol and packaging smoke-test engine. Production serving refuses this
/// engine unless the caller passes `--development-deterministic-engine`.
public final class DeterministicEmbeddingEngine: EmbeddingEngine, @unchecked Sendable {
    public let metadata = EngineMetadata(
        backend: "deterministic-contract-test",
        packageFingerprint: "deterministic-test-only",
        embeddingSpaceTemplate: "deterministic-test-{dimension}"
    )

    private let delayMS: Int
    public init(delayMS: Int = 0) throws {
        guard (0...10_000).contains(delayMS) else { throw ServiceError.invalidRequest("Invalid deterministic test delay") }
        self.delayMS = delayMS
    }

    public func embed(
        messages: [JSONValue],
        dimension: Int,
        cancellation: CancellationToken
    ) throws -> EmbeddingResult {
        let deadline = ProcessInfo.processInfo.systemUptime + Double(delayMS) / 1000
        while ProcessInfo.processInfo.systemUptime < deadline {
            try cancellation.check()
            Thread.sleep(forTimeInterval: 0.005)
        }
        try cancellation.check()
        guard !messages.isEmpty else { throw ServiceError.invalidRequest("messages 必须是非空数组") }
        var vector = [Float](repeating: 0, count: dimension)
        vector[0] = 1
        return EmbeddingResult(
            vector: vector,
            timingsMS: ["preprocess": 0, "vision": 0, "language": 0, "total": 0],
            modality: containsImage(messages) ? "image" : "text"
        )
    }

    private func containsImage(_ values: [JSONValue]) -> Bool {
        values.contains { value in
            switch value {
            case .string(let value): value.contains("image")
            case .array(let values): containsImage(values)
            case .object(let values): containsImage(Array(values.values))
            default: false
            }
        }
    }
}
