import Foundation

public let officialDimensions = [64, 128, 256, 512, 1024, 2048]
public let modelPrefix = "wemm-embedding-2b-apple"

public enum JSONValue: Codable, Sendable, Equatable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let value = try? container.decode(Bool.self) { self = .bool(value) }
        else if let value = try? container.decode(Double.self) { self = .number(value) }
        else if let value = try? container.decode(String.self) { self = .string(value) }
        else if let value = try? container.decode([JSONValue].self) { self = .array(value) }
        else { self = .object(try container.decode([String: JSONValue].self)) }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }

    public var stringValue: String? {
        guard case .string(let value) = self else { return nil }
        return value
    }

    public var objectValue: [String: JSONValue]? {
        guard case .object(let value) = self else { return nil }
        return value
    }

    public var arrayValue: [JSONValue]? {
        guard case .array(let value) = self else { return nil }
        return value
    }

    public var numberValue: Double? {
        guard case .number(let value) = self else { return nil }
        return value
    }
}

public struct EmbeddingRequest: Codable, Sendable {
    public var model: String?
    public var input: String?
    public var messages: [JSONValue]?
    public var requestID: String?

    enum CodingKeys: String, CodingKey {
        case model, input, messages
        case requestID = "request_id"
    }
}

public struct EmbeddingResult: Sendable {
    public let vector: [Float]
    public let timingsMS: [String: Double]
    public let modality: String
    public let promptTokens: Int

    public init(
        vector: [Float],
        timingsMS: [String: Double],
        modality: String,
        promptTokens: Int = 0
    ) {
        self.vector = vector
        self.timingsMS = timingsMS
        self.modality = modality
        self.promptTokens = promptTokens
    }
}

public struct EngineMetadata: Sendable {
    public let backend: String
    public let loadSeconds: Double
    public let warmupSeconds: Double
    public let allocatedBytes: Int
    public let packageFingerprint: String?
    public let embeddingSpaceTemplate: String?
    public let computeDevices: [String]
    public let decoderSegmentLoaded: Bool
    public let decoderBundleLoaded: Bool
    public let decoderMinimumTokens: Int?
    public let decoderMaximumTokens: Int?
    public let decoderBucketTokenLimits: [Int]
    public let decoderSegmentFingerprint: String?
    public let decoderBundleFingerprint: String?
    public let decoderLoadingStrategy: String?
    public let languageCompute: String
    public let mlxDecoderLoaded: Bool
    public let mlxDevice: String
    public let maxModelLength: Int
    public let privateANE: JSONValue?
    public let scheduling: JSONValue?

    public init(
        backend: String,
        loadSeconds: Double = 0,
        warmupSeconds: Double = 0,
        allocatedBytes: Int = 0,
        packageFingerprint: String? = nil,
        embeddingSpaceTemplate: String? = nil,
        computeDevices: [String] = [],
        decoderSegmentLoaded: Bool = false,
        decoderBundleLoaded: Bool = false,
        decoderMinimumTokens: Int? = nil,
        decoderMaximumTokens: Int? = nil,
        decoderBucketTokenLimits: [Int] = [],
        decoderSegmentFingerprint: String? = nil,
        decoderBundleFingerprint: String? = nil,
        decoderLoadingStrategy: String? = nil,
        languageCompute: String = "gpu",
        mlxDecoderLoaded: Bool = true,
        mlxDevice: String = "gpu",
        maxModelLength: Int = 8192,
        privateANE: JSONValue? = nil,
        scheduling: JSONValue? = nil
    ) {
        self.backend = backend
        self.loadSeconds = loadSeconds
        self.warmupSeconds = warmupSeconds
        self.allocatedBytes = allocatedBytes
        self.packageFingerprint = packageFingerprint
        self.embeddingSpaceTemplate = embeddingSpaceTemplate
        self.computeDevices = computeDevices
        self.decoderSegmentLoaded = decoderSegmentLoaded
        self.decoderBundleLoaded = decoderBundleLoaded
        self.decoderMinimumTokens = decoderMinimumTokens
        self.decoderMaximumTokens = decoderMaximumTokens
        self.decoderBucketTokenLimits = decoderBucketTokenLimits
        self.decoderSegmentFingerprint = decoderSegmentFingerprint
        self.decoderBundleFingerprint = decoderBundleFingerprint
        self.decoderLoadingStrategy = decoderLoadingStrategy
        self.languageCompute = languageCompute
        self.mlxDecoderLoaded = mlxDecoderLoaded
        self.mlxDevice = mlxDevice
        self.maxModelLength = maxModelLength
        self.privateANE = privateANE
        self.scheduling = scheduling
    }
}

public protocol EmbeddingEngine: AnyObject, Sendable {
    var metadata: EngineMetadata { get }
    /// Engines opting into concurrency must serialize their shared stage state.
    var maxConcurrentRequests: Int { get }
    func embed(messages: [JSONValue], dimension: Int, cancellation: CancellationToken) throws
        -> EmbeddingResult
}

public extension EmbeddingEngine {
    var maxConcurrentRequests: Int { 1 }
}

public final class CancellationToken: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private var deadline: Double?

    public init(timeoutMS: Int? = nil) {
        deadline = timeoutMS.map { ProcessInfo.processInfo.systemUptime + Double($0) / 1000 }
    }
    /// Deadlines may only shrink. Kernel control flow still receives cancelled
    /// so a timeout cannot latch a healthy ANE kernel as failed.
    public func limit(toMilliseconds milliseconds: Int) {
        lock.withLock {
            let next = ProcessInfo.processInfo.systemUptime + Double(milliseconds) / 1000
            deadline = min(deadline ?? next, next)
        }
    }
    public var hasExpired: Bool {
        lock.withLock { deadline.map { ProcessInfo.processInfo.systemUptime >= $0 } ?? false }
    }
    var remainingSeconds: Double {
        lock.withLock { max(0.001, (deadline ?? (ProcessInfo.processInfo.systemUptime + 300)) - ProcessInfo.processInfo.systemUptime) }
    }

    public func cancel() { lock.withLock { cancelled = true } }
    public var isCancelled: Bool { lock.withLock { cancelled || (deadline.map { ProcessInfo.processInfo.systemUptime >= $0 } ?? false) } }

    public func check() throws {
        if isCancelled { throw ServiceError.cancelled }
    }
}

public enum ServiceError: Error, LocalizedError, Sendable {
    case invalidRequest(String, status: Int = 400)
    case notFound(String)
    case overloaded(String)
    case unauthorized(String)
    case cancelled
    case deadlineExceeded
    case model(String)
    case internalFailure(String)

    public var status: Int {
        switch self {
        case .invalidRequest(_, let status): status
        case .notFound: 404
        case .overloaded: 429
        case .unauthorized: 401
        case .cancelled: 499
        case .deadlineExceeded: 504
        case .model, .internalFailure: 500
        }
    }

    public var errorDescription: String? {
        switch self {
        case .invalidRequest(let message, _), .notFound(let message),
             .overloaded(let message), .model(let message), .internalFailure(let message):
            message
        case .unauthorized(let message): message
        case .cancelled: "请求已取消"
        case .deadlineExceeded: "请求超时"
        }
    }

    public var responseType: String {
        if case .unauthorized = self { return "authentication_error" }
        return status < 500 ? "invalid_request_error" : "server_error"
    }

    public var responseCode: String {
        switch self {
        case .invalidRequest: "invalid_request"
        case .notFound: "not_found"
        case .overloaded: "service_overloaded"
        case .unauthorized: "authentication_failed"
        case .cancelled: "request_cancelled"
        case .deadlineExceeded: "request_timeout"
        case .model: "model_error"
        case .internalFailure: "internal_error"
        }
    }
}

public func modelID(dimension: Int) throws -> String {
    guard officialDimensions.contains(dimension) else {
        throw ServiceError.invalidRequest("不支持的 WeMM 维度：\(dimension)")
    }
    return "\(modelPrefix)-\(dimension)"
}

public func dimension(for selectedModel: String?, default defaultDimension: Int) throws -> Int {
    let selected = (selectedModel ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased()
    if selected.isEmpty || selected == modelPrefix { return defaultDimension }
    let marker = "\(modelPrefix)-"
    guard selected.hasPrefix(marker), let value = Int(selected.dropFirst(marker.count)) else {
        throw ServiceError.notFound("未知模型：\(selected)")
    }
    guard officialDimensions.contains(value) else {
        throw ServiceError.invalidRequest(
            "模型 \(selected) 的维度不受支持；可选 \(officialDimensions)"
        )
    }
    return value
}

public func normalizedMessages(from request: EmbeddingRequest) throws -> [JSONValue] {
    if let messages = request.messages, !messages.isEmpty { return messages }
    if let input = request.input {
        return [
            .object([
                "role": .string("user"),
                "content": .array([.object(["type": .string("text"), "text": .string(input)])]),
            ])
        ]
    }
    throw ServiceError.invalidRequest("messages 必须是非空数组")
}
