import Foundation

public struct ServiceConfiguration: Sendable {
    public var host: String
    public var port: UInt16
    public var defaultDimension: Int
    public var maxRequestBytes: Int
    public var maxConcurrentRequests: Int
    public var maxQueuedRequests: Int
    public var maxRequestDurationMS: Int
    public var authToken: String?

    public init(
        host: String = "127.0.0.1",
        port: UInt16 = 18_768,
        defaultDimension: Int = 2048,
        maxRequestBytes: Int = 64 * 1024 * 1024,
        maxConcurrentRequests: Int = 1,
        maxQueuedRequests: Int = 16,
        maxRequestDurationMS: Int = 300_000,
        authToken: String? = nil
    ) {
        self.host = host
        self.port = port
        self.defaultDimension = defaultDimension
        self.maxRequestBytes = maxRequestBytes
        self.maxConcurrentRequests = maxConcurrentRequests
        self.maxQueuedRequests = maxQueuedRequests
        self.authToken = authToken
        self.maxRequestDurationMS = maxRequestDurationMS
    }
}

public struct HTTPRequest: Sendable {
    public let method: String
    public let path: String
    public let headers: [String: String]
    public let body: Data
}

public struct HTTPResponse: Sendable {
    public let status: Int
    public let body: Data
    public let headers: [String: String]

    public init(status: Int, body: Data = Data(), headers: [String: String] = [:]) {
        self.status = status
        self.body = body
        self.headers = headers
    }
}

public final class ServiceRouter: @unchecked Sendable {
    private let engine: EmbeddingEngine
    private let configuration: ServiceConfiguration
    private let startedAt = Date()
    private let requestSlots: DispatchSemaphore
    private let requestCapacity: DispatchSemaphore
    private let requestsLock = NSLock()
    private let loadLock = NSLock()
    private let metricsLock = NSLock()
    private var activeRequests: [String: CancellationToken] = [:]
    private var queuedRequests = 0
    private var runningRequests = 0
    private var requestsTotal = 0
    private var requestsSucceeded = 0
    private var requestsFailed = 0
    private var cumulativeRequestMilliseconds = 0.0
    private var lastRequestMilliseconds = 0.0

    public init(engine: EmbeddingEngine, configuration: ServiceConfiguration) throws {
        guard officialDimensions.contains(configuration.defaultDimension) else {
            throw ServiceError.invalidRequest("默认维度不受支持")
        }
        guard configuration.maxConcurrentRequests > 0 else {
            throw ServiceError.invalidRequest("maxConcurrentRequests 必须大于零")
        }
        guard configuration.maxConcurrentRequests <= engine.maxConcurrentRequests else {
            throw ServiceError.invalidRequest("Requested concurrency exceeds engine capacity")
        }
        guard configuration.maxQueuedRequests >= 0 else {
            throw ServiceError.invalidRequest("maxQueuedRequests 不能小于零")
        }
        guard (1...600_000).contains(configuration.maxRequestDurationMS) else {
            throw ServiceError.invalidRequest("Invalid maximum request duration")
        }
        self.engine = engine
        self.configuration = configuration
        self.requestSlots = DispatchSemaphore(value: configuration.maxConcurrentRequests)
        self.requestCapacity = DispatchSemaphore(
            value: configuration.maxConcurrentRequests + configuration.maxQueuedRequests
        )
    }

    public func route(_ request: HTTPRequest, cancellation: CancellationToken = CancellationToken()) -> HTTPResponse {
        do {
            if request.method == "OPTIONS" { return try response(status: 204, value: nil) }
            try authenticate(request)
            if request.method == "GET", request.path == "/health" { return try health() }
            if request.method == "GET", request.path == "/v1/models" { return try models() }
            if request.method == "DELETE", request.path.hasPrefix("/v1/requests/") {
                return try cancel(request.path)
            }
            if request.method == "POST", request.path == "/v1/embeddings" {
                return try embeddings(request, cancellation: cancellation)
            }
            throw ServiceError.notFound("Not found")
        } catch ServiceError.cancelled {
            return errorResponse(cancellation.hasExpired ? .deadlineExceeded : .cancelled)
        } catch let error as ServiceError {
            return errorResponse(error)
        } catch {
            fputs("[apple-embedding] request failed: \(error)\n", stderr)
            return errorResponse(.internalFailure(String(describing: error)))
        }
    }

    func configureDeadline(_ request: HTTPRequest, cancellation: CancellationToken) throws {
        var duration = configuration.maxRequestDurationMS
        if let header = request.headers["x-indexed-timeout-ms"] {
            guard !header.isEmpty, header.utf8.allSatisfy({ (48...57).contains($0) }),
                  let value = Int(header), (1...600_000).contains(value) else {
                throw ServiceError.invalidRequest("X-Indexed-Timeout-MS must be an integer from 1 to 600000")
            }
            duration = min(duration, value)
        }
        cancellation.limit(toMilliseconds: duration)
    }

    private func authenticate(_ request: HTTPRequest) throws {
        guard let expected = configuration.authToken, !expected.isEmpty else { return }
        let bearer = request.headers["authorization"].flatMap { value -> String? in
            let prefix = "Bearer "
            guard value.count >= prefix.count,
                  value.prefix(prefix.count).caseInsensitiveCompare(prefix) == .orderedSame else {
                return nil
            }
            return String(value.dropFirst(prefix.count))
        }
        let supplied = bearer ?? request.headers["x-indexed-token"]
        guard let supplied, constantTimeEqual(supplied, expected) else {
            throw ServiceError.unauthorized("缺少或无效的本地服务令牌")
        }
    }

    private func constantTimeEqual(_ left: String, _ right: String) -> Bool {
        let a = Array(left.utf8)
        let b = Array(right.utf8)
        var difference = a.count ^ b.count
        for index in 0 ..< max(a.count, b.count) {
            difference |= Int((index < a.count ? a[index] : 0)
                ^ (index < b.count ? b[index] : 0))
        }
        return difference == 0
    }

    private func health() throws -> HTTPResponse {
        let metadata = engine.metadata
        let load = loadLock.withLock { (queued: queuedRequests, running: runningRequests) }
        let metrics = metricsLock.withLock {
            (
                total: requestsTotal,
                succeeded: requestsSucceeded,
                failed: requestsFailed,
                cumulativeMS: cumulativeRequestMilliseconds,
                lastMS: lastRequestMilliseconds
            )
        }
        return try response(status: 200, value: JSONValue.object([
            "status": .string("ready"),
            "backend": .string(metadata.backend),
            "uptime_seconds": .number(Date().timeIntervalSince(startedAt)),
            "load_seconds": .number(metadata.loadSeconds),
            "warmup_seconds": .number(metadata.warmupSeconds),
            "allocated_bytes": .number(Double(metadata.allocatedBytes)),
            "mlx_allocated_bytes": .number(Double(metadata.allocatedBytes)),
            "mps_allocated_bytes": .number(
                Double(metadata.mlxDevice == "gpu" ? metadata.allocatedBytes : 0)
            ),
            "package_fingerprint": metadata.packageFingerprint.map(JSONValue.string) ?? .null,
            "compute_devices": .array(metadata.computeDevices.map(JSONValue.string)),
            "decoder_segment_loaded": .bool(metadata.decoderSegmentLoaded),
            "decoder_bundle_loaded": .bool(metadata.decoderBundleLoaded),
            "decoder_minimum_tokens": metadata.decoderMinimumTokens
                .map { .number(Double($0)) } ?? .null,
            "decoder_maximum_tokens": metadata.decoderMaximumTokens
                .map { .number(Double($0)) } ?? .null,
            "decoder_bucket_token_limits": .array(
                metadata.decoderBucketTokenLimits.map { .number(Double($0)) }
            ),
            "decoder_segment_fingerprint": metadata.decoderSegmentFingerprint
                .map(JSONValue.string) ?? .null,
            "decoder_bundle_fingerprint": metadata.decoderBundleFingerprint
                .map(JSONValue.string) ?? .null,
            "decoder_loading_strategy": metadata.decoderLoadingStrategy
                .map(JSONValue.string) ?? .null,
            "language_compute": .string(metadata.languageCompute),
            "mlx_decoder_loaded": .bool(metadata.mlxDecoderLoaded),
            "mlx_device": .string(metadata.mlxDevice),
            "max_model_len": .number(Double(metadata.maxModelLength)),
            "private_ane": metadata.privateANE ?? .null,
            "scheduling": metadata.scheduling ?? .null,
            "running_requests": .number(Double(load.running)),
            "queued_requests": .number(Double(load.queued)),
            "max_concurrent_requests": .number(Double(configuration.maxConcurrentRequests)),
            "max_queued_requests": .number(Double(configuration.maxQueuedRequests)),
            "requests_total": .number(Double(metrics.total)),
            "requests_succeeded": .number(Double(metrics.succeeded)),
            "requests_failed": .number(Double(metrics.failed)),
            "request_latency_ms_average": .number(
                metrics.total > 0 ? metrics.cumulativeMS / Double(metrics.total) : 0
            ),
            "request_latency_ms_last": .number(metrics.lastMS),
        ]))
    }

    private func models() throws -> HTTPResponse {
        let metadata = engine.metadata
        let ordered = [configuration.defaultDimension]
            + officialDimensions.filter { $0 != configuration.defaultDimension }
        let data = try ordered.map { dimension -> JSONValue in
            .object([
                "id": .string(try modelID(dimension: dimension)),
                "object": .string("model"),
                "owned_by": .string("indexed-apple"),
                "dimension": .number(Double(dimension)),
                "embedding_space": .string(embeddingSpace(dimension)),
                "max_model_len": .number(Double(metadata.maxModelLength)),
                "modalities": .array([.string("text"), .string("image"), .string("video")]),
                "backend": .string(metadata.backend),
                "language_compute": .string(metadata.languageCompute),
            ])
        }
        return try response(status: 200, value: .object([
            "object": .string("list"), "data": .array(data),
        ]))
    }

    private func embeddings(_ request: HTTPRequest, cancellation: CancellationToken) throws -> HTTPResponse {
        try configureDeadline(request, cancellation: cancellation)
        try cancellation.check()
        let requestStartedAt = Date()
        var succeeded = false
        defer {
            let elapsed = Date().timeIntervalSince(requestStartedAt) * 1_000
            metricsLock.withLock {
                requestsTotal += 1
                if succeeded { requestsSucceeded += 1 }
                else { requestsFailed += 1 }
                cumulativeRequestMilliseconds += elapsed
                lastRequestMilliseconds = elapsed
            }
        }
        guard !request.body.isEmpty else { throw ServiceError.invalidRequest("请求体为空") }
        guard request.body.count <= configuration.maxRequestBytes else {
            throw ServiceError.invalidRequest("请求体超过 Apple helper 限制", status: 413)
        }
        let input: EmbeddingRequest
        do { input = try JSONDecoder().decode(EmbeddingRequest.self, from: request.body) }
        catch { throw ServiceError.invalidRequest("请求体不是有效 JSON 对象") }

        let selectedDimension = try dimension(
            for: input.model,
            default: configuration.defaultDimension
        )
        let messages = try normalizedMessages(from: input)
        let requestID = input.requestID ?? UUID().uuidString.lowercased()
        guard (1...128).contains(requestID.utf8.count), requestID.utf8.allSatisfy({
            (65...90).contains($0) || (97...122).contains($0) || (48...57).contains($0)
                || [45, 46, 58, 95].contains($0)
        }) else { throw ServiceError.invalidRequest("request_id must contain 1–128 ASCII letters, digits, or -._:") }
        guard requestCapacity.wait(timeout: .now()) == .success else {
            throw ServiceError.overloaded("原生 embedding 服务队列已满，请稍后重试")
        }
        defer { requestCapacity.signal() }

        try requestsLock.withLock {
            guard activeRequests[requestID] == nil else {
                throw ServiceError.invalidRequest("request_id is already active", status: 409)
            }
            activeRequests[requestID] = cancellation
        }
        defer { requestsLock.withLock { activeRequests[requestID] = nil } }

        loadLock.withLock { queuedRequests += 1 }
        var acquiredRequestSlot = false
        defer {
            loadLock.withLock {
                if acquiredRequestSlot { runningRequests -= 1 }
                else { queuedRequests -= 1 }
            }
            if acquiredRequestSlot { requestSlots.signal() }
        }
        while requestSlots.wait(timeout: .now() + .milliseconds(50)) != .success {
            try cancellation.check()
        }
        loadLock.withLock {
            queuedRequests -= 1
            runningRequests += 1
        }
        acquiredRequestSlot = true

        try cancellation.check()
        let result = try engine.embed(
            messages: messages, dimension: selectedDimension, cancellation: cancellation
        )
        guard result.vector.count == selectedDimension else {
            throw ServiceError.internalFailure(
                "推理引擎返回 \(result.vector.count) 维，预期 \(selectedDimension) 维"
            )
        }
        let vector = result.vector.map { JSONValue.number(Double($0)) }
        let timings = result.timingsMS.mapValues(JSONValue.number)
        let output = try response(status: 200, value: .object([
            "object": .string("list"),
            "model": .string(try modelID(dimension: selectedDimension)),
            "data": .array([.object([
                "object": .string("embedding"),
                "index": .number(0),
                "embedding": .array(vector),
            ])]),
            "usage": .object([
                "prompt_tokens": .number(Double(result.promptTokens)),
                "total_tokens": .number(Double(result.promptTokens)),
            ]),
            "indexed": .object([
                "backend": .string(engine.metadata.backend),
                "modality": .string(result.modality),
                "dimension": .number(Double(selectedDimension)),
                "embedding_space": .string(embeddingSpace(selectedDimension)),
                "request_id": .string(requestID),
                "timings_ms": .object(timings),
            ]),
        ]))
        succeeded = true
        return output
    }

    private func cancel(_ path: String) throws -> HTTPResponse {
        let requestID = String(path.dropFirst("/v1/requests/".count))
        guard !requestID.isEmpty else { throw ServiceError.notFound("Not found") }
        let token = requestsLock.withLock { activeRequests[requestID] }
        guard let token else { throw ServiceError.notFound("未找到运行中的请求：\(requestID)") }
        token.cancel()
        return try response(status: 202, value: .object([
            "status": .string("cancelling"), "request_id": .string(requestID),
        ]))
    }

    private func embeddingSpace(_ dimension: Int) -> String {
        if let template = engine.metadata.embeddingSpaceTemplate {
            return template.replacingOccurrences(of: "{dimension}", with: String(dimension))
        }
        return "\(modelPrefix)-\(dimension)-wemm-indexed-v1"
    }

    private func response(status: Int, value: JSONValue?) throws -> HTTPResponse {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        let body = try value.map { try encoder.encode($0) } ?? Data()
        return HTTPResponse(status: status, body: body, headers: defaultHeaders)
    }

    private func errorResponse(_ error: ServiceError) -> HTTPResponse {
        let value = JSONValue.object(["error": .object([
            "message": .string(error.localizedDescription),
            "type": .string(error.responseType),
            "code": .string(error.responseCode),
        ])])
        return (try? response(status: error.status, value: value))
            ?? HTTPResponse(status: 500, body: Data())
    }

    private var defaultHeaders: [String: String] {
        [
            "Content-Type": "application/json; charset=utf-8",
            "Cache-Control": "no-store",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Indexed-Token, X-Indexed-Timeout-MS",
            "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
        ]
    }
}
