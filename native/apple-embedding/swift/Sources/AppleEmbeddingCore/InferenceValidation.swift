import Dispatch
import Foundation
import MLX

/// Maintainer checks for scheduling and the CPU/MLX preprocessing boundary.
/// No model weights or private ANE hardware are required.
public enum InferenceValidation {
    private final class Results: @unchecked Sendable {
        let lock = NSLock()
        var errors: [String] = []
        var order: [String] = []
        func record(_ value: String) { lock.withLock { order.append(value) } }
        func run(_ body: () throws -> Void) {
            do { try body() } catch { lock.withLock { errors.append(String(describing: error)) } }
        }
    }
    private static func require(_ value: Bool, _ message: String) throws {
        if !value { throw ServiceError.internalFailure(message) }
    }
    private static func wait(_ semaphore: DispatchSemaphore) throws {
        try require(semaphore.wait(timeout: .now() + 5) == .success, "Scheduling check timed out")
    }
    private static func until(_ predicate: () -> Bool) throws {
        let deadline = ProcessInfo.processInfo.systemUptime + 5
        while !predicate() {
            try require(ProcessInfo.processInfo.systemUptime < deadline, "Waiter did not reach gate")
            Thread.sleep(forTimeInterval: 0.001)
        }
    }

    public static func checkScheduling() throws {
        let gate = InferenceGate(), group = DispatchGroup(), result = Results()
        let entered = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        DispatchQueue.global().async(group: group) {
            result.run {
                try gate.withPermit(cancellation: CancellationToken()) {
                    entered.signal(); try wait(release)
                }
            }
        }
        defer { release.signal() }
        try wait(entered)
        func enqueue(_ label: String, priority: Bool = false) {
            DispatchQueue.global().async(group: group) {
                result.run { try gate.withPermit(priority: priority, cancellation: CancellationToken()) { result.record(label) } }
            }
        }
        enqueue("normal-1"); try until { gate.snapshot.waiting == 1 }
        enqueue("normal-2"); try until { gate.snapshot.waiting == 2 }
        enqueue("priority", priority: true); try until { gate.snapshot.priorityWaiting == 1 }
        let token = CancellationToken(), cancelled = DispatchSemaphore(value: 0)
        DispatchQueue.global().async(group: group) {
            defer { cancelled.signal() }
            result.run {
                do {
                    try gate.withPermit(cancellation: token) { result.record("unexpected-cancelled") }
                    throw ServiceError.internalFailure("Cancelled waiter entered gate")
                } catch ServiceError.cancelled { }
            }
        }
        try until { gate.snapshot.waiting == 4 }
        token.cancel(); try wait(cancelled)
        try require(gate.snapshot.waiting == 3, "Cancelled waiter leaked a queue entry")
        release.signal()
        try require(group.wait(timeout: .now() + 5) == .success, "Gate workers did not finish")
        try require(result.errors.isEmpty && result.order == ["priority", "normal-1", "normal-2"], "FIFO/priority behavior changed: \(result.errors) \(result.order)")
        do {
            try gate.withPermit(cancellation: CancellationToken()) { throw ServiceError.cancelled }
        } catch ServiceError.cancelled { }
        try gate.withPermit(cancellation: CancellationToken()) { }
        try require(gate.snapshot.active == 0 && gate.snapshot.peak == 1, "Failure leaked a permit")

        let pipeline = InferencePipeline(depth: 2), stages = DispatchGroup(), staged = Results()
        let languageEntered = DispatchSemaphore(value: 0), visionEntered = DispatchSemaphore(value: 0)
        let releaseLanguage = DispatchSemaphore(value: 0), releaseVision = DispatchSemaphore(value: 0)
        defer { releaseLanguage.signal(); releaseVision.signal() }
        DispatchQueue.global().async(group: stages) {
            staged.run {
                try pipeline.admission.withPermit(cancellation: CancellationToken()) {
                    try pipeline.withStage(.vision, cancellation: CancellationToken()) { }
                    try pipeline.withStage(.language, cancellation: CancellationToken()) {
                        languageEntered.signal(); try wait(releaseLanguage)
                    }
                }
            }
        }
        try wait(languageEntered)
        DispatchQueue.global().async(group: stages) {
            staged.run {
                try pipeline.admission.withPermit(cancellation: CancellationToken()) {
                    try pipeline.withStage(.vision, cancellation: CancellationToken()) {
                        visionEntered.signal(); try wait(releaseVision)
                    }
                    try pipeline.withStage(.language, cancellation: CancellationToken()) { }
                }
            }
        }
        try wait(visionEntered)
        let third = CancellationToken(), thirdDone = DispatchSemaphore(value: 0)
        DispatchQueue.global().async(group: stages) {
            defer { thirdDone.signal() }
            staged.run {
                do {
                    try pipeline.admission.withPermit(cancellation: third) { staged.record("unexpected-third") }
                    throw ServiceError.internalFailure("Third request exceeded pipeline capacity")
                } catch ServiceError.cancelled { }
            }
        }
        try until { pipeline.admission.snapshot.waiting == 1 }
        third.cancel(); try wait(thirdDone)
        releaseVision.signal()
        try until { pipeline.language.snapshot.waiting == 1 }
        try require(pipeline.language.snapshot.active == 1, "Two language stages ran together")
        releaseLanguage.signal()
        try require(stages.wait(timeout: .now() + 5) == .success, "Pipeline workers did not finish")
        let metrics = pipeline.diagnostics.objectValue!
        try require(staged.errors.isEmpty && staged.order.isEmpty, "Pipeline errors: \(staged.errors)")
        try require(metrics["peak_in_flight"] == .number(2)
            && metrics["peak_language_active"] == .number(1)
            && metrics["peak_vision_active"] == .number(1)
            && metrics["vision_language_overlap_events"] == .number(1)
            && pipeline.admission.snapshot.active == 0, "Pipeline evidence is missing or unbounded")
    }

    private final class ControlledEngine: EmbeddingEngine, @unchecked Sendable {
        let entered = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        private let lock = NSLock()
        private var first = true
        var metadata: EngineMetadata { EngineMetadata(backend: "isolated-scheduling-fixture") }
        func embed(messages: [JSONValue], dimension: Int, cancellation: CancellationToken) throws -> EmbeddingResult {
            let block = lock.withLock { let value = first; first = false; return value }
            if block { entered.signal(); try InferenceValidation.wait(release) }
            try cancellation.check()
            return EmbeddingResult(vector: [1] + [Float](repeating: 0, count: dimension - 1),
                timingsMS: [:], modality: "text", promptTokens: 1)
        }
    }

    public static func checkRouter() throws {
        let engine = ControlledEngine(), token = "isolated-fixture-token"
        let router = try ServiceRouter(engine: engine, configuration: ServiceConfiguration(
            defaultDimension: 64, maxQueuedRequests: 2, authToken: token))
        do {
            _ = try ServiceRouter(engine: engine, configuration: ServiceConfiguration(maxConcurrentRequests: 2))
            throw ServiceError.internalFailure("Router exceeded engine concurrency")
        } catch ServiceError.invalidRequest { }
        func request(_ method: String, _ path: String, body: JSONValue? = nil, auth: String? = nil) throws -> HTTPRequest {
            HTTPRequest(method: method, path: path, headers: ["authorization": "Bearer \(auth ?? token)"],
                body: try body.map { try JSONEncoder().encode($0) } ?? Data())
        }
        func embedding(_ id: String) throws -> HTTPRequest {
            try request("POST", "/v1/embeddings", body: .object(["input": .string("fixture"), "request_id": .string(id)]))
        }
        try require(router.route(try request("GET", "/health", auth: token + String(repeating: "\0", count: 256))).status == 401,
            "Token length difference was truncated")
        try require(router.route(try embedding("invalid/id")).status == 400, "Invalid cancellation ID was accepted")
        let group = DispatchGroup(), result = Results()
        let first = try embedding("first"), queued = try embedding("queued")
        DispatchQueue.global().async(group: group) {
            result.run { try require(router.route(first).status == 499, "Original request lost its cancellation token") }
        }
        defer { engine.release.signal() }
        try wait(engine.entered)
        DispatchQueue.global().async(group: group) {
            result.run { try require(router.route(queued).status == 499, "Queued cancellation did not finish") }
        }
        try until {
            let health = router.route(HTTPRequest(method: "GET", path: "/health", headers: ["authorization": "Bearer \(token)"], body: Data()))
            let value = try? JSONDecoder().decode(JSONValue.self, from: health.body)
            return value?.objectValue?["queued_requests"] == .number(1)
        }
        try require(router.route(first).status == 409, "Duplicate active request ID was accepted")
        try require(router.route(try request("DELETE", "/v1/requests/queued")).status == 202, "Queued request was not cancellable")
        try require(router.route(try request("DELETE", "/v1/requests/first")).status == 202, "Original request token was overwritten")
        engine.release.signal()
        try require(group.wait(timeout: .now() + 5) == .success && result.errors.isEmpty, "Router cancellation failed: \(result.errors)")
        try require(router.route(try embedding("first")).status == 200, "Completed cancellation leaked capacity or request ID")
    }

    public static func checkVisionPatches() throws -> Int {
        let table = (MLXArray((0...255).map(UInt8.init)).asType(.float32)
            / MLXArray(127.5) - MLXArray(1.0)).asArray(Float.self)
        var checked = 0
        for side in [32, 448] {
            let firstRGBA = (0..<(side * side * 4)).map { UInt8(truncatingIfNeeded: $0 * 17 + $0 / 7) }
            let secondRGBA = firstRGBA.map { $0 ^ 157 }
            func legacy(_ rgba: [UInt8]) -> MLXArray {
                let rgb = MLXArray(Data(rgba), [side, side, 4], type: UInt8.self)[0..., 0..., ..<3]
                return (rgb.asType(.float32) / MLXArray(127.5) - MLXArray(1.0)).transposed(2, 0, 1)
            }
            let first = legacy(firstRGBA), second = legacy(secondRGBA)
            let cpuFirst = VisionPatches.normalizedCHW(firstRGBA, side: side, table: table)
            let cpuSecond = VisionPatches.normalizedCHW(secondRGBA, side: side, table: table)
            try require(cpuFirst.map(\.bitPattern) == first.asArray(Float.self).map(\.bitPattern)
                && cpuSecond.map(\.bitPattern) == second.asArray(Float.self).map(\.bitPattern), "CPU normalization differs from legacy MLX")
            let grid = side / 16
            let expected = stacked([first, second])
                .reshaped(1, 2, 3, grid / 2, 2, 16, grid / 2, 2, 16)
                .transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)
                .reshaped(grid * grid, 1536).asArray(Float.self)
            let actual = VisionPatches.pair(cpuFirst, cpuSecond, side: side)
            try require(actual.values.map(\.bitPattern) == expected.map(\.bitPattern), "CPU temporal patches differ from legacy MLX")
            checked += cpuFirst.count + cpuSecond.count + actual.values.count
        }
        return checked
    }
}
