import AppleEmbeddingCore
import Darwin
import Dispatch
import Foundation

/// Isolated maintainer endpoint. This executable is excluded from the product
/// distribution; Node's product C/D routing stays on the frozen Python backend.
func runReferenceServer() async throws {
    let arguments = CommandLine.arguments
    func argument(_ name: String) -> String? {
        guard let index = arguments.firstIndex(of: name), index + 1 < arguments.count else { return nil }
        return arguments[index + 1]
    }
    guard let path = argument("--package"),
          let token = ProcessInfo.processInfo.environment["INDEXED_APPLE_EMBEDDING_AUTH_TOKEN"], !token.isEmpty else {
        throw ServiceError.invalidRequest("Private C reference requires --package and an isolated helper token")
    }
    let configuration = ServiceConfiguration(port: 0, defaultDimension: 2048, maxConcurrentRequests: 2, maxQueuedRequests: 2, authToken: token)
    let mode = argument("--reference-mode") ?? "c"
    guard ["c", "d"].contains(mode) else { throw ServiceError.invalidRequest("Unknown private reference mode") }
    let profile: Data?
    if mode == "d" {
        guard let profilePath = argument("--recurrence-profile") else { throw ServiceError.invalidRequest("Private D requires --recurrence-profile") }
        profile = try Data(contentsOf: URL(fileURLWithPath: profilePath))
    } else { profile = nil }
    let engine = try await MLXWeMMEngine.load(
        packageURL: URL(fileURLWithPath: path),
        coreMLCacheDirectory: argument("--coreml-cache").map { URL(fileURLWithPath: $0, isDirectory: true) },
        languageComputeTarget: .gpu, developmentPrivateMLP: PrivateMLPConfiguration(), developmentGDNProfile: profile, developmentPipelineDepth: 2)
    let router = try ServiceRouter(engine: engine, configuration: configuration)
    let server = LoopbackHTTPServer(router: router, configuration: configuration)
    try server.start()
    let metadata = engine.metadata
    let ready: JSONValue = .object([
        "status": .string("ready"), "url": .string("http://127.0.0.1:\(server.boundPort)"),
        "backend": .string(metadata.backend), "candidate": .string("swift-\(mode)-pipeline-candidate"),
        "package_fingerprint": metadata.packageFingerprint.map(JSONValue.string) ?? .null,
        "private_ane": metadata.privateANE ?? .null,
        "scheduling": metadata.scheduling ?? .null,
        "known_deviations": .array([]),
    ])
    let (stream, continuation) = AsyncStream<Void>.makeStream()
    let signals = [SIGINT, SIGTERM].map { signal -> DispatchSourceSignal in
        Darwin.signal(signal, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: signal, queue: .main)
        source.setEventHandler { continuation.finish() }
        source.resume()
        return source
    }
    FileHandle.standardOutput.write(try JSONEncoder().encode(ready))
    FileHandle.standardOutput.write(Data("\n".utf8))
    for await _ in stream { }
    for source in signals { source.cancel() }
    server.stop()
}
