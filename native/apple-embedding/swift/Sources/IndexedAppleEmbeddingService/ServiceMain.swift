import AppleEmbeddingCore
import Darwin
import Dispatch
import Foundation

private func argument(_ name: String, in arguments: [String]) -> String? {
    guard let index = arguments.firstIndex(of: name), index + 1 < arguments.count else { return nil }
    return arguments[index + 1]
}

private func usage() -> Never {
    fputs(
        """
        Usage:
          indexed-apple-embedding validate-model --package PATH [--full]
          indexed-apple-embedding install-model --source PATH --models-dir PATH
          indexed-apple-embedding prepare-coreml --package PATH
            [--coreml-cache PATH]
          indexed-apple-embedding serve --package PATH [--port 18768] [--default-dimension 2048]
            [--coreml-cache PATH]
            [--execution-mode a|b|c|d]
            [--vision-compute ane|gpu]
            [--language-compute gpu]
            [--max-queued-requests 16] [--auth-token TOKEN]
            [--skip-warmup] [--development-deterministic-engine]

        """,
        stderr
    )
    exit(64)
}

private final class SignalAwaiter: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Never>?
    private var stopped = false

    func wait() async {
        await withCheckedContinuation { continuation in
            let resumeNow = lock.withLock { () -> Bool in
                if stopped { return true }
                self.continuation = continuation
                return false
            }
            if resumeNow { continuation.resume() }
        }
    }

    func signal() {
        let continuation = lock.withLock { () -> CheckedContinuation<Void, Never>? in
            guard !stopped else { return nil }
            stopped = true
            defer { self.continuation = nil }
            return self.continuation
        }
        continuation?.resume()
    }
}

@main
private enum ServiceMain {
    static func main() async {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard let command = arguments.first else { usage() }
        do {
            if arguments.contains(where: { $0.hasPrefix("--decoder-") })
                || (argument("--language-compute", in: arguments).map { $0 != "gpu" } ?? false)
                || argument("--execution-mode", in: arguments)?.lowercased() == "e" {
                throw ServiceError.invalidRequest("E 模式及完整 Core ML 语言塔参数已退出产品；请通过 Indexed 选择 B 或 A/C/D")
            }
            if let mode = argument("--execution-mode", in: arguments)?.lowercased(), !["a", "b", "c", "d"].contains(mode) {
                throw ServiceError.invalidRequest("--execution-mode 必须是 a、b、c 或 d")
            }
            switch command {
            case "validate-model":
        guard let path = argument("--package", in: arguments) else { usage() }
        let depth: ValidationDepth = arguments.contains("--full") ? .full : .quick
        let package = try ModelPackageValidator.validate(
            at: URL(fileURLWithPath: path, isDirectory: true),
            depth: depth
        )
        let value: JSONValue = .object([
            "status": .string("valid"),
            "depth": .string(depth.rawValue),
            "package_fingerprint": .string(package.manifest.packageFingerprint),
            "model": .string(package.manifest.model),
        ])
        FileHandle.standardOutput.write(try JSONEncoder().encode(value))
        FileHandle.standardOutput.write(Data("\n".utf8))

            case "serve":
        guard let path = argument("--package", in: arguments) else { usage() }
        let package = try ModelPackageValidator.validate(
            at: URL(fileURLWithPath: path, isDirectory: true),
            depth: .quick
        )
        let port = UInt16(argument("--port", in: arguments) ?? "18768") ?? 18_768
        let defaultDimension = Int(argument("--default-dimension", in: arguments) ?? "2048") ?? 2048
        let selectedMode = argument("--execution-mode", in: arguments)?.lowercased()
        let selectedVision = selectedMode == "a" ? "gpu" : "ane"
        if let explicit = argument("--vision-compute", in: arguments), selectedMode != nil, explicit != selectedVision {
            throw ServiceError.invalidRequest("--execution-mode 与 --vision-compute 冲突")
        }
        guard let visionComputeTarget = VisionComputeTarget(
            rawValue: argument("--vision-compute", in: arguments) ?? selectedVision
        ) else {
            throw ServiceError.invalidRequest("--vision-compute 必须是 ane 或 gpu")
        }
        let privateOptions = try PrivateANEOptions(arguments: arguments, mode: selectedMode ?? (visionComputeTarget == .gpu ? "a" : "b"))
        let configuration = ServiceConfiguration(
            port: port,
            defaultDimension: defaultDimension,
            maxConcurrentRequests: privateOptions.pipelineDepth,
            maxQueuedRequests: Int(argument("--max-queued-requests", in: arguments) ?? "16") ?? 16,
            authToken: argument("--auth-token", in: arguments)
                ?? ProcessInfo.processInfo.environment["INDEXED_APPLE_EMBEDDING_AUTH_TOKEN"]
        )
        if arguments.contains("--development-delay-ms"), !arguments.contains("--development-deterministic-engine") {
            throw ServiceError.invalidRequest("Test delay requires the deterministic development engine")
        }
        let engine: any EmbeddingEngine
        if arguments.contains("--development-deterministic-engine") {
            guard let delay = Int(argument("--development-delay-ms", in: arguments) ?? "0") else {
                throw ServiceError.invalidRequest("Invalid deterministic test delay")
            }
            engine = try DeterministicEmbeddingEngine(delayMS: delay)
        } else {
            fputs("[apple-embedding] loading WeMM MLX Swift language model...\n", stderr)
            engine = try await MLXWeMMEngine.load(
                packageURL: package.root,
                coreMLCacheDirectory: argument("--coreml-cache", in: arguments).map {
                    URL(fileURLWithPath: $0, isDirectory: true)
                },
                visionComputeTarget: visionComputeTarget,
                languageComputeTarget: .gpu,
                warmup: !arguments.contains("--skip-warmup"),
                developmentPrivateMLP: privateOptions.mlp,
                developmentGDNProfile: privateOptions.recurrenceProfile,
                developmentPipelineDepth: privateOptions.pipelineDepth
            )
        }
        let router = try ServiceRouter(engine: engine, configuration: configuration)
        let server = LoopbackHTTPServer(router: router, configuration: configuration)
        try server.start()

        let stop = SignalAwaiter()
        let signals = [SIGINT, SIGTERM].map { signal -> DispatchSourceSignal in
            Darwin.signal(signal, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: signal, queue: .main)
            source.setEventHandler { stop.signal() }
            source.resume()
            return source
        }
        let ready: JSONValue = .object([
            "status": .string("ready"),
            "url": .string("http://127.0.0.1:\(server.boundPort)"),
            "default_model": .string(try modelID(dimension: defaultDimension)),
            "backend": .string(engine.metadata.backend),
            "execution_mode": .string(selectedMode ?? (visionComputeTarget == .gpu ? "a" : "b")),
            "private_ane": engine.metadata.privateANE ?? .null,
            "scheduling": engine.metadata.scheduling ?? .null,
            "load_seconds": .number(engine.metadata.loadSeconds),
            "package_fingerprint": .string(package.manifest.packageFingerprint),
            "default_embedding_space": .string(
                (engine.metadata.embeddingSpaceTemplate
                    ?? package.manifest.embeddingSpaceTemplate).replacingOccurrences(
                        of: "{dimension}", with: String(defaultDimension)
                    )
            ),
        ])
        FileHandle.standardOutput.write(try JSONEncoder().encode(ready))
        FileHandle.standardOutput.write(Data("\n".utf8))
        await stop.wait()
        for source in signals { source.cancel() }
        server.stop()

            case "prepare-coreml":
        guard let path = argument("--package", in: arguments) else { usage() }
        let prepared = try await CoreMLAssetPreparer.prepare(
            packageURL: URL(fileURLWithPath: path, isDirectory: true),
            cacheDirectory: argument("--coreml-cache", in: arguments).map {
                URL(fileURLWithPath: $0, isDirectory: true)
            }
        )
        let value: JSONValue = .object([
            "status": .string("prepared"),
            "package_fingerprint": .string(prepared.packageFingerprint),
            "vision_compiled_path": .string(prepared.visionCompiledURL.path),
            "decoder_segment_fingerprint": prepared.decoderSegmentFingerprint
                .map(JSONValue.string) ?? .null,
            "decoder_bundle_fingerprint": prepared.decoderBundleFingerprint
                .map(JSONValue.string) ?? .null,
            "decoder_bundle_fingerprints": .array(
                prepared.decoderBundleFingerprints.map(JSONValue.string)
            ),
        ])
        FileHandle.standardOutput.write(try JSONEncoder().encode(value))
        FileHandle.standardOutput.write(Data("\n".utf8))

            case "install-model":
        guard let source = argument("--source", in: arguments),
              let modelsDirectory = argument("--models-dir", in: arguments) else { usage() }
        let installed = try ModelInstaller.install(
            source: URL(fileURLWithPath: source, isDirectory: true),
            modelsDirectory: URL(fileURLWithPath: modelsDirectory, isDirectory: true)
        )
        let value: JSONValue = .object([
            "status": .string("installed"),
            "path": .string(installed.root.path),
            "model": .string(installed.manifest.model),
            "package_fingerprint": .string(installed.manifest.packageFingerprint),
        ])
        FileHandle.standardOutput.write(try JSONEncoder().encode(value))
        FileHandle.standardOutput.write(Data("\n".utf8))

            default: usage()
            }
        } catch {
            fputs("indexed-apple-embedding: \(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }
}
