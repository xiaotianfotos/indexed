import CoreML
import Foundation

enum CoreMLCompilationCache {
    static func compiledModel(
        from sourceURL: URL,
        prefix: String,
        expectedDigest: String? = nil,
        cacheDirectory: URL? = nil
    ) async throws -> URL {
        if sourceURL.pathExtension == "mlmodelc" { return sourceURL }
        guard sourceURL.pathExtension == "mlpackage" else {
            throw ServiceError.model("Core ML 模型必须是 .mlpackage 或 .mlmodelc")
        }
        let digest = try expectedDigest ?? ModelPackageValidator.sha256Tree(of: sourceURL)
        let cacheDirectory = try cacheDirectory ?? defaultDirectory()
        let destination = cacheDirectory.appendingPathComponent(
            "\(prefix)-\(digest.prefix(24)).mlmodelc",
            isDirectory: true
        )
        if FileManager.default.fileExists(atPath: destination.path) { return destination }
        try FileManager.default.createDirectory(
            at: cacheDirectory,
            withIntermediateDirectories: true
        )
        let compiled = try await MLModel.compileModel(at: sourceURL)
        // Core ML places this directory in the boot volume's temporary area.
        // Keeping it after copying into INDEXED_HOME silently duplicates every
        // decoder graph and can exhaust a small system SSD.
        defer { try? FileManager.default.removeItem(at: compiled) }
        let temporary = cacheDirectory.appendingPathComponent(
            ".install-\(UUID().uuidString.lowercased()).mlmodelc",
            isDirectory: true
        )
        do {
            try FileManager.default.copyItem(at: compiled, to: temporary)
            do {
                try FileManager.default.moveItem(at: temporary, to: destination)
            } catch where FileManager.default.fileExists(atPath: destination.path) {
                try? FileManager.default.removeItem(at: temporary)
            }
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
        guard FileManager.default.fileExists(atPath: destination.path) else {
            throw ServiceError.model("Core ML 编译缓存写入失败")
        }
        return destination
    }

    private static func defaultDirectory() throws -> URL {
        let root = try FileManager.default.url(
            for: .cachesDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        return root
            .appendingPathComponent("com.xiaotian.indexed", isDirectory: true)
            .appendingPathComponent("AppleEmbedding", isDirectory: true)
            .appendingPathComponent("CoreML", isDirectory: true)
    }
}
