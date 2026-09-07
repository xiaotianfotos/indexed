import CoreML
import CryptoKit
import Darwin
import Foundation

public struct WeMMManifest: Codable, Sendable {
    public struct Language: Codable, Sendable {
        public struct Quantization: Codable, Sendable {
            public let bits: Int
            public let groupSize: Int
            public let mode: String

            enum CodingKeys: String, CodingKey {
                case bits, mode
                case groupSize = "group_size"
            }
        }

        public let backend: String
        public let path: String
        public let precision: String
        public let sha256: String
        public let sizeBytes: Int64
        public let tensorCount: Int
        public let quantization: Quantization?

        enum CodingKeys: String, CodingKey {
            case backend, path, precision, sha256, quantization
            case sizeBytes = "size_bytes"
            case tensorCount = "tensor_count"
        }
    }

    public struct Vision: Codable, Sendable {
        public let backend: String
        public let compiled: Bool
        public let imageSize: Int
        public let packagedSHA256Tree: String
        public let path: String
        public let sizeBytes: Int64

        enum CodingKeys: String, CodingKey {
            case backend, compiled, path
            case imageSize = "image_size"
            case packagedSHA256Tree = "packaged_sha256_tree"
            case sizeBytes = "size_bytes"
        }
    }

    public let schemaVersion: Int
    public let runtimeSemantics: String
    public let model: String
    public let packageFingerprint: String
    public let embeddingSpaceTemplate: String
    public let embeddingTokenID: Int
    public let imageTokenID: Int
    public let videoTokenID: Int
    public let matryoshkaDimensions: [Int]
    public let tokenizerFiles: [String]
    public let language: Language
    public let vision: Vision

    enum CodingKeys: String, CodingKey {
        case model, language, vision
        case schemaVersion = "schema_version"
        case runtimeSemantics = "runtime_semantics"
        case packageFingerprint = "package_fingerprint"
        case embeddingSpaceTemplate = "embedding_space_template"
        case embeddingTokenID = "embedding_token_id"
        case imageTokenID = "image_token_id"
        case videoTokenID = "video_token_id"
        case matryoshkaDimensions = "matryoshka_dimensions"
        case tokenizerFiles = "tokenizer_files"
    }
}

public struct ValidatedModelPackage: Sendable {
    public let root: URL
    public let manifest: WeMMManifest
    public let languageURL: URL
    public let visionURL: URL
}

public enum ValidationDepth: String, Sendable { case quick, full }

public enum ModelPackageValidator {
    public static func validate(at root: URL, depth: ValidationDepth) throws -> ValidatedModelPackage {
        let root = try canonicalURL(root)
        let manifestURL = root.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ServiceError.model("WeMM Apple manifest 不存在：\(manifestURL.path)")
        }
        let manifest = try JSONDecoder().decode(
            WeMMManifest.self,
            from: Data(contentsOf: manifestURL, options: .mappedIfSafe)
        )
        guard manifest.schemaVersion == 1,
              manifest.runtimeSemantics == "wemm-apple-embedding-v1" else {
            throw ServiceError.model("WeMM Apple package schema/runtime_semantics 不受支持")
        }
        guard manifest.matryoshkaDimensions == officialDimensions else {
            throw ServiceError.model("模型的 Matryoshka 维度与服务契约不一致")
        }

        let languageURL = try containedURL(root: root, relativePath: manifest.language.path)
        let visionURL = try containedURL(root: root, relativePath: manifest.vision.path)
        let expectedVisionExtension = manifest.vision.compiled ? "mlmodelc" : "mlpackage"
        guard visionURL.pathExtension == expectedVisionExtension else {
            throw ServiceError.model(
                "vision.compiled 与 Core ML 模型扩展名不一致：\(visionURL.lastPathComponent)"
            )
        }
        try requireRegularFile(languageURL, expectedSize: manifest.language.sizeBytes)
        guard FileManager.default.fileExists(atPath: visionURL.path) else {
            throw ServiceError.model("Core ML 视觉模型不存在：\(visionURL.path)")
        }
        for relativePath in manifest.tokenizerFiles {
            try requireRegularFile(try containedURL(root: root, relativePath: relativePath))
        }

        if depth == .full {
            let digest = try sha256(of: languageURL)
            guard digest == manifest.language.sha256.lowercased() else {
                throw ServiceError.model("language/model.safetensors SHA-256 校验失败")
            }
            let treeDigest = try sha256Tree(of: visionURL)
            guard treeDigest == manifest.vision.packagedSHA256Tree.lowercased() else {
                throw ServiceError.model("Core ML 视觉模型目录 SHA-256 校验失败")
            }
        }

        return .init(
            root: root,
            manifest: manifest,
            languageURL: languageURL,
            visionURL: visionURL
        )
    }

    private static func containedURL(root: URL, relativePath: String) throws -> URL {
        guard !relativePath.isEmpty, !relativePath.hasPrefix("/") else {
            throw ServiceError.model("manifest 包含非法绝对路径")
        }
        let candidate = try canonicalURL(
            root.appendingPathComponent(relativePath).standardizedFileURL
        )
        let prefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
        guard candidate.path.hasPrefix(prefix) else {
            throw ServiceError.model("manifest 路径越出模型目录：\(relativePath)")
        }
        return candidate
    }

    private static func requireRegularFile(_ url: URL, expectedSize: Int64? = nil) throws {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values.isRegularFile == true else {
            throw ServiceError.model("模型文件不存在：\(url.path)")
        }
        if let expectedSize, Int64(values.fileSize ?? -1) != expectedSize {
            throw ServiceError.model("模型文件大小不符：\(url.lastPathComponent)")
        }
    }

    public static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let data = try handle.read(upToCount: 4 * 1024 * 1024), !data.isEmpty {
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Matches convert_model.py: SHA256(path byte count + relative path + file bytes).
    public static func sha256Tree(of root: URL) throws -> String {
        let root = try canonicalURL(root)
        let keys: [URLResourceKey] = [.isRegularFileKey]
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles]
        ) else { throw ServiceError.model("无法枚举模型目录：\(root.path)") }
        let files = enumerator.compactMap { $0 as? URL }.filter {
            (try? $0.resourceValues(forKeys: Set(keys)).isRegularFile) == true
        }.sorted { $0.path < $1.path }
        var hasher = SHA256()
        for file in files {
            let relative = String(file.path.dropFirst(root.path.count + 1))
            let relativeData = Data(relative.utf8)
            var length = UInt64(relativeData.count).bigEndian
            withUnsafeBytes(of: &length) { hasher.update(bufferPointer: $0) }
            hasher.update(data: relativeData)
            let handle = try FileHandle(forReadingFrom: file)
            defer { try? handle.close() }
            while let data = try handle.read(upToCount: 8 * 1024 * 1024), !data.isEmpty {
                hasher.update(data: data)
            }
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func canonicalURL(_ url: URL) throws -> URL {
        var buffer = [CChar](repeating: 0, count: Int(PATH_MAX))
        guard realpath(url.path, &buffer) != nil else {
            throw ServiceError.model(
                "模型路径不存在或无法解析：\(url.path)（\(String(cString: strerror(errno)))）"
            )
        }
        let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return URL(
            fileURLWithPath: String(decoding: bytes, as: UTF8.self),
            isDirectory: true
        )
    }
}
