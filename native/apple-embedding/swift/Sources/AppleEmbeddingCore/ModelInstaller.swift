import Foundation

public enum ModelInstaller {
    /// Fully verifies a model package and atomically installs it under a
    /// fingerprinted directory. Existing valid installs are reused; no model is
    /// overwritten in place.
    public static func install(source: URL, modelsDirectory: URL) throws
        -> ValidatedModelPackage
    {
        let source = try ModelPackageValidator.validate(at: source, depth: .full)
        try FileManager.default.createDirectory(
            at: modelsDirectory,
            withIntermediateDirectories: true
        )
        let safeModel = source.manifest.model.map {
            $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" ? $0 : "-"
        }
        let name = "\(String(safeModel))-\(source.manifest.packageFingerprint.prefix(16))"
        let destination = modelsDirectory.appendingPathComponent(name, isDirectory: true)
        if FileManager.default.fileExists(atPath: destination.path) {
            return try ModelPackageValidator.validate(at: destination, depth: .full)
        }

        let temporary = modelsDirectory.appendingPathComponent(
            ".install-\(UUID().uuidString.lowercased())",
            isDirectory: true
        )
        do {
            try FileManager.default.copyItem(at: source.root, to: temporary)
            _ = try ModelPackageValidator.validate(at: temporary, depth: .full)
            do {
                try FileManager.default.moveItem(at: temporary, to: destination)
            } catch where FileManager.default.fileExists(atPath: destination.path) {
                try? FileManager.default.removeItem(at: temporary)
            }
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
        return try ModelPackageValidator.validate(at: destination, depth: .full)
    }
}
