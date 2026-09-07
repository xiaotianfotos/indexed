import Foundation

public struct PreparedCoreMLAssets: Sendable {
    public let packageFingerprint: String
    public let visionCompiledURL: URL
    public let decoderSegmentFingerprint: String?
    public let decoderBundleFingerprints: [String]

    public var decoderBundleFingerprint: String? {
        if decoderBundleFingerprints.isEmpty { return nil }
        if decoderBundleFingerprints.count == 1 {
            return decoderBundleFingerprints[0].split(separator: ":", maxSplits: 1)
                .last.map(String.init)
        }
        return decoderBundleFingerprints.joined(separator: ",")
    }
}

public enum CoreMLAssetPreparer {
    /// Fully verifies the downloaded package, compiles portable Core ML assets
    /// into the persistent cache, and exercises decoder model loading before
    /// the interactive service is started.
    public static func prepare(
        packageURL: URL,
        decoderSegmentURL: URL? = nil,
        decoderBundleURLs: [URL] = [],
        cacheDirectory: URL? = nil
    ) async throws -> PreparedCoreMLAssets {
        guard decoderSegmentURL == nil || decoderBundleURLs.isEmpty else {
            throw ServiceError.invalidRequest(
                "decoder segment 与完整 decoder bundle 不能同时准备"
            )
        }
        let package = try ModelPackageValidator.validate(at: packageURL, depth: .full)
        let visionCompiledURL = try await CoreMLCompilationCache.compiledModel(
            from: package.visionURL,
            prefix: "vision",
            expectedDigest: package.manifest.vision.packagedSHA256Tree,
            cacheDirectory: cacheDirectory
        )
        let decoder: CoreMLDecoderSegment?
        if let decoderSegmentURL {
            decoder = try await CoreMLDecoderSegment.load(
                from: decoderSegmentURL,
                modelPackageFingerprint: package.manifest.packageFingerprint,
                cacheDirectory: cacheDirectory
            )
        } else {
            decoder = nil
        }
        var bundles: [CoreMLDecoderBundle] = []
        for decoderBundleURL in decoderBundleURLs {
            bundles.append(try await CoreMLDecoderBundle.load(
                from: decoderBundleURL,
                modelPackageFingerprint: package.manifest.packageFingerprint,
                cacheDirectory: cacheDirectory
            ))
        }
        bundles.sort { $0.sequenceLength < $1.sequenceLength }
        guard Set(bundles.map(\.sequenceLength)).count == bundles.count else {
            throw ServiceError.invalidRequest("decoder bundle 的 sequence_length 不能重复")
        }
        return PreparedCoreMLAssets(
            packageFingerprint: package.manifest.packageFingerprint,
            visionCompiledURL: visionCompiledURL,
            decoderSegmentFingerprint: decoder?.fingerprint,
            decoderBundleFingerprints: bundles.map {
                "\($0.sequenceLength):\($0.fingerprint)"
            }
        )
    }
}
