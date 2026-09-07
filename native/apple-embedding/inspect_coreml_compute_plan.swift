import CoreML
import Foundation

@available(macOS 14.0, *)
private func deviceName(_ device: MLComputeDevice) -> String {
    switch device {
    case .cpu: "cpu"
    case .gpu: "gpu"
    case .neuralEngine: "neural_engine"
    @unknown default: "unknown"
    }
}

@available(macOS 15.0, *)
private func operations(
    in block: MLModelStructure.Program.Block
) -> [MLModelStructure.Program.Operation] {
    block.operations.flatMap { operation in
        [operation] + operation.blocks.flatMap(operations(in:))
    }
}

@main
private enum InspectCoreMLComputePlan {
    static func main() async throws {
        guard #available(macOS 15.0, *) else {
            throw NSError(domain: "ComputePlan", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "MLComputePlan requires macOS 15 or later"
            ])
        }
        guard CommandLine.arguments.count > 1 else {
            fputs("usage: inspect-coreml-compute-plan MODEL.mlpackage [...]\n", stderr)
            exit(64)
        }
        var reports: [[String: Any]] = []
        for path in CommandLine.arguments.dropFirst() {
            let url = URL(fileURLWithPath: path)
            let planURL = url.pathExtension == "mlmodelc"
                ? url : try await MLModel.compileModel(at: url)
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine
            var hints = MLOptimizationHints()
            hints.specializationStrategy = .fastPrediction
            configuration.optimizationHints = hints
            let plan = try await MLComputePlan.load(
                contentsOf: planURL,
                configuration: configuration
            )
            guard case .program(let program) = plan.modelStructure else {
                throw NSError(domain: "ComputePlan", code: 2, userInfo: [
                    NSLocalizedDescriptionKey: "model is not an ML Program: \(path)"
                ])
            }
            var counts: [String: Int] = [:]
            var estimatedCosts: [String: Double] = [:]
            var operatorCounts: [String: [String: Int]] = [:]
            for function in program.functions.values {
                for operation in operations(in: function.block) {
                    guard let usage = plan.deviceUsage(for: operation) else { continue }
                    let device = deviceName(usage.preferred)
                    counts[device, default: 0] += 1
                    estimatedCosts[device, default: 0] += plan.estimatedCost(of: operation)?.weight ?? 0
                    operatorCounts[device, default: [:]][operation.operatorName, default: 0] += 1
                }
            }
            reports.append([
                "path": url.standardizedFileURL.path,
                "preferred_device_operation_counts": counts,
                "estimated_cost_weights": estimatedCosts,
                "operator_counts_by_preferred_device": operatorCounts,
            ])
        }
        let data = try JSONSerialization.data(
            withJSONObject: reports,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data("\n".utf8))
    }
}
