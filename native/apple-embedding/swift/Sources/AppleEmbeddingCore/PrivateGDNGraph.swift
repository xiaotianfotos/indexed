import Foundation
import CryptoKit

/// Fixed D kernel. Geometry and masks match the pre-port Python reference;
/// unsupported block sizes/scales require a separately validated graph.
enum PrivateGDNGraph {
    static let size = 128, heads = 16, chunk = 64, block = 8
    static let queryScale: Float = 4096
    static let dynamicNames = ["log_decay_diagonal", "k", "kt", "beta", "state", "v", "q"]
    static let sharedNames = ["lower_inclusive", "ones", "upper_inclusive", "negative_ones", "negative_strict_lower", "last_row_selector", "solve_block_diagonal", "solve_block_strict_lower", "identity", "solve_rows_0", "solve_rows_1", "solve_rows_2", "solve_rows_3", "solve_rows_4", "solve_rows_5", "solve_rows_6", "solve_rows_7", "query_inverse_scale"]
    static let outputNames = ["final_state", "output"]

    static let mil: String = {
        var lines = [
            "program(1.3)",
            "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]",
            "{",
            "  func main<ios18>(tensor<fp16, [7,16,128,128]> dynamic, tensor<fp16, [18,1,128,128]> shared) {",
            "    tensor<int32, [4]> dynamic_slice_shape = const()[name=string(\"dynamic_slice_shape\"), val=tensor<int32, [4]>([1,16,128,128])];",
            "    tensor<int32, [4]> shared_slice_shape = const()[name=string(\"shared_slice_shape\"), val=tensor<int32, [4]>([1,1,128,128])];",
            "    tensor<int32, [4]> dynamic_matrix_shape = const()[name=string(\"dynamic_matrix_shape\"), val=tensor<int32, [4]>([16,1,128,128])];",
            "    tensor<int32, [4]> shared_matrix_shape = const()[name=string(\"shared_matrix_shape\"), val=tensor<int32, [4]>([1,1,128,128])];",
            "    tensor<int32, [4]> live_shape = const()[name=string(\"live_shape\"), val=tensor<int32, [4]>([1,16,128,128])];",
            "    bool f = const()[name=string(\"f\"), val=bool(false)];",
        ]
        var aliases: [String: String] = [:]
        for (kind, names) in [("dynamic", dynamicNames), ("shared", sharedNames)] {
            let batch = kind == "dynamic" ? 16 : 1
            for (index, name) in names.enumerated() {
                let alias = "\(kind)_input_\(index)"
                aliases[name] = alias
                lines += [
                    "    tensor<int32, [4]> \(kind)_begin_\(index) = const()[name=string(\"\(kind)_begin_\(index)\"), val=tensor<int32, [4]>([\(index),0,0,0])];",
                    "    tensor<fp16, [1,\(batch),128,128]> \(kind)_raw_\(index) = slice_by_size(x=\(kind), begin=\(kind)_begin_\(index), size=\(kind)_slice_shape)[name=string(\"\(kind)_slice_\(index)\")];",
                    "    tensor<fp16, [\(batch),1,128,128]> \(alias) = reshape(shape=\(kind)_matrix_shape, x=\(kind)_raw_\(index))[name=string(\"\(kind)_reshape_\(index)\")];",
                ]
            }
        }
        for (index, operation) in operations.enumerated() {
            let (output, op, left, right) = operation
            let lhs = aliases[left]!
            let rhs = right.map { aliases[$0]! }
            let expression: String
            if op == "matmul" { expression = "matmul(transpose_x=f, transpose_y=f, x=\(lhs), y=\(rhs!))" }
            else if op == "exp" { expression = "exp(x=\(lhs))" }
            else { expression = "\(op)(x=\(lhs), y=\(rhs!))" }
            lines.append("    tensor<fp16, [16,1,128,128]> value_\(index) = \(expression)[name=string(\"op_\(index)\")];")
            aliases[output] = "value_\(index)"
        }
        for (index, output) in outputNames.enumerated() {
            lines.append("    tensor<fp16, [1,16,128,128]> live_\(index) = reshape(shape=live_shape, x=\(aliases[output]!))[name=string(\"live_\(index)\")];")
        }
        lines += [
            "    int32 concat_axis = const()[name=string(\"concat_axis\"), val=int32(0)];",
            "    bool concat_interleave = const()[name=string(\"concat_interleave\"), val=bool(false)];",
            "    tensor<fp16, [2,16,128,128]> combined = concat(axis=concat_axis, interleave=concat_interleave, values=(live_0,live_1))[name=string(\"pack_outputs\")];",
            "  } -> (combined);", "}", "",
        ]
        return lines.joined(separator: "\n")
    }()

    static let shared: [Float16] = {
        var result = [Float16]()
        result.reserveCapacity(sharedNames.count * size * size)
        for name in sharedNames {
            for row in 0..<size {
                for column in 0..<size {
                    let value: Float
                    switch name {
                    case "lower_inclusive": value = column <= row ? 1 : 0
                    case "upper_inclusive": value = column >= row ? 1 : 0
                    case "ones": value = 1
                    case "negative_ones": value = -1
                    case "negative_strict_lower": value = column < row ? -1 : 0
                    case "last_row_selector": value = column == size - 1 ? 1 : 0
                    case "identity": value = row == column ? 1 : 0
                    case "solve_block_diagonal": value = row < chunk && column < chunk && row / block == column / block ? 1 : 0
                    case "solve_block_strict_lower": value = row < chunk && column < chunk && row / block > column / block ? 1 : 0
                    case "query_inverse_scale": value = 1 / queryScale
                    default:
                        let slot = Int(name.dropFirst("solve_rows_".count))!
                        value = row >= slot * block && row < (slot + 1) * block ? 1 : 0
                    }
                    result.append(Float16(value))
                }
            }
        }
        return result
    }()

    static func validateFrozenIdentity() throws {
        let milHash = SHA256.hash(data: Data(mil.utf8)).map { String(format: "%02x", $0) }.joined()
        let sharedHash = shared.withUnsafeBytes { SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined() }
        guard milHash == "55097b6a94edf8648d87b269bef997a6dc44d9bec98201b07824901e5c68d4fe",
              sharedHash == "9be1e1bb8bc74d399e949d675018eb95656e61ff0cbd122ba07200c156f3a0c6" else {
            throw ServiceError.model("Swift GDN graph/constants differ from the frozen Python kernel")
        }
    }
}
