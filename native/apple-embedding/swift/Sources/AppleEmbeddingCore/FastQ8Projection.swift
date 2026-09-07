import MLX
import MLXNN
import Metal

/// The already validated classic variant 8 also improves these long WeMM Q8
/// projections on M4. Preserve packed weights/scales verbatim and retain stock
/// MLX for short requests and layouts outside the measured contract.
final class FastQ8Projection: QuantizedLinear {
    // Only the base M4 has been measured. In particular, preserve stock MLX's
    // NAX selection on M5 instead of substituting an unmeasured classic kernel.
    static let isSupportedDevice = MTLCreateSystemDefaultDevice()?.name == "Apple M4"
    private let context: PrivateMLPContext

    init(_ original: QuantizedLinear, context: PrivateMLPContext) {
        self.context = context
        super.init(weight: original.weight, bias: original.bias,
            scales: original.scales, biases: original.biases,
            groupSize: original.groupSize, bits: original.bits, mode: original.mode)
        self.freeze()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard x.ndim == 3, x.dim(0) == 1, x.dim(1) >= 2112,
              x.dim(-1) == shape.1, shape.0.isMultiple(of: 64), shape.1.isMultiple(of: 64),
              bits == 8, groupSize == 64, mode == .affine, bias == nil,
              let biases, x.dtype == scales.dtype,
              [DType.float16, .bfloat16].contains(x.dtype) else { return super.callAsFunction(x) }
        context.recordGPUProjection()
        return PrivateMLPKernels.affine(x, weight: weight, scales: scales, biases: biases)
    }

    static func installGDN(in model: Module, context: PrivateMLPContext) throws {
        guard isSupportedDevice else { return }
        let names = ["in_proj_qkv", "in_proj_z", "out_proj"]
        let modules = model.namedModules().filter { $0.0.hasSuffix(".linear_attn") }
        guard modules.count == 18 else { throw ServiceError.model("Long Q8 projection tuning requires the 18 WeMM GDN layers") }
        var replacements: [(Module, [(String, Module)])] = []
        for (_, module) in modules {
            let children = Dictionary(uniqueKeysWithValues: module.children().flattened())
            var projections: [(String, Module)] = []
            for name in names {
                guard let linear = children[name] as? QuantizedLinear, linear.bits == 8, linear.groupSize == 64,
                      linear.mode == .affine, linear.bias == nil, linear.biases != nil else {
                    throw ServiceError.model("Long Q8 projection tuning requires unchanged affine Q8/G64 weights")
                }
                projections.append((name, FastQ8Projection(linear, context: context)))
            }
            replacements.append((module, projections))
        }
        // Patch each validated GDN container: flattening the sparse 18-of-24
        // layer indices would lose the array shape at the model root.
        for (module, projections) in replacements {
            try module.update(modules: .unflattened(projections), verify: .all)
        }
    }
}
