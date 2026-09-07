import Foundation
import MLX
import PrivateANEBridge

/// oMLX's classic variant 8: Q8 affine, group 64, BM=64/BK=32/BN=64.
/// Uses the pinned MLX implementation of qmm_t_impl at the same fixed tile.
/// NAX is a separate, currently unsupported candidate and is never claimed here.
enum PrivateMLPKernels {
    /// ANE consumes channel-major FP16. Materialize that layout on the GPU;
    /// exporting a transposed view through asArray otherwise copies one scalar
    /// at a time in MLX Swift's generic CPU strided-copy implementation.
    static func packedANEInput(_ x: MLXArray) -> MLXArray {
        contiguous(x.reshaped(x.size / x.dim(-1), x.dim(-1)).transposed().asType(.float16))
    }

    static let qmm = MLXFast.metalKernel(
        name: "indexed_private_q8_variant8_v1",
        inputNames: ["w", "scales", "biases", "x", "K", "N", "M"],
        outputNames: ["y"],
        source: """
        constexpr int padded = 32 + 16 / sizeof(T);
        threadgroup T Xs[64 * padded];
        threadgroup T Ws[64 * padded];
        qmm_t_impl<T, 64, 8, true, 64, 32, 64>(
            w, scales, biases, x, y, Xs, Ws, K, N, M,
            threadgroup_position_in_grid, thread_index_in_threadgroup,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """,
        header: String(cString: indexed_mlx_affine_kernel_header())
    )

    static func affine(_ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray) -> MLXArray {
        let rows = x.size / x.dim(-1)
        let columns = weight.dim(0)
        return qmm([weight, scales, biases, x, Int32(x.dim(-1)), Int32(columns), Int32(rows)],
            template: [("T", x.dtype)],
            grid: (((columns + 63) / 64) * 32, ((rows + 63) / 64) * 2, 2),
            threadGroup: (32, 2, 2), outputShapes: [[1, rows, columns]], outputDTypes: [x.dtype])[0]
    }

    // Match the original merge's FP32 gate*up/(1+exp(-gate)), with one cast at
    // the end. Separate FP16 silu/multiply operations change this rounding.
    static let merge = MLXFast.metalKernel(
        name: "indexed_private_ane_merge_swiglu_v1",
        inputNames: ["ane", "gpu", "M", "A", "G"], outputNames: ["out"],
        source: """
        uint n = thread_position_in_grid.x;
        uint m = thread_position_in_grid.y;
        if (m >= uint(M) || n >= uint(A + G)) return;
        float gate, up;
        if (n < uint(A)) {
            gate = float(ane[n * M + m]);
            up = float(ane[(A + n) * M + m]);
        } else {
            uint suffix = n - A;
            uint base = m * (2 * G);
            gate = float(gpu[base + suffix]);
            up = float(gpu[base + G + suffix]);
        }
        out[m * (A + G) + n] = T(gate * up / (1.0f + exp(-gate)));
        """
    )

    static func activation(_ ane: MLXArray, gpu: MLXArray, rows: Int, aneHidden: Int, gpuHidden: Int) -> MLXArray {
        merge([ane, gpu, Int32(rows), Int32(aneHidden), Int32(gpuHidden)],
            template: [("T", gpu.dtype)], grid: (aneHidden + gpuHidden, rows, 1),
            threadGroup: (16, 16, 1), outputShapes: [[1, rows, aneHidden + gpuHidden]], outputDTypes: [gpu.dtype])[0]
    }
}
