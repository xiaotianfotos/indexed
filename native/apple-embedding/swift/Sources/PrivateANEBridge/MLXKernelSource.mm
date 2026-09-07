// SPDX-License-Identifier: Apache-2.0
#include "PrivateANEBridge.h"
#include <string>

// These generated-source accessors belong to the pinned mlx-swift/Cmlx library.
// MLXFast supplies utils() itself. Keep this narrow ABI covered by a linked
// hardware test: an upstream symbol/layout change must fail the native build.
namespace mlx { namespace core { namespace metal {
const char *gemm();
const char *quantized_utils();
const char *quantized();
} } }

const char *indexed_mlx_affine_kernel_header(void) {
    static const std::string source = std::string(mlx::core::metal::gemm()) +
        mlx::core::metal::quantized_utils() + mlx::core::metal::quantized();
    return source.c_str();
}
