"""Video-only FP16 cache for the GPU down projection left outside ANE MLP.

The Q8 weights remain the source of truth. Images, text, and short tails keep
their existing quantized path; no frame/token is removed. This module is loaded
only by the optional private-ANE Python service.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class VideoProjectionRuntime:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.video_active = False
        self.layers = 0
        self.dense_bytes = 0
        self.fp16_calls = 0
        self.q8_calls = 0

    def profile(self):
        return {"mode": "fp16" if self.enabled else "q8", "minimum_tokens": 1024,
                "layers": self.layers, "dense_weight_bytes": self.dense_bytes,
                "fp16_calls": self.fp16_calls, "q8_calls": self.q8_calls}


class VideoDownProjection(nn.Module):
    def __init__(self, original, runtime: VideoProjectionRuntime):
        super().__init__()
        self.original = original
        self._runtime = runtime
        self.dense = mx.dequantize(original.weight, original.scales, original.biases,
                                   group_size=original.group_size, bits=original.bits).astype(mx.float16)
        mx.eval(self.dense)
        runtime.layers += 1
        runtime.dense_bytes += self.dense.nbytes

    def __call__(self, x):
        runtime = self._runtime
        if runtime.enabled and runtime.video_active and x.ndim >= 2 and x.shape[-2] >= 1024:
            runtime.fp16_calls += 1
            result = x.astype(mx.float16) @ self.dense.T
            if "bias" in self.original:
                result = result + self.original.bias.astype(mx.float16)
            return result.astype(x.dtype)
        runtime.q8_calls += 1
        return self.original(x)


def install_video_projection(model, mode: str):
    if mode not in {"q8", "fp16"}:
        raise ValueError("video down projection must be q8 or fp16")
    runtime = VideoProjectionRuntime(mode == "fp16")
    if runtime.enabled:
        for _, module in list(model.named_modules()):
            if not all(hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj")):
                continue
            original = module.down_proj
            if isinstance(original, nn.QuantizedLinear) and original.bits == 8 and original.mode == "affine":
                module.down_proj = VideoDownProjection(original, runtime)
        model.eval()
    return runtime
