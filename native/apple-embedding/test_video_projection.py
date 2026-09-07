import unittest
import mlx.core as mx
import mlx.nn as nn
from video_projection import VideoDownProjection, VideoProjectionRuntime, install_video_projection


class VideoProjectionTests(unittest.TestCase):
    def test_only_long_video_uses_cache_and_keeps_output_dtype(self):
        mx.random.seed(3)
        original = nn.QuantizedLinear.from_linear(nn.Linear(64, 64, bias=True), group_size=64, bits=8)
        runtime = VideoProjectionRuntime(True)
        layer = VideoDownProjection(original, runtime)
        x = mx.random.normal((1, 1024, 64)).astype(mx.bfloat16)
        self.assertTrue(bool(mx.array_equal(layer(x), original(x)).item()))
        runtime.video_active = True
        y = layer(x)
        ref = original(x)
        self.assertEqual(y.dtype, x.dtype)
        self.assertTrue(bool(mx.all(mx.isfinite(y)).item()))
        error = mx.linalg.norm(y.astype(mx.float32) - ref.astype(mx.float32)) / mx.linalg.norm(ref.astype(mx.float32))
        self.assertLess(error.item(), 0.01)
        self.assertTrue(bool(mx.array_equal(layer(x[:, :8]), original(x[:, :8])).item()))
        self.assertEqual(runtime.fp16_calls, 1)
        self.assertEqual(runtime.q8_calls, 2)
        self.assertEqual(runtime.dense_bytes, 64 * 64 * 2)

    def test_q8_mode_does_not_allocate_or_replace_layers(self):
        model = nn.Sequential(nn.Linear(64, 64))
        runtime = install_video_projection(model, "q8")
        self.assertEqual(runtime.dense_bytes, 0)
        self.assertEqual(runtime.layers, 0)
        with self.assertRaises(ValueError): install_video_projection(model, "invalid")


if __name__ == '__main__': unittest.main()
