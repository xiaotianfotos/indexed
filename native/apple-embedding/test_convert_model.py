#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from convert_model import _write_modifications_notice, prepare_language_weights


class PrepareLanguageWeightsTests(unittest.TestCase):
    def test_filters_and_converts_qwen35_authoring_layout(self) -> None:
        source = {
            "model.language_model.layers.0.linear_attn.conv1d.weight": mx.arange(
                24, dtype=mx.float32
            ).reshape(3, 1, 8),
            "model.language_model.layers.0.input_layernorm.weight": mx.array(
                [0.0, 1.0], dtype=mx.float32
            ),
            "model.language_model.norm.weight": mx.array(
                [2.0, 3.0], dtype=mx.float32
            ),
            "model.language_model.layers.0.self_attn.q_proj.weight": mx.ones(
                (2, 2), dtype=mx.float32
            ),
            "model.visual.blocks.0.weight": mx.ones((1,), dtype=mx.float32),
            "lm_head.weight": mx.ones((4, 2), dtype=mx.float32),
        }

        prepared = prepare_language_weights(source)
        self.assertEqual(
            set(prepared),
            {
                "model.layers.0.linear_attn.conv1d.weight",
                "model.layers.0.input_layernorm.weight",
                "model.norm.weight",
                "model.layers.0.self_attn.q_proj.weight",
            },
        )
        self.assertEqual(
            prepared["model.layers.0.linear_attn.conv1d.weight"].shape,
            (3, 8, 1),
        )
        self.assertEqual(
            prepared["model.layers.0.input_layernorm.weight"].tolist(), [1.0, 2.0]
        )
        self.assertEqual(prepared["model.norm.weight"].tolist(), [3.0, 4.0])

    def test_rejects_non_wemm_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "No tensors start"):
            prepare_language_weights({"model.layers.0.weight": mx.ones((1,))})

    def test_writes_derivative_model_notice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_modifications_notice(
                root,
                source_description="Tencent/WeMM-Embedding-2B",
                language_precision="q8-g64 affine quantization",
                vision_artifact="vision/WeMMVision448.mlmodelc",
            )
            notice = (root / "MODIFICATIONS.md").read_text(encoding="utf-8")
            self.assertIn("not an official Tencent distribution", notice)
            self.assertIn("language/model.safetensors", notice)
            self.assertIn("q8-g64 affine quantization", notice)
            self.assertIn("vision/WeMMVision448.mlmodelc", notice)


if __name__ == "__main__":
    unittest.main()
