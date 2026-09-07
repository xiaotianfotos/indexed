#!/usr/bin/env python3
"""Benchmark the same real WeMM decoder block with public MLX/Metal."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm.models.qwen3_5.config import ModelConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    return parser.parse_args()


def agreement(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    left = actual.reshape(-1).astype(np.float64)
    right = expected.reshape(-1).astype(np.float64)
    return {
        "cosine": float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right))),
        "maximum_absolute_error": float(np.max(np.abs(left - right))),
        "mean_absolute_error": float(np.mean(np.abs(left - right))),
    }


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def load_layer(package: Path, layer_index: int):
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    language_path = package / str(manifest["language"]["path"])
    config_path = language_path.parent / "config.json"
    config_dict = json.loads(config_path.read_text(encoding="utf-8"))
    config = ModelConfig.from_dict(config_dict)
    language = LanguageModel(config.text_config, config)
    if hasattr(language, "lm_head"):
        del language.lm_head
    weights = mx.load(str(language_path))
    language.load_weights(list(weights.items()), strict=True)
    language.eval()
    mx.eval(language.parameters())
    del weights
    return language.model.layers[layer_index], language


def benchmark_shape(
    layer,
    hidden: np.ndarray,
    expected: np.ndarray,
    sequence_length: int,
    valid_tokens: int,
    warmup: int,
    runs: int,
) -> dict[str, object]:
    padded = np.zeros((1, sequence_length, hidden.shape[-1]), dtype=np.float32)
    padded[:, :valid_tokens] = hidden[:, :valid_tokens]
    values = mx.array(padded).astype(mx.bfloat16)
    mask_array = np.zeros((1, sequence_length), dtype=np.bool_)
    mask_array[:, :valid_tokens] = True
    mask = mx.array(mask_array)

    @mx.compile
    def forward(x: mx.array, token_mask: mx.array) -> mx.array:
        return layer(x, mask=token_mask, cache=None)

    for _ in range(warmup):
        mx.eval(forward(values, mask))
    timings: list[float] = []
    output = None
    for _ in range(runs):
        started = time.perf_counter_ns()
        output = forward(values, mask)
        mx.eval(output)
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    assert output is not None
    output_array = np.asarray(output.astype(mx.float32))[:, :valid_tokens]
    return {
        "sequence_length": sequence_length,
        "valid_tokens": valid_tokens,
        "latency": {
            "median_ms": statistics.median(timings),
            "p95_ms": percentile(timings, 95),
            "minimum_ms": min(timings),
            "maximum_ms": max(timings),
            "runs": runs,
        },
        "agreement_vs_bf16_oracle": agreement(
            output_array, expected[:, :valid_tokens]
        ),
    }


def main() -> None:
    args = parse_args()
    with np.load(args.oracle.resolve()) as data:
        hidden = np.asarray(data["decoder_input"], dtype=np.float32)
        expected = np.asarray(data["decoder_output"], dtype=np.float32)
    valid = hidden.shape[1]
    load_started = time.perf_counter()
    layer, language_owner = load_layer(args.package.resolve(), args.layer)
    load_seconds = time.perf_counter() - load_started
    results = [
        benchmark_shape(
            layer,
            hidden,
            expected,
            sequence_length,
            valid,
            args.warmup,
            args.runs,
        )
        for sequence_length in (valid, args.sequence_length)
    ]
    payload = {
        "backend": "MLX GPU",
        "package": str(args.package.resolve()),
        "oracle": str(args.oracle.resolve()),
        "layer": args.layer,
        "weight_dtype": str(layer.mlp.gate_proj.weight.dtype),
        "load_seconds": load_seconds,
        "active_memory_bytes": int(mx.get_active_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "results": results,
    }
    del language_owner
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
