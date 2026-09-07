#!/usr/bin/env python3
"""Replace one MLX WeMM decoder block with a public Core ML block end to end."""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import time
from pathlib import Path
from typing import Any

import coremltools as ct
import mlx.core as mx
import numpy as np

from service import MLXWeMMEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--block-model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument(
        "--last-layer",
        type=int,
        default=None,
        help="Replace a contiguous layer range; later layers become pass-through.",
    )
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    return parser.parse_args()


def image_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def vector_agreement(left: list[float], right: list[float]) -> dict[str, float]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return {
        "cosine": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
        "maximum_absolute_error": float(np.max(np.abs(a - b))),
        "mean_absolute_error": float(np.mean(np.abs(a - b))),
    }


class CoreMLDecoderBlock:
    def __init__(
        self,
        model_path: Path,
        sequence_length: int,
        original_layer: Any,
        rotary_provider: Any | None = None,
    ) -> None:
        self.model = ct.models.MLModel(
            str(model_path),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            optimization_hints={
                "specializationStrategy": ct.SpecializationStrategy.FastPrediction
            },
        )
        self.sequence_length = sequence_length
        self.original_layer = original_layer
        self.rotary_provider = rotary_provider
        self.is_linear = original_layer.is_linear
        self.input_names = {
            item.name for item in self.model.get_spec().description.input
        }
        self.dynamic_positions = {
            "position_cos",
            "position_sin",
        }.issubset(self.input_names)
        self.calls: list[dict[str, float]] = []

    def __getattr__(self, name: str) -> Any:
        """Preserve model-level introspection used by the Qwen3.5 dispatcher."""
        return getattr(self.original_layer, name)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Any | None = None,
        cache: Any | None = None,
        position_ids: Any | None = None,
        position_embeddings: Any | None = None,
    ) -> mx.array:
        del mask
        if cache is not None:
            raise ValueError("Core ML block integration only supports cache-free prefill")
        if hidden_states.shape[0] != 1 or hidden_states.shape[2] != 2048:
            raise ValueError(f"unexpected decoder input shape: {hidden_states.shape}")
        tokens = int(hidden_states.shape[1])
        if tokens > self.sequence_length:
            raise ValueError(f"{tokens} tokens exceed Core ML shape {self.sequence_length}")

        total_started = time.perf_counter_ns()
        transfer_started = time.perf_counter_ns()
        value = hidden_states.astype(mx.float16)
        mx.eval(value)
        padded = np.zeros((1, 2048, 1, self.sequence_length), dtype=np.float16)
        padded[:, :, 0, :tokens] = np.asarray(value).transpose(0, 2, 1)
        inputs: dict[str, np.ndarray] = {"hidden_states": padded}
        if self.dynamic_positions:
            if position_embeddings is None:
                if self.rotary_provider is None:
                    raise ValueError(
                        "dynamic Core ML segment requires positions or a rotary provider"
                    )
                if position_ids is None:
                    # The optimized MLX-LM text path keeps ordinary 1-D RoPE
                    # internal to attention instead of materializing it.
                    position_ids = mx.arange(tokens)[None, :]
                position_embeddings = self.rotary_provider(
                    hidden_states, position_ids
                )
            cos, sin = position_embeddings
            cos = cos.astype(mx.float16)
            sin = sin.astype(mx.float16)
            mx.eval(cos, sin)
            padded_cos = np.ones(
                (1, 64, 1, self.sequence_length), dtype=np.float16
            )
            padded_sin = np.zeros(
                (1, 64, 1, self.sequence_length), dtype=np.float16
            )
            padded_cos[:, :, 0, :tokens] = np.asarray(cos).transpose(0, 2, 1)
            padded_sin[:, :, 0, :tokens] = np.asarray(sin).transpose(0, 2, 1)
            inputs["position_cos"] = padded_cos
            inputs["position_sin"] = padded_sin
        transfer_in_ms = (time.perf_counter_ns() - transfer_started) / 1_000_000.0

        predict_started = time.perf_counter_ns()
        prediction = self.model.predict(inputs)
        predict_ms = (time.perf_counter_ns() - predict_started) / 1_000_000.0

        transfer_started = time.perf_counter_ns()
        output = np.asarray(prediction["output"], dtype=np.float16)
        output = output[:, :, 0, :tokens].transpose(0, 2, 1)
        result = mx.array(output).astype(hidden_states.dtype)
        transfer_out_ms = (time.perf_counter_ns() - transfer_started) / 1_000_000.0
        self.calls.append(
            {
                "tokens": float(tokens),
                "transfer_in": transfer_in_ms,
                "predict": predict_ms,
                "transfer_out": transfer_out_ms,
                "total": (time.perf_counter_ns() - total_started) / 1_000_000.0,
            }
        )
        return result


class PassthroughDecoderBlock:
    """Keep Qwen3.5 layer metadata while skipping a layer included in a segment."""

    def __init__(self, original_layer: Any) -> None:
        self.original_layer = original_layer
        self.is_linear = original_layer.is_linear

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original_layer, name)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Any | None = None,
        cache: Any | None = None,
        position_ids: Any | None = None,
        position_embeddings: Any | None = None,
    ) -> mx.array:
        del mask, position_ids, position_embeddings
        if cache is not None:
            raise ValueError("Core ML segment integration only supports cache-free prefill")
        return hidden_states


def run_engine(
    engine: MLXWeMMEngine,
    messages: list[dict[str, Any]],
    dimension: int,
    warmup: int,
    runs: int,
) -> tuple[list[float], dict[str, dict[str, float]]]:
    for _ in range(warmup):
        engine.embed(messages, dimension=dimension)
    vectors: list[list[float]] = []
    timings: list[dict[str, float]] = []
    for _ in range(runs):
        result = engine.embed(messages, dimension=dimension)
        vectors.append(result.vector)
        timings.append(result.timings_ms)
    return vectors[-1], {
        key: summarize([row[key] for row in timings]) for key in timings[0]
    }


def main() -> None:
    args = parse_args()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri(args.image)}},
                {"type": "text", "text": "Represent this image."},
            ],
        }
    ]
    engine = MLXWeMMEngine(
        package_path=args.package,
        image_size=448,
        compute_unit="cpu_ane",
        warmup=False,
    )
    baseline_vector, baseline_timings = run_engine(
        engine, messages, args.dimension, args.warmup, args.runs
    )

    last_layer = args.last_layer if args.last_layer is not None else args.layer
    if last_layer < args.layer:
        raise ValueError("last-layer must be greater than or equal to layer")
    original_layers = list(engine.language.model.layers[args.layer : last_layer + 1])
    original_layer = original_layers[0]
    rotary_provider = next(
        (
            original.self_attn.rotary_emb
            for original in original_layers
            if not original.is_linear
        ),
        None,
    )
    coreml_block = CoreMLDecoderBlock(
        args.block_model.resolve(),
        args.sequence_length,
        original_layer,
        rotary_provider,
    )
    engine.language.model.layers[args.layer] = coreml_block
    for offset, original in enumerate(original_layers[1:], start=1):
        engine.language.model.layers[args.layer + offset] = PassthroughDecoderBlock(
            original
        )
    hybrid_vector, hybrid_timings = run_engine(
        engine, messages, args.dimension, args.warmup, args.runs
    )
    for offset, original in enumerate(original_layers):
        engine.language.model.layers[args.layer + offset] = original

    calls = coreml_block.calls[-args.runs :]
    payload = {
        "experiment": {
            "package": str(args.package.resolve()),
            "coreml_block": str(args.block_model.resolve()),
            "image": str(args.image.resolve()),
            "layer": args.layer,
            "last_layer": last_layer,
            "fixed_sequence_length": args.sequence_length,
            "actual_tokens": int(calls[-1]["tokens"]),
            "dimension": args.dimension,
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
        },
        "baseline_mlx": baseline_timings,
        "hybrid_one_coreml_block": hybrid_timings,
        "coreml_boundary": {
            key: summarize([row[key] for row in calls])
            for key in ("transfer_in", "predict", "transfer_out", "total")
        },
        "final_vector_agreement": vector_agreement(hybrid_vector, baseline_vector),
        "language_speedup": (
            baseline_timings["language"]["median"]
            / hybrid_timings["language"]["median"]
        ),
        "end_to_end_speedup": (
            baseline_timings["total"]["median"]
            / hybrid_timings["total"]["median"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
