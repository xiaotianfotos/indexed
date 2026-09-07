#!/usr/bin/env python3
"""Replace one MLX WeMM MLP with a public Core ML/ANE layer end to end."""

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
    parser.add_argument("--mlp-model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=8)
    return parser.parse_args()


def image_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


class CoreMLMLP:
    def __init__(self, model_path: Path, sequence_length: int) -> None:
        self.model = ct.models.MLModel(
            str(model_path),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            optimization_hints={
                "specializationStrategy": ct.SpecializationStrategy.FastPrediction
            },
        )
        self.sequence_length = sequence_length
        self.calls: list[dict[str, float]] = []

    def __call__(self, hidden_states: mx.array) -> mx.array:
        if hidden_states.shape[0] != 1 or hidden_states.shape[2] != 2048:
            raise ValueError(f"unexpected MLP input shape: {hidden_states.shape}")
        tokens = int(hidden_states.shape[1])
        if tokens > self.sequence_length:
            raise ValueError(f"{tokens} tokens exceed Core ML shape {self.sequence_length}")

        total_started = time.perf_counter_ns()
        transfer_started = time.perf_counter_ns()
        value = hidden_states.astype(mx.float16)
        mx.eval(value)
        token_array = np.asarray(value)
        padded = np.zeros(
            (1, 2048, 1, self.sequence_length), dtype=np.float16
        )
        padded[:, :, 0, :tokens] = token_array.transpose(0, 2, 1)
        transfer_in_ms = (time.perf_counter_ns() - transfer_started) / 1_000_000.0

        predict_started = time.perf_counter_ns()
        prediction = self.model.predict({"hidden_states": padded})
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


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


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


def vector_agreement(left: list[float], right: list[float]) -> dict[str, float]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return {
        "cosine": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
        "maximum_absolute_error": float(np.max(np.abs(a - b))),
        "mean_absolute_error": float(np.mean(np.abs(a - b))),
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
        engine,
        messages,
        args.dimension,
        args.warmup,
        args.runs,
    )
    layer = engine.language.model.layers[args.layer]
    original_mlp = layer.mlp
    coreml_mlp = CoreMLMLP(args.mlp_model.resolve(), args.sequence_length)
    layer.mlp = coreml_mlp
    hybrid_vector, hybrid_timings = run_engine(
        engine,
        messages,
        args.dimension,
        args.warmup,
        args.runs,
    )
    layer.mlp = original_mlp

    measured_calls = coreml_mlp.calls[-args.runs :]
    payload = {
        "experiment": {
            "package": str(args.package.resolve()),
            "coreml_mlp": str(args.mlp_model.resolve()),
            "image": str(args.image.resolve()),
            "layer": args.layer,
            "fixed_sequence_length": args.sequence_length,
            "actual_tokens": int(measured_calls[-1]["tokens"]),
            "dimension": args.dimension,
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
        },
        "baseline_mlx": baseline_timings,
        "hybrid_one_coreml_ane_mlp": hybrid_timings,
        "coreml_boundary": {
            key: summarize([row[key] for row in measured_calls])
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
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
