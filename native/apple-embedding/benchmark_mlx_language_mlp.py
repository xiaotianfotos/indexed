#!/usr/bin/env python3
"""Benchmark the same real WeMM Qwen3.5 MLP layer with public MLX GPU."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--coreml-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    return parser.parse_args()


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def agreement(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    left = actual.reshape(-1).astype(np.float64)
    right = expected.reshape(-1).astype(np.float64)
    return {
        "cosine": float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right))),
        "maximum_absolute_error": float(np.max(np.abs(left - right))),
        "mean_absolute_error": float(np.mean(np.abs(left - right))),
    }


def main() -> None:
    args = parse_args()
    prefix = f"model.layers.{args.layer}.mlp"
    loaded = mx.load(str(args.weights.resolve()))
    gate = loaded[f"{prefix}.gate_proj.weight"]
    up = loaded[f"{prefix}.up_proj.weight"]
    down = loaded[f"{prefix}.down_proj.weight"]
    mx.eval(gate, up, down)
    del loaded

    nchw = np.load(args.input.resolve())
    tokens = mx.array(nchw[0, :, 0, :].T)

    @mx.compile
    def mlp(hidden_states: mx.array) -> mx.array:
        gate_projection = hidden_states @ gate.T
        gate_values = gate_projection * mx.sigmoid(gate_projection)
        return (gate_values * (hidden_states @ up.T)) @ down.T

    for _ in range(args.warmup):
        mx.eval(mlp(tokens))

    samples: list[float] = []
    output: mx.array | None = None
    for _ in range(args.runs):
        started = time.perf_counter_ns()
        output = mlp(tokens)
        mx.eval(output)
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    assert output is not None
    output_array = np.asarray(output.astype(mx.float32)).T[None, :, None, :]
    coreml_output = np.load(args.coreml_output.resolve())

    payload = {
        "backend": "MLX GPU",
        "layer": args.layer,
        "input_shape": list(nchw.shape),
        "weight_dtype": str(gate.dtype),
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "latency": {
            "median_ms": statistics.median(samples),
            "p95_ms": percentile(samples, 95),
            "minimum_ms": min(samples),
            "maximum_ms": max(samples),
        },
        "agreement_vs_coreml_cpu_ane": agreement(output_array, coreml_output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
