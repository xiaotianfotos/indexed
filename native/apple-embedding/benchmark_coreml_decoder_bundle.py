#!/usr/bin/env python3
"""Benchmark and numerically validate a chained L0-L23 Core ML decoder bundle."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_coreml_decoder_block import agreement


COMPUTE_UNITS = {
    "cpu_only": ct.ComputeUnit.CPU_ONLY,
    "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu_ane": ct.ComputeUnit.CPU_AND_NE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--compute", choices=COMPUTE_UNITS, default="cpu_ane")
    parser.add_argument(
        "--cpu-only-segments",
        default="",
        help="comma-separated labels such as l8_l11,l20_l23",
    )
    return parser.parse_args()


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values), value))


def main() -> None:
    args = parse_args()
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("runtime_semantics") != "wemm-coreml-decoder-bundle-v1":
        raise ValueError("unsupported decoder bundle")
    sequence_length = int(manifest["sequence_length"])
    boundary_layers = [int(segment["last_layer"]) for segment in manifest["segments"]]
    cpu_only_segments = {
        value.strip() for value in args.cpu_only_segments.split(",") if value.strip()
    }

    with np.load(args.oracle.resolve()) as oracle:
        source = np.asarray(oracle["layer_0_input"], dtype=np.float32)
        expected = np.asarray(oracle["layer_23_output"], dtype=np.float32)
        cos = np.asarray(oracle["position_cos"], dtype=np.float32)
        sin = np.asarray(oracle["position_sin"], dtype=np.float32)
        expected_boundaries = {
            last: np.asarray(oracle[f"layer_{last}_output"], dtype=np.float32)
            for last in boundary_layers
        }
    valid_tokens = source.shape[1]
    if valid_tokens > sequence_length:
        raise ValueError("oracle is longer than the decoder bundle")

    hidden = np.zeros((1, source.shape[2], 1, sequence_length), dtype=np.float16)
    hidden[:, :, 0, :valid_tokens] = source.transpose(0, 2, 1).astype(np.float16)
    position_cos = np.ones((1, cos.shape[2], 1, sequence_length), dtype=np.float16)
    position_sin = np.zeros((1, sin.shape[2], 1, sequence_length), dtype=np.float16)
    position_cos[:, :, 0, :valid_tokens] = cos.transpose(0, 2, 1).astype(np.float16)
    position_sin[:, :, 0, :valid_tokens] = sin.transpose(0, 2, 1).astype(np.float16)

    models: list[tuple[dict[str, object], ct.models.MLModel]] = []
    load_timings: dict[str, float] = {}
    for segment in manifest["segments"]:
        label = f"l{segment['first_layer']}_l{segment['last_layer']}"
        compute_units = (
            ct.ComputeUnit.CPU_ONLY
            if label in cpu_only_segments
            else COMPUTE_UNITS[args.compute]
        )
        path = (bundle / segment["path"]).resolve()
        started = time.perf_counter_ns()
        model = ct.models.MLModel(
            str(path),
            compute_units=compute_units,
            optimization_hints={
                "specializationStrategy": ct.SpecializationStrategy.FastPrediction
            },
        )
        load_timings[label] = (time.perf_counter_ns() - started) / 1_000_000.0
        models.append((segment, model))

    def predict_once(
        record: bool,
    ) -> tuple[np.ndarray, dict[str, float], dict[int, np.ndarray]]:
        value = hidden
        timings: dict[str, float] = {}
        boundaries: dict[int, np.ndarray] = {}
        for segment, model in models:
            label = f"l{segment['first_layer']}_l{segment['last_layer']}"
            started = time.perf_counter_ns()
            value = np.asarray(
                model.predict({
                    "hidden_states": value,
                    "position_cos": position_cos,
                    "position_sin": position_sin,
                })["output"],
                dtype=np.float16,
            )
            if record:
                timings[label] = (time.perf_counter_ns() - started) / 1_000_000.0
                boundaries[int(segment["last_layer"])] = value.astype(np.float32)
        return value, timings, boundaries

    for _ in range(args.warmup):
        predict_once(record=False)
    runs: list[dict[str, float]] = []
    boundaries: dict[int, np.ndarray] = {}
    output: np.ndarray | None = None
    for _ in range(args.runs):
        output, timings, boundaries = predict_once(record=True)
        runs.append(timings)
    assert output is not None
    output_valid = output[:, :, :, :valid_tokens].astype(np.float32)
    expected_nchw = expected.transpose(0, 2, 1)[:, :, None, :]
    totals = [sum(run.values()) for run in runs]
    segment_timings = {
        label: {
            "median_ms": statistics.median(run[label] for run in runs),
            "p95_ms": percentile([run[label] for run in runs], 95),
        }
        for label in runs[0]
    }
    report = {
        "experiment": {
            "bundle": str(bundle),
            "oracle": str(args.oracle.resolve()),
            "sequence_length": sequence_length,
            "valid_tokens": valid_tokens,
            "warmup": args.warmup,
            "runs": args.runs,
            "compute_units": args.compute,
            "cpu_only_segments": sorted(cpu_only_segments),
        },
        "machine": {
            "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
            "memory_bytes": command("sysctl", "-n", "hw.memsize"),
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "coremltools": ct.__version__,
        },
        "load_ms": load_timings,
        "total": {
            "median_ms": statistics.median(totals),
            "p95_ms": percentile(totals, 95),
            "minimum_ms": min(totals),
            "maximum_ms": max(totals),
        },
        "segments": segment_timings,
        "agreement_vs_bf16_oracle": agreement(output_valid, expected_nchw),
        "last_token_agreement_vs_bf16_oracle": agreement(
            output_valid[..., -1:], expected_nchw[..., -1:]
        ),
        "chained_boundary_agreement_vs_bf16_oracle": {
            f"l{last}": {
                "all_tokens": agreement(
                    boundaries[last][..., :valid_tokens],
                    expected_boundaries[last].transpose(0, 2, 1)[:, :, None, :],
                ),
                "last_token": agreement(
                    boundaries[last][..., valid_tokens - 1 : valid_tokens],
                    expected_boundaries[last]
                        .transpose(0, 2, 1)[:, :, None, valid_tokens - 1 : valid_tokens],
                ),
            }
            for last in expected_boundaries
        },
    }
    suffix = "_" + "-".join(sorted(cpu_only_segments)) if cpu_only_segments else ""
    output_path = bundle / f"decoder_bundle_benchmark_{args.compute}{suffix}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()
