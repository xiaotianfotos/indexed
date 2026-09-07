#!/usr/bin/env python3
"""Isolate WeMM GDN gated RMSNorm and output projection in Core ML."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from safetensors import safe_open


POLICIES = {
    "cpu_only": ct.ComputeUnit.CPU_ONLY,
    "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu_ane": ct.ComputeUnit.CPU_AND_NE,
    "all": ct.ComputeUnit.ALL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--force-convert", action="store_true")
    return parser.parse_args()


def agreement(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    left = actual.reshape(-1).astype(np.float64)
    right = expected.reshape(-1).astype(np.float64)
    return {
        "cosine": float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right))),
        "maximum_absolute_error": float(np.max(np.abs(left - right))),
        "mean_absolute_error": float(np.mean(np.abs(left - right))),
    }


class GDNPostprocessor(torch.nn.Module):
    def __init__(self, norm_weight: torch.Tensor, out_weight: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("norm_weight", norm_weight.reshape(1, 1, 1, 128))
        self.out_proj = torch.nn.Conv2d(2048, 2048, 1, bias=False)
        with torch.no_grad():
            self.out_proj.weight.copy_(out_weight[:, :, None, None])
        self.half().eval()

    def forward(self, recurrence: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        variance = torch.mean(recurrence * recurrence, dim=-1, keepdim=True)
        normalized = recurrence * torch.rsqrt(variance + 1e-6)
        normalized = normalized * self.norm_weight
        normalized = normalized * torch.nn.functional.silu(z)
        channels = normalized.permute(0, 1, 3, 2).reshape(
            1, 2048, 1, recurrence.shape[2]
        )
        return self.out_proj(channels)


def load_module(weights: Path, layer: int) -> GDNPostprocessor:
    prefix = f"model.layers.{layer}.linear_attn"
    with safe_open(weights, framework="pt", device="cpu") as handle:
        norm = handle.get_tensor(f"{prefix}.norm.weight").float()
        out = handle.get_tensor(f"{prefix}.out_proj.weight").float()
    return GDNPostprocessor(norm, out)


def load_inputs(
    oracle: Path, sequence_length: int
) -> tuple[dict[str, np.ndarray], int, np.ndarray]:
    with np.load(oracle) as data:
        recurrence = np.asarray(data["recurrent_output"], dtype=np.float32)
        z = np.asarray(data["norm_gate_z"], dtype=np.float32).reshape(
            recurrence.shape
        )
        expected = np.asarray(data["gdn_output"], dtype=np.float32)
    valid = recurrence.shape[1]
    inputs: dict[str, np.ndarray] = {}
    for name, source in (("recurrence", recurrence), ("z", z)):
        padded = np.zeros((1, 16, sequence_length, 128), dtype=np.float16)
        padded[:, :, :valid] = source.transpose(0, 2, 1, 3).astype(np.float16)
        inputs[name] = padded
    expected = expected.transpose(0, 2, 1)[:, :, None, :]
    return inputs, valid, expected


def convert_model(
    module: GDNPostprocessor, inputs: dict[str, np.ndarray], package: Path
) -> None:
    samples = tuple(torch.from_numpy(inputs[name]) for name in ("recurrence", "z"))
    with torch.inference_mode():
        traced = torch.jit.freeze(torch.jit.trace(module, samples, strict=True).eval())
    model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name=name, shape=inputs[name].shape, dtype=np.float16)
            for name in ("recurrence", "z")
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    model.save(str(package))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs, valid, expected = load_inputs(args.oracle.resolve(), args.sequence_length)
    module = load_module(args.weights.resolve(), args.layer)
    samples = tuple(torch.from_numpy(inputs[name]) for name in ("recurrence", "z"))
    with torch.inference_mode():
        torch_output = module(*samples).float().numpy()
    package = output_dir / f"WeMM2B-Layer{args.layer}-GDNPost-S{args.sequence_length}.mlpackage"
    if args.force_convert or not package.exists():
        convert_model(module, inputs, package)

    results: dict[str, dict[str, float | int]] = {}
    agreements: dict[str, dict[str, float]] = {}
    for name, units in POLICIES.items():
        print(f"benchmarking {name}...", flush=True)
        model = ct.models.MLModel(
            str(package),
            compute_units=units,
            optimization_hints={
                "specializationStrategy": ct.SpecializationStrategy.FastPrediction
            },
        )
        for _ in range(args.warmup):
            model.predict(inputs)
        timings: list[float] = []
        output: np.ndarray | None = None
        for _ in range(args.runs):
            started = time.perf_counter_ns()
            output = np.asarray(model.predict(inputs)["output"], dtype=np.float32)
            timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
        assert output is not None
        results[name] = {
            "median_ms": statistics.median(timings),
            "p95_ms": float(np.percentile(timings, 95)),
            "runs": args.runs,
        }
        agreements[name] = agreement(output[..., :valid], expected[..., :valid])

    report = {
        "experiment": {
            "weights": str(args.weights.resolve()),
            "oracle": str(args.oracle.resolve()),
            "valid_tokens": valid,
            "sequence_length": args.sequence_length,
            "coreml_package": str(package),
        },
        "torch_fp16_agreement_vs_bf16_oracle": agreement(
            torch_output[..., :valid], expected[..., :valid]
        ),
        "policies": results,
        "coreml_agreement_vs_bf16_oracle": agreements,
    }
    report_path = output_dir / "gdn_postprocessor_benchmark.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report_path)


if __name__ == "__main__":
    main()
