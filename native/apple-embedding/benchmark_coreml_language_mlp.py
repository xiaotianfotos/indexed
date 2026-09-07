#!/usr/bin/env python3
"""Convert and benchmark one real WeMM Qwen3.5 MLP layer with public Core ML."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument(
        "--input-scale",
        type=float,
        default=1.0,
        help="RMSNorm 后的隐藏状态通常接近单位方差",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--force-convert", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class ConvMLP(torch.nn.Module):
    """Qwen SwiGLU MLP in the rank-4 layout preferred by Core ML/ANE."""

    def __init__(
        self,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        intermediate, hidden = gate_weight.shape
        self.gate_proj = torch.nn.Conv2d(hidden, intermediate, 1, bias=False)
        self.up_proj = torch.nn.Conv2d(hidden, intermediate, 1, bias=False)
        self.down_proj = torch.nn.Conv2d(intermediate, hidden, 1, bias=False)
        with torch.no_grad():
            self.gate_proj.weight.copy_(gate_weight[:, :, None, None])
            self.up_proj.weight.copy_(up_weight[:, :, None, None])
            self.down_proj.weight.copy_(down_weight[:, :, None, None])
        self.half().eval()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(hidden_states))
        return self.down_proj(gate * self.up_proj(hidden_states))


def load_layer(weights_path: Path, layer: int) -> ConvMLP:
    prefix = f"model.layers.{layer}.mlp"
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        gate = handle.get_tensor(f"{prefix}.gate_proj.weight").float()
        up = handle.get_tensor(f"{prefix}.up_proj.weight").float()
        down = handle.get_tensor(f"{prefix}.down_proj.weight").float()
    if gate.shape != up.shape or down.shape != (gate.shape[1], gate.shape[0]):
        raise ValueError(
            f"unexpected MLP shapes: gate={gate.shape}, up={up.shape}, down={down.shape}"
        )
    return ConvMLP(gate, up, down)


def convert_model(
    module: ConvMLP,
    sample: torch.Tensor,
    package_path: Path,
) -> None:
    with torch.inference_mode():
        traced = torch.jit.trace(module, sample, strict=True)
        traced = torch.jit.freeze(traced.eval())
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="hidden_states",
                shape=tuple(sample.shape),
                dtype=np.float16,
            )
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    converted.short_description = (
        "One real WeMM-Embedding-2B Qwen3.5 SwiGLU MLP layer for ANE placement testing"
    )
    converted.save(str(package_path))


def benchmark_policy(
    package_path: Path,
    policy_name: str,
    sample: np.ndarray,
    warmup: int,
    runs: int,
) -> tuple[dict[str, Any], np.ndarray]:
    loaded_at = time.perf_counter()
    model = ct.models.MLModel(
        str(package_path),
        compute_units=POLICIES[policy_name],
        optimization_hints={
            "specializationStrategy": ct.SpecializationStrategy.FastPrediction
        },
    )
    load_ms = (time.perf_counter() - loaded_at) * 1000.0
    for _ in range(warmup):
        model.predict({"hidden_states": sample})

    samples: list[float] = []
    output: np.ndarray | None = None
    for _ in range(runs):
        started = time.perf_counter_ns()
        prediction = model.predict({"hidden_states": sample})
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
        output = np.asarray(prediction["output"], dtype=np.float32)
    assert output is not None
    result = {
        "policy": policy_name,
        "load_ms": load_ms,
        "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 95),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "runs": runs,
    }
    del model
    gc.collect()
    return result, output


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
    weights_path = args.weights.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    package_path = output_dir / (
        f"WeMM2B-Layer{args.layer}-MLP-S{args.sequence_length}.mlpackage"
    )

    module = load_layer(weights_path, args.layer)
    hidden = module.gate_proj.in_channels
    generator = torch.Generator(device="cpu").manual_seed(20260901)
    sample = (
        torch.randn(
            (1, hidden, 1, args.sequence_length),
            generator=generator,
            dtype=torch.float32,
        )
        * args.input_scale
    ).half()
    sample_array = sample.numpy()
    np.save(output_dir / "input.npy", sample_array)

    with torch.inference_mode():
        torch_started = time.perf_counter_ns()
        torch_output = module(sample).float().numpy()
        torch_ms = (time.perf_counter_ns() - torch_started) / 1_000_000.0
    np.save(output_dir / "torch_output.npy", torch_output)

    if args.force_convert or not package_path.exists():
        convert_model(module, sample, package_path)

    policy_results: dict[str, dict[str, Any]] = {}
    policy_outputs: dict[str, np.ndarray] = {}
    for policy_name in POLICIES:
        print(f"benchmarking {policy_name}...", flush=True)
        result, output = benchmark_policy(
            package_path,
            policy_name,
            sample_array,
            args.warmup,
            args.runs,
        )
        policy_results[policy_name] = result
        policy_outputs[policy_name] = output
        np.save(output_dir / f"{policy_name}_output.npy", output)

    payload = {
        "experiment": {
            "weights": str(weights_path),
            "layer": args.layer,
            "sequence_length": args.sequence_length,
            "input_shape": list(sample.shape),
            "hidden_size": hidden,
            "intermediate_size": module.gate_proj.out_channels,
            "parameter_count": sum(p.numel() for p in module.parameters()),
            "input_scale": args.input_scale,
            "coreml_package": str(package_path),
            "precision": "source BF16 -> Core ML FP16",
            "layout": "NCHW 1x1 convolution SwiGLU",
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
        },
        "machine": {
            "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
            "memory_bytes": command("sysctl", "-n", "hw.memsize"),
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "coremltools": ct.__version__,
            "torch": torch.__version__,
        },
        "torch_cpu_first_run_ms": torch_ms,
        "policies": policy_results,
        "agreement_vs_torch": {
            name: agreement(output, torch_output)
            for name, output in policy_outputs.items()
        },
        "cross_policy_agreement": {
            name: agreement(output, policy_outputs["cpu_only"])
            for name, output in policy_outputs.items()
        },
        "comparison": {
            "cpu_ane_speedup_over_cpu_gpu": (
                policy_results["cpu_gpu"]["median_ms"]
                / policy_results["cpu_ane"]["median_ms"]
            ),
            "all_speedup_over_cpu_gpu": (
                policy_results["cpu_gpu"]["median_ms"]
                / policy_results["all"]["median_ms"]
            ),
        },
    }
    report_path = output_dir / "coreml_benchmark.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
