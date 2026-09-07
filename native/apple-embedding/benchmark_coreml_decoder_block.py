#!/usr/bin/env python3
"""Convert and benchmark one complete real WeMM Qwen3.5 decoder block."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_coreml_full_gdn import load_complete_gdn
from benchmark_coreml_language_mlp import load_layer as load_mlp


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
    parser.add_argument(
        "--gdn-representation",
        choices=("chunked_fp32", "chunked_fp32_decomposed", "chunked_fp32_sensitive"),
        default="chunked_fp32_decomposed",
    )
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--force-convert", action="store_true")
    return parser.parse_args()


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def agreement(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    left = actual.reshape(-1).astype(np.float64)
    right = expected.reshape(-1).astype(np.float64)
    return {
        "cosine": float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right))),
        "maximum_absolute_error": float(np.max(np.abs(left - right))),
        "mean_absolute_error": float(np.mean(np.abs(left - right))),
        "root_mean_square_error": float(np.sqrt(np.mean((left - right) ** 2))),
    }


class RMSNormChannels(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.register_buffer("weight", weight.reshape(1, -1, 1, 1))
        self.epsilon = epsilon

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = torch.mean(hidden_states * hidden_states, dim=1, keepdim=True)
        return hidden_states * torch.rsqrt(variance + self.epsilon) * self.weight


class CompleteDecoderBlock(torch.nn.Module):
    def __init__(
        self,
        input_norm: RMSNormChannels,
        gdn: torch.nn.Module,
        post_norm: RMSNormChannels,
        mlp: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.input_norm = input_norm
        self.gdn = gdn
        self.post_norm = post_norm
        self.mlp = mlp
        self.half().eval()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        after_gdn = hidden_states + self.gdn(self.input_norm(hidden_states))
        return after_gdn + self.mlp(self.post_norm(after_gdn))


def load_block(
    weights: Path,
    layer: int,
    representation: str,
    sequence_length: int,
    valid_tokens: int,
) -> CompleteDecoderBlock:
    prefix = f"model.layers.{layer}"
    with safe_open(weights, framework="pt", device="cpu") as handle:
        input_weight = handle.get_tensor(f"{prefix}.input_layernorm.weight").float()
        post_weight = handle.get_tensor(f"{prefix}.post_attention_layernorm.weight").float()
    return CompleteDecoderBlock(
        RMSNormChannels(input_weight),
        load_complete_gdn(
            weights,
            layer,
            representation,
            sequence_length,
            valid_tokens,
        ),
        RMSNormChannels(post_weight),
        load_mlp(weights, layer),
    )


def load_input(
    oracle: Path, sequence_length: int
) -> tuple[torch.Tensor, int, np.ndarray]:
    with np.load(oracle) as data:
        hidden = np.asarray(data["decoder_input"], dtype=np.float32)
        expected = np.asarray(data["decoder_output"], dtype=np.float32)
    valid = hidden.shape[1]
    padded = np.zeros((1, hidden.shape[2], 1, sequence_length), dtype=np.float16)
    padded[:, :, 0, :valid] = hidden.transpose(0, 2, 1).astype(np.float16)
    expected = expected.transpose(0, 2, 1)[:, :, None, :]
    return torch.from_numpy(padded), valid, expected


def convert_model(
    module: CompleteDecoderBlock, sample: torch.Tensor, package: Path
) -> None:
    with torch.inference_mode():
        traced = torch.jit.freeze(torch.jit.trace(module, sample, strict=True).eval())
    precision = ct.transform.FP16ComputePrecision(
        op_selector=lambda op: "recurrence" not in str(op.scopes)
    )
    model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="hidden_states", shape=tuple(sample.shape), dtype=np.float16
            )
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    model.short_description = "One complete real WeMM-Embedding-2B Qwen3.5 decoder block"
    model.save(str(package))


def benchmark(
    package: Path,
    units: ct.ComputeUnit,
    sample: np.ndarray,
    warmup: int,
    runs: int,
) -> tuple[dict[str, float | int], np.ndarray]:
    loaded_at = time.perf_counter()
    model = ct.models.MLModel(
        str(package),
        compute_units=units,
        optimization_hints={
            "specializationStrategy": ct.SpecializationStrategy.FastPrediction
        },
    )
    load_ms = (time.perf_counter() - loaded_at) * 1000.0
    for _ in range(warmup):
        model.predict({"hidden_states": sample})
    timings: list[float] = []
    output: np.ndarray | None = None
    for _ in range(runs):
        started = time.perf_counter_ns()
        output = np.asarray(
            model.predict({"hidden_states": sample})["output"], dtype=np.float32
        )
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    assert output is not None
    return {
        "load_ms": load_ms,
        "median_ms": statistics.median(timings),
        "p95_ms": float(np.percentile(timings, 95)),
        "minimum_ms": min(timings),
        "maximum_ms": max(timings),
        "runs": runs,
    }, output


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample, valid, expected = load_input(args.oracle.resolve(), args.sequence_length)
    module = load_block(
        args.weights.resolve(),
        args.layer,
        args.gdn_representation,
        args.sequence_length,
        valid,
    )
    with torch.inference_mode():
        started = time.perf_counter_ns()
        torch_output = module(sample).float().numpy()
        torch_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    package = output_dir / (
        f"WeMM2B-Layer{args.layer}-DecoderBlock-{args.gdn_representation}"
        f"-S{args.sequence_length}.mlpackage"
    )
    if args.force_convert or not package.exists():
        convert_model(module, sample, package)

    policies: dict[str, dict[str, float | int]] = {}
    agreements: dict[str, dict[str, float]] = {}
    expected_valid = expected[..., :valid]
    for name, units in POLICIES.items():
        print(f"benchmarking {name}...", flush=True)
        result, output = benchmark(
            package, units, sample.numpy(), args.warmup, args.runs
        )
        policies[name] = result
        agreements[name] = agreement(output[..., :valid], expected_valid)

    report = {
        "experiment": {
            "weights": str(args.weights.resolve()),
            "oracle": str(args.oracle.resolve()),
            "layer": args.layer,
            "gdn_representation": args.gdn_representation,
            "sequence_length": args.sequence_length,
            "valid_tokens": valid,
            "coreml_package": str(package),
        },
        "machine": {
            "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
            "memory_bytes": command("sysctl", "-n", "hw.memsize"),
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "coremltools": ct.__version__,
            "torch": torch.__version__,
        },
        "torch_fp16_first_run_ms": torch_ms,
        "torch_fp16_agreement_vs_bf16_oracle": agreement(
            torch_output[..., :valid], expected_valid
        ),
        "policies": policies,
        "coreml_agreement_vs_bf16_oracle": agreements,
    }
    report_path = output_dir / (
        f"decoder_block_{args.gdn_representation}_benchmark.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report_path)


if __name__ == "__main__":
    main()
