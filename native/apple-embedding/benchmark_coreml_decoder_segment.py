#!/usr/bin/env python3
"""Benchmark a contiguous 3xGDN + 1xattention WeMM decoder segment."""

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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_coreml_attention_block import load_block as load_attention_block
from benchmark_coreml_decoder_block import agreement, load_block as load_gdn_block


POLICIES = {
    "cpu_only": ct.ComputeUnit.CPU_ONLY,
    "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu_ane": ct.ComputeUnit.CPU_AND_NE,
    "all": ct.ComputeUnit.ALL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--first-oracle", type=Path)
    parser.add_argument("--last-oracle", type=Path)
    parser.add_argument(
        "--oracle",
        type=Path,
        help="combined video oracle with layer_<N>_input/output tensors",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--last-layer", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument(
        "--accuracy-profile",
        choices=("balanced", "gdn_high", "attention_high", "high"),
        default="balanced",
        help="select which numerically sensitive subgraphs remain FP32",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--policies",
        default=",".join(POLICIES),
        help="comma-separated subset of cpu_only,cpu_gpu,cpu_ane,all",
    )
    parser.add_argument("--skip-torch-benchmark", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    return parser.parse_args()


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class DecoderSegment(torch.nn.Module):
    def __init__(self, layers: list[torch.nn.Module]) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.half().eval()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            if hasattr(layer, "attention"):
                hidden_states = layer(hidden_states, position_cos, position_sin)
            else:
                hidden_states = layer(hidden_states)
        return hidden_states


def load_segment(
    weights: Path,
    last_oracle: Path,
    first_layer: int,
    last_layer: int,
    sequence_length: int,
    valid_tokens: int,
    accuracy_profile: str,
) -> DecoderSegment:
    high_gdn = accuracy_profile in ("gdn_high", "high")
    high_attention = accuracy_profile in ("attention_high", "high")
    layers: list[torch.nn.Module] = []
    for layer in range(first_layer, last_layer + 1):
        if (layer + 1) % 4 == 0:
            layers.append(
                load_attention_block(
                    weights,
                    last_oracle,
                    layer,
                    sequence_length,
                    "fp32_softmax" if high_attention else "fp16",
                )
            )
        else:
            layers.append(
                load_gdn_block(
                    weights,
                    layer,
                    "chunked_fp32_sensitive"
                    if high_gdn
                    else "chunked_fp32_decomposed",
                    sequence_length,
                    valid_tokens,
                )
            )
    return DecoderSegment(layers)


def load_input(
    first_oracle: Path,
    last_oracle: Path,
    sequence_length: int,
    first_layer: int,
    last_layer: int,
    combined_oracle: bool = False,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], int, np.ndarray]:
    with np.load(first_oracle) as first:
        hidden = np.asarray(
            first[f"layer_{first_layer}_input"]
            if combined_oracle else first["decoder_input"],
            dtype=np.float32,
        )
    with np.load(last_oracle) as last:
        expected = np.asarray(
            last[f"layer_{last_layer}_output"]
            if combined_oracle else last["decoder_output"],
            dtype=np.float32,
        )
        oracle_cos = np.asarray(last["position_cos"], dtype=np.float32)
        oracle_sin = np.asarray(last["position_sin"], dtype=np.float32)
    valid = hidden.shape[1]
    if expected.shape[1] != valid:
        raise ValueError("oracle sequence lengths differ")
    padded = np.zeros((1, hidden.shape[2], 1, sequence_length), dtype=np.float16)
    padded[:, :, 0, :valid] = hidden.transpose(0, 2, 1).astype(np.float16)
    position_cos = np.ones((1, 64, 1, sequence_length), dtype=np.float16)
    position_sin = np.zeros((1, 64, 1, sequence_length), dtype=np.float16)
    position_cos[:, :, 0, :valid] = oracle_cos.transpose(0, 2, 1).astype(np.float16)
    position_sin[:, :, 0, :valid] = oracle_sin.transpose(0, 2, 1).astype(np.float16)
    expected = expected.transpose(0, 2, 1)[:, :, None, :]
    return (
        (
            torch.from_numpy(padded),
            torch.from_numpy(position_cos),
            torch.from_numpy(position_sin),
        ),
        valid,
        expected,
    )


def convert_model(
    module: torch.nn.Module,
    sample: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    package: Path,
    accuracy_profile: str,
) -> None:
    with torch.inference_mode():
        traced = torch.jit.freeze(
            torch.jit.trace(
                module, sample, strict=True, check_trace=False
            ).eval()
        )
    protected_scopes = ("recurrence",)
    if accuracy_profile in ("gdn_high", "high"):
        protected_scopes += (
            "qkv_activation",
            "gate_activation",
        )
    if accuracy_profile in ("attention_high", "high"):
        protected_scopes += (
            "attention",
        )
    precision = ct.transform.FP16ComputePrecision(
        op_selector=lambda op: not any(
            scope in str(op.scopes) for scope in protected_scopes
        )
    )
    model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="hidden_states", shape=tuple(sample[0].shape), dtype=np.float16
            ),
            ct.TensorType(
                name="position_cos", shape=tuple(sample[1].shape), dtype=np.float16
            ),
            ct.TensorType(
                name="position_sin", shape=tuple(sample[2].shape), dtype=np.float16
            ),
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    model.short_description = "WeMM-Embedding-2B contiguous Core ML decoder segment"
    model.save(str(package))


def benchmark(
    package: Path,
    units: ct.ComputeUnit,
    sample: dict[str, np.ndarray],
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
        model.predict(sample)
    timings: list[float] = []
    output: np.ndarray | None = None
    for _ in range(runs):
        started = time.perf_counter_ns()
        output = np.asarray(
            model.predict(sample)["output"], dtype=np.float32
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
    if args.oracle is not None:
        first_oracle = last_oracle = args.oracle.resolve()
        combined_oracle = True
    elif args.first_oracle is not None and args.last_oracle is not None:
        first_oracle = args.first_oracle.resolve()
        last_oracle = args.last_oracle.resolve()
        combined_oracle = False
    else:
        raise SystemExit("provide --oracle or both --first-oracle and --last-oracle")
    selected_policies = [name.strip() for name in args.policies.split(",") if name.strip()]
    unknown_policies = set(selected_policies) - POLICIES.keys()
    if unknown_policies:
        raise SystemExit(f"unknown policies: {sorted(unknown_policies)}")
    if not selected_policies:
        raise SystemExit("--policies cannot be empty")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample, valid, expected = load_input(
        first_oracle,
        last_oracle,
        args.sequence_length,
        args.first_layer,
        args.last_layer,
        combined_oracle,
    )
    module = load_segment(
        args.weights.resolve(),
        last_oracle,
        args.first_layer,
        args.last_layer,
        args.sequence_length,
        valid,
        args.accuracy_profile,
    )
    expected_valid = expected[..., :valid]
    torch_ms: float | None = None
    torch_agreement: dict[str, float] | None = None
    if not args.skip_torch_benchmark:
        with torch.inference_mode():
            started = time.perf_counter_ns()
            torch_output = module(*sample).float().numpy()
            torch_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        torch_agreement = agreement(torch_output[..., :valid], expected_valid)
        print(f"torch segment oracle cosine: {torch_agreement['cosine']:.9f}", flush=True)
    profile_suffix = {
        "balanced": "",
        "gdn_high": "-GDNHigh",
        "attention_high": "-AttentionHigh",
        "high": "-HighAccuracy",
    }[args.accuracy_profile]
    package = output_dir / (
        f"WeMM2B-DecoderSegment-L{args.first_layer}-L{args.last_layer}-DynamicRoPE"
        f"-S{args.sequence_length}{profile_suffix}.mlpackage"
    )
    if args.force_convert or not package.exists():
        convert_model(module, sample, package, args.accuracy_profile)

    policies: dict[str, dict[str, float | int]] = {}
    agreements: dict[str, dict[str, float]] = {}
    for name in selected_policies:
        units = POLICIES[name]
        print(f"benchmarking {name}...", flush=True)
        coreml_inputs = {
            "hidden_states": sample[0].numpy(),
            "position_cos": sample[1].numpy(),
            "position_sin": sample[2].numpy(),
        }
        result, output = benchmark(package, units, coreml_inputs, args.warmup, args.runs)
        policies[name] = result
        agreements[name] = agreement(output[..., :valid], expected_valid)
    report = {
        "experiment": {
            "weights": str(args.weights.resolve()),
            "first_oracle": str(first_oracle),
            "last_oracle": str(last_oracle),
            "combined_oracle": combined_oracle,
            "first_layer": args.first_layer,
            "last_layer": args.last_layer,
            "sequence_length": args.sequence_length,
            "valid_tokens": valid,
            "accuracy_profile": args.accuracy_profile,
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
        "torch_fp16_agreement_vs_bf16_oracle": torch_agreement,
        "policies": policies,
        "coreml_agreement_vs_bf16_oracle": agreements,
    }
    report_path = output_dir / (
        f"decoder_segment_l{args.first_layer}_l{args.last_layer}_benchmark.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report_path)


if __name__ == "__main__":
    main()
