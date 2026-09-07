#!/usr/bin/env python3
"""Convert and benchmark one complete real WeMM Qwen3.5 GDN layer."""

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
from benchmark_coreml_gdn_preprocessor import load_layer as load_preprocessor
from benchmark_coreml_gdn_recurrence import ChunkedRecurrence, UnrolledRecurrence


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
        "--representation",
        choices=(
            "chunked",
            "chunked_fp32",
            "chunked_fp32_decomposed",
            "chunked_fp32_sensitive",
            "unrolled",
        ),
        default="chunked",
    )
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--force-convert", action="store_true")
    return parser.parse_args()


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def agreement(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    left = actual.reshape(-1).astype(np.float64)
    right = expected.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return {
        "cosine": float(left @ right / denominator),
        "maximum_absolute_error": float(np.max(np.abs(left - right))),
        "mean_absolute_error": float(np.mean(np.abs(left - right))),
        "root_mean_square_error": float(np.sqrt(np.mean((left - right) ** 2))),
    }


class CompleteGDN(torch.nn.Module):
    def __init__(
        self,
        preprocessor: torch.nn.Module,
        recurrence: torch.nn.Module,
        norm_weight: torch.Tensor,
        out_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.recurrence = recurrence
        self.register_buffer("norm_weight", norm_weight.reshape(1, 1, 1, 128))
        self.out_proj = torch.nn.Conv2d(2048, 2048, 1, bias=False)
        with torch.no_grad():
            self.out_proj.weight.copy_(out_weight[:, :, None, None])
        self.half().eval()

    @staticmethod
    def channels_to_heads(value: torch.Tensor) -> torch.Tensor:
        # [B, H*D, 1, S] -> [B, H, S, D]
        return value.reshape(1, 16, 128, value.shape[-1]).permute(0, 1, 3, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query, key, value, z, g, beta = self.preprocessor(hidden_states)
        query = self.channels_to_heads(query)
        key = self.channels_to_heads(key)
        value = self.channels_to_heads(value)
        z = self.channels_to_heads(z)
        recurrence_output, _ = self.recurrence(
            query,
            key,
            value,
            g[:, :, 0, :],
            beta[:, :, 0, :],
        )

        variance = torch.mean(recurrence_output * recurrence_output, dim=-1, keepdim=True)
        normalized = recurrence_output * torch.rsqrt(variance + 1e-6)
        normalized = normalized * self.norm_weight
        normalized = normalized * torch.nn.functional.silu(z)
        channels = normalized.permute(0, 1, 3, 2).reshape(
            1, 2048, 1, hidden_states.shape[-1]
        )
        return self.out_proj(channels)


class FP32ChunkedRecurrence(torch.nn.Module):
    """Keep the numerically sensitive chunk recurrence in explicit FP32."""

    def __init__(self, sequence_length: int) -> None:
        super().__init__()
        self.inner = ChunkedRecurrence(sequence_length)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, state = self.inner(
            query.float(), key.float(), value.float(), g.float(), beta.float()
        )
        return output.half(), state.half()


def load_complete_gdn(
    weights_path: Path,
    layer: int,
    representation: str,
    sequence_length: int,
    valid_tokens: int,
) -> CompleteGDN:
    preprocessor = load_preprocessor(
        weights_path,
        layer,
        sensitive_activations_fp32=representation == "chunked_fp32_sensitive",
        decomposed_silu=representation == "chunked_fp32_decomposed",
    )
    if representation == "chunked":
        recurrence = ChunkedRecurrence(sequence_length)
    elif representation in (
        "chunked_fp32",
        "chunked_fp32_decomposed",
        "chunked_fp32_sensitive",
    ):
        recurrence = FP32ChunkedRecurrence(sequence_length)
    else:
        recurrence = UnrolledRecurrence(sequence_length, valid_tokens)
    prefix = f"model.layers.{layer}.linear_attn"
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        norm = handle.get_tensor(f"{prefix}.norm.weight").float()
        out = handle.get_tensor(f"{prefix}.out_proj.weight").float()
    return CompleteGDN(preprocessor, recurrence, norm, out)


def load_real_input(
    oracle_path: Path, sequence_length: int
) -> tuple[torch.Tensor, int, np.ndarray]:
    with np.load(oracle_path) as data:
        hidden = np.asarray(data["gdn_input"], dtype=np.float32)
        expected = np.asarray(data["gdn_output"], dtype=np.float32)
    valid = hidden.shape[1]
    if valid > sequence_length:
        raise ValueError(f"oracle length {valid} exceeds model length {sequence_length}")
    padded = np.zeros((1, hidden.shape[2], 1, sequence_length), dtype=np.float16)
    padded[:, :, 0, :valid] = hidden.transpose(0, 2, 1).astype(np.float16)
    expected = expected.transpose(0, 2, 1)[:, :, None, :]
    return torch.from_numpy(padded), valid, expected


def convert_model(
    module: CompleteGDN,
    sample: torch.Tensor,
    package_path: Path,
    preserve_recurrence_fp32: bool = False,
) -> None:
    with torch.inference_mode():
        traced = torch.jit.trace(module, sample, strict=True)
        traced = torch.jit.freeze(traced.eval())
    if preserve_recurrence_fp32:
        compute_precision: Any = ct.transform.FP16ComputePrecision(
            op_selector=lambda op: not any(
                scope in str(op.scopes)
                for scope in ("recurrence", "qkv_activation", "gate_activation")
            )
        )
    else:
        compute_precision = ct.precision.FLOAT16
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="hidden_states", shape=tuple(sample.shape), dtype=np.float16
            )
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        compute_precision=compute_precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    converted.short_description = "One complete real WeMM-Embedding-2B Qwen3.5 GDN layer"
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
    timings: list[float] = []
    output: np.ndarray | None = None
    for _ in range(runs):
        started = time.perf_counter_ns()
        prediction = model.predict({"hidden_states": sample})
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
        output = np.asarray(prediction["output"], dtype=np.float32)
    assert output is not None
    result = {
        "policy": policy_name,
        "load_ms": load_ms,
        "median_ms": statistics.median(timings),
        "p95_ms": percentile(timings, 95),
        "minimum_ms": min(timings),
        "maximum_ms": max(timings),
        "runs": runs,
    }
    del model
    gc.collect()
    return result, output


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample, valid_tokens, expected = load_real_input(
        args.oracle.resolve(), args.sequence_length
    )
    module = load_complete_gdn(
        args.weights.resolve(),
        args.layer,
        args.representation,
        args.sequence_length,
        valid_tokens,
    )
    sample_array = sample.numpy()
    with torch.inference_mode():
        started = time.perf_counter_ns()
        torch_output = module(sample).float().numpy()
        torch_ms = (time.perf_counter_ns() - started) / 1_000_000.0

    package_path = output_dir / (
        f"WeMM2B-Layer{args.layer}-CompleteGDN-{args.representation}"
        f"-S{args.sequence_length}.mlpackage"
    )
    if args.force_convert or not package_path.exists():
        convert_model(
            module,
            sample,
            package_path,
            preserve_recurrence_fp32=args.representation
            in (
                "chunked_fp32",
                "chunked_fp32_decomposed",
                "chunked_fp32_sensitive",
            ),
        )

    policies: dict[str, dict[str, Any]] = {}
    outputs: dict[str, np.ndarray] = {}
    for policy_name in POLICIES:
        print(f"benchmarking {policy_name}...", flush=True)
        result, output = benchmark_policy(
            package_path, policy_name, sample_array, args.warmup, args.runs
        )
        policies[policy_name] = result
        outputs[policy_name] = output

    expected_valid = expected[..., :valid_tokens]
    report = {
        "experiment": {
            "weights": str(args.weights.resolve()),
            "oracle": str(args.oracle.resolve()),
            "layer": args.layer,
            "representation": args.representation,
            "sequence_length": args.sequence_length,
            "valid_tokens": valid_tokens,
            "coreml_package": str(package_path),
            "precision": "source BF16 -> Core ML FP16",
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
        "torch_fp16_first_run_ms": torch_ms,
        "torch_fp16_agreement_vs_bf16_oracle": agreement(
            torch_output[..., :valid_tokens], expected_valid
        ),
        "policies": policies,
        "coreml_agreement_vs_bf16_oracle": {
            policy: agreement(output[..., :valid_tokens], expected_valid)
            for policy, output in outputs.items()
        },
        "coreml_agreement_vs_torch_fp16": {
            policy: agreement(output[..., :valid_tokens], torch_output[..., :valid_tokens])
            for policy, output in outputs.items()
        },
    }
    report_path = output_dir / f"complete_gdn_{args.representation}_benchmark.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report_path)


if __name__ == "__main__":
    main()
