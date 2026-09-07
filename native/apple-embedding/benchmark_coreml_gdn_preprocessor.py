#!/usr/bin/env python3
"""Convert and benchmark the real WeMM GDN pre-recurrence subgraph."""

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
OUTPUT_NAMES = ("query", "key", "value", "z", "g", "beta")


def stable_softplus(value: torch.Tensor) -> torch.Tensor:
    """Softplus without the ANE softplus operator's large-positive overflow.

    On the M4, the direct Core ML softplus implementation returned zero for a
    few real gate inputs above roughly 10.  The equivalent identity below keeps
    both exponent and logarithm in their well-conditioned ranges.
    """

    return torch.relu(value) + torch.log1p(torch.exp(-torch.abs(value)))


class FP32SiLU(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(value.float())


class DecomposedSiLU(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.sigmoid(value)


class FP32Gates(torch.nn.Module):
    def forward(
        self,
        b: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        negative_a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        beta = torch.sigmoid(b.float())
        g = negative_a.float() * stable_softplus(a.float() + dt_bias.float())
        return g, beta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--force-convert", action="store_true")
    return parser.parse_args()


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


class GDNPreprocessor(torch.nn.Module):
    """GDN projections, causal depthwise convolution, and decay gates.

    Rank-4 NCHW tensors and 1x1/1x4 convolutions are intentional: they expose
    the large projections and causal convolution in a form Core ML can place
    on the Neural Engine.  The output remains pre-recurrence, so this experiment
    isolates operator placement from the gated-delta state update.
    """

    def __init__(
        self,
        qkv_weight: torch.Tensor,
        z_weight: torch.Tensor,
        b_weight: torch.Tensor,
        a_weight: torch.Tensor,
        conv_weight: torch.Tensor,
        dt_bias: torch.Tensor,
        a_log: torch.Tensor,
        sensitive_activations_fp32: bool = False,
        decomposed_silu: bool = False,
    ) -> None:
        super().__init__()
        conv_dim, hidden = qkv_weight.shape
        value_dim = z_weight.shape[0]
        heads = b_weight.shape[0]
        if conv_dim != value_dim * 3:
            raise ValueError(f"expected equal Q/K/V dimensions, got {conv_dim=} {value_dim=}")
        if a_weight.shape != b_weight.shape or b_weight.shape[1] != hidden:
            raise ValueError("unexpected A/B projection shapes")
        if conv_weight.shape != (conv_dim, 1, 4):
            raise ValueError(f"unexpected causal convolution shape: {conv_weight.shape}")

        self.qkv_proj = torch.nn.Conv2d(hidden, conv_dim, 1, bias=False)
        self.z_proj = torch.nn.Conv2d(hidden, value_dim, 1, bias=False)
        # Fuse the two tiny 16-channel projections.  Besides saving one launch,
        # this avoids a Core ML Torch-frontend dtype bug that can materialize an
        # FP16 zero bias next to a promoted FP32 weight for very small convs.
        self.ab_proj = torch.nn.Conv2d(hidden, heads * 2, 1, bias=False)
        self.causal_conv = torch.nn.Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=(1, 4),
            groups=conv_dim,
            bias=False,
        )
        with torch.no_grad():
            self.qkv_proj.weight.copy_(qkv_weight[:, :, None, None])
            self.z_proj.weight.copy_(z_weight[:, :, None, None])
            self.ab_proj.weight.copy_(
                torch.cat((b_weight, a_weight), dim=0)[:, :, None, None]
            )
            self.causal_conv.weight.copy_(conv_weight[:, :, None, :])

        # The reference performs these two gate calculations in float32.  Here
        # their constants are deliberately folded to FP16 so the whole subgraph
        # remains eligible for ANE.  Agreement with the captured FP32 gate is
        # measured explicitly below.
        self.register_buffer("dt_bias", dt_bias.reshape(1, heads, 1, 1))
        self.register_buffer("negative_a", -torch.exp(a_log).reshape(1, heads, 1, 1))
        if sensitive_activations_fp32:
            self.qkv_activation = FP32SiLU()
        elif decomposed_silu:
            self.qkv_activation = DecomposedSiLU()
        else:
            self.qkv_activation = torch.nn.SiLU()
        self.gate_activation: torch.nn.Module | None = (
            FP32Gates() if sensitive_activations_fp32 else None
        )
        self.value_dim = value_dim
        self.half().eval()

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed_qkv = self.qkv_proj(hidden_states)
        mixed_qkv = torch.nn.functional.pad(mixed_qkv, (3, 0, 0, 0))
        mixed_qkv = self.qkv_activation(self.causal_conv(mixed_qkv))
        query, key, value = torch.split(mixed_qkv, self.value_dim, dim=1)
        z = self.z_proj(hidden_states)
        b, a = torch.split(self.ab_proj(hidden_states), self.dt_bias.shape[1], dim=1)
        if self.gate_activation is None:
            beta = torch.sigmoid(b)
            g = self.negative_a * stable_softplus(a + self.dt_bias)
        else:
            g, beta = self.gate_activation(b, a, self.dt_bias, self.negative_a)
        return query, key, value, z, g, beta


def load_layer(
    weights_path: Path,
    layer: int,
    sensitive_activations_fp32: bool = False,
    decomposed_silu: bool = False,
) -> GDNPreprocessor:
    prefix = f"model.layers.{layer}.linear_attn"
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        qkv = handle.get_tensor(f"{prefix}.in_proj_qkv.weight").float()
        z = handle.get_tensor(f"{prefix}.in_proj_z.weight").float()
        b = handle.get_tensor(f"{prefix}.in_proj_b.weight").float()
        a = handle.get_tensor(f"{prefix}.in_proj_a.weight").float()
        # MLX stores Conv1d as [out, kernel, in/groups].  Restore PyTorch's
        # [out, in/groups, kernel] before copying into a depthwise Conv2d.
        conv = handle.get_tensor(f"{prefix}.conv1d.weight").float().permute(0, 2, 1)
        dt_bias = handle.get_tensor(f"{prefix}.dt_bias").float()
        a_log = handle.get_tensor(f"{prefix}.A_log").float()
    return GDNPreprocessor(
        qkv,
        z,
        b,
        a,
        conv,
        dt_bias,
        a_log,
        sensitive_activations_fp32=sensitive_activations_fp32,
        decomposed_silu=decomposed_silu,
    )


def load_real_input(
    oracle_path: Path, sequence_length: int
) -> tuple[torch.Tensor, int, dict[str, np.ndarray]]:
    with np.load(oracle_path) as data:
        original = np.asarray(data["gdn_input"], dtype=np.float32)
        valid = original.shape[1]
        if original.shape[0] != 1 or original.shape[2] != 2048:
            raise ValueError(f"unexpected GDN input shape: {original.shape}")
        if valid > sequence_length:
            raise ValueError(f"oracle length {valid} exceeds requested length {sequence_length}")
        padded = np.zeros((1, 2048, 1, sequence_length), dtype=np.float16)
        padded[:, :, 0, :valid] = original.transpose(0, 2, 1).astype(np.float16)

        expected: dict[str, np.ndarray] = {}
        for name in ("recurrent_query", "recurrent_key", "recurrent_value"):
            value = np.asarray(data[name], dtype=np.float32)
            value = value.reshape(1, valid, 2048).transpose(0, 2, 1)[:, :, None, :]
            expected[name.removeprefix("recurrent_")] = value
        expected["z"] = np.asarray(data["in_proj_z"], dtype=np.float32).transpose(0, 2, 1)[
            :, :, None, :
        ]
        expected["g"] = np.asarray(data["recurrent_g"], dtype=np.float32).transpose(0, 2, 1)[
            :, :, None, :
        ]
        expected["beta"] = np.asarray(data["recurrent_beta"], dtype=np.float32).transpose(
            0, 2, 1
        )[:, :, None, :]
    return torch.from_numpy(padded), valid, expected


def convert_model(
    module: GDNPreprocessor, sample: torch.Tensor, package_path: Path
) -> None:
    with torch.inference_mode():
        traced = torch.jit.trace(module, sample, strict=True)
        traced = torch.jit.freeze(traced.eval())
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="hidden_states", shape=tuple(sample.shape), dtype=np.float16
            )
        ],
        outputs=[ct.TensorType(name=name, dtype=np.float16) for name in OUTPUT_NAMES],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    converted.short_description = (
        "Real WeMM-Embedding-2B GDN projections, causal convolution, and gates"
    )
    converted.save(str(package_path))


def agreement(
    actual: np.ndarray, expected: np.ndarray, valid_tokens: int
) -> dict[str, float]:
    actual = actual[..., :valid_tokens].reshape(-1).astype(np.float64)
    expected = expected[..., :valid_tokens].reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(actual) * np.linalg.norm(expected)
    return {
        "cosine": float(actual @ expected / denominator),
        "maximum_absolute_error": float(np.max(np.abs(actual - expected))),
        "mean_absolute_error": float(np.mean(np.abs(actual - expected))),
    }


def benchmark_policy(
    package_path: Path,
    policy_name: str,
    sample: np.ndarray,
    warmup: int,
    runs: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
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
    outputs: dict[str, np.ndarray] = {}
    for _ in range(runs):
        started = time.perf_counter_ns()
        prediction = model.predict({"hidden_states": sample})
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
        outputs = {
            name: np.asarray(prediction[name], dtype=np.float32)
            for name in OUTPUT_NAMES
        }
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
    return result, outputs


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    package_path = output_dir / (
        f"WeMM2B-Layer{args.layer}-GDNPre-S{args.sequence_length}.mlpackage"
    )
    module = load_layer(args.weights.resolve(), args.layer)
    sample, valid_tokens, expected = load_real_input(
        args.oracle.resolve(), args.sequence_length
    )
    sample_array = sample.numpy()
    np.save(output_dir / "input.npy", sample_array)

    with torch.inference_mode():
        started = time.perf_counter_ns()
        torch_values = module(sample)
        torch_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    torch_outputs = {
        name: value.float().numpy()
        for name, value in zip(OUTPUT_NAMES, torch_values, strict=True)
    }

    if args.force_convert or not package_path.exists():
        convert_model(module, sample, package_path)

    policy_results: dict[str, dict[str, Any]] = {}
    policy_outputs: dict[str, dict[str, np.ndarray]] = {}
    for policy_name in POLICIES:
        print(f"benchmarking {policy_name}...", flush=True)
        result, outputs = benchmark_policy(
            package_path,
            policy_name,
            sample_array,
            args.warmup,
            args.runs,
        )
        policy_results[policy_name] = result
        policy_outputs[policy_name] = outputs

    payload = {
        "experiment": {
            "weights": str(args.weights.resolve()),
            "oracle": str(args.oracle.resolve()),
            "layer": args.layer,
            "sequence_length": args.sequence_length,
            "valid_tokens": valid_tokens,
            "input_shape": list(sample.shape),
            "coreml_package": str(package_path),
            "precision": "source BF16 -> Core ML FP16",
            "layout": "NCHW Conv2d projections plus causal depthwise 1x4 Conv2d",
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
        "torch_fp16_agreement_vs_bf16_oracle": {
            name: agreement(torch_outputs[name], expected[name], valid_tokens)
            for name in OUTPUT_NAMES
        },
        "policies": policy_results,
        "coreml_agreement_vs_bf16_oracle": {
            policy: {
                name: agreement(outputs[name], expected[name], valid_tokens)
                for name in OUTPUT_NAMES
            }
            for policy, outputs in policy_outputs.items()
        },
        "coreml_agreement_vs_torch_fp16": {
            policy: {
                name: agreement(outputs[name], torch_outputs[name], valid_tokens)
                for name in OUTPUT_NAMES
            }
            for policy, outputs in policy_outputs.items()
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
    report_path = output_dir / "gdn_preprocessor_benchmark.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report_path)


if __name__ == "__main__":
    main()
