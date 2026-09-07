#!/usr/bin/env python3
"""Convert and benchmark one real WeMM Qwen3.5 full-attention decoder block."""

from __future__ import annotations

import argparse
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
from benchmark_coreml_decoder_block import RMSNormChannels, agreement
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
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument(
        "--representation", choices=("fp16", "fp32_softmax"), default="fp16"
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--force-convert", action="store_true")
    return parser.parse_args()


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class FullAttention(torch.nn.Module):
    def __init__(
        self,
        weights: Path,
        layer: int,
        sequence_length: int,
        position_cos: np.ndarray,
        position_sin: np.ndarray,
        fp32_softmax: bool,
    ) -> None:
        super().__init__()
        self.num_heads = 8
        self.num_kv_heads = 2
        self.head_dim = 256
        self.fp32_softmax = fp32_softmax
        prefix = f"model.layers.{layer}.self_attn"
        self.q_proj = torch.nn.Conv2d(2048, 4096, 1, bias=False)
        self.k_proj = torch.nn.Conv2d(2048, 512, 1, bias=False)
        self.v_proj = torch.nn.Conv2d(2048, 512, 1, bias=False)
        self.o_proj = torch.nn.Conv2d(2048, 2048, 1, bias=False)
        with safe_open(weights, framework="pt", device="cpu") as handle:
            with torch.no_grad():
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    target = getattr(self, name)
                    value = handle.get_tensor(f"{prefix}.{name}.weight").float()
                    target.weight.copy_(value[:, :, None, None])
            q_norm = handle.get_tensor(f"{prefix}.q_norm.weight").float()
            k_norm = handle.get_tensor(f"{prefix}.k_norm.weight").float()
        self.register_buffer("q_norm_weight", q_norm.reshape(1, 1, 1, -1))
        self.register_buffer("k_norm_weight", k_norm.reshape(1, 1, 1, -1))

        cos = np.ones((1, 1, sequence_length, 64), dtype=np.float32)
        sin = np.zeros((1, 1, sequence_length, 64), dtype=np.float32)
        valid = position_cos.shape[1]
        cos[:, :, :valid, :] = position_cos[:, None, :, :]
        sin[:, :, :valid, :] = position_sin[:, None, :, :]
        self.register_buffer("position_cos", torch.from_numpy(cos))
        self.register_buffer("position_sin", torch.from_numpy(sin))
        mask = torch.triu(
            torch.full((sequence_length, sequence_length), -10_000.0), diagonal=1
        )
        self.register_buffer("causal_mask", mask.reshape(1, 1, sequence_length, sequence_length))
        self.half().eval()

    @staticmethod
    def rms_norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        variance = torch.mean(value * value, dim=-1, keepdim=True)
        return value * torch.rsqrt(variance + 1e-6) * weight

    @staticmethod
    def rotate_half(value: torch.Tensor) -> torch.Tensor:
        left, right = torch.chunk(value, 2, dim=-1)
        return torch.cat((-right, left), dim=-1)

    def apply_rope(
        self,
        value: torch.Tensor,
        position_cos: torch.Tensor | None = None,
        position_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if position_cos is None or position_sin is None:
            position_cos = self.position_cos
            position_sin = self.position_sin
        else:
            # Public Core ML boundary is NCHW: [B, 64, 1, S].
            position_cos = position_cos.permute(0, 2, 3, 1)
            position_sin = position_sin.permute(0, 2, 3, 1)
        rotary = value[..., :64]
        passthrough = value[..., 64:]
        rotary = (
            rotary * position_cos
            + self.rotate_half(rotary) * position_sin
        )
        return torch.cat((rotary, passthrough), dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_cos: torch.Tensor | None = None,
        position_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = hidden_states.shape[0]
        tokens = hidden_states.shape[-1]
        q_projected = self.q_proj(hidden_states).permute(0, 3, 2, 1)
        q_projected = q_projected.reshape(batch, tokens, self.num_heads, 512)
        query, gate = torch.chunk(q_projected, 2, dim=-1)
        gate = gate.reshape(batch, tokens, 2048)

        key = self.k_proj(hidden_states).permute(0, 3, 2, 1)
        key = key.reshape(batch, tokens, self.num_kv_heads, self.head_dim)
        value = self.v_proj(hidden_states).permute(0, 3, 2, 1)
        value = value.reshape(batch, tokens, self.num_kv_heads, self.head_dim)
        query = self.rms_norm(query, self.q_norm_weight).permute(0, 2, 1, 3)
        key = self.rms_norm(key, self.k_norm_weight).permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        query = self.apply_rope(query, position_cos, position_sin)
        key = self.apply_rope(key, position_cos, position_sin)

        key = key[:, :, None, :, :].expand(-1, -1, 4, -1, -1)
        key = key.reshape(batch, self.num_heads, tokens, self.head_dim)
        value = value[:, :, None, :, :].expand(-1, -1, 4, -1, -1)
        value = value.reshape(batch, self.num_heads, tokens, self.head_dim)
        scores = torch.matmul(query, key.transpose(-1, -2)) * 0.0625
        scores = scores + self.causal_mask
        if self.fp32_softmax:
            probabilities = torch.softmax(scores.float(), dim=-1).half()
        else:
            probabilities = torch.softmax(scores, dim=-1)
        output = torch.matmul(probabilities, value)
        output = output.permute(0, 2, 1, 3).reshape(batch, tokens, 2048)
        output = output * torch.sigmoid(gate)
        output = output.permute(0, 2, 1)[:, :, None, :]
        return self.o_proj(output)


class AttentionDecoderBlock(torch.nn.Module):
    def __init__(
        self,
        input_norm: RMSNormChannels,
        attention: FullAttention,
        post_norm: RMSNormChannels,
        mlp: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.input_norm = input_norm
        self.attention = attention
        self.post_norm = post_norm
        self.mlp = mlp
        self.half().eval()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_cos: torch.Tensor | None = None,
        position_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        after_attention = hidden_states + self.attention(
            self.input_norm(hidden_states), position_cos, position_sin
        )
        return after_attention + self.mlp(self.post_norm(after_attention))


def load_block(
    weights: Path,
    oracle: Path,
    layer: int,
    sequence_length: int,
    representation: str,
) -> AttentionDecoderBlock:
    prefix = f"model.layers.{layer}"
    with safe_open(weights, framework="pt", device="cpu") as handle:
        input_weight = handle.get_tensor(f"{prefix}.input_layernorm.weight").float()
        post_weight = handle.get_tensor(f"{prefix}.post_attention_layernorm.weight").float()
    with np.load(oracle) as data:
        position_cos = np.asarray(data["position_cos"], dtype=np.float32)
        position_sin = np.asarray(data["position_sin"], dtype=np.float32)
    return AttentionDecoderBlock(
        RMSNormChannels(input_weight),
        FullAttention(
            weights,
            layer,
            sequence_length,
            position_cos,
            position_sin,
            representation == "fp32_softmax",
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


def convert_model(module: torch.nn.Module, sample: torch.Tensor, package: Path) -> None:
    with torch.inference_mode():
        traced = torch.jit.freeze(torch.jit.trace(module, sample, strict=True).eval())
    model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="hidden_states", shape=tuple(sample.shape), dtype=np.float16
            )
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    model.short_description = "One complete WeMM-Embedding-2B full-attention decoder block"
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
        args.oracle.resolve(),
        args.layer,
        args.sequence_length,
        args.representation,
    )
    with torch.inference_mode():
        started = time.perf_counter_ns()
        torch_output = module(sample).float().numpy()
        torch_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    expected_valid = expected[..., :valid]
    torch_agreement = agreement(torch_output[..., :valid], expected_valid)
    print(f"torch oracle cosine: {torch_agreement['cosine']:.9f}", flush=True)
    package = output_dir / (
        f"WeMM2B-Layer{args.layer}-AttentionBlock-{args.representation}"
        f"-S{args.sequence_length}.mlpackage"
    )
    if args.force_convert or not package.exists():
        convert_model(module, sample, package)

    policies: dict[str, dict[str, float | int]] = {}
    agreements: dict[str, dict[str, float]] = {}
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
            "representation": args.representation,
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
        "torch_fp16_agreement_vs_bf16_oracle": torch_agreement,
        "policies": policies,
        "coreml_agreement_vs_bf16_oracle": agreements,
    }
    report_path = output_dir / f"attention_block_{args.representation}_benchmark.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report_path)


if __name__ == "__main__":
    main()
