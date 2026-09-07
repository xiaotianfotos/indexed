#!/usr/bin/env python3
"""Test public-Core-ML representations of WeMM's gated-delta recurrence."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


POLICIES = {
    "cpu_only": ct.ComputeUnit.CPU_ONLY,
    "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu_ane": ct.ComputeUnit.CPU_AND_NE,
    "all": ct.ComputeUnit.ALL,
}
INPUT_NAMES = ("query", "key", "value", "g", "beta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--representation", choices=("while", "unrolled", "chunked"), default="while"
    )
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument(
        "--valid-tokens",
        type=int,
        default=None,
        help="constant loop bound; defaults to the oracle sequence length",
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


def load_oracle(
    oracle_path: Path, sequence_length: int, valid_override: int | None
) -> tuple[dict[str, np.ndarray], int, dict[str, np.ndarray]]:
    with np.load(oracle_path) as data:
        query = np.asarray(data["recurrent_query"], dtype=np.float32)
        valid = query.shape[1] if valid_override is None else valid_override
        if valid > query.shape[1]:
            raise ValueError(f"valid tokens {valid} exceeds oracle length {query.shape[1]}")
        if valid > sequence_length:
            raise ValueError(f"valid tokens {valid} exceeds model length {sequence_length}")
        heads, dimension = query.shape[2:]
        inputs: dict[str, np.ndarray] = {}
        for name in ("query", "key", "value"):
            source = np.asarray(data[f"recurrent_{name}"], dtype=np.float32)
            padded = np.zeros((1, heads, sequence_length, dimension), dtype=np.float16)
            padded[:, :, :valid, :] = source[:, :valid].transpose(0, 2, 1, 3).astype(
                np.float16
            )
            inputs[name] = padded
        for name in ("g", "beta"):
            source = np.asarray(data[f"recurrent_{name}"], dtype=np.float32)
            padded = np.zeros((1, heads, sequence_length), dtype=np.float16)
            padded[:, :, :valid] = source[:, :valid].transpose(0, 2, 1).astype(np.float16)
            inputs[name] = padded

        expected_output = np.asarray(data["recurrent_output"], dtype=np.float32)[
            :, :valid
        ].transpose(0, 2, 1, 3)
        expected_state = np.asarray(data["recurrent_final_state"], dtype=np.float32)
    return inputs, valid, {"output": expected_output, "final_state": expected_state}


def sequential_reference(
    inputs: dict[str, np.ndarray], valid_tokens: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Float32 recurrence matching Transformers' simple reference algorithm."""

    query = inputs["query"].astype(np.float32)[:, :, :valid_tokens]
    key = inputs["key"].astype(np.float32)[:, :, :valid_tokens]
    value = inputs["value"].astype(np.float32)[:, :, :valid_tokens]
    g = inputs["g"].astype(np.float32)[:, :, :valid_tokens]
    beta = inputs["beta"].astype(np.float32)[:, :, :valid_tokens]
    query /= np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), 1e-6)
    key /= np.maximum(np.linalg.norm(key, axis=-1, keepdims=True), 1e-6)
    query *= 1.0 / math.sqrt(query.shape[-1])

    state = np.zeros(
        (query.shape[0], query.shape[1], query.shape[-1], value.shape[-1]),
        dtype=np.float32,
    )
    output = np.empty_like(value)
    started = time.perf_counter_ns()
    for index in range(valid_tokens):
        q_t = query[:, :, index]
        k_t = key[:, :, index]
        v_t = value[:, :, index]
        state *= np.exp(g[:, :, index])[:, :, None, None]
        kv_memory = np.einsum("bhk,bhkv->bhv", k_t, state, optimize=True)
        delta = (v_t - kv_memory) * beta[:, :, index, None]
        state += k_t[:, :, :, None] * delta[:, :, None, :]
        output[:, :, index] = np.einsum(
            "bhk,bhkv->bhv", q_t, state, optimize=True
        )
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return output, state, elapsed_ms


def normalized_vectors(query: Any, key: Any) -> tuple[Any, Any]:
    epsilon = np.float16(1e-6)
    q_square = mb.mul(x=query, y=query, name="q_square")
    q_norm = mb.rsqrt(
        x=mb.add(
            x=mb.reduce_sum(x=q_square, axes=[3], keep_dims=True),
            y=epsilon,
        ),
        name="q_inverse_norm",
    )
    k_square = mb.mul(x=key, y=key, name="k_square")
    k_norm = mb.rsqrt(
        x=mb.add(
            x=mb.reduce_sum(x=k_square, axes=[3], keep_dims=True),
            y=epsilon,
        ),
        name="k_inverse_norm",
    )
    query = mb.mul(
        x=mb.mul(x=query, y=q_norm),
        y=np.float16(1.0 / math.sqrt(128.0)),
        name="normalized_scaled_query",
    )
    key = mb.mul(x=key, y=k_norm, name="normalized_key")
    return query, key


def build_while_program(sequence_length: int, valid_tokens: int) -> Any:
    specs = [
        mb.TensorSpec(shape=(1, 16, sequence_length, 128), dtype=types.fp16),
        mb.TensorSpec(shape=(1, 16, sequence_length, 128), dtype=types.fp16),
        mb.TensorSpec(shape=(1, 16, sequence_length, 128), dtype=types.fp16),
        mb.TensorSpec(shape=(1, 16, sequence_length), dtype=types.fp16),
        mb.TensorSpec(shape=(1, 16, sequence_length), dtype=types.fp16),
    ]

    @mb.program(input_specs=specs, opset_version=ct.target.iOS18)
    def program(query: Any, key: Any, value: Any, g: Any, beta: Any) -> Any:
        query, key = normalized_vectors(query, key)
        decay = mb.exp(x=g, name="decay")
        initial_index = mb.const(val=np.int32(0), name="initial_index")
        initial_state = mb.const(
            val=np.zeros((1, 16, 128, 128), dtype=np.float16),
            name="initial_state",
        )
        initial_output = mb.const(
            val=np.zeros((1, 16, sequence_length, 128), dtype=np.float16),
            name="initial_output",
        )

        def condition(index: Any, _state: Any, _output: Any) -> Any:
            return mb.less(x=index, y=np.int32(valid_tokens))

        def body(index: Any, state: Any, output: Any) -> tuple[Any, Any, Any]:
            q_t = mb.expand_dims(
                x=mb.gather(x=query, indices=index, axis=2), axes=[2]
            )
            k_t = mb.expand_dims(
                x=mb.gather(x=key, indices=index, axis=2), axes=[2]
            )
            v_t = mb.expand_dims(
                x=mb.gather(x=value, indices=index, axis=2), axes=[2]
            )
            decay_t = mb.expand_dims(
                x=mb.gather(x=decay, indices=index, axis=2), axes=[2, 3]
            )
            beta_t = mb.expand_dims(
                x=mb.gather(x=beta, indices=index, axis=2), axes=[2, 3]
            )
            state = mb.mul(x=state, y=decay_t)
            memory = mb.matmul(x=k_t, y=state)
            delta = mb.mul(x=mb.sub(x=v_t, y=memory), y=beta_t)
            state = mb.add(
                x=state,
                y=mb.matmul(
                    x=mb.transpose(x=k_t, perm=[0, 1, 3, 2]), y=delta
                ),
            )
            output_t = mb.matmul(x=q_t, y=state)
            next_index = mb.add(x=index, y=np.int32(1))
            begin = mb.concat(
                values=(np.array([0, 0], dtype=np.int32), mb.expand_dims(x=index, axes=[0]), np.array([0], dtype=np.int32)),
                axis=0,
            )
            end = mb.concat(
                values=(np.array([1, 16], dtype=np.int32), mb.expand_dims(x=next_index, axes=[0]), np.array([128], dtype=np.int32)),
                axis=0,
            )
            output = mb.slice_update(
                x=output,
                update=output_t,
                begin=begin,
                end=end,
                name="write_output",
            )
            return next_index, state, output

        _, final_state, output = mb.while_loop(
            _cond=condition,
            _body=body,
            loop_vars=(initial_index, initial_state, initial_output),
            name="gated_delta_loop",
        )
        return (
            mb.identity(x=output, name="output"),
            mb.identity(x=final_state, name="final_state"),
        )

    return program


class UnrolledRecurrence(torch.nn.Module):
    def __init__(self, sequence_length: int, valid_tokens: int) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.valid_tokens = valid_tokens

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = query * torch.rsqrt(torch.sum(query * query, dim=-1, keepdim=True) + 1e-6)
        query = query * (1.0 / math.sqrt(128.0))
        key = key * torch.rsqrt(torch.sum(key * key, dim=-1, keepdim=True) + 1e-6)
        decay = torch.exp(g)
        state = torch.zeros((1, 16, 128, 128), dtype=query.dtype, device=query.device)
        outputs: list[torch.Tensor] = []
        for index in range(self.valid_tokens):
            q_t = query[:, :, index : index + 1]
            k_t = key[:, :, index : index + 1]
            v_t = value[:, :, index : index + 1]
            state = state * decay[:, :, index, None, None]
            memory = torch.matmul(k_t, state)
            delta = (v_t - memory) * beta[:, :, index, None, None]
            state = state + torch.matmul(k_t.transpose(-1, -2), delta)
            outputs.append(torch.matmul(q_t, state))
        if self.valid_tokens < self.sequence_length:
            outputs.append(
                torch.zeros(
                    (1, 16, self.sequence_length - self.valid_tokens, 128),
                    dtype=query.dtype,
                    device=query.device,
                )
            )
        return torch.cat(outputs, dim=2), state


class ChunkedRecurrence(torch.nn.Module):
    """Transformers' 64-token parallel gated-delta algorithm, statically traced."""

    def __init__(self, sequence_length: int, chunk_size: int = 64) -> None:
        super().__init__()
        if sequence_length % chunk_size:
            raise ValueError("chunked representation requires a multiple of 64")
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        lower_strict = torch.tril(torch.ones(chunk_size, chunk_size), diagonal=-1)
        lower_inclusive = torch.tril(torch.ones(chunk_size, chunk_size), diagonal=0)
        self.register_buffer("lower_strict", lower_strict)
        self.register_buffer("lower_inclusive", lower_inclusive)
        self.register_buffer("identity", torch.eye(chunk_size))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = query * torch.rsqrt(torch.sum(query * query, dim=-1, keepdim=True) + 1e-6)
        query = query * (1.0 / math.sqrt(128.0))
        key = key * torch.rsqrt(torch.sum(key * key, dim=-1, keepdim=True) + 1e-6)
        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)

        chunks = self.sequence_length // self.chunk_size
        query = query.reshape(1, 16, chunks, self.chunk_size, 128)
        key = key.reshape(1, 16, chunks, self.chunk_size, 128)
        value = value.reshape(1, 16, chunks, self.chunk_size, 128)
        v_beta = v_beta.reshape(1, 16, chunks, self.chunk_size, 128)
        k_beta = k_beta.reshape(1, 16, chunks, self.chunk_size, 128)
        g = torch.cumsum(g.reshape(1, 16, chunks, self.chunk_size), dim=-1)

        decay_difference = (g.unsqueeze(-1) - g.unsqueeze(-2)) * self.lower_inclusive
        decay_mask = torch.exp(decay_difference) * self.lower_inclusive
        raw_attn = -torch.matmul(k_beta, key.transpose(-1, -2))
        raw_attn = raw_attn * decay_mask * self.lower_strict

        # The reference implementation updates one lower-triangular row at a
        # time.  Building a fresh row list avoids in-place slice_update ops and
        # lets Core ML see the surrounding 64x64 matrix multiplications.
        rows: list[torch.Tensor] = [raw_attn[..., 0:1, :] * 0.0]
        for index in range(1, self.chunk_size):
            previous = torch.cat(rows, dim=-2)
            prefix = raw_attn[..., index : index + 1, :index]
            prefix = prefix + torch.matmul(prefix, previous[..., :index])
            suffix = raw_attn[..., index : index + 1, index:] * 0.0
            rows.append(torch.cat((prefix, suffix), dim=-1))
        attn = torch.cat(rows, dim=-2) + self.identity
        value = torch.matmul(attn, v_beta)
        k_cumdecay = torch.matmul(
            attn, k_beta * torch.exp(g).unsqueeze(-1)
        )

        state = torch.zeros((1, 16, 128, 128), dtype=query.dtype, device=query.device)
        outputs: list[torch.Tensor] = []
        for index in range(chunks):
            q_i = query[:, :, index]
            k_i = key[:, :, index]
            v_i = value[:, :, index]
            local_attn = torch.matmul(q_i, k_i.transpose(-1, -2))
            local_attn = local_attn * decay_mask[:, :, index] * self.lower_inclusive
            v_prime = torch.matmul(k_cumdecay[:, :, index], state)
            v_new = v_i - v_prime
            inter = torch.matmul(
                q_i * torch.exp(g[:, :, index]).unsqueeze(-1), state
            )
            outputs.append(inter + torch.matmul(local_attn, v_new))
            last_g = g[:, :, index, -1:]
            state = state * torch.exp(last_g).unsqueeze(-1)
            decayed_key = k_i * torch.exp(last_g - g[:, :, index]).unsqueeze(-1)
            state = state + torch.matmul(decayed_key.transpose(-1, -2), v_new)
        return torch.cat(outputs, dim=2), state


def convert_model(
    representation: str,
    inputs: dict[str, np.ndarray],
    sequence_length: int,
    valid_tokens: int,
    package_path: Path,
) -> None:
    if representation == "while":
        source = build_while_program(sequence_length, valid_tokens)
    else:
        if representation == "unrolled":
            module = UnrolledRecurrence(sequence_length, valid_tokens)
        else:
            module = ChunkedRecurrence(sequence_length)
        module = module.half().eval()
        samples = tuple(torch.from_numpy(inputs[name]) for name in INPUT_NAMES)
        with torch.inference_mode():
            source = torch.jit.trace(module, samples, strict=True)
            source = torch.jit.freeze(source.eval())
    converted = ct.convert(
        source,
        convert_to="mlprogram",
        inputs=None
        if representation == "while"
        else [
            ct.TensorType(name=name, shape=inputs[name].shape, dtype=np.float16)
            for name in INPUT_NAMES
        ],
        outputs=None
        if representation == "while"
        else [
            ct.TensorType(name="output", dtype=np.float16),
            ct.TensorType(name="final_state", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    converted.short_description = (
        f"WeMM-Embedding-2B gated-delta recurrence ({representation}, S={sequence_length})"
    )
    converted.save(str(package_path))


def benchmark_policy(
    package_path: Path,
    policy_name: str,
    inputs: dict[str, np.ndarray],
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
        model.predict(inputs)
    timings: list[float] = []
    outputs: dict[str, np.ndarray] = {}
    for _ in range(runs):
        started = time.perf_counter_ns()
        prediction = model.predict(inputs)
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
        outputs = {
            "output": np.asarray(prediction["output"], dtype=np.float32),
            "final_state": np.asarray(prediction["final_state"], dtype=np.float32),
        }
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
    return result, outputs


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs, valid_tokens, oracle = load_oracle(
        args.oracle.resolve(), args.sequence_length, args.valid_tokens
    )
    reference_output, reference_state, reference_ms = sequential_reference(
        inputs, valid_tokens
    )
    reference = {"output": reference_output, "final_state": reference_state}
    package_path = output_dir / (
        f"WeMM2B-GDNRecurrence-{args.representation}-S{args.sequence_length}"
        f"-V{valid_tokens}.mlpackage"
    )
    if args.force_convert or not package_path.exists():
        convert_model(
            args.representation,
            inputs,
            args.sequence_length,
            valid_tokens,
            package_path,
        )

    policies: dict[str, dict[str, Any]] = {}
    policy_outputs: dict[str, dict[str, np.ndarray]] = {}
    for policy_name in POLICIES:
        print(f"benchmarking {policy_name}...", flush=True)
        result, output = benchmark_policy(
            package_path, policy_name, inputs, args.warmup, args.runs
        )
        policies[policy_name] = result
        policy_outputs[policy_name] = output

    oracle_sliced = {
        "output": oracle["output"][:, :, :valid_tokens],
        "final_state": oracle["final_state"],
    }
    report = {
        "experiment": {
            "oracle": str(args.oracle.resolve()),
            "representation": args.representation,
            "sequence_length": args.sequence_length,
            "valid_tokens": valid_tokens,
            "coreml_package": str(package_path),
            "precision": "Core ML FP16 state; float32 NumPy sequential reference",
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
        "numpy_sequential_ms": reference_ms,
        "numpy_sequential_agreement_vs_chunk_oracle": {
            name: agreement(reference[name], oracle_sliced[name])
            for name in ("output", "final_state")
        },
        "policies": policies,
        "coreml_agreement_vs_sequential_reference": {
            policy: {
                "output": agreement(
                    values["output"][:, :, :valid_tokens], reference["output"]
                ),
                "final_state": agreement(values["final_state"], reference["final_state"]),
            }
            for policy, values in policy_outputs.items()
        },
        "coreml_agreement_vs_chunk_oracle": {
            policy: {
                "output": agreement(
                    values["output"][:, :, :valid_tokens], oracle_sliced["output"]
                ),
                "final_state": agreement(
                    values["final_state"], oracle_sliced["final_state"]
                ),
            }
            for policy, values in policy_outputs.items()
        },
    }
    report_path = output_dir / f"recurrence_{args.representation}_benchmark.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report_path)


if __name__ == "__main__":
    main()
