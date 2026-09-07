#!/usr/bin/env python3
"""Prepare one real WeMM Gated DeltaNet head for the H16G C128 fixture.

WeMM uses 64-token chunks and 128-dimensional heads.  The research compiler's
currently measured DeltaNet geometry is square C128/D128, so this script pads
one real 64-token chunk to 128 rows.  The padded rows have beta=0 and g=0;
therefore they neither update nor decay the state, and q=0 makes their output
zero.  This preserves the first chunk's output and final state while making the
geometry legal for the current compiler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np


PADDED_CHUNK = 128
HEAD_DIMENSION = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--token-start", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--benchmark-runs", type=int, default=20)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_vectors(query: np.ndarray, key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query = query.astype(np.float32, copy=True)
    key = key.astype(np.float32, copy=True)
    query /= np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), 1.0e-6)
    key /= np.maximum(np.linalg.norm(key, axis=-1, keepdims=True), 1.0e-6)
    query *= 1.0 / math.sqrt(HEAD_DIMENSION)
    return query, key


def sequential_reference(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    log_decay: np.ndarray,
    beta: np.ndarray,
    initial_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    state = initial_state.astype(np.float32, copy=True)
    output = np.empty_like(value, dtype=np.float32)
    for token in range(query.shape[0]):
        state *= np.exp(np.float32(log_decay[token]))
        memory = key[token] @ state
        delta = (value[token] - memory) * beta[token]
        state += np.outer(key[token], delta)
        output[token] = query[token] @ state
    return output, state


def agreement(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    left = actual.reshape(-1).astype(np.float64)
    right = expected.reshape(-1).astype(np.float64)
    error = left - right
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return {
        "cosine": float(left @ right / max(denominator, 1.0e-30)),
        "relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(right), 1.0e-30)),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "mean_absolute_error": float(np.mean(np.abs(error))),
    }


def summary(value: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "minimum": float(value.min()),
        "maximum": float(value.max()),
        "mean": float(value.mean()),
        "l2_norm": float(np.linalg.norm(value.astype(np.float64).reshape(-1))),
    }


def write_array(path: Path, value: np.ndarray, dtype: Any) -> None:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    contiguous.tofile(path)


def main() -> None:
    args = parse_args()
    if args.tokens <= 0 or args.tokens > PADDED_CHUNK:
        raise ValueError("tokens must be in [1, 128]")
    oracle_path = args.oracle.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(oracle_path) as oracle:
        source_query = np.asarray(oracle["recurrent_query"], dtype=np.float32)
        source_key = np.asarray(oracle["recurrent_key"], dtype=np.float32)
        source_value = np.asarray(oracle["recurrent_value"], dtype=np.float32)
        source_g = np.asarray(oracle["recurrent_g"], dtype=np.float32)
        source_beta = np.asarray(oracle["recurrent_beta"], dtype=np.float32)
        source_output = np.asarray(oracle["recurrent_output"], dtype=np.float32)

    if source_query.ndim != 4 or source_query.shape[-1] != HEAD_DIMENSION:
        raise ValueError(f"unexpected WeMM query shape: {source_query.shape}")
    if not 0 <= args.head < source_query.shape[2]:
        raise ValueError(f"head {args.head} outside [0, {source_query.shape[2]})")
    token_stop = args.token_start + args.tokens
    if args.token_start < 0 or token_stop > source_query.shape[1]:
        raise ValueError("requested token range exceeds the oracle")

    index = (0, slice(args.token_start, token_stop), args.head)
    raw_query = source_query[index]
    raw_key = source_key[index]
    raw_value = source_value[index]
    raw_g = source_g[0, args.token_start:token_stop, args.head]
    raw_beta = source_beta[0, args.token_start:token_stop, args.head]
    oracle_output = source_output[index]
    normalized_query, normalized_key = normalize_vectors(raw_query, raw_key)

    padded: dict[str, np.ndarray] = {
        "q": np.zeros((PADDED_CHUNK, HEAD_DIMENSION), dtype=np.float16),
        "k": np.zeros((PADDED_CHUNK, HEAD_DIMENSION), dtype=np.float16),
        "v": np.zeros((PADDED_CHUNK, HEAD_DIMENSION), dtype=np.float16),
        "beta": np.zeros((PADDED_CHUNK,), dtype=np.float16),
        "log_decay": np.zeros((PADDED_CHUNK,), dtype=np.float16),
        "state": np.zeros((HEAD_DIMENSION, HEAD_DIMENSION), dtype=np.float16),
    }
    padded["q"][: args.tokens] = normalized_query.astype(np.float16)
    padded["k"][: args.tokens] = normalized_key.astype(np.float16)
    padded["v"][: args.tokens] = raw_value.astype(np.float16)
    padded["beta"][: args.tokens] = raw_beta.astype(np.float16)
    padded["log_decay"][: args.tokens] = raw_g.astype(np.float16)

    output, final_state = sequential_reference(
        *(padded[name].astype(np.float32) for name in ("q", "k", "v", "log_decay", "beta", "state"))
    )
    if not np.all(output[args.tokens:] == 0):
        raise RuntimeError("padded query rows should produce exactly zero output")

    timings_ms: list[float] = []
    for _ in range(args.benchmark_runs):
        started = time.perf_counter_ns()
        sequential_reference(
            *(padded[name].astype(np.float32) for name in ("q", "k", "v", "log_decay", "beta", "state"))
        )
        timings_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)

    for name, value in padded.items():
        write_array(output_dir / f"{name}.f16", value, np.float16)
    write_array(output_dir / "expected_output.f32", output, np.float32)
    write_array(output_dir / "expected_final_state.f32", final_state, np.float32)
    np.savez_compressed(
        output_dir / "reference.npz",
        **padded,
        expected_output=output,
        expected_final_state=final_state,
        oracle_output=oracle_output,
    )

    receipt = {
        "experiment": "real WeMM layer-0 Gated DeltaNet head in H16G C128/D128 fixture",
        "source_oracle": str(oracle_path),
        "source_oracle_sha256": sha256(oracle_path),
        "layer": 0,
        "head": args.head,
        "token_range": [args.token_start, token_stop],
        "real_chunk_tokens": args.tokens,
        "compiled_geometry": {"chunk": PADDED_CHUNK, "head_dimension": HEAD_DIMENSION},
        "padding_contract": {
            "q_k_v": "zero after the real token range",
            "beta": "zero, so padded rows do not update state",
            "log_decay": "zero, so padded rows do not decay state",
            "invariant": "first-64 output and final state equal the 64-token recurrence for the FP16 inputs",
        },
        "input_preprocessing": {
            "query": "L2 normalized then scaled by 1/sqrt(128), cast to FP16",
            "key": "L2 normalized, cast to FP16",
            "value_g_beta": "cast from the captured WeMM tensors to FP16",
            "initial_state": "zero",
        },
        "state_bytes_per_head_fp16": int(HEAD_DIMENSION * HEAD_DIMENSION * 2),
        "state_bytes_all_16_heads_fp16": int(16 * HEAD_DIMENSION * HEAD_DIMENSION * 2),
        "cpu_sequential_reference": {
            "runs": args.benchmark_runs,
            "median_ms": statistics.median(timings_ms),
            "minimum_ms": min(timings_ms),
            "maximum_ms": max(timings_ms),
            "scope": "one padded head/chunk, NumPy FP32 recurrence from FP16 inputs",
        },
        "agreement_fp16_sequential_vs_captured_model_output_first_64": agreement(
            output[: args.tokens], oracle_output
        ),
        "tensors": {
            **{name: summary(value) for name, value in padded.items()},
            "expected_output": summary(output),
            "expected_final_state": summary(final_state),
            "captured_model_output_first_64": summary(oracle_output),
        },
    }
    receipt_path = output_dir / "data_manifest.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(receipt_path)


if __name__ == "__main__":
    main()
