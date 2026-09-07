#!/usr/bin/env python3
"""Verify all private-ANE GDN layers and find a fallback subset if needed."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_private_ane import _messages, _run
from service import MLXWeMMEngine


GDN_DECODER_LAYERS = [
    layer for layer in range(24) if (layer + 1) % 4 != 0
]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.dot(left, right)
        / (np.linalg.norm(left) * np.linalg.norm(right))
    )


def median(rows: list[dict[str, float]], key: str, discard: int) -> float:
    return float(statistics.median(row[key] for row in rows[discard:]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument(
        "--block-size",
        type=int,
        choices=(2, 4, 8, 16, 32, 64),
        default=8,
    )
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument("--discard", type=int, default=2)
    args = parser.parse_args()
    if args.benchmark_runs <= args.discard:
        parser.error("--benchmark-runs must be greater than --discard")
    return args


def main() -> None:
    args = parse_args()
    messages = _messages(args.image.resolve())
    engine = MLXWeMMEngine(
        package_path=args.package.resolve(),
        image_size=448,
        compute_unit="cpu_ane",
        warmup=False,
    )
    baseline_rows, baseline_vector = _run(
        engine,
        messages,
        dimension=args.dimension,
        runs=args.benchmark_runs,
    )

    root = args.experiment_root.resolve()
    sys.path.insert(0, str(root))
    from wemm_mlx_recurrence_backend import PrivateANEGatedDeltaPrefill
    from mlx_vlm.models.qwen3_5.gated_delta import (
        register_qwen3_5_gated_delta_prefill_backend,
    )

    backend = PrivateANEGatedDeltaPrefill(
        bridge_path=root / ".cache/ane-private-runtime/bridge/libane_bridge.dylib",
        mil_path=root / "results/real_g_safe_c64_specialized.mil",
        max_tokens=args.sequence_length,
        layer_slots=[],
        solve_block_size=args.block_size,
        verify_reference=True,
    )
    register_qwen3_5_gated_delta_prefill_backend(backend)

    def evaluate(slots: list[int]) -> dict[str, Any]:
        backend.set_layer_slots(slots)
        backend.reset_profile()
        rows, vector = _run(engine, messages, dimension=args.dimension, runs=1)
        return {
            "slots": slots,
            "decoder_layers": [GDN_DECODER_LAYERS[slot] for slot in slots],
            "vector_cosine": cosine(baseline_vector, vector),
            "vector_max_absolute_error": float(
                np.max(np.abs(baseline_vector - vector))
            ),
            "language_ms": rows[0]["language"],
            "profile": backend.profile(),
        }

    try:
        individual = [evaluate([slot]) for slot in range(18)]
        ranked = sorted(
            individual,
            key=lambda row: (-row["vector_cosine"], row["slots"][0]),
        )
        selected: list[int] = []
        greedy_trials = []
        for row in ranked:
            candidate = sorted(selected + row["slots"])
            trial = evaluate(candidate)
            trial["accepted"] = trial["vector_cosine"] >= args.min_cosine
            greedy_trials.append(trial)
            if trial["accepted"]:
                selected = candidate

        backend.set_layer_slots(selected)
        backend.reset_profile()
        final_rows, final_vector = _run(
            engine,
            messages,
            dimension=args.dimension,
            runs=args.benchmark_runs,
        )
        final_profile = backend.profile()
    finally:
        register_qwen3_5_gated_delta_prefill_backend(None)
        backend.close()

    payload = {
        "schema_version": 1,
        "scope": "unsupported-private-ane-experiment-not-for-app-store-build",
        "quality_gate": {"minimum_vector_cosine": args.min_cosine},
        "package": str(args.package.resolve()),
        "package_fingerprint": engine.package_fingerprint,
        "image": str(args.image.resolve()),
        "dimension": args.dimension,
        "recurrence_algorithm": "block-forward-substitution-v1",
        "solve_block_size": args.block_size,
        "individual_layer_trials": individual,
        "greedy_trials": greedy_trials,
        "selected": {
            "slots": selected,
            "decoder_layers": [GDN_DECODER_LAYERS[slot] for slot in selected],
            "vector_cosine": cosine(baseline_vector, final_vector),
            "vector_max_absolute_error": float(
                np.max(np.abs(baseline_vector - final_vector))
            ),
            "profile": final_profile,
        },
        "benchmark": {
            "runs": args.benchmark_runs,
            "discard": args.discard,
            "baseline_language_median_ms": median(
                baseline_rows, "language", args.discard
            ),
            "private_ane_language_median_ms": median(
                final_rows, "language", args.discard
            ),
            "baseline_total_median_ms": median(
                baseline_rows, "total", args.discard
            ),
            "private_ane_total_median_ms": median(
                final_rows, "total", args.discard
            ),
            "baseline_rows": baseline_rows,
            "private_ane_rows": final_rows,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
