#!/usr/bin/env python3
"""Compare MLX and a dynamic-RoPE Core ML segment on the mixed query set."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_coreml_block_integration import (
    CoreMLDecoderBlock,
    PassthroughDecoderBlock,
    image_uri,
)
from service import MLXWeMMEngine
from validate_mixed_queries import normalize, parse_weights, rank_row, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--segment", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vectors", type=Path)
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--last-layer", type=int, default=3)
    parser.add_argument(
        "--fusion-weights",
        type=parse_weights,
        default=parse_weights("0.5,0.65,0.75"),
    )
    return parser.parse_args()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))


def agreement_summary(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    values = [cosine(a, b) for a, b in zip(left, right)]
    return {
        "count": len(values),
        "minimum_cosine": min(values),
        "median_cosine": statistics.median(values),
        "mean_cosine": statistics.mean(values),
        "maximum_cosine": max(values),
    }


def image_message(path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri(path)}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def text_message(text: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def mixed_message(path: Path, text: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri(path)}},
                {"type": "text", "text": text},
            ],
        }
    ]


def run_embeddings(
    engine: MLXWeMMEngine,
    manifest: dict[str, Any],
    root: Path,
    dimension: int,
) -> dict[str, Any]:
    images = manifest["images"]
    queries = manifest["queries"]
    image_by_id = {item["id"]: item for item in images}
    prompt = str(manifest.get("corpus_prompt") or "Represent this image.")
    started = time.perf_counter()
    corpus_results = [
        engine.embed(image_message(root / item["path"], prompt), dimension=dimension)
        for item in images
    ]
    text_results = [
        engine.embed(text_message(query["text"]), dimension=dimension)
        for query in queries
    ]
    mixed_results = [
        engine.embed(
            mixed_message(
                root / image_by_id[query["reference_image"]]["path"], query["text"]
            ),
            dimension=dimension,
        )
        for query in queries
    ]
    return {
        "corpus": np.asarray([result.vector for result in corpus_results]),
        "text": np.asarray([result.vector for result in text_results]),
        "mixed": np.asarray([result.vector for result in mixed_results]),
        "elapsed_seconds": time.perf_counter() - started,
        "timings": {
            "corpus": [result.timings_ms for result in corpus_results],
            "text": [result.timings_ms for result in text_results],
            "mixed": [result.timings_ms for result in mixed_results],
        },
    }


def evaluate(
    manifest: dict[str, Any],
    vectors: dict[str, Any],
    fusion_weights: list[float],
) -> dict[str, Any]:
    images = manifest["images"]
    queries = manifest["queries"]
    image_ids = [item["id"] for item in images]
    corpus = vectors["corpus"]
    strategies: dict[str, list[np.ndarray]] = {
        "image_only": [
            corpus[image_ids.index(query["reference_image"])] for query in queries
        ],
        "text_only": list(vectors["text"]),
        "native_mixed": list(vectors["mixed"]),
    }
    for text_weight in fusion_weights:
        strategies[f"late_fusion_text_{text_weight:.2f}"] = [
            normalize(
                (1.0 - text_weight)
                * corpus[image_ids.index(query["reference_image"])]
                + text_weight * text_vector
            )
            for query, text_vector in zip(queries, vectors["text"])
        ]
    rows = {
        name: [
            rank_row(query, vector, corpus, image_ids)
            for query, vector in zip(queries, strategy_vectors)
        ]
        for name, strategy_vectors in strategies.items()
    }
    return {
        "summaries": {name: summarize(value) for name, value in rows.items()},
        "top1": {
            name: [value["top1"] for value in strategy_rows]
            for name, strategy_rows in rows.items()
        },
        "rows": rows,
    }


def timing_medians(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: statistics.median(float(row[key]) for row in rows) for key in rows[0]
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    engine = MLXWeMMEngine(
        package_path=args.package.resolve(),
        image_size=448,
        compute_unit="cpu_ane",
        warmup=False,
    )
    baseline = run_embeddings(
        engine, manifest, manifest_path.parent, args.dimension
    )

    original_layers = list(
        engine.language.model.layers[args.first_layer : args.last_layer + 1]
    )
    # ``pipeline_layers`` is a slice copy in MLX-LM; mutate the owning list.
    text_layers = engine.text_language.language_model.model.layers
    original_text_layers = list(text_layers[args.first_layer : args.last_layer + 1])
    rotary_provider = next(
        original.self_attn.rotary_emb
        for original in original_layers
        if not original.is_linear
    )
    coreml_segment = CoreMLDecoderBlock(
        args.segment.resolve(),
        args.sequence_length,
        original_layers[0],
        rotary_provider,
    )
    engine.language.model.layers[args.first_layer] = coreml_segment
    for offset, original in enumerate(original_layers[1:], start=1):
        engine.language.model.layers[args.first_layer + offset] = (
            PassthroughDecoderBlock(original)
        )
    text_layers[args.first_layer] = coreml_segment
    for offset, original in enumerate(original_text_layers[1:], start=1):
        text_layers[args.first_layer + offset] = PassthroughDecoderBlock(original)
    hybrid = run_embeddings(engine, manifest, manifest_path.parent, args.dimension)
    for offset, original in enumerate(original_layers):
        engine.language.model.layers[args.first_layer + offset] = original
    for offset, original in enumerate(original_text_layers):
        text_layers[args.first_layer + offset] = original

    baseline_eval = evaluate(manifest, baseline, args.fusion_weights)
    hybrid_eval = evaluate(manifest, hybrid, args.fusion_weights)
    payload = {
        "experiment": {
            "package": str(args.package.resolve()),
            "segment": str(args.segment.resolve()),
            "manifest": str(manifest_path),
            "dimension": args.dimension,
            "first_layer": args.first_layer,
            "last_layer": args.last_layer,
            "fixed_sequence_length": args.sequence_length,
            "coreml_calls": len(coreml_segment.calls),
            "minimum_tokens": int(min(row["tokens"] for row in coreml_segment.calls)),
            "maximum_tokens": int(max(row["tokens"] for row in coreml_segment.calls)),
        },
        "elapsed_seconds": {
            "baseline_mlx": baseline["elapsed_seconds"],
            "hybrid_coreml_segment": hybrid["elapsed_seconds"],
            "speedup": baseline["elapsed_seconds"] / hybrid["elapsed_seconds"],
        },
        "vector_agreement": {
            name: agreement_summary(baseline[name], hybrid[name])
            for name in ("corpus", "text", "mixed")
        },
        "latency_median_ms": {
            mode: {
                kind: timing_medians(values["timings"][kind])
                for kind in ("corpus", "text", "mixed")
            }
            for mode, values in (("baseline_mlx", baseline), ("hybrid", hybrid))
        },
        "retrieval": {
            "baseline_mlx": baseline_eval,
            "hybrid_coreml_segment": hybrid_eval,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.vectors:
        args.vectors.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.vectors,
            baseline_corpus=baseline["corpus"],
            baseline_text=baseline["text"],
            baseline_mixed=baseline["mixed"],
            hybrid_corpus=hybrid["corpus"],
            hybrid_text=hybrid["text"],
            hybrid_mixed=hybrid["mixed"],
        )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
