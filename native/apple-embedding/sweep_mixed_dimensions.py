#!/usr/bin/env python3
"""Re-evaluate saved 2048-d WeMM mixed-query vectors at Matryoshka prefixes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from validate_mixed_queries import normalize, rank_row, summarize


def parse_dimensions(value: str) -> list[int]:
    dimensions = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not dimensions or any(dimension <= 0 for dimension in dimensions):
        raise argparse.ArgumentTypeError("dimensions 必须是正整数")
    return dimensions


def parse_weights(value: str) -> list[float]:
    weights = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not weights or any(weight < 0 or weight > 1 for weight in weights):
        raise argparse.ArgumentTypeError("fusion weights 必须位于0到1之间")
    return weights


def normalized_prefixes(values: np.ndarray, dimension: int) -> np.ndarray:
    return np.stack([normalize(row[:dimension]) for row in values])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dimensions",
        type=parse_dimensions,
        default=parse_dimensions("64,128,256,512,1024,2048"),
    )
    parser.add_argument(
        "--fusion-weights",
        type=parse_weights,
        default=parse_weights("0.5,0.65,0.75"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    queries = manifest["queries"]
    image_ids = [item["id"] for item in manifest["images"]]
    saved = np.load(args.vectors)
    if list(saved["image_ids"]) != image_ids:
        raise ValueError("vectors image_ids 与 manifest 不一致")
    if list(saved["query_ids"]) != [query["id"] for query in queries]:
        raise ValueError("vectors query_ids 与 manifest 不一致")
    max_dimension = int(saved["corpus"].shape[1])
    if max(args.dimensions) > max_dimension:
        raise ValueError(f"最大可用维度为 {max_dimension}")

    results: dict[str, Any] = {}
    for dimension in args.dimensions:
        corpus = normalized_prefixes(saved["corpus"], dimension)
        text = normalized_prefixes(saved["text"], dimension)
        mixed = normalized_prefixes(saved["native_mixed"], dimension)
        image = np.stack(
            [corpus[image_ids.index(query["reference_image"])] for query in queries]
        )
        strategies: dict[str, np.ndarray] = {
            "image_only": image,
            "text_only": text,
            "native_mixed": mixed,
        }
        for text_weight in args.fusion_weights:
            strategies[f"late_fusion_text_{text_weight:.2f}"] = np.stack(
                [
                    normalize((1.0 - text_weight) * image_row + text_weight * text_row)
                    for image_row, text_row in zip(image, text)
                ]
            )
        rows = {
            name: [
                rank_row(query, vector, corpus, image_ids)
                for query, vector in zip(queries, vectors)
            ]
            for name, vectors in strategies.items()
        }
        summaries = {name: summarize(value) for name, value in rows.items()}
        fusion_names = [name for name in summaries if name.startswith("late_fusion_")]
        best_fusion = max(
            fusion_names,
            key=lambda name: (
                summaries[name]["accuracy"],
                summaries[name]["mean_reciprocal_rank"],
                summaries[name]["mean_target_margin"],
            ),
        )
        results[str(dimension)] = {
            "strategies": summaries,
            "best_late_fusion": best_fusion,
            "native_failures": [row for row in rows["native_mixed"] if not row["correct"]],
        }

    report = {
        "dataset": manifest["name"],
        "vectors": str(args.vectors.resolve()),
        "source_dimension": max_dimension,
        "dimensions": args.dimensions,
        "results": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
