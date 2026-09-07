#!/usr/bin/env python3
"""Evaluate native image+text queries and same-space late fusion for WeMM."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from validate_service import embedding, image_uri, post


def parse_weights(value: str) -> list[float]:
    weights = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not weights or any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise argparse.ArgumentTypeError("fusion weights 必须位于0到1之间")
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18768")
    parser.add_argument("--model", default="wemm-embedding-2b-apple-256")
    parser.add_argument(
        "--dimensions",
        type=int,
        help="请求 OpenAI/vLLM 兼容服务返回指定 Matryoshka 维度",
    )
    parser.add_argument(
        "--embedding-space",
        help=(
            "服务未返回 indexed.embedding_space 时必须显式提供；"
            "建议包含模型权重指纹、精度和维度"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("mixed-query-set") / "manifest.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--vectors", type=Path)
    parser.add_argument(
        "--fusion-weights",
        type=parse_weights,
        default=parse_weights("0.5,0.65,0.75"),
        help="文本向量的凸组合权重，逗号分隔",
    )
    parser.add_argument("--min-native-accuracy", type=float)
    return parser.parse_args()


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("不能归一化零向量")
    return vector / norm


def rank_row(
    query: dict[str, Any],
    vector: np.ndarray,
    corpus: np.ndarray,
    image_ids: list[str],
) -> dict[str, Any]:
    scores = corpus @ vector
    order = np.argsort(-scores)
    target_index = image_ids.index(query["target_image"])
    source_index = image_ids.index(query["reference_image"])
    target_rank = int(np.where(order == target_index)[0][0]) + 1
    target_score = float(scores[target_index])
    source_score = float(scores[source_index])
    best_other = max(
        float(score) for index, score in enumerate(scores) if index != target_index
    )
    return {
        "query": query["id"],
        "edit_type": query["edit_type"],
        "language": query["language"],
        "source": query["reference_image"],
        "target": query["target_image"],
        "top1": image_ids[int(order[0])],
        "correct": bool(order[0] == target_index),
        "target_rank": target_rank,
        "reciprocal_rank": 1.0 / target_rank,
        "target_score": target_score,
        "source_score": source_score,
        "target_minus_source": target_score - source_score,
        "target_margin": target_score - best_other,
        "scores": {
            image_id: float(score) for image_id, score in zip(image_ids, scores)
        },
    }


def _basic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "correct": sum(int(row["correct"]) for row in rows),
        "total": len(rows),
        "accuracy": sum(int(row["correct"]) for row in rows) / len(rows),
        "mean_reciprocal_rank": statistics.mean(row["reciprocal_rank"] for row in rows),
        "mean_target_minus_source": statistics.mean(
            row["target_minus_source"] for row in rows
        ),
        "mean_target_margin": statistics.mean(row["target_margin"] for row in rows),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "edit_type": defaultdict(list),
        "language": defaultdict(list),
    }
    for row in rows:
        grouped["edit_type"][row["edit_type"]].append(row)
        grouped["language"][row["language"]].append(row)
    result = _basic_summary(rows)
    result["by_edit_type"] = {
        key: _basic_summary(group) for key, group in grouped["edit_type"].items()
    }
    result["by_language"] = {
        key: _basic_summary(group) for key, group in grouped["language"].items()
    }
    return result


def timing_summary(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: float(statistics.median(row[key] for row in rows)) for key in rows[0]
    }


def timed_post(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    dimensions: int | None,
) -> tuple[dict[str, Any], dict[str, float]]:
    started = time.perf_counter()
    response = post(
        base_url,
        model,
        messages,
        dimensions=dimensions,
    )
    client_total = (time.perf_counter() - started) * 1000.0
    indexed = response.get("indexed")
    if isinstance(indexed, dict) and isinstance(indexed.get("timings_ms"), dict):
        return response, {
            key: float(value) for key, value in indexed["timings_ms"].items()
        }
    return response, {"client_total": client_total}


def response_space(response: dict[str, Any], declared_space: str | None) -> str:
    indexed = response.get("indexed")
    if isinstance(indexed, dict) and indexed.get("embedding_space"):
        return str(indexed["embedding_space"])
    if declared_space:
        return declared_space
    raise RuntimeError(
        "服务没有返回 indexed.embedding_space；请用 --embedding-space 显式声明模型空间"
    )


def checked_embedding(
    response: dict[str, Any],
    *,
    model: str,
    dimensions: int | None,
) -> np.ndarray:
    response_model = response.get("model")
    if response_model is not None and str(response_model) != model:
        raise RuntimeError(f"响应模型不匹配：请求 {model}，返回 {response_model}")
    vector = embedding(response)
    if dimensions is not None and vector.shape != (dimensions,):
        raise RuntimeError(
            f"响应维度不匹配：请求 {dimensions}，返回 {vector.shape}"
        )
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or abs(norm - 1.0) > 1e-3:
        raise RuntimeError(f"响应不是有限的 L2 单位向量：norm={norm}")
    return vector


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    images = manifest["images"]
    queries = manifest["queries"]
    image_ids = [item["id"] for item in images]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("语料图片 ID 重复")

    spaces: set[str] = set()
    corpus_vectors: list[np.ndarray] = []
    corpus_timings: list[dict[str, float]] = []
    corpus_prompt = str(manifest.get("corpus_prompt") or "Represent this image.")
    for item in images:
        response, timing = timed_post(
            args.base_url,
            args.model,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_uri(root / item["path"])},
                        },
                        {"type": "text", "text": corpus_prompt},
                    ],
                }
            ],
            dimensions=args.dimensions,
        )
        corpus_vectors.append(
            checked_embedding(response, model=args.model, dimensions=args.dimensions)
        )
        corpus_timings.append(timing)
        spaces.add(response_space(response, args.embedding_space))
    corpus = np.stack(corpus_vectors)

    text_vectors: list[np.ndarray] = []
    mixed_vectors: list[np.ndarray] = []
    text_timings: list[dict[str, float]] = []
    mixed_timings: list[dict[str, float]] = []
    for query in queries:
        text_response, text_timing = timed_post(
            args.base_url,
            args.model,
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": query["text"]}],
                }
            ],
            dimensions=args.dimensions,
        )
        reference = images[image_ids.index(query["reference_image"])]
        mixed_response, mixed_timing = timed_post(
            args.base_url,
            args.model,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_uri(root / reference["path"])},
                        },
                        {"type": "text", "text": query["text"]},
                    ],
                }
            ],
            dimensions=args.dimensions,
        )
        text_vectors.append(
            checked_embedding(
                text_response, model=args.model, dimensions=args.dimensions
            )
        )
        mixed_vectors.append(
            checked_embedding(
                mixed_response, model=args.model, dimensions=args.dimensions
            )
        )
        text_timings.append(text_timing)
        mixed_timings.append(mixed_timing)
        spaces.add(response_space(text_response, args.embedding_space))
        spaces.add(response_space(mixed_response, args.embedding_space))

    if len(spaces) != 1 or not next(iter(spaces)):
        raise RuntimeError(f"请求没有保持同一个非空 embedding_space：{sorted(spaces)}")

    strategy_vectors: dict[str, list[np.ndarray]] = {
        "image_only": [corpus[image_ids.index(query["reference_image"])] for query in queries],
        "text_only": text_vectors,
        "native_mixed": mixed_vectors,
    }
    for text_weight in args.fusion_weights:
        strategy_vectors[f"late_fusion_text_{text_weight:.2f}"] = [
            normalize(
                (1.0 - text_weight)
                * corpus[image_ids.index(query["reference_image"])]
                + text_weight * text_vector
            )
            for query, text_vector in zip(queries, text_vectors)
        ]

    strategy_rows = {
        name: [
            rank_row(query, vector, corpus, image_ids)
            for query, vector in zip(queries, vectors)
        ]
        for name, vectors in strategy_vectors.items()
    }
    summaries = {name: summarize(rows) for name, rows in strategy_rows.items()}
    fusion_names = [name for name in summaries if name.startswith("late_fusion_")]
    best_fusion = max(
        fusion_names,
        key=lambda name: (
            summaries[name]["accuracy"],
            summaries[name]["mean_reciprocal_rank"],
            summaries[name]["mean_target_margin"],
        ),
    )

    native_rows = strategy_rows["native_mixed"]
    image_rows = strategy_rows["image_only"]
    promotion = [
        native["target_minus_source"] - image["target_minus_source"]
        for native, image in zip(native_rows, image_rows)
    ]
    image_source_correct = sum(
        int(row["top1"] == row["source"]) for row in image_rows
    )

    report: dict[str, Any] = {
        "dataset": manifest["name"],
        "model": args.model,
        "base_url": args.base_url,
        "embedding_space": next(iter(spaces)),
        "dimension": int(corpus.shape[1]),
        "corpus_images": len(images),
        "queries": len(queries),
        "measured_at_unix": time.time(),
        "sanity": {
            "image_only_recovers_reference": image_source_correct,
            "total": len(image_rows),
            "accuracy": image_source_correct / len(image_rows),
        },
        "strategies": summaries,
        "best_late_fusion": best_fusion,
        "native_text_conditioning": {
            "mean_target_minus_source_gain_over_image_only": statistics.mean(
                promotion
            ),
            "positive_gain": sum(int(value > 0) for value in promotion),
            "total": len(promotion),
        },
        "latency_median_ms": {
            "corpus_image": timing_summary(corpus_timings),
            "text_only": timing_summary(text_timings),
            "native_mixed": timing_summary(mixed_timings),
        },
        "rows": strategy_rows,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.vectors:
        args.vectors.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.vectors,
            image_ids=np.asarray(image_ids),
            query_ids=np.asarray([query["id"] for query in queries]),
            corpus=corpus,
            text=np.stack(text_vectors),
            native_mixed=np.stack(mixed_vectors),
        )
    if (
        args.min_native_accuracy is not None
        and summaries["native_mixed"]["accuracy"] < args.min_native_accuracy
    ):
        raise SystemExit(
            "native mixed accuracy "
            f"{summaries['native_mixed']['accuracy']:.3f} < {args.min_native_accuracy:.3f}"
        )


if __name__ == "__main__":
    main()
