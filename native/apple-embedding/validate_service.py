#!/usr/bin/env python3
"""Run an Indexed-shaped text/image retrieval smoke test against the helper."""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18768")
    parser.add_argument("--model", default="wemm-embedding-2b-apple-256")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def post(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    dimensions: int | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "messages": messages}
    if dimensions is not None:
        body["dimensions"] = dimensions
    payload = json.dumps(body).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def image_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def embedding(response: dict[str, Any]) -> np.ndarray:
    return np.asarray(response["data"][0]["embedding"], dtype=np.float32)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.manifest.resolve().parent
    image_vectors: list[np.ndarray] = []
    text_vectors: list[np.ndarray] = []
    image_timings: list[dict[str, float]] = []
    text_timings: list[dict[str, float]] = []

    for item in manifest["images"]:
        path = root / item["path"]
        response = post(
            args.base_url,
            args.model,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_uri(path)}},
                        {"type": "text", "text": "Represent this image."},
                    ],
                }
            ],
        )
        image_vectors.append(embedding(response))
        image_timings.append(response["indexed"]["timings_ms"])

    instruction = "Find an image or video that best matches the following description:"
    for item in manifest["queries"]:
        response = post(
            args.base_url,
            args.model,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{instruction} {item['text']}"}
                    ],
                }
            ],
        )
        text_vectors.append(embedding(response))
        text_timings.append(response["indexed"]["timings_ms"])

    images = np.stack(image_vectors)
    rows = []
    for item, query in zip(manifest["queries"], text_vectors):
        scores = images @ query
        top = manifest["images"][int(np.argmax(scores))]["id"]
        rows.append(
            {
                "query": item["id"],
                "target": item["target_image"],
                "top1": top,
                "correct": top == item["target_image"],
                "scores": {
                    image["id"]: float(score)
                    for image, score in zip(manifest["images"], scores)
                },
            }
        )

    result: dict[str, Any] = {
        "model": args.model,
        "base_url": args.base_url,
        "dimension": int(images.shape[1]),
        "retrieval": {
            "correct": sum(int(row["correct"]) for row in rows),
            "total": len(rows),
            "accuracy": sum(int(row["correct"]) for row in rows) / len(rows),
            "rows": rows,
        },
        "latency_ms": {
            "images": image_timings,
            "texts": text_timings,
        },
        "measured_at_unix": time.time(),
    }

    if args.baseline:
        baseline = np.load(args.baseline)
        baseline_name = f"vectors_{images.shape[1]}"
        if baseline_name in baseline.files and "keys" in baseline.files:
            expected = {
                str(key): vector
                for key, vector in zip(baseline["keys"], baseline[baseline_name])
            }
        elif {"image_ids", "query_ids", "corpus", "text"}.issubset(baseline.files):
            if baseline["corpus"].shape[1] != images.shape[1]:
                raise SystemExit(
                    f"baseline 维度为 {baseline['corpus'].shape[1]}，服务维度为 {images.shape[1]}"
                )
            expected = {
                **{
                    f"image:{key}": vector
                    for key, vector in zip(baseline["image_ids"], baseline["corpus"])
                },
                **{
                    f"text:{key}": vector
                    for key, vector in zip(baseline["query_ids"], baseline["text"])
                },
            }
        else:
            raise SystemExit(
                "baseline 必须包含 keys/vectors_<dimension>，或 "
                "image_ids/query_ids/corpus/text；"
                f"可用数组: {', '.join(baseline.files)}"
            )
        parity: list[dict[str, Any]] = []
        for item in manifest["images"]:
            key = f"image:{item['id']}"
            actual = embedding(
                post(
                    args.base_url,
                    args.model,
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_uri(root / item["path"])},
                                }
                            ],
                        }
                    ],
                )
            )
            parity.append({"key": key, "cosine": cosine(actual, expected[key])})
        for item in manifest["queries"]:
            key = f"text:{item['id']}"
            actual = embedding(
                post(
                    args.base_url,
                    args.model,
                    [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": item["text"]}],
                        }
                    ],
                )
            )
            parity.append({"key": key, "cosine": cosine(actual, expected[key])})
        result["baseline_parity"] = {
            "source": str(args.baseline.resolve()),
            "minimum_cosine": min(row["cosine"] for row in parity),
            "rows": parity,
        }

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
