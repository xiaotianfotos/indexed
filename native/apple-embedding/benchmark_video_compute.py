#!/usr/bin/env python3
"""Benchmark identical WeMM video embeddings with Core ML ANE and GPU vision."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm)


def request_json(url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        return json.load(response)


def start_helper(binary: Path, package: Path, target: str) -> tuple[subprocess.Popen[str], dict[str, Any]]:
    process = subprocess.Popen(
        [
            str(binary),
            "serve",
            "--package",
            str(package),
            "--port",
            "0",
            "--default-dimension",
            "2048",
            "--vision-compute",
            target,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "MLX_METAL_PREWARM": "1"},
    )
    assert process.stdout is not None
    line = process.stdout.readline()
    if not line:
        stderr = process.stderr.read() if process.stderr is not None else ""
        raise RuntimeError(f"helper failed before ready: {stderr}")
    ready = json.loads(line)
    if ready.get("status") != "ready":
        raise RuntimeError(f"unexpected helper startup output: {ready}")
    return process, ready


def stop_helper(process: subprocess.Popen[str]) -> str:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    return process.stderr.read() if process.stderr is not None else ""


def run_request(base_url: str, body: dict[str, Any]) -> tuple[dict[str, Any], list[float]]:
    started = time.perf_counter()
    response = request_json(f"{base_url}/v1/embeddings", body)
    wall_ms = (time.perf_counter() - started) * 1000
    indexed = response["indexed"]
    timings = indexed["timings_ms"]
    vector = response["data"][0]["embedding"]
    return (
        {
            "wall_ms": wall_ms,
            "preprocess_ms": timings["preprocess"],
            "vision_ms": timings["vision"],
            "language_ms": timings["language"],
            "total_ms": timings["total"],
            "video_frames": int(timings["video_frames"]),
            "prompt_tokens": response["usage"]["prompt_tokens"],
            "dimension": indexed["dimension"],
            "embedding_space": indexed["embedding_space"],
            "backend": indexed["backend"],
        },
        vector,
    )


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("wall_ms", "preprocess_ms", "vision_ms", "language_ms", "total_ms"):
        values = [float(sample[field]) for sample in samples]
        result[field] = {
            "median": statistics.median(values),
            "p95": percentile(values, 0.95),
            "min": min(values),
            "max": max(values),
        }
    first = samples[0]
    for field in ("video_frames", "prompt_tokens", "dimension", "embedding_space", "backend"):
        result[field] = first[field]
    result["videos_per_second_median"] = 1000.0 / result["wall_ms"]["median"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--target-order",
        choices=("ane,gpu", "gpu,ane"),
        default="ane,gpu",
        help="Run both targets in this order so thermal/order bias can be checked.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    arguments = parser.parse_args()

    encoded_video = base64.b64encode(arguments.video.read_bytes()).decode()
    request_body = {
        "model": "wemm-embedding-2b-apple-2048",
        "dimensions": 2048,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{encoded_video}"},
                    },
                    {"type": "text", "text": "Represent this video."},
                ],
            }
        ],
    }

    target_results: dict[str, Any] = {}
    target_vectors: dict[str, list[list[float]]] = {}
    target_order = arguments.target_order.split(",")
    for target in target_order:
        process, ready = start_helper(arguments.binary.resolve(), arguments.package.resolve(), target)
        try:
            base_url = ready["url"].replace("\\/", "/")
            for _ in range(arguments.warmup):
                run_request(base_url, request_body)
            samples: list[dict[str, Any]] = []
            vectors: list[list[float]] = []
            for run_index in range(arguments.runs):
                sample, vector = run_request(base_url, request_body)
                sample["run"] = run_index + 1
                samples.append(sample)
                vectors.append(vector)
                print(
                    f"{target} run {run_index + 1}: wall={sample['wall_ms']:.1f} ms "
                    f"vision={sample['vision_ms']:.1f} ms language={sample['language_ms']:.1f} ms",
                    flush=True,
                )
            health = request_json(f"{base_url}/health")
            target_results[target] = {
                "ready": ready,
                "samples": samples,
                "summary": summarize(samples),
                "health_after": health,
            }
            target_vectors[target] = vectors
        finally:
            stderr = stop_helper(process)
            if stderr:
                target_results.setdefault(target, {})["helper_stderr"] = stderr.strip()

    ane_vectors = target_vectors["ane"]
    gpu_vectors = target_vectors["gpu"]
    ane_median = target_results["ane"]["summary"]
    gpu_median = target_results["gpu"]["summary"]
    comparisons = {
        "ane_repeat_cosine_min": min(cosine(ane_vectors[0], vector) for vector in ane_vectors),
        "gpu_repeat_cosine_min": min(cosine(gpu_vectors[0], vector) for vector in gpu_vectors),
        "ane_vs_gpu_cosine": cosine(ane_vectors[0], gpu_vectors[0]),
        "ane_vs_gpu_max_abs": max(
            abs(left - right) for left, right in zip(ane_vectors[0], gpu_vectors[0], strict=True)
        ),
        "wall_speedup_ane_over_gpu": gpu_median["wall_ms"]["median"]
        / ane_median["wall_ms"]["median"],
        "vision_speedup_ane_over_gpu": gpu_median["vision_ms"]["median"]
        / ane_median["vision_ms"]["median"],
        "wall_ms_saved_by_ane": gpu_median["wall_ms"]["median"]
        - ane_median["wall_ms"]["median"],
        "vision_ms_saved_by_ane": gpu_median["vision_ms"]["median"]
        - ane_median["vision_ms"]["median"],
    }
    reference_metadata: dict[str, Any] | None = None
    if arguments.reference:
        reference = json.loads(arguments.reference.read_text())
        reference_vector = reference.pop("vector")
        reference_metadata = reference
        for target, vectors in target_vectors.items():
            comparisons[f"{target}_vs_official_bf16_cosine"] = cosine(
                vectors[0], reference_vector
            )
            comparisons[f"{target}_vs_official_bf16_max_abs"] = max(
                abs(left - right)
                for left, right in zip(vectors[0], reference_vector, strict=True)
            )
    output = {
        "schema": "indexed-wemm-video-compute-benchmark/v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "fixture": {
            "path": str(arguments.video.resolve()),
            "bytes": arguments.video.stat().st_size,
            "duration_seconds": 10.0,
            "source_fps": 30,
            "sample_fps": 2,
        },
        "configuration": {
            "dimension": 2048,
            "warmup_requests_per_target": arguments.warmup,
            "measured_requests_per_target": arguments.runs,
            "language_backend_both_targets": "mlx-swift-gpu",
            "target_order": target_order,
        },
        "targets": target_results,
        "comparisons": comparisons,
        "official_reference": reference_metadata,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
