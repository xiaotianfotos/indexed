#!/usr/bin/env python3
"""Compare MLX GPU and complete Core ML CPU+ANE language paths on one image/video."""

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

import numpy as np


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True)) / math.sqrt(
        sum(value * value for value in left) * sum(value * value for value in right)
    )


def request_json(url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def start_helper(
    binary: Path,
    package: Path,
    bundles: list[Path],
    cache: Path | None,
    minimum_tokens: int,
) -> tuple[subprocess.Popen[str], dict[str, Any], float]:
    command = [
        str(binary), "serve", "--package", str(package), "--port", "0",
        "--default-dimension", "2048", "--vision-compute", "ane", "--skip-warmup",
    ]
    for bundle in bundles:
        command.extend(("--decoder-bundle", str(bundle)))
    if bundles:
        command.extend(("--decoder-minimum-tokens", str(minimum_tokens)))
    if cache is not None:
        command.extend(("--coreml-cache", str(cache)))
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=binary.parent,
        env={**os.environ, "MLX_METAL_PREWARM": "1"},
    )
    assert process.stdout is not None
    line = process.stdout.readline()
    startup_seconds = time.perf_counter() - started
    if not line:
        stderr = process.stderr.read() if process.stderr is not None else ""
        raise RuntimeError(f"helper failed before ready: {stderr}")
    ready = json.loads(line)
    if ready.get("status") != "ready":
        raise RuntimeError(f"unexpected helper output: {ready}")
    return process, ready, startup_seconds


def stop_helper(process: subprocess.Popen[str]) -> str:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    return process.stderr.read() if process.stderr is not None else ""


def rss_bytes(process: subprocess.Popen[str]) -> int:
    output = subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(process.pid)], text=True
    ).strip()
    return int(output) * 1024 if output else 0


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    timing_keys = sorted(samples[0]["timings"].keys())
    return {
        "median_ms": {
            key: statistics.median(sample["timings"].get(key, 0.0) for sample in samples)
            for key in timing_keys
        },
        "wall_median_ms": statistics.median(sample["wall_ms"] for sample in samples),
        "prompt_tokens": samples[0]["prompt_tokens"],
        "backend": samples[0]["backend"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, action="append", required=True)
    media = parser.add_mutually_exclusive_group(required=True)
    media.add_argument("--video", type=Path)
    media.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--coreml-cache", type=Path)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    media_path = args.video or args.image
    assert media_path is not None
    media_kind = "video" if args.video else "image"
    mime_type = "video/mp4" if args.video else "image/png"
    media_uri = f"data:{mime_type};base64," + base64.b64encode(media_path.read_bytes()).decode()
    content = (
        {"type": "video_url", "video_url": {"url": media_uri}}
        if args.video else {"type": "image_url", "image_url": {"url": media_uri}}
    )
    body = {
        "model": "wemm-embedding-2b-apple-2048",
        "messages": [{
            "role": "user",
            "content": [
                content,
                {"type": "text", "text": f"Represent this {media_kind}."},
            ],
        }],
    }
    results: dict[str, Any] = {}
    vectors: dict[str, list[float]] = {}
    resolved_bundles = [bundle.resolve() for bundle in args.bundle]
    for name, bundles in (("mlx_gpu", []), ("coreml_cpu_ane", resolved_bundles)):
        process, ready, startup_seconds = start_helper(
            args.binary.resolve(),
            args.package.resolve(),
            bundles,
            args.coreml_cache.resolve() if args.coreml_cache else None,
            1024 if args.video else 128,
        )
        try:
            base_url = ready["url"]
            for _ in range(args.warmup):
                request_json(f"{base_url}/v1/embeddings", body)
            samples: list[dict[str, Any]] = []
            for run in range(args.runs):
                started = time.perf_counter_ns()
                response = request_json(f"{base_url}/v1/embeddings", body)
                wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
                sample = {
                    "run": run + 1,
                    "wall_ms": wall_ms,
                    "timings": response["indexed"]["timings_ms"],
                    "prompt_tokens": response["usage"]["prompt_tokens"],
                    "backend": response["indexed"]["backend"],
                }
                samples.append(sample)
                vectors.setdefault(name, response["data"][0]["embedding"])
                print(
                    f"{name} run {run + 1}: wall={wall_ms:.1f} "
                    f"language={sample['timings']['language']:.1f} "
                    f"decoder={sample['timings']['coreml_decoder']:.1f} ms",
                    flush=True,
                )
            health = request_json(f"{base_url}/health")
            if name == "coreml_cpu_ane":
                assert health["decoder_bundle_loaded"] is True
                assert all(sample["timings"]["coreml_decoder"] > 0 for sample in samples)
            results[name] = {
                "ready": ready,
                "startup_seconds": startup_seconds,
                "rss_bytes_after_runs": rss_bytes(process),
                "health": health,
                "samples": samples,
                "summary": summarize(samples),
            }
        finally:
            stderr = stop_helper(process)
            if stderr:
                results.setdefault(name, {})["stderr"] = stderr.strip()

    comparisons = {
        "coreml_vs_mlx_cosine": cosine(vectors["coreml_cpu_ane"], vectors["mlx_gpu"]),
        "language_speedup_coreml_over_mlx": (
            results["mlx_gpu"]["summary"]["median_ms"]["language"]
            / results["coreml_cpu_ane"]["summary"]["median_ms"]["language"]
        ),
        "wall_speedup_coreml_over_mlx": (
            results["mlx_gpu"]["summary"]["wall_median_ms"]
            / results["coreml_cpu_ane"]["summary"]["wall_median_ms"]
        ),
    }
    reference_metadata = None
    if args.reference:
        reference = json.loads(args.reference.read_text())
        reference_vector = reference.pop("vector")
        reference_metadata = reference
        for name, vector in vectors.items():
            comparisons[f"{name}_vs_official_bf16_cosine"] = cosine(
                vector, reference_vector
            )
    elif args.oracle:
        with np.load(args.oracle) as oracle:
            reference_vector = np.asarray(oracle["final_embedding"], dtype=np.float32).reshape(-1)
        reference_vector /= np.linalg.norm(reference_vector)
        for name, vector in vectors.items():
            comparisons[f"{name}_vs_official_bf16_cosine"] = cosine(
                vector, reference_vector.tolist()
            )
        reference_metadata = {"oracle": str(args.oracle.resolve())}
    output = {
        "schema": f"indexed-wemm-{media_kind}-language-backends/v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "configuration": {
            media_kind: str(media_path.resolve()),
            "package": str(args.package.resolve()),
            "bundles": [str(bundle) for bundle in resolved_bundles],
            "warmup": args.warmup,
            "runs": args.runs,
            "vision_backend_both": "coreml-cpu-ane",
        },
        "results": results,
        "comparisons": comparisons,
        "official_reference": reference_metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
