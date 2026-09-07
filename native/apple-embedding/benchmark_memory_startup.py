#!/usr/bin/env python3
"""Measure startup and memory for the MLX + public Core ML segment runtime."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path

import mlx.core as mx

from benchmark_coreml_block_integration import CoreMLDecoderBlock, image_uri
from service import MLXWeMMEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--segment", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    return parser.parse_args()


def rss_bytes() -> int:
    value = subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
    ).strip()
    return int(value) * 1024


def peak_rss_bytes() -> int:
    # Darwin reports ru_maxrss in bytes.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def memory_snapshot() -> dict[str, int]:
    return {
        "rss_bytes": rss_bytes(),
        "peak_rss_bytes": peak_rss_bytes(),
        "mlx_active_bytes": int(mx.get_active_memory()),
        "mlx_peak_bytes": int(mx.get_peak_memory()),
    }


def main() -> None:
    args = parse_args()
    package = args.package.resolve()
    segment = args.segment.resolve()
    snapshots: dict[str, dict[str, int]] = {"process_start": memory_snapshot()}

    started = time.perf_counter()
    engine = MLXWeMMEngine(
        package_path=package,
        image_size=448,
        compute_unit="cpu_ane",
        warmup=False,
    )
    engine_wall_seconds = time.perf_counter() - started
    snapshots["engine_loaded"] = memory_snapshot()

    original_layers = list(engine.language.model.layers[:4])
    rotary_provider = next(
        original.self_attn.rotary_emb
        for original in original_layers
        if not original.is_linear
    )
    started = time.perf_counter()
    coreml_segment = CoreMLDecoderBlock(
        segment,
        args.sequence_length,
        original_layers[0],
        rotary_provider,
    )
    segment_load_seconds = time.perf_counter() - started
    snapshots["segment_loaded"] = memory_snapshot()

    image_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_uri(args.image.resolve())},
                },
                {"type": "text", "text": "Represent this image."},
            ],
        }
    ]
    started = time.perf_counter()
    baseline = engine.embed(image_messages, dimension=2048)
    baseline_first_seconds = time.perf_counter() - started
    snapshots["after_baseline_first_inference"] = memory_snapshot()

    from benchmark_coreml_block_integration import PassthroughDecoderBlock

    engine.language.model.layers[0] = coreml_segment
    for index, original in enumerate(original_layers[1:], start=1):
        engine.language.model.layers[index] = PassthroughDecoderBlock(original)
    started = time.perf_counter()
    hybrid = engine.embed(image_messages, dimension=2048)
    hybrid_first_seconds = time.perf_counter() - started
    snapshots["after_hybrid_first_inference"] = memory_snapshot()
    started = time.perf_counter()
    hybrid_steady = engine.embed(image_messages, dimension=2048)
    hybrid_steady_seconds = time.perf_counter() - started
    snapshots["after_hybrid_steady_inference"] = memory_snapshot()

    payload = {
        "machine": {
            "chip": subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip(),
            "physical_memory_bytes": int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            ),
            "macos": platform.mac_ver()[0],
        },
        "artifacts": {
            "apple_package": str(package),
            "apple_package_bytes": directory_bytes(package),
            "coreml_segment": str(segment),
            "coreml_segment_bytes": directory_bytes(segment),
        },
        "startup": {
            "engine_constructor_wall_seconds": engine_wall_seconds,
            "engine_reported_load_seconds": engine.load_seconds,
            "coreml_segment_load_seconds": segment_load_seconds,
            "baseline_first_embedding_seconds": baseline_first_seconds,
            "hybrid_first_embedding_seconds": hybrid_first_seconds,
            "hybrid_steady_embedding_seconds": hybrid_steady_seconds,
            "baseline_timings_ms": baseline.timings_ms,
            "hybrid_first_timings_ms": hybrid.timings_ms,
            "hybrid_steady_timings_ms": hybrid_steady.timings_ms,
        },
        "memory": snapshots,
        "coreml_segment_calls": coreml_segment.calls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
