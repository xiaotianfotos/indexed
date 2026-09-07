#!/usr/bin/env python3
"""Compare WeMM's MLX language path with oMLX's private ANE experiment.

This file is deliberately separate from ``service.py``.  The product service
uses public Core ML APIs for the vision tower and MLX/Metal for the language
model.  This benchmark imports oMLX only after measuring the GPU baseline and
never changes the HTTP product path.

For WeMM's 202-token image prompt, the useful experimental tile is 256.  oMLX
0.6.4 intentionally accepts only tiles >= 1024, so the repository also ships
``omlx-short-sequence.patch`` for an explicitly unsupported local experiment.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from service import MLXWeMMEngine, normalize_messages


def _messages(media_path: Path, media_kind: str) -> list[dict[str, Any]]:
    if media_kind == "video":
        with tempfile.TemporaryDirectory(prefix="indexed-private-ane-frames-") as temporary:
            pattern = Path(temporary) / "frame-%06d.jpg"
            subprocess.run(
                [
                    os.environ.get("INDEXED_FFMPEG_BINARY", "ffmpeg"),
                    "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(media_path), "-vf", "fps=2", "-q:v", "5",
                    "-f", "image2", "-start_number", "0", str(pattern),
                ],
                check=True,
            )
            frames = []
            for index, frame_path in enumerate(sorted(Path(temporary).glob("frame-*.jpg"))):
                payload = base64.b64encode(frame_path.read_bytes()).decode("ascii")
                frames.append({
                    "image_url": {"url": f"data:image/jpeg;base64,{payload}"},
                    "timestamp": index / 2.0,
                })
        if not frames:
            raise RuntimeError(f"video produced no frames: {media_path}")
        media_item = {"type": "video_frames", "frames": frames}
    else:
        mime = "image/png" if media_path.suffix.lower() == ".png" else "image/jpeg"
        media_item = {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{base64.b64encode(media_path.read_bytes()).decode('ascii')}"
            },
        }
    return [
        {
            "role": "user",
            "content": [
                media_item
            ],
        }
    ]


def _median(rows: list[dict[str, float]], discard: int) -> dict[str, float]:
    kept = rows[discard:]
    return {
        key: float(statistics.median(row[key] for row in kept)) for key in rows[0]
    }


def _run(
    engine: MLXWeMMEngine,
    messages: list[dict[str, Any]],
    *,
    dimension: int,
    runs: int,
) -> tuple[list[dict[str, float]], np.ndarray]:
    import mlx.core as mx

    rows: list[dict[str, float]] = []
    result = None
    for _ in range(runs):
        result = engine.embed(messages, dimension=dimension)
        mx.synchronize()
        rows.append(result.timings_ms)
    assert result is not None
    return rows, np.asarray(result.vector, dtype=np.float32)


def _speedup(before_ms: float, after_ms: float) -> dict[str, float]:
    return {
        "ratio": before_ms / after_ms,
        "percent": (before_ms - after_ms) / before_ms * 100.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated private-ANE benchmark for converted WeMM Q8 packages"
    )
    parser.add_argument("--package", type=Path, required=True)
    media = parser.add_mutually_exclusive_group(required=True)
    media.add_argument("--image", type=Path)
    media.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--discard", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument(
        "--mlp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--fraction", type=float, default=0.75)
    parser.add_argument("--max-layers", type=int, default=24)
    parser.add_argument("--gdn", action="store_true")
    parser.add_argument("--gdn-fraction", type=float, default=0.50)
    parser.add_argument("--gdn-max-layers", type=int, default=18)
    parser.add_argument("--min-vector-cosine", type=float, default=0.999)
    parser.add_argument(
        "--order",
        choices=("baseline-first", "ane-first"),
        default="baseline-first",
        help="run the reverse order as a drift/thermal control",
    )
    parser.add_argument("--recurrence-ane", action="store_true")
    parser.add_argument("--recurrence-max-layers", type=int, default=18)
    parser.add_argument(
        "--recurrence-max-tokens",
        type=int,
        help="maximum input length for the chunked recurrence backend; defaults to --sequence-length",
    )
    parser.add_argument(
        "--recurrence-block-size",
        type=int,
        choices=(2, 4, 8, 16, 32, 64),
        default=8,
    )
    parser.add_argument(
        "--recurrence-io-dtype",
        choices=("fp16", "fp32"),
        default="fp16",
        help="IOSurface element type used at the MLX/private-ANE boundary",
    )
    parser.add_argument("--recurrence-verify-reference", action="store_true")
    parser.add_argument(
        "--recurrence-layer-slots",
        help="comma-separated zero-based GDN call slots; overrides max-layers",
    )
    parser.add_argument(
        "--recurrence-experiment-root",
        type=Path,
        help="wemm_ane_hwx_experiment directory containing the private bridge",
    )
    parser.add_argument(
        "--compute-unit", choices=("cpu_ane", "cpu_gpu", "all"), default="cpu_ane"
    )
    args = parser.parse_args()
    if args.runs <= args.discard:
        parser.error("--runs 必须大于 --discard")
    if args.sequence_length % 64:
        parser.error("--sequence-length 必须是 64 的倍数")
    if args.recurrence_max_tokens is None:
        args.recurrence_max_tokens = args.sequence_length
    if args.recurrence_max_tokens < 64 or args.recurrence_max_tokens % 64:
        parser.error("--recurrence-max-tokens 必须是正的 64 倍数")
    if not args.mlp and not args.recurrence_ane:
        parser.error("--no-mlp 需要同时启用 --recurrence-ane")
    if args.recurrence_ane and args.recurrence_experiment_root is None:
        parser.error("--recurrence-ane 需要 --recurrence-experiment-root")
    if not 0 <= args.recurrence_max_layers <= 18:
        parser.error("--recurrence-max-layers 必须在 0 到 18 之间")
    if args.recurrence_layer_slots:
        try:
            args.recurrence_layer_slots = sorted(
                {int(value) for value in args.recurrence_layer_slots.split(",")}
            )
        except ValueError:
            parser.error("--recurrence-layer-slots 必须是逗号分隔的整数")
        if any(value < 0 or value >= 18 for value in args.recurrence_layer_slots):
            parser.error("--recurrence-layer-slots 必须在 0 到 17 之间")
    return args


def main() -> None:
    args = parse_args()
    package = args.package.resolve()
    media_path = (args.video or args.image).resolve()
    media_kind = "video" if args.video else "image"
    if not media_path.is_file():
        raise FileNotFoundError(media_path)

    messages = _messages(media_path, media_kind)
    engine = MLXWeMMEngine(
        package_path=package,
        image_size=448,
        compute_unit=args.compute_unit,
        warmup=False,
    )
    normalized, declared_images = normalize_messages(messages, image_size=448)
    input_ids = engine._inputs(normalized, declared_images)[0]
    actual_tokens = int(input_ids.shape[-1])

    gpu_rows = None
    gpu_vector = None
    if args.order == "baseline-first":
        gpu_rows, gpu_vector = _run(
            engine, messages, dimension=args.dimension, runs=args.runs
        )

    compile_started = time.perf_counter()
    fast = None
    status: dict[str, Any] = {
        "configured": False,
        "mlp_layers": [],
        "gdn_layers": [],
    }
    enabled_layers: list[int] = []
    transient_bytes = 0
    release_qwen35_ane_prefill = None
    if args.mlp:
        # Importing here keeps the baseline independent of oMLX's process-wide
        # dispatch registration.
        try:
            from omlx.custom_kernels.qwen35_prefill import fast
            from omlx.patches.qwen35_ane_prefill import (
                ane_prefill_transient_bytes,
                enable_qwen35_ane_prefill,
                qwen35_ane_prefill_status,
                release_qwen35_ane_prefill,
            )
        except ImportError as exc:
            raise RuntimeError(
                "缺少带 qwen35_prefill 原生扩展的 oMLX；请使用独立 .venv-ane 环境"
            ) from exc

        try:
            enabled_layers = enable_qwen35_ane_prefill(
                engine.language,
                sequence_length=args.sequence_length,
                fraction=args.fraction,
                variant=8,
                max_layers=args.max_layers,
                gdn=args.gdn,
                gdn_fraction=args.gdn_fraction,
                gdn_max_layers=args.gdn_max_layers,
                dual_ane=False,
                tail_padding_min_tokens=1,
            )
        except ValueError as exc:
            if args.sequence_length < 1024 and ">= 1024" in str(exc):
                patch_path = Path(__file__).with_name("omlx-short-sequence.patch")
                raise RuntimeError(
                    "当前 oMLX 拒绝小于 1024 的实验形状。请只在隔离实验环境中把 "
                    f"{patch_path} 应用到 oMLX 0.6.4 源码后重试。"
                ) from exc
            raise
        status = qwen35_ane_prefill_status(engine.language)
        if not status["configured"] or not enabled_layers:
            raise RuntimeError(f"oMLX 未配置任何 ANE MLP：{status}")
        transient_bytes = ane_prefill_transient_bytes(engine.language)

    recurrence = None
    if args.recurrence_ane:
        experiment_root = args.recurrence_experiment_root.resolve()
        sys.path.insert(0, str(experiment_root))
        from wemm_mlx_recurrence_backend import PrivateANEGatedDeltaPrefill
        from mlx_vlm.models.qwen3_5.gated_delta import (
            register_qwen3_5_gated_delta_prefill_backend,
        )

        recurrence = PrivateANEGatedDeltaPrefill(
            bridge_path=(
                experiment_root
                / ".cache/ane-private-runtime/bridge/libane_bridge.dylib"
            ),
            mil_path=(
                experiment_root / "results/real_g_safe_c64_specialized.mil"
            ),
            max_tokens=args.recurrence_max_tokens,
            enabled_layer_slots=args.recurrence_max_layers,
            layer_slots=args.recurrence_layer_slots,
            solve_block_size=args.recurrence_block_size,
            verify_reference=args.recurrence_verify_reference,
            io_dtype=args.recurrence_io_dtype,
        )
        register_qwen3_5_gated_delta_prefill_backend(recurrence)
    compile_seconds = time.perf_counter() - compile_started
    if fast is not None:
        fast.qwen35_ane_profile_reset()
        fast.qwen35_ane_profile_set_enabled(True)
    if recurrence is not None:
        recurrence.reset_profile()
    try:
        ane_rows, ane_vector = _run(
            engine, messages, dimension=args.dimension, runs=args.runs
        )
    finally:
        if recurrence is not None:
            register_qwen3_5_gated_delta_prefill_backend(None)
    ane_median = _median(ane_rows, args.discard)
    profile = (
        fast.qwen35_ane_profile_snapshot()
        if fast is not None
        else {
            "mlp": {"operations": 0},
            "gdn": {"operations": 0},
        }
    )
    recurrence_profile = recurrence.profile() if recurrence is not None else None
    if recurrence is not None:
        recurrence.close()
    if fast is not None:
        fast.qwen35_ane_profile_set_enabled(False)
        if release_qwen35_ane_prefill is not None:
            release_qwen35_ane_prefill(engine.language)

    if args.order == "ane-first":
        gpu_rows, gpu_vector = _run(
            engine, messages, dimension=args.dimension, runs=args.runs
        )
    assert gpu_rows is not None and gpu_vector is not None
    gpu_median = _median(gpu_rows, args.discard)

    cosine = float(
        np.dot(gpu_vector, ane_vector)
        / (np.linalg.norm(gpu_vector) * np.linalg.norm(ane_vector))
    )
    reference_gate = None
    verification = (
        recurrence_profile.get("reference_verification")
        if recurrence_profile is not None
        else None
    )
    if verification is not None:
        reference_gate = {
            "minimum_cosine": args.min_vector_cosine,
            "maximum_relative_l2": 0.05,
            "minimum_output_cosine": verification["minimum_output_cosine"],
            "minimum_state_cosine": verification["minimum_state_cosine"],
            "maximum_output_relative_l2": verification[
                "maximum_output_relative_l2"
            ],
            "maximum_state_relative_l2": verification[
                "maximum_state_relative_l2"
            ],
            "nonfinite": verification["nonfinite"],
        }
        reference_gate["passed"] = bool(
            reference_gate["nonfinite"] == 0
            and reference_gate["minimum_output_cosine"]
            >= reference_gate["minimum_cosine"]
            and reference_gate["minimum_state_cosine"]
            >= reference_gate["minimum_cosine"]
            and reference_gate["maximum_output_relative_l2"]
            <= reference_gate["maximum_relative_l2"]
            and reference_gate["maximum_state_relative_l2"]
            <= reference_gate["maximum_relative_l2"]
        )
    quality_passed = bool(
        cosine >= args.min_vector_cosine
        and (reference_gate is None or reference_gate["passed"])
    )
    report: dict[str, Any] = {
        "scope": "unsupported-private-ane-experiment-not-for-app-store-build",
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "macos": platform.mac_ver()[0],
        },
        "package": str(package),
        "package_fingerprint": engine.package_fingerprint,
        "media": str(media_path),
        "media_kind": media_kind,
        "image": str(media_path) if media_kind == "image" else None,
        "video": str(media_path) if media_kind == "video" else None,
        "dimension": args.dimension,
        "actual_input_tokens": actual_tokens,
        "quality_gate": {
            "minimum_vector_cosine": args.min_vector_cosine,
            "reference": reference_gate,
            "passed": quality_passed,
        },
        "settings": {
            "sequence_length": args.sequence_length,
            "mlp": args.mlp,
            "fraction": args.fraction,
            "max_layers": args.max_layers,
            "gdn": args.gdn,
            "gdn_fraction": args.gdn_fraction,
            "gdn_max_layers": args.gdn_max_layers,
            "recurrence_ane": args.recurrence_ane,
            "recurrence_max_layers": args.recurrence_max_layers,
            "recurrence_max_tokens": args.recurrence_max_tokens,
            "recurrence_layer_slots": args.recurrence_layer_slots,
            "recurrence_block_size": args.recurrence_block_size,
            "recurrence_verify_reference": args.recurrence_verify_reference,
            "order": args.order,
            "dual_ane": False,
            "runs": args.runs,
            "discard": args.discard,
        },
        "model": {
            "mlx_active_bytes_before_ane": engine.allocated_bytes,
            "ane_transient_surface_bytes": transient_bytes,
            "compile_seconds": compile_seconds,
            "status": status,
            "recurrence": recurrence_profile,
        },
        "gpu_language_baseline": {
            "median_ms": gpu_median,
            "rows": gpu_rows,
        },
        "private_ane_hybrid": {
            "median_ms": ane_median,
            "rows": ane_rows,
            "native_profile": profile,
        },
        "comparison": {
            "language_speedup": _speedup(
                gpu_median["language"], ane_median["language"]
            ),
            "end_to_end_speedup": _speedup(
                gpu_median["total"], ane_median["total"]
            ),
            "vector_cosine": cosine,
            "vector_max_absolute_error": float(
                np.max(np.abs(gpu_vector - ane_vector))
            ),
        },
        "observations": {
            "mlp_ane_operations": profile["mlp"]["operations"],
            "gdn_ane_operations": profile["gdn"]["operations"],
            "gdn_configured_but_not_observed": bool(
                status["gdn_layers"] and not profile["gdn"]["operations"]
            ),
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
