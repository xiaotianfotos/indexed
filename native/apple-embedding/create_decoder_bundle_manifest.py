#!/usr/bin/env python3
"""Create a signed-by-digest manifest for a complete Core ML decoder bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from convert_model import _sha256_tree


def parse_segment(value: str) -> tuple[int, int, Path]:
    try:
        first_text, last_text, path_text = value.split(":", 2)
        first = int(first_text)
        last = int(last_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "segment must use FIRST:LAST:/path/to/model.mlpackage"
        ) from error
    path = Path(path_text).resolve()
    if first < 0 or last < first:
        raise argparse.ArgumentTypeError("segment layer range is invalid")
    if not path.is_dir() or path.suffix != ".mlpackage":
        raise argparse.ArgumentTypeError(f"Core ML package does not exist: {path}")
    return first, last, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument(
        "--segment",
        action="append",
        type=parse_segment,
        required=True,
        help="repeat FIRST:LAST:/path/to/model.mlpackage",
    )
    parser.add_argument("--sequence-length", type=int, default=2112)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--rotary-size", type=int, default=64)
    parser.add_argument("--source-precision", default="q8-g64-dequantized")
    parser.add_argument(
        "--accuracy-profile",
        choices=("balanced", "gdn_high", "attention_high", "high", "production"),
        default="high",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = args.bundle.resolve()
    model_package = args.model_package.resolve()
    model_manifest = json.loads((model_package / "manifest.json").read_text())
    if model_manifest.get("schema_version") != 1:
        raise ValueError("unsupported model manifest schema")

    segments = sorted(args.segment, key=lambda item: item[0])
    expected_first = 0
    values: list[dict[str, object]] = []
    bundle.mkdir(parents=True, exist_ok=True)
    for first, last, path in segments:
        if first != expected_first:
            raise ValueError(
                f"decoder segments must be contiguous; expected layer {expected_first}, got {first}"
            )
        values.append(
            {
                "path": os.path.relpath(path, bundle),
                "segment_sha256_tree": _sha256_tree(path),
                "first_layer": first,
                "last_layer": last,
            }
        )
        expected_first = last + 1
    if expected_first != 24:
        raise ValueError(f"complete WeMM 2B decoder must end at layer 23, got {expected_first - 1}")

    value = {
        "schema_version": 1,
        "runtime_semantics": "wemm-coreml-decoder-bundle-v1",
        "model": model_manifest["model"],
        "model_package_fingerprint": model_manifest["package_fingerprint"],
        "sequence_length": args.sequence_length,
        "hidden_size": args.hidden_size,
        "rotary_size": args.rotary_size,
        "first_layer": 0,
        "last_layer": 23,
        "next_layer": 24,
        "source_precision": args.source_precision,
        "accuracy_profile": args.accuracy_profile,
        "coreml_weight_precision": "fp16",
        "compute_units": "cpu-and-neural-engine",
        "segments": values,
    }
    output = bundle / "manifest.json"
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
