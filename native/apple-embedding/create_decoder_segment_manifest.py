#!/usr/bin/env python3
"""Bind a Core ML decoder segment to one validated WeMM Apple model package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from convert_model import _sha256_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", type=Path, required=True)
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--last-layer", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--rotary-size", type=int, default=64)
    parser.add_argument("--source-precision", default="q8-g64-dequantized")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segment = args.segment.resolve()
    model_package = args.model_package.resolve()
    manifest = json.loads((model_package / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported model manifest schema")
    if args.first_layer != 0 or args.last_layer != 3:
        raise ValueError("current native runtime requires the contiguous L0-L3 segment")
    output = args.output or Path(f"{segment}.manifest.json")
    value = {
        "schema_version": 1,
        "runtime_semantics": "wemm-coreml-decoder-segment-v1",
        "model": manifest["model"],
        "model_package_fingerprint": manifest["package_fingerprint"],
        "segment_sha256_tree": _sha256_tree(segment),
        "sequence_length": args.sequence_length,
        "hidden_size": args.hidden_size,
        "rotary_size": args.rotary_size,
        "first_layer": args.first_layer,
        "last_layer": args.last_layer,
        "next_layer": args.last_layer + 1,
        "source_precision": args.source_precision,
        "coreml_weight_precision": "fp16",
        "compute_units": "cpu-and-neural-engine",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(output.resolve())


if __name__ == "__main__":
    main()
