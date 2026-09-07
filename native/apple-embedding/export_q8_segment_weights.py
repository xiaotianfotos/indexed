#!/usr/bin/env python3
"""Export selected MLX Q8 decoder layers as dequantized safetensors for Core ML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--last-layer", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = args.model.resolve()
    config = json.loads((model / "language" / "config.json").read_text())
    quantization = config["quantization"]
    bits = int(quantization["bits"])
    group_size = int(quantization["group_size"])
    mode = str(quantization.get("mode", "affine"))
    source = mx.load(str(model / "language" / "model.safetensors"))
    prefixes = tuple(
        f"model.layers.{layer}."
        for layer in range(args.first_layer, args.last_layer + 1)
    )
    selected = {key: value for key, value in source.items() if key.startswith(prefixes)}
    output: dict[str, mx.array] = {}

    for key, value in selected.items():
        if key.endswith((".scales", ".biases")):
            continue
        if key.endswith(".weight"):
            base = key.removesuffix(".weight")
            scales = selected.get(f"{base}.scales")
            biases = selected.get(f"{base}.biases")
            if scales is not None:
                value = mx.dequantize(
                    value,
                    scales,
                    biases,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                    dtype=mx.bfloat16,
                )
        output[key] = value

    if not output:
        raise SystemExit("没有找到目标 decoder layer 权重")
    mx.eval(*output.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.output),
        output,
        metadata={
            "source": str(model),
            "quantization": f"{mode}-q{bits}-g{group_size}",
            "layers": f"{args.first_layer}-{args.last_layer}",
            "purpose": "Core ML decoder segment conversion",
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "tensors": len(output),
                "bytes": args.output.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
