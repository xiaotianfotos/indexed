#!/usr/bin/env python3
"""Create an isolated affine-quantized variant of a WeMM Apple package."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from convert_model import (
    RUNTIME_SEMANTICS,
    _package_fingerprint,
    _read_json,
    _sha256_file,
    _sha256_tree,
    _total_file_size,
    _write_modifications_notice,
    _write_json,
)


def quantize_package(
    source: Path,
    output: Path,
    *,
    bits: int,
    group_size: int,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_vlm.models.qwen3_5.config import ModelConfig
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if bits not in (4, 5, 6, 8):
        raise ValueError("Affine bits must be one of 4/5/6/8")
    if group_size not in (64, 128):
        raise ValueError("Affine group size must be 64 or 128")
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")

    manifest = _read_json(source / "manifest.json")
    if manifest.get("runtime_semantics") != RUNTIME_SEMANTICS:
        raise ValueError("Unsupported source package runtime semantics")
    language_path = source / manifest["language"]["path"]
    language_config_path = language_path.parent / "config.json"
    vision_path = source / manifest["vision"]["path"]
    if not language_path.is_file() or not vision_path.exists():
        raise FileNotFoundError("Source package language or vision artifact is missing")

    config = _read_json(language_config_path)
    if config.get("quantization") is not None:
        raise ValueError("Source package is already quantized")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        language_dir = staging / "language"
        language_dir.mkdir()
        weights = mx.load(str(language_path))
        model_config = ModelConfig.from_dict(config)
        model = LanguageModel(model_config.text_config, model_config)
        if hasattr(model, "lm_head"):
            del model.lm_head
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())
        del weights

        nn.quantize(
            model,
            group_size=group_size,
            bits=bits,
            mode="affine",
        )
        quantized_weights = dict(tree_flatten(model.parameters()))
        mx.eval(quantized_weights)
        language_output = language_dir / "model.safetensors"
        precision = f"q{bits}-g{group_size}"
        mx.save_safetensors(
            str(language_output),
            quantized_weights,
            metadata={
                "format": "mlx",
                "runtime_semantics": RUNTIME_SEMANTICS,
                "precision": precision,
                "quantization_mode": "affine",
            },
        )
        tensor_count = len(quantized_weights)
        del quantized_weights, model

        for item in source.iterdir():
            if item.name in {"manifest.json", "language", "vision"}:
                continue
            destination = staging / item.name
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)
        shutil.copytree(source / "vision", staging / "vision")
        _write_modifications_notice(
            staging,
            source_description=(
                "an Indexed WeMM Apple package with fingerprint "
                f"{manifest['package_fingerprint']}"
            ),
            language_precision=f"{precision} affine quantization",
            vision_artifact=str(manifest["vision"]["path"]),
        )

        quantization = {
            "group_size": group_size,
            "bits": bits,
            "mode": "affine",
        }
        config["quantization"] = quantization
        config["quantization_config"] = quantization
        apple_config = dict(config.get("_wemm_apple") or {})
        apple_config["language_precision"] = precision
        apple_config["source_package_fingerprint"] = manifest["package_fingerprint"]

        language_hash = _sha256_file(language_output)
        vision_source_hash = manifest["vision"].get("source_sha256_tree")
        if not vision_source_hash:
            vision_source_hash = _sha256_tree(vision_path)
        fingerprint = _package_fingerprint(
            source_hashes=list(manifest["source_weight_sha256"]),
            language_hash=language_hash,
            vision_hash=str(vision_source_hash),
        )
        apple_config["package_fingerprint"] = fingerprint
        config["_wemm_apple"] = apple_config
        _write_json(language_dir / "config.json", config)

        result = dict(manifest)
        result["source_package_fingerprint"] = manifest["package_fingerprint"]
        result["package_fingerprint"] = fingerprint
        result["quantization"] = quantization
        result["modifications_notice"] = "MODIFICATIONS.md"
        result["embedding_space_template"] = (
            f"wemm-2b-apple-{precision}-{{dimension}}-{fingerprint[:16]}"
        )
        result["language"] = {
            "backend": "mlx-gpu",
            "precision": precision,
            "quantization": quantization,
            "path": "language/model.safetensors",
            "sha256": language_hash,
            "size_bytes": _total_file_size(language_output),
            "tensor_count": tensor_count,
            "lm_head_included": False,
            "vision_weights_included": False,
        }
        _write_json(staging / "manifest.json", result)
        os.replace(staging, output)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bits", type=int, choices=(4, 5, 6, 8), default=8)
    parser.add_argument("--group-size", type=int, choices=(64, 128), default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = quantize_package(
        args.source,
        args.output,
        bits=args.bits,
        group_size=args.group_size,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
