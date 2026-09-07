#!/usr/bin/env python3
"""Build a product-oriented Apple runtime package from WeMM HF weights.

The public runtime is intentionally heterogeneous:

* the fixed WeMM vision tower is shipped as a Core ML ``.mlpackage``;
* the Qwen3.5 language backbone is stored in MLX-native tensor layout;
* the generation-only vocabulary head and duplicate vision weights are removed.

The first conversion target is lossless BF16. Quantized variants must be emitted
as separate packages so Indexed never silently mixes incompatible vector spaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
RUNTIME_SEMANTICS = "wemm-apple-embedding-v1"
MODEL_TYPE = "qwen3_5"
LANGUAGE_PREFIX = "model.language_model."
OFFICIAL_DIMENSIONS = (64, 128, 256, 512, 1024, 2048)
NORM_WEIGHT_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)
TOKENIZER_FILES = (
    "LICENSE",
    "README.md",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "embedding_chat_template.jinja",
    "processor_config.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    """Hash a directory deterministically, including relative file names."""

    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _total_file_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _find_weight_files(source: Path) -> list[Path]:
    files = sorted(source.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No model*.safetensors found in {source}")
    return files


def _load_source_weights(paths: Iterable[Path]) -> dict[str, Any]:
    try:
        import mlx.core as mx
    except ImportError as error:
        raise RuntimeError("MLX is required; run this with the MLX environment") from error

    weights: dict[str, Any] = {}
    for path in paths:
        for key, value in mx.load(str(path)).items():
            if key in weights:
                raise ValueError(f"Duplicate tensor key across shards: {key}")
            weights[key] = value
    return weights


def prepare_language_weights(source_weights: dict[str, Any]) -> dict[str, Any]:
    """Return Qwen3.5 language tensors in the MLX-VLM ``model.*`` layout.

    Current Qwen3.5 HF checkpoints store depthwise-convolution kernels and
    RMSNorm weights in their authoring convention. These two transformations
    match the upstream MLX Qwen3.5 sanitizer and are required for parity.
    """

    try:
        import mlx.core as mx
    except ImportError as error:
        raise RuntimeError("MLX is required; run this with the MLX environment") from error

    candidates = {
        key: value
        for key, value in source_weights.items()
        if key.startswith(LANGUAGE_PREFIX) and "mtp." not in key
    }
    if not candidates:
        raise ValueError(f"No tensors start with {LANGUAGE_PREFIX!r}")

    shift_norm_weights = any(
        "conv1d.weight" in key and value.shape[-1] != 1
        for key, value in candidates.items()
    )
    prepared: dict[str, Any] = {}
    for source_key, source_value in candidates.items():
        key = "model." + source_key.removeprefix(LANGUAGE_PREFIX)
        value = source_value
        if "conv1d.weight" in key and value.shape[-1] != 1:
            value = value.moveaxis(2, 1)
        if shift_norm_weights and value.ndim == 1 and any(
            key.endswith(suffix) for suffix in NORM_WEIGHT_SUFFIXES
        ):
            value = value + 1.0
        prepared[key] = mx.contiguous(value)
    return prepared


def _token_id(tokenizer_path: Path, token: str) -> int:
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError("tokenizers is required for package validation") from error

    value = Tokenizer.from_file(str(tokenizer_path)).token_to_id(token)
    if value is None:
        raise ValueError(f"Tokenizer does not contain required token {token!r}")
    return int(value)


def _copy_required_files(source: Path, destination: Path) -> list[str]:
    copied: list[str] = []
    for name in TOKENIZER_FILES:
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, destination / name)
            copied.append(name)
    if "tokenizer.json" not in copied or "tokenizer_config.json" not in copied:
        raise FileNotFoundError("WeMM tokenizer.json/tokenizer_config.json are required")
    return copied


def _write_modifications_notice(
    destination: Path,
    *,
    source_description: str,
    language_precision: str,
    vision_artifact: str,
) -> None:
    """Record the mechanically transformed files required by Apache-2.0 §4(b)."""

    content = f"""# Modifications to WeMM-Embedding-2B

This is a derivative runtime package produced for Indexed from
`{source_description}`. It is not an official Tencent distribution.

The following files were changed or created by the conversion pipeline:

- `language/model.safetensors`: extracted the Qwen3.5 language backbone,
  removed the vocabulary head and duplicate vision weights, transformed tensors
  to the MLX runtime layout, and stored them as `{language_precision}`.
- `language/config.json`: added Indexed Apple runtime and fingerprint metadata.
- `{vision_artifact}`: converted the WeMM vision tower to a Core ML artifact for
  public Core ML CPU/ANE execution.
- `manifest.json`: added the reproducible hashes, runtime contract, supported
  Matryoshka dimensions, and semantic-space fingerprint.
- `MODIFICATIONS.md`: added this notice.

The upstream `LICENSE`, `README.md`, tokenizer, processor, and prompt templates
are copied without semantic changes. WeMM-Embedding-2B and its weights remain
subject to the license and third-party notices in `LICENSE`.
"""
    with (destination / "MODIFICATIONS.md").open("w", encoding="utf-8") as handle:
        handle.write(content)


def _package_fingerprint(
    *, source_hashes: list[str], language_hash: str, vision_hash: str
) -> str:
    canonical = json.dumps(
        {
            "runtime_semantics": RUNTIME_SEMANTICS,
            "source_weights": source_hashes,
            "language": language_hash,
            "vision": vision_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def convert(
    source: Path,
    vision_model: Path,
    output: Path,
    *,
    compile_vision: bool = True,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    vision_model = vision_model.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source model directory not found: {source}")
    if not vision_model.is_dir() or vision_model.suffix != ".mlpackage":
        raise FileNotFoundError(f"Core ML .mlpackage not found: {vision_model}")
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")

    source_config = _read_json(source / "config.json")
    if source_config.get("model_type") != MODEL_TYPE:
        raise ValueError(
            f"Expected model_type={MODEL_TYPE!r}, got {source_config.get('model_type')!r}"
        )
    configured_dimensions = tuple(source_config.get("matryoshka_dimensions", ()))
    if configured_dimensions != OFFICIAL_DIMENSIONS:
        raise ValueError(
            "Unexpected WeMM Matryoshka dimensions: " f"{configured_dimensions!r}"
        )

    weight_files = _find_weight_files(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        language_dir = staging / "language"
        vision_dir = staging / "vision"
        language_dir.mkdir()
        vision_dir.mkdir()

        source_weights = _load_source_weights(weight_files)
        language_weights = prepare_language_weights(source_weights)
        del source_weights
        language_tensor_count = len(language_weights)

        import mlx.core as mx

        language_path = language_dir / "model.safetensors"
        mx.save_safetensors(
            str(language_path),
            language_weights,
            metadata={
                "format": "mlx",
                "runtime_semantics": RUNTIME_SEMANTICS,
                "source_model_type": MODEL_TYPE,
                "precision": "bfloat16",
            },
        )
        del language_weights

        copied_tokenizer_files = _copy_required_files(source, staging)
        source_vision_hash = _sha256_tree(vision_model)
        if compile_vision:
            import coremltools as ct

            packaged_vision = vision_dir / f"{vision_model.stem}.mlmodelc"
            ct.models.utils.compile_model(
                str(vision_model), destination_path=str(packaged_vision)
            )
        else:
            packaged_vision = vision_dir / vision_model.name
            shutil.copytree(vision_model, packaged_vision)

        _write_modifications_notice(
            staging,
            source_description=f"Tencent/WeMM-Embedding-2B ({source.name})",
            language_precision="bfloat16",
            vision_artifact=f"vision/{packaged_vision.name}",
        )

        source_hashes = [_sha256_file(path) for path in weight_files]
        language_hash = _sha256_file(language_path)
        packaged_vision_hash = _sha256_tree(packaged_vision)
        fingerprint = _package_fingerprint(
            source_hashes=source_hashes,
            language_hash=language_hash,
            vision_hash=source_vision_hash,
        )
        embedding_token_id = _token_id(staging / "tokenizer.json", "<embedding>")

        packaged_config = dict(source_config)
        packaged_config["_wemm_apple"] = {
            "schema_version": SCHEMA_VERSION,
            "runtime_semantics": RUNTIME_SEMANTICS,
            "language_layout": "mlx-vlm-qwen3.5-model",
            "language_precision": "bfloat16",
            "language_has_lm_head": False,
            "language_has_vision_tower": False,
            "embedding_token": "<embedding>",
            "embedding_token_id": embedding_token_id,
            "package_fingerprint": fingerprint,
        }
        _write_json(language_dir / "config.json", packaged_config)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "runtime_semantics": RUNTIME_SEMANTICS,
            "model": "Tencent/WeMM-Embedding-2B",
            "source": source.name,
            "source_weight_files": [path.name for path in weight_files],
            "source_weight_sha256": source_hashes,
            "package_fingerprint": fingerprint,
            "embedding_space_template": f"wemm-2b-apple-{{dimension}}-{fingerprint[:16]}",
            "matryoshka_dimensions": list(OFFICIAL_DIMENSIONS),
            "embedding_token_id": embedding_token_id,
            "image_token_id": int(source_config["image_token_id"]),
            "video_token_id": int(source_config["video_token_id"]),
            "language": {
                "backend": "mlx-gpu",
                "precision": "bfloat16",
                "path": "language/model.safetensors",
                "sha256": language_hash,
                "size_bytes": _total_file_size(language_path),
                "tensor_count": language_tensor_count,
                "lm_head_included": False,
                "vision_weights_included": False,
            },
            "vision": {
                "backend": "coreml-cpu-and-neural-engine",
                "compiled": compile_vision,
                "path": f"vision/{packaged_vision.name}",
                "source_sha256_tree": source_vision_hash,
                "packaged_sha256_tree": packaged_vision_hash,
                "size_bytes": _total_file_size(packaged_vision),
                "image_size": 448,
            },
            "modifications_notice": "MODIFICATIONS.md",
            "tokenizer_files": copied_tokenizer_files,
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert WeMM-Embedding-2B into an Apple Core ML + MLX package"
    )
    parser.add_argument("--source", required=True, type=Path, help="HF model directory")
    parser.add_argument(
        "--vision-model", required=True, type=Path, help="Converted Core ML .mlpackage"
    )
    parser.add_argument("--output", required=True, type=Path, help="New package directory")
    parser.add_argument(
        "--portable-vision",
        action="store_true",
        help="Copy the portable .mlpackage instead of compiling a local .mlmodelc",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = convert(
        args.source,
        args.vision_model,
        args.output,
        compile_vision=not args.portable_vision,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
