#!/usr/bin/env python3
"""Capture WeMM decoder boundaries for one 448 px image."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModel, AutoProcessor


SEGMENTS = [(0, 7), (8, 15), (16, 23)]


def tensor_summary(value: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()),
        "minimum": float(value.min()),
        "maximum": float(value.max()),
        "mean": float(value.mean()),
        "standard_deviation": float(value.std()),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_image_inputs(processor: Any, image_path: Path) -> tuple[Any, str]:
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": str(image_path),
                "resized_height": 448,
                "resized_width": 448,
            },
            {"type": "text", "text": "Represent this image."},
        ],
    }]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if videos is not None:
        videos, video_metadata = zip(*videos)
        videos = list(videos)
        video_metadata = list(video_metadata)
    else:
        video_metadata = None
    inputs = processor(
        text=prompt,
        images=images,
        videos=videos,
        video_metadata=video_metadata,
        return_tensors="pt",
        **video_kwargs,
    )
    return inputs, prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()

    model_path = args.model.resolve()
    image_path = args.image.resolve()
    processor = AutoProcessor.from_pretrained(
        str(model_path), trust_remote_code=True, use_fast=True
    )
    inputs, prompt = build_image_inputs(processor, image_path)

    load_started = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    ).eval()
    model.lm_head = None
    if args.device == "mps":
        torch.mps.synchronize()
    load_seconds = time.perf_counter() - load_started

    captured: dict[str, np.ndarray] = {}
    handles: list[Any] = []

    def save(name: str, value: Any) -> None:
        if isinstance(value, tuple):
            value = value[0]
        captured[name] = value.detach().float().cpu().numpy()

    layers = model.model.language_model.layers
    for first, last in SEGMENTS:
        handles.append(layers[first].register_forward_pre_hook(
            lambda _module, hook_args, hook_kwargs, layer=first: save(
                f"layer_{layer}_input",
                hook_kwargs.get("hidden_states", hook_args[0] if hook_args else None),
            ),
            with_kwargs=True,
        ))
        handles.append(layers[last].register_forward_hook(
            lambda _module, _hook_args, output, layer=last: save(
                f"layer_{layer}_output", output
            )
        ))

    def attention_input_hook(
        _module: Any, _hook_args: tuple[Any, ...], hook_kwargs: dict[str, Any]
    ) -> None:
        position_embeddings = hook_kwargs.get("position_embeddings")
        if position_embeddings is None:
            raise RuntimeError("full-attention layer did not receive position embeddings")
        save("position_cos", position_embeddings[0])
        save("position_sin", position_embeddings[1])

    handles.append(
        layers[3].self_attn.register_forward_pre_hook(
            attention_input_hook, with_kwargs=True
        )
    )

    input_ids = inputs.input_ids.detach().cpu().numpy().astype(np.int32)
    image_grid_thw = inputs.image_grid_thw.detach().cpu().tolist()
    inputs = inputs.to(args.device)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        final_embedding = model.embedding(**inputs)
    if args.device == "mps":
        torch.mps.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    save("final_embedding", final_embedding)
    captured["input_ids"] = input_ids
    for handle in handles:
        handle.remove()

    required = {"position_cos", "position_sin", "final_embedding", "input_ids"}
    for first, last in SEGMENTS:
        required.update((f"layer_{first}_input", f"layer_{last}_output"))
    missing = required - captured.keys()
    if missing:
        raise RuntimeError(f"incomplete image decoder oracle: {sorted(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **captured)
    metadata = {
        "schema": "indexed-wemm-image-decoder-oracle/v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "model": str(model_path),
        "image": str(image_path),
        "image_sha256": sha256(image_path),
        "device": args.device,
        "sequence_length": int(input_ids.shape[1]),
        "image_grid_thw": image_grid_thw,
        "segments": [list(value) for value in SEGMENTS],
        "prompt": prompt,
        "load_seconds": load_seconds,
        "inference_seconds_with_capture": inference_seconds,
        "tensors": {name: tensor_summary(value) for name, value in captured.items()},
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(metadata_path)


if __name__ == "__main__":
    main()
