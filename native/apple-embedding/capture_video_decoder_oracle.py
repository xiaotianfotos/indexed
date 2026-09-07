#!/usr/bin/env python3
"""Capture all six WeMM Qwen3.5 decoder-segment boundaries for one video."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoProcessor


SEGMENTS = [(start, start + 3) for start in range(0, 24, 4)]


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


def build_video_inputs(processor: Any, video_path: Path) -> tuple[Any, dict[str, Any]]:
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames = [
        torch.from_numpy(frame.to_rgb().to_ndarray()).permute(2, 0, 1)
        for frame in container.decode(stream)
    ]
    source_fps = float(stream.average_rate)
    requested = max(2, int((len(frames) / source_fps) * 2.0))
    frame_count = min(64, requested)
    frame_count -= frame_count % 2
    indices = torch.linspace(0, len(frames) - 1, frame_count).round().long().tolist()
    video = torch.stack([frames[index] for index in indices])
    video = functional.interpolate(
        video.float(), size=(448, 448), mode="bicubic", align_corners=False, antialias=True
    )
    metadata = {
        "fps": source_fps,
        "frames_indices": indices,
        "total_num_frames": len(frames),
        "video_backend": "pyav-parity",
    }
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": str(video_path)},
            {"type": "text", "text": "Represent this video."},
        ],
    }]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    inputs = processor(
        text=prompt,
        videos=[video],
        video_metadata=[metadata],
        do_sample_frames=False,
        return_tensors="pt",
    )
    return inputs, {
        "prompt": prompt,
        "source_frames": len(frames),
        "source_fps": source_fps,
        "sampled_frame_indices": indices,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()

    model_path = args.model.resolve()
    video_path = args.video.resolve()
    processor = AutoProcessor.from_pretrained(
        str(model_path), trust_remote_code=True, use_fast=True
    )
    inputs, media = build_video_inputs(processor, video_path)

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
        raise RuntimeError(f"incomplete video decoder oracle: {sorted(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **captured)
    metadata = {
        "schema": "indexed-wemm-video-decoder-oracle/v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "model": str(model_path),
        "video": str(video_path),
        "video_sha256": sha256(video_path),
        "device": args.device,
        "sequence_length": int(input_ids.shape[1]),
        "video_grid_thw": inputs.video_grid_thw.detach().cpu().tolist(),
        "segments": [list(value) for value in SEGMENTS],
        "load_seconds": load_seconds,
        "inference_seconds_with_capture": inference_seconds,
        **media,
        "tensors": {name: tensor_summary(value) for name, value in captured.items()},
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(metadata_path)


if __name__ == "__main__":
    main()
