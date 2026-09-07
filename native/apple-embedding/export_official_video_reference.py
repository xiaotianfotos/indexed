#!/usr/bin/env python3
"""Export one official PyTorch WeMM video embedding for native-runtime parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import av
import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    arguments = parser.parse_args()

    container = av.open(str(arguments.video))
    stream = container.streams.video[0]
    all_frames = [
        torch.from_numpy(frame.to_rgb().to_ndarray()).permute(2, 0, 1)
        for frame in container.decode(stream)
    ]
    source_fps = float(stream.average_rate)
    requested_frames = max(2, int((len(all_frames) / source_fps) * 2.0))
    frame_count = min(64, requested_frames)
    frame_count -= frame_count % 2
    indices = torch.linspace(0, len(all_frames) - 1, frame_count).round().long().tolist()
    video = torch.stack([all_frames[index] for index in indices])
    video = functional.interpolate(
        video.float(),
        size=(448, 448),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    metadata = {
        "fps": source_fps,
        "frames_indices": indices,
        "total_num_frames": len(all_frames),
        "video_backend": "pyav-parity",
    }

    processor = AutoProcessor.from_pretrained(
        str(arguments.model), trust_remote_code=True, use_fast=True
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(arguments.video)},
                {"type": "text", "text": "Represent this video."},
            ],
        }
    ]
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

    load_started = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(arguments.model),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": arguments.device},
    ).eval()
    if arguments.device == "mps":
        torch.mps.synchronize()
    load_seconds = time.perf_counter() - load_started

    inputs = inputs.to(arguments.device)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        vector = model.embedding(**inputs)
    if arguments.device == "mps":
        torch.mps.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    values = vector[0].float().cpu().tolist()
    input_ids = inputs.input_ids[0].cpu().tolist()
    output = {
        "schema": "indexed-wemm-official-video-reference/v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "model": str(arguments.model.resolve()),
        "video": str(arguments.video.resolve()),
        "device": arguments.device,
        "dtype": "bfloat16",
        "source_frames": len(all_frames),
        "source_fps": source_fps,
        "sampled_frame_indices": indices,
        "video_grid_thw": inputs.video_grid_thw.cpu().tolist(),
        "prompt_tokens": len(input_ids),
        "input_ids_sha256": hashlib.sha256(
            b"".join(int(token).to_bytes(4, "little") for token in input_ids)
        ).hexdigest(),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "dimension": len(values),
        "vector": values,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in output.items() if key != "vector"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
