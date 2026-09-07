#!/usr/bin/env python3
"""Capture real WeMM Qwen3.5 GDN tensors as a Core ML conversion oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModel, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=448)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_array(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def tensor_summary(value: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(finite.all()),
        "minimum": float(value.min()),
        "maximum": float(value.max()),
        "mean": float(value.mean()),
        "standard_deviation": float(value.std()),
        "l2_norm": float(np.linalg.norm(value.reshape(-1))),
    }


def build_inputs(
    processor: Any,
    image_path: Path,
    image_size: int,
) -> tuple[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path),
                    "resized_height": image_size,
                    "resized_width": image_size,
                },
                {"type": "text", "text": "Represent this image."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
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
    return prompt, inputs


def main() -> None:
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    model_path = args.model.resolve()
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "mps"},
    ).eval()
    processor = AutoProcessor.from_pretrained(
        str(model_path), trust_remote_code=True, use_fast=True
    )
    model.lm_head = None
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - load_started

    layer = model.model.language_model.layers[args.layer]
    if layer.layer_type != "linear_attention":
        raise ValueError(f"layer {args.layer} is {layer.layer_type}, not linear_attention")
    gdn = layer.linear_attn
    captured: dict[str, np.ndarray] = {}
    handles: list[Any] = []

    def save(name: str, value: torch.Tensor) -> None:
        captured[name] = cpu_array(value)

    def gdn_pre_hook(
        _module: Any,
        hook_args: tuple[Any, ...],
        hook_kwargs: dict[str, Any],
    ) -> None:
        hidden = hook_kwargs.get("hidden_states")
        if hidden is None:
            hidden = hook_args[0]
        save("gdn_input", hidden)

    def gdn_output_hook(
        _module: Any, _hook_args: tuple[Any, ...], output: torch.Tensor
    ) -> None:
        save("gdn_output", output)

    def projection_hook(name: str):
        def hook(
            _module: Any, _hook_args: tuple[Any, ...], output: torch.Tensor
        ) -> None:
            save(name, output)

        return hook

    def norm_pre_hook(
        _module: Any, hook_args: tuple[Any, ...]
    ) -> None:
        save("norm_input", hook_args[0])
        save("norm_gate_z", hook_args[1])

    handles.append(gdn.register_forward_pre_hook(gdn_pre_hook, with_kwargs=True))
    handles.append(gdn.register_forward_hook(gdn_output_hook))
    handles.append(
        layer.register_forward_pre_hook(
            lambda _module, hook_args, hook_kwargs: save(
                "decoder_input",
                hook_kwargs.get("hidden_states", hook_args[0] if hook_args else None),
            ),
            with_kwargs=True,
        )
    )
    handles.append(
        layer.register_forward_hook(
            lambda _module, _hook_args, output: save(
                "decoder_output", output[0] if isinstance(output, tuple) else output
            )
        )
    )
    handles.append(
        layer.input_layernorm.register_forward_hook(
            projection_hook("input_layernorm_output")
        )
    )
    handles.append(
        layer.post_attention_layernorm.register_forward_hook(
            projection_hook("post_attention_layernorm_output")
        )
    )
    handles.append(
        layer.mlp.register_forward_pre_hook(
            lambda _module, hook_args: save("mlp_input", hook_args[0])
        )
    )
    handles.append(layer.mlp.register_forward_hook(projection_hook("mlp_output")))
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
        handles.append(
            getattr(gdn, name).register_forward_hook(projection_hook(name))
        )
    handles.append(gdn.conv1d.register_forward_hook(projection_hook("conv1d_raw")))
    handles.append(gdn.norm.register_forward_pre_hook(norm_pre_hook))
    handles.append(gdn.norm.register_forward_hook(projection_hook("norm_output")))
    handles.append(gdn.out_proj.register_forward_hook(projection_hook("out_proj")))

    original_rule = gdn.chunk_gated_delta_rule

    def capture_rule(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        save("recurrent_query", query)
        save("recurrent_key", key)
        save("recurrent_value", value)
        save("recurrent_g", kwargs["g"])
        save("recurrent_beta", kwargs["beta"])
        output, state = original_rule(query, key, value, **kwargs)
        save("recurrent_output", output)
        if state is not None:
            save("recurrent_final_state", state)
        return output, state

    gdn.chunk_gated_delta_rule = capture_rule
    prompt, inputs = build_inputs(processor, image_path, args.image_size)
    input_ids = inputs["input_ids"]
    inputs = inputs.to("mps")
    inference_started = time.perf_counter()
    with torch.inference_mode():
        embedding = model.embedding(**inputs)
    torch.mps.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    save("final_embedding", embedding)

    gdn.chunk_gated_delta_rule = original_rule
    for handle in handles:
        handle.remove()

    required = {
        "gdn_input",
        "decoder_input",
        "decoder_output",
        "input_layernorm_output",
        "post_attention_layernorm_output",
        "mlp_input",
        "mlp_output",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "conv1d_raw",
        "recurrent_query",
        "recurrent_key",
        "recurrent_value",
        "recurrent_g",
        "recurrent_beta",
        "recurrent_output",
        "norm_input",
        "norm_gate_z",
        "norm_output",
        "out_proj",
        "gdn_output",
        "final_embedding",
    }
    missing = required - captured.keys()
    if missing:
        raise RuntimeError(f"oracle capture incomplete: {sorted(missing)}")
    if not np.array_equal(captured["out_proj"], captured["gdn_output"]):
        raise RuntimeError("out_proj and GDN output differ")

    np.savez_compressed(output_dir / "gdn_oracle.npz", **captured)
    metadata = {
        "model": str(model_path),
        "model_weights_sha256": sha256(model_path / "model.safetensors"),
        "image": str(image_path),
        "image_sha256": sha256(image_path),
        "layer": args.layer,
        "layer_type": layer.layer_type,
        "image_size": args.image_size,
        "sequence_length": int(input_ids.shape[1]),
        "input_ids": input_ids.tolist(),
        "prompt": prompt,
        "load_seconds": load_seconds,
        "inference_seconds_with_capture": inference_seconds,
        "tensors": {name: tensor_summary(value) for name, value in captured.items()},
    }
    metadata_path = output_dir / "gdn_oracle.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(metadata_path)


if __name__ == "__main__":
    main()
