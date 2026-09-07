#!/usr/bin/env python3
"""Capture a real WeMM Qwen3.5 full-attention decoder layer oracle."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from capture_gdn_oracle import build_inputs, cpu_array, sha256, tensor_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=448)
    return parser.parse_args()


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
    if layer.layer_type != "full_attention":
        raise ValueError(f"layer {args.layer} is {layer.layer_type}, not full_attention")
    attention = layer.self_attn
    captured: dict[str, np.ndarray] = {}
    handles: list[Any] = []

    def save(name: str, value: torch.Tensor | None) -> None:
        if value is not None:
            captured[name] = cpu_array(value)

    def output_hook(name: str):
        def hook(_module: Any, _hook_args: tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            save(name, value)

        return hook

    def attention_pre_hook(
        _module: Any,
        hook_args: tuple[Any, ...],
        hook_kwargs: dict[str, Any],
    ) -> None:
        save(
            "attention_input",
            hook_kwargs.get("hidden_states", hook_args[0] if hook_args else None),
        )
        position_embeddings = hook_kwargs.get("position_embeddings")
        if position_embeddings is not None:
            save("position_cos", position_embeddings[0])
            save("position_sin", position_embeddings[1])
        save("attention_mask", hook_kwargs.get("attention_mask"))
        save("position_ids", hook_kwargs.get("position_ids"))

    handles.append(
        layer.register_forward_pre_hook(
            lambda _module, hook_args, hook_kwargs: save(
                "decoder_input",
                hook_kwargs.get("hidden_states", hook_args[0] if hook_args else None),
            ),
            with_kwargs=True,
        )
    )
    handles.append(layer.register_forward_hook(output_hook("decoder_output")))
    handles.append(
        layer.input_layernorm.register_forward_hook(
            output_hook("input_layernorm_output")
        )
    )
    handles.append(
        layer.post_attention_layernorm.register_forward_hook(
            output_hook("post_attention_layernorm_output")
        )
    )
    handles.append(
        layer.mlp.register_forward_pre_hook(
            lambda _module, hook_args: save("mlp_input", hook_args[0])
        )
    )
    handles.append(layer.mlp.register_forward_hook(output_hook("mlp_output")))
    handles.append(
        attention.register_forward_pre_hook(attention_pre_hook, with_kwargs=True)
    )
    handles.append(attention.register_forward_hook(output_hook("attention_output")))
    for name in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj"):
        handles.append(
            getattr(attention, name).register_forward_hook(output_hook(name))
        )

    prompt, inputs = build_inputs(processor, image_path, args.image_size)
    input_ids = inputs["input_ids"]
    inputs = inputs.to("mps")
    inference_started = time.perf_counter()
    with torch.inference_mode():
        embedding = model.embedding(**inputs)
    torch.mps.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    save("final_embedding", embedding)
    for handle in handles:
        handle.remove()

    required = {
        "decoder_input",
        "decoder_output",
        "input_layernorm_output",
        "attention_input",
        "position_cos",
        "position_sin",
        "q_proj",
        "k_proj",
        "v_proj",
        "q_norm",
        "k_norm",
        "o_proj",
        "attention_output",
        "post_attention_layernorm_output",
        "mlp_input",
        "mlp_output",
        "final_embedding",
    }
    missing = required - captured.keys()
    if missing:
        raise RuntimeError(f"oracle capture incomplete: {sorted(missing)}")
    if not np.array_equal(captured["o_proj"], captured["attention_output"]):
        raise RuntimeError("o_proj and attention output differ")

    np.savez_compressed(output_dir / "attention_oracle.npz", **captured)
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
    metadata_path = output_dir / "attention_oracle.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(metadata_path)


if __name__ == "__main__":
    main()
