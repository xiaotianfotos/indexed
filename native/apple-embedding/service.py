#!/usr/bin/env python3
"""Apple-local WeMM embedding service for Indexed.

The public service contract stays smaller than the runtime behind it. Indexed
sends OpenAI-style ``messages`` to ``/v1/embeddings``; the default backend runs
the fixed WeMM vision tower with Core ML/ANE and the Qwen3.5 language model
with MLX/Metal. The older PyTorch MPS backend remains a correctness baseline.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import io
import json
import os
import re
import signal
import sys
import threading
import time
import types
from dataclasses import dataclass
from contextlib import nullcontext
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


OFFICIAL_DIMENSIONS = (64, 128, 256, 512, 1024, 2048)
MODEL_PREFIX = "wemm-embedding-2b-apple"
DEFAULT_PORT = 18768
DEFAULT_IMAGE_SIZE = 448


def inference_modules(*modules: Any) -> None:
    """MLX defaults to training mode; array evaluation does not change it."""
    for module in modules:
        module.eval()


def tokenize_embedding_prompt(tokenizer: Any, prompt: str) -> Any:
    # The explicit chat template already ends in <embedding>. The package's
    # tokenizer post-processor appends another one unless disabled, diverging
    # from Swift's addSpecialTokens:false and the intended pooled position.
    return tokenizer(prompt, return_tensors="np", add_special_tokens=False)


def execution_settings(mode: str | None) -> tuple[str, str]:
    """Map the article's A-D modes to public and private runtime controls."""

    if mode == "a":
        return "cpu_gpu", "none"
    if mode == "b":
        return "cpu_ane", "none"
    if mode == "c":
        return "cpu_ane", "mlp"
    if mode == "d":
        return "cpu_ane", "quality-gated"
    if mode == "e":
        raise ValueError("E mode is retired; explicitly choose B or experimental A/C/D")
    raise ValueError(f"unknown execution mode: {mode}")


def parse_layer_slots(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    slots = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not slots or any(slot < 0 or slot >= 18 for slot in slots):
        raise ValueError("private ANE recurrence layer slots must be within 0..17")
    return slots


class RequestError(ValueError):
    """A client-visible request failure with an HTTP status."""

    def __init__(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = int(status)


def model_id(dimension: int) -> str:
    if dimension not in OFFICIAL_DIMENSIONS:
        raise ValueError(f"Unsupported WeMM dimension: {dimension}")
    return f"{MODEL_PREFIX}-{dimension}"


def dimension_for_model(value: object, default_dimension: int) -> int:
    selected = str(value or "").strip().lower()
    if not selected or selected == MODEL_PREFIX:
        return default_dimension
    match = re.fullmatch(rf"{re.escape(MODEL_PREFIX)}-(\d+)", selected)
    if not match:
        raise RequestError(f"未知模型：{selected}", HTTPStatus.NOT_FOUND)
    dimension = int(match.group(1))
    if dimension not in OFFICIAL_DIMENSIONS:
        raise RequestError(
            f"模型 {selected} 的维度不受支持；可选 {list(OFFICIAL_DIMENSIONS)}"
        )
    return dimension


def _content_items(content: object) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise RequestError("message.content 必须是字符串或数组")
    output: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            raise RequestError("message.content 中的每一项必须是对象")
        output.append(item)
    return output


def _data_uri(value: object) -> tuple[str, bytes]:
    raw = str(value or "").strip()
    match = re.fullmatch(r"data:([^;,]+);base64,(.*)", raw, flags=re.DOTALL)
    if not match:
        raise RequestError("Apple helper 只接受 base64 data URI 图片")
    mime_type = match.group(1).lower()
    if not mime_type.startswith("image/"):
        raise RequestError(f"不支持的媒体类型：{mime_type}")
    try:
        payload = base64.b64decode(match.group(2), validate=True)
    except (binascii.Error, ValueError) as error:
        raise RequestError("图片 base64 无效") from error
    if not payload:
        raise RequestError("图片内容为空")
    return mime_type, payload


def normalize_messages(
    messages: object,
    *,
    image_size: int,
) -> tuple[list[dict[str, Any]], int]:
    """Convert Indexed/OpenAI message items to qwen-vl-utils input objects."""

    if not isinstance(messages, list) or not messages:
        raise RequestError("messages 必须是非空数组")

    # Pillow stays a runtime dependency instead of being required for contract
    # tests that only exercise text inputs.
    from PIL import Image

    normalized: list[dict[str, Any]] = []
    image_count = 0
    visual_count = 0
    for message in messages:
        if not isinstance(message, dict):
            raise RequestError("messages 中的每一项必须是对象")
        role = str(message.get("role") or "user").strip().lower()
        if role not in {"system", "user"}:
            raise RequestError(f"不支持的 message role：{role}")
        content: list[dict[str, Any]] = []
        for item in _content_items(message.get("content")):
            kind = str(item.get("type") or "text").strip().lower()
            if kind == "text":
                content.append({"type": "text", "text": str(item.get("text") or "")})
                continue
            if kind in {"image", "image_url"}:
                source: object = item.get("image")
                if kind == "image_url":
                    image_url = item.get("image_url")
                    source = image_url.get("url") if isinstance(image_url, dict) else image_url
                _, payload = _data_uri(source)
                try:
                    image = Image.open(io.BytesIO(payload)).convert("RGB")
                    image.load()
                except Exception as error:
                    raise RequestError("无法解码图片") from error
                image_count += 1
                visual_count += 1
                content.append(
                    {
                        "type": "image",
                        "image": image,
                        "resized_height": image_size,
                        "resized_width": image_size,
                    }
                )
                continue
            if kind == "video_frames":
                frame_values = item.get("frames")
                if not isinstance(frame_values, list) or not frame_values:
                    raise RequestError("video_frames.frames 必须是非空数组")
                if len(frame_values) > 64:
                    raise RequestError("视频帧包最多支持 64 帧", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                frames: list[dict[str, Any]] = []
                previous_timestamp = -1.0
                for frame_index, frame in enumerate(frame_values):
                    if not isinstance(frame, dict):
                        raise RequestError("video_frames 中的每一帧必须是对象")
                    image_url = frame.get("image_url")
                    source = image_url.get("url") if isinstance(image_url, dict) else image_url
                    if source is None:
                        source = frame.get("url")
                    _, payload = _data_uri(source)
                    try:
                        image = Image.open(io.BytesIO(payload)).convert("RGB")
                        image.load()
                    except Exception as error:
                        raise RequestError(f"视频第 {frame_index + 1} 帧无法解码") from error
                    raw_timestamp = frame.get("timestamp", frame_index / 2.0)
                    try:
                        timestamp = float(raw_timestamp)
                    except (TypeError, ValueError) as error:
                        raise RequestError("视频帧时间戳必须是数字") from error
                    if not (timestamp >= 0.0 and timestamp >= previous_timestamp):
                        raise RequestError("视频帧时间戳必须递增且不能为负数")
                    previous_timestamp = timestamp
                    frames.append({"image": image, "timestamp": timestamp})
                visual_count += 1
                content.append({"type": "video_frames", "frames": frames})
                continue
            if kind in {"video", "video_url"}:
                raise RequestError(
                    "Apple helper MVP 暂不支持视频；请先为该项目关闭 video 索引",
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                )
            raise RequestError(f"不支持的 content type：{kind}")
        if not content:
            raise RequestError("message.content 不能为空")
        normalized.append({"role": role, "content": content})

    if visual_count > 1:
        raise RequestError(
            "Apple helper 每次只支持一个视觉输入",
            HTTPStatus.UNPROCESSABLE_ENTITY,
        )
    return normalized, image_count


@dataclass(frozen=True)
class EmbeddingResult:
    vector: list[float]
    timings_ms: dict[str, float]
    modality: str
    prompt_tokens: int = 0


def private_ane_tail_threshold(sequence_length: int) -> int:
    """Do not inflate a short request to more than twice its useful tile rows."""
    return max(1, (sequence_length + 1) // 2)


class AppleWeMMEngine:
    """Persistent Core ML vision + MPS language WeMM runtime."""

    def __init__(
        self,
        *,
        model_path: Path,
        coreml_model_path: Path,
        image_size: int,
        compute_unit: str,
        warmup: bool,
    ) -> None:
        import coremltools as ct
        import torch
        from transformers import AutoModel, AutoProcessor
        from transformers.utils import logging as transformers_logging

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS 不可用；Apple helper 只能运行在 Apple Silicon Mac")
        if not model_path.is_dir():
            raise FileNotFoundError(f"WeMM 模型目录不存在：{model_path}")
        if not coreml_model_path.exists():
            raise FileNotFoundError(f"Core ML 视觉模型不存在：{coreml_model_path}")

        compute_units = {
            "cpu_ane": ct.ComputeUnit.CPU_AND_NE,
            "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU,
            "all": ct.ComputeUnit.ALL,
        }
        if compute_unit not in compute_units:
            raise ValueError(f"未知 compute unit：{compute_unit}")

        self._torch = torch
        self._ct = ct
        self._image_size = image_size
        self._lock = threading.Lock()
        self._image_features: Any | None = None

        transformers_logging.disable_progress_bar()
        started = time.perf_counter()
        self.model = AutoModel.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": "mps"},
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=True, use_fast=True
        )
        torch.mps.synchronize()

        # Embedding never touches the vocabulary projection.  Core ML owns the
        # visual tower, so neither module needs to remain resident on MPS.
        self.model.lm_head = None
        self.model.model.visual = None

        def get_image_features(
            _model: Any,
            pixel_values: Any,
            image_grid_thw: Any | None = None,
            **kwargs: Any,
        ) -> Any:
            del _model, pixel_values, image_grid_thw, kwargs
            from transformers.modeling_outputs import BaseModelOutputWithPooling

            if self._image_features is None:
                raise RuntimeError("Core ML image features 尚未安装")
            return BaseModelOutputWithPooling(pooler_output=(self._image_features,))

        self.model.model.get_image_features = types.MethodType(
            get_image_features, self.model.model
        )
        torch.mps.empty_cache()
        torch.mps.synchronize()

        model_class = (
            ct.models.CompiledMLModel
            if coreml_model_path.suffix == ".mlmodelc"
            else ct.models.MLModel
        )
        self.vision = model_class(
            str(coreml_model_path),
            compute_units=compute_units[compute_unit],
            optimization_hints={
                "specializationStrategy": ct.SpecializationStrategy.FastPrediction
            },
        )
        self.warmup_seconds = 0.0
        if warmup:
            warmup_started = time.perf_counter()
            self._warmup()
            self.warmup_seconds = time.perf_counter() - warmup_started
        self.load_seconds = time.perf_counter() - started
        self.mps_allocated_bytes = int(torch.mps.current_allocated_memory())

    def _warmup(self) -> None:
        """Pay the largest MPS/Core ML first-evaluation costs before ready."""

        from PIL import Image

        image = Image.new("RGB", (8, 8), (127, 127, 127))
        output = io.BytesIO()
        image.save(output, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
        self.embed(
            [{"role": "user", "content": [{"type": "text", "text": "warmup"}]}],
            dimension=256,
        )
        self.embed(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": uri}},
                        {"type": "text", "text": "Represent this image."},
                    ],
                }
            ],
            dimension=256,
        )

    def _inputs(self, messages: list[dict[str, Any]]) -> tuple[Any, int]:
        from qwen_vl_utils import process_vision_info

        prompt = self.processor.apply_chat_template(
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
        video_metadata = None
        if videos is not None:
            videos, video_metadata = zip(*videos)
            videos = list(videos)
            video_metadata = list(video_metadata)
        inputs = self.processor(
            text=prompt,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            return_tensors="pt",
            **video_kwargs,
        )
        image_count = 0 if images is None else len(images)
        return inputs, image_count

    def embed(self, raw_messages: object, *, dimension: int) -> EmbeddingResult:
        import numpy as np
        import torch.nn.functional as functional

        messages, declared_images = normalize_messages(
            raw_messages, image_size=self._image_size
        )
        total_started = time.perf_counter()
        with self._lock, self._torch.inference_mode():
            preprocess_started = time.perf_counter()
            inputs, image_count = self._inputs(messages)
            preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
            if image_count != declared_images:
                raise RuntimeError(
                    f"processor 图片数量不一致：{image_count} != {declared_images}"
                )

            vision_ms = 0.0
            self._image_features = None
            if image_count:
                patches = inputs["pixel_values"].float().numpy()
                vision_started = time.perf_counter()
                prediction = self.vision.predict({"patches": patches})
                vision_ms = (time.perf_counter() - vision_started) * 1000
                visual_output = np.asarray(prediction["image_embeds"], dtype=np.float32)
                self._image_features = self._torch.from_numpy(visual_output).to(
                    device="mps", dtype=self._torch.bfloat16
                )

            mps_inputs = inputs.to("mps")
            self._torch.mps.synchronize()
            language_started = time.perf_counter()
            embedding = self.model.embedding(**mps_inputs)
            self._torch.mps.synchronize()
            language_ms = (time.perf_counter() - language_started) * 1000

            # The released model already normalizes the full vector.  Slice the
            # Matryoshka prefix and normalize once more in FP32 for storage.
            vector = functional.normalize(
                embedding[0].float().cpu()[:dimension], dim=-1
            ).numpy()
            self._image_features = None

        total_ms = (time.perf_counter() - total_started) * 1000
        return EmbeddingResult(
            vector=[float(value) for value in vector],
            timings_ms={
                "preprocess": preprocess_ms,
                "vision": vision_ms,
                "language": language_ms,
                "total": total_ms,
            },
            modality="image" if declared_images else "text",
        )


class MLXWeMMEngine:
    """Core ML vision + MLX Qwen3.5 language runtime without PyTorch."""

    backend = "coreml-ane+mlx-gpu"

    def __init__(
        self,
        *,
        package_path: Path,
        image_size: int,
        compute_unit: str,
        warmup: bool,
    ) -> None:
        import coremltools as ct
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.models.qwen3_5 import Model as MLXLMModel
        from mlx_lm.models.qwen3_5 import ModelArgs as MLXLMModelArgs
        from mlx_vlm.models.qwen3_5.config import ModelConfig
        from mlx_vlm.models.qwen3_5.language import LanguageModel
        from transformers import AutoTokenizer, Qwen2VLImageProcessorPil

        package_path = package_path.resolve()
        self.package_path = package_path
        manifest_path = package_path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"WeMM Apple manifest 不存在：{manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("runtime_semantics") != "wemm-apple-embedding-v1":
            raise RuntimeError("WeMM Apple package runtime_semantics 不受支持")

        language = manifest.get("language") or {}
        vision = manifest.get("vision") or {}
        language_path = package_path / str(language.get("path") or "")
        config_path = language_path.parent / "config.json"
        coreml_model_path = package_path / str(vision.get("path") or "")
        if not language_path.is_file() or not config_path.is_file():
            raise FileNotFoundError("WeMM Apple language package 不完整")
        if not coreml_model_path.exists():
            raise FileNotFoundError(f"Core ML 视觉模型不存在：{coreml_model_path}")

        compute_units = {
            "cpu_ane": ct.ComputeUnit.CPU_AND_NE,
            "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU,
            "all": ct.ComputeUnit.ALL,
        }
        if compute_unit not in compute_units:
            raise ValueError(f"未知 compute unit：{compute_unit}")

        self._mx = mx
        self._vision_lock = threading.Lock()
        self._pipeline_gate = threading.BoundedSemaphore(1)
        self._ct = ct
        self._image_size = image_size
        self._lock = threading.Lock()
        self.compute_unit = compute_unit
        self.backend = {
            "cpu_ane": "coreml-ane+mlx-gpu",
            "cpu_gpu": "coreml-gpu+mlx-gpu",
            "all": "coreml-auto+mlx-gpu",
        }[compute_unit]
        self.manifest = manifest
        self.package_fingerprint = str(manifest["package_fingerprint"])
        self.embedding_space_template = str(manifest["embedding_space_template"])
        self.image_token_id = int(manifest["image_token_id"])
        self.video_token_id = int(manifest["video_token_id"])
        self.embedding_token_id = int(manifest["embedding_token_id"])

        started = time.perf_counter()
        config_dict = json.loads(config_path.read_text(encoding="utf-8"))
        config = ModelConfig.from_dict(config_dict)
        self.language = LanguageModel(config.text_config, config)
        # WeMM returns hidden states and never computes vocabulary logits.
        if hasattr(self.language, "lm_head"):
            del self.language.lm_head
        weights = mx.load(str(language_path))
        quantization = config_dict.get("quantization")
        if quantization:
            nn.quantize(
                self.language,
                group_size=int(quantization["group_size"]),
                bits=int(quantization["bits"]),
                mode=str(quantization.get("mode", "affine")),
                class_predicate=lambda path, _module: f"{path}.scales" in weights,
            )
        self.language.load_weights(list(weights.items()), strict=True)
        inference_modules(self.language)
        mx.eval(self.language.parameters())

        # The Apple MLX-LM text path avoids the general multimodal MRoPE
        # dispatcher. Both module trees reference the same MLX arrays, so this
        # improves short-text latency without duplicating the 3.76 GB weights.
        self.text_language = MLXLMModel(MLXLMModelArgs.from_dict(config_dict))
        if hasattr(self.text_language.language_model, "lm_head"):
            del self.text_language.language_model.lm_head
        text_weights = {f"language_model.{key}": value for key, value in weights.items()}
        if quantization:
            nn.quantize(
                self.text_language,
                group_size=int(quantization["group_size"]),
                bits=int(quantization["bits"]),
                mode=str(quantization.get("mode", "affine")),
                class_predicate=lambda path, _module: f"{path}.scales"
                in text_weights,
            )
        self.text_language.load_weights(list(text_weights.items()), strict=True)
        inference_modules(self.text_language)
        mx.eval(self.text_language.parameters())
        del text_weights
        del weights

        self.tokenizer = AutoTokenizer.from_pretrained(str(package_path))
        self.image_processor = Qwen2VLImageProcessorPil.from_pretrained(
            str(package_path)
        )
        model_class = (
            ct.models.CompiledMLModel
            if coreml_model_path.suffix == ".mlmodelc"
            else ct.models.MLModel
        )
        self.vision = model_class(
            str(coreml_model_path),
            compute_units=compute_units[compute_unit],
            optimization_hints={
                # FastPrediction made first startup ~15-25 seconds longer for
                # this fixed vision tower without improving its ~120 ms steady
                # prediction on the M4 test machine.
                "specializationStrategy": ct.SpecializationStrategy.Default
            },
        )
        self.warmup_seconds = 0.0
        if warmup:
            warmup_started = time.perf_counter()
            self._warmup()
            self.warmup_seconds = time.perf_counter() - warmup_started
        self.load_seconds = time.perf_counter() - started
        self.allocated_bytes = int(mx.get_active_memory())
        # Kept for the first HTTP contract; health also exposes allocated_bytes.
        self.mps_allocated_bytes = self.allocated_bytes

    def _warmup(self) -> None:
        from PIL import Image

        image = Image.new("RGB", (8, 8), (127, 127, 127))
        output = io.BytesIO()
        image.save(output, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
        self.embed(
            [{"role": "user", "content": [{"type": "text", "text": "warmup"}]}],
            dimension=256,
        )
        self.embed(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": uri}},
                        {"type": "text", "text": "Represent this image."},
                    ],
                }
            ],
            dimension=256,
        )

    def _inputs(
        self, messages: list[dict[str, Any]], image_count: int
    ) -> tuple[Any, list[Any], Any | None, Any | None, int | None, str]:
        import mlx.core as mx
        import numpy as np
        from PIL import Image

        # Keep this byte-for-byte aligned with embedding_chat_template.jinja and
        # the native Swift implementation. The tokenizer's default chat template
        # is a generation template and does not append WeMM's <embedding> token.
        prompt_parts: list[str] = []
        for message_index, message in enumerate(messages):
            prompt_parts.append(f"<|im_start|>{message['role']}")
            for item_index, item in enumerate(message["content"]):
                kind = item.get("type")
                if kind == "text":
                    if item_index == 0:
                        prompt_parts.append("\n")
                    prompt_parts.append(str(item.get("text") or ""))
                elif kind == "image":
                    prompt_parts.append("<|vision_start|><|image_pad|><|vision_end|>")
                elif kind == "video_frames":
                    prompt_parts.append("\n<|video_pad|>")
                else:
                    raise RuntimeError(f"MLX prompt 不支持视觉类型：{kind}")
            prompt_parts.append("<|im_end|>")
            if message_index + 1 < len(messages):
                prompt_parts.append("\n")
        prompt_parts.append("<embedding>")
        prompt = "".join(prompt_parts)

        patch_batches: list[Any] = []
        image_grid_thw = None
        video_grid_thw = None
        visual_token_id = None
        modality = "text"
        if image_count:
            images = [
                item["image"]
                for message in messages
                for item in message["content"]
                if item.get("type") == "image"
            ]
            if len(images) != 1:
                raise RuntimeError(f"MLX processor 预期一张图片，实际 {len(images)}")
            image = images[0].resize(
                (self._image_size, self._image_size), Image.Resampling.BICUBIC
            )
            image_inputs = self.image_processor(images=[image], return_tensors="np")
            patch_batches = [np.asarray(image_inputs["pixel_values"], dtype=np.float32)]
            image_grid_thw = np.asarray(image_inputs["image_grid_thw"], dtype=np.int64)
            merge_length = int(self.image_processor.merge_size) ** 2
            image_tokens = int(image_grid_thw[0].prod()) // merge_length
            prompt = prompt.replace(
                self.tokenizer.image_token,
                self.tokenizer.image_token * image_tokens,
                1,
            )
            visual_token_id = self.image_token_id
            modality = "image"
        else:
            video_items = [
                item
                for message in messages
                for item in message["content"]
                if item.get("type") == "video_frames"
            ]
            if video_items:
                if len(video_items) != 1:
                    raise RuntimeError(f"MLX processor 预期一个视频帧包，实际 {len(video_items)}")
                frames = list(video_items[0]["frames"])
                if len(frames) == 1:
                    frames.append(frames[0])
                elif len(frames) % 2:
                    frames.pop()
                timestamps: list[float] = []
                for index in range(0, len(frames), 2):
                    rgb_frames: list[Any] = []
                    for frame in frames[index : index + 2]:
                        image = frame["image"].resize(
                            (self._image_size, self._image_size), Image.Resampling.BICUBIC
                        )
                        rgb = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
                        rgb_frames.append(np.transpose(rgb, (2, 0, 1)))
                    # Qwen3.5 groups two sampled frames into one temporal patch.
                    # This layout is identical to Qwen2VLImageProcessorPil for
                    # a duplicated still image and to the native Swift backend.
                    pair = np.stack(rgb_frames, axis=0)
                    grid_h = self._image_size // 16
                    grid_w = self._image_size // 16
                    pair = pair.reshape(
                        1, 2, 3,
                        grid_h // 2, 2, 16,
                        grid_w // 2, 2, 16,
                    )
                    pair = pair.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
                    patch_batches.append(
                        np.asarray(
                            pair.reshape(grid_h * grid_w, 3 * 2 * 16 * 16),
                            dtype=np.float32,
                        )
                    )
                    timestamps.append(
                        (float(frames[index]["timestamp"]) + float(frames[index + 1]["timestamp"])) / 2.0
                    )
                video_tokens_per_pair = (self._image_size // 16) ** 2 // 4
                video_pads = self.tokenizer.video_token * video_tokens_per_pair
                video_prompt = "".join(
                    f"<{timestamp:.1f} seconds><|vision_start|>{video_pads}<|vision_end|>"
                    for timestamp in timestamps
                )
                prompt = prompt.replace(self.tokenizer.video_token, video_prompt, 1)
                # mlx-vlm's Python Qwen3.5 MRoPE implementation expects one
                # [1,h,w] row per timestamped temporal pair.
                video_grid_thw = np.asarray(
                    [[1, grid_h, grid_w] for _ in timestamps], dtype=np.int64
                )
                visual_token_id = self.video_token_id
                modality = "video"

        tokenized = tokenize_embedding_prompt(self.tokenizer, prompt)
        input_ids_np = np.asarray(tokenized["input_ids"], dtype=np.int32)
        if input_ids_np.shape[1] > 8192:
            raise RequestError("输入超过 WeMM 8192 token 限制", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        if int(input_ids_np[0, -1]) != self.embedding_token_id:
            raise RuntimeError("WeMM prompt 最后一个 token 不是 <embedding>")
        return (
            mx.array(input_ids_np),
            patch_batches,
            None if image_grid_thw is None else mx.array(image_grid_thw),
            None if video_grid_thw is None else mx.array(video_grid_thw),
            visual_token_id,
            modality,
        )

    def _hidden(
        self,
        input_ids: Any,
        image_features: Any | None,
        image_grid_thw: Any | None,
        video_grid_thw: Any | None = None,
        visual_token_id: int | None = None,
    ) -> Any:
        mx = self._mx
        if image_features is None and image_grid_thw is None and video_grid_thw is None:
            # The dedicated text tree is the fastest public MLX path, but the
            # private ANE patches are installed on the multimodal language
            # tree. Both trees reference the same weights, so C/D must use the
            # patched tree for text as well as image requests.
            runtime = getattr(self, "_private_ane_runtime", None)
            if runtime is not None and int(input_ids.shape[1]) >= private_ane_tail_threshold(runtime["sequence_length"]):
                position_ids, _ = self.language.get_rope_index(
                    input_ids,
                    image_grid_thw=None,
                    attention_mask=None,
                )
                return self.language.model(input_ids, position_ids=position_ids)
            return self.text_language.model(input_ids)

        inputs_embeds = None
        if image_features is not None:
            token_id = self.image_token_id if visual_token_id is None else visual_token_id
            positions = [
                index
                for index, value in enumerate(input_ids[0].tolist())
                if int(value) == token_id
            ]
            if not positions:
                raise RuntimeError("WeMM 视觉 token 为空")
            if len(positions) != int(image_features.shape[0]):
                raise RuntimeError(
                    "Core ML 视觉特征数量与视觉 token 不一致："
                    f"{image_features.shape[0]} != {len(positions)}"
                )
            token_embeddings = self.language.model.embed_tokens(input_ids)
            # Video tokens appear in multiple timestamped vision blocks. Build
            # contiguous slices so their Core ML features can be injected while
            # retaining the text and boundary embeddings between blocks.
            parts: list[Any] = []
            cursor = 0
            feature_cursor = 0
            run_start = positions[0]
            run_end = positions[0]
            for position in positions[1:] + [positions[-1] + 2]:
                if position == run_end + 1:
                    run_end = position
                    continue
                if cursor < run_start:
                    parts.append(token_embeddings[:, cursor:run_start, :])
                run_length = run_end - run_start + 1
                parts.append(image_features[None, feature_cursor : feature_cursor + run_length, :])
                feature_cursor += run_length
                cursor = run_end + 1
                run_start = position
                run_end = position
            if cursor < int(input_ids.shape[1]):
                parts.append(token_embeddings[:, cursor:, :])
            inputs_embeds = mx.concatenate(parts, axis=1)

        position_ids, _ = self.language.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=None,
        )
        return self.language.model(
            input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )

    def embed(self, raw_messages: object, *, dimension: int) -> EmbeddingResult:
        import numpy as np

        messages, declared_images = normalize_messages(
            raw_messages, image_size=self._image_size
        )
        total_started = time.perf_counter()
        with self._pipeline_gate:
            with self._vision_lock:
                preprocess_started = time.perf_counter()
                input_ids, patch_batches, image_grid_thw, video_grid_thw, visual_token_id, modality = self._inputs(
                    messages, declared_images
                )
                preprocess_ms = (time.perf_counter() - preprocess_started) * 1000

                vision_ms = 0.0
                image_features = None
                if patch_batches:
                    vision_started = time.perf_counter()
                    feature_parts: list[Any] = []
                    for patches in patch_batches:
                        lane = getattr(self, "_ane_lane", None)
                        with lane.claim() if lane is not None else nullcontext():
                            prediction = self.vision.predict({"patches": patches})
                        feature_parts.append(
                            np.asarray(prediction["image_embeds"], dtype=np.float32)
                        )
                    vision_ms = (time.perf_counter() - vision_started) * 1000
                    visual_output = np.concatenate(feature_parts, axis=0)
                    image_features = self._mx.array(visual_output).astype(
                        self._mx.bfloat16
                    )

            with self._lock:
                language_started = time.perf_counter()
                video_projection = getattr(self, "_video_projection", None)
                if video_projection is not None:
                    video_projection.video_active = video_grid_thw is not None
                hidden = self._hidden(
                    input_ids,
                    image_features,
                    image_grid_thw,
                    video_grid_thw,
                    visual_token_id,
                )
                vector = hidden[0, -1, :dimension].astype(self._mx.float32)
                vector = vector / self._mx.linalg.norm(vector)
                self._mx.eval(vector)
                language_ms = (time.perf_counter() - language_started) * 1000
                vector_np = np.asarray(vector, dtype=np.float32)

        total_ms = (time.perf_counter() - total_started) * 1000
        return EmbeddingResult(
            vector=[float(value) for value in vector_np],
            timings_ms={
                "preprocess": preprocess_ms,
                "vision": vision_ms,
                "language": language_ms,
                "total": total_ms,
            },
            modality=modality,
            prompt_tokens=int(input_ids.shape[1]),
        )


def enable_private_ane_runtime(
    engine: MLXWeMMEngine,
    *,
    mode: str,
    sequence_length: int,
    mlp_fraction: float,
    mlp_variant: int,
    mlp_max_layers: int,
    experiment_root: Path | None,
    recurrence_profile_path: Path | None,
    bridge_path: Path | None,
    mil_path: Path | None,
    recurrence_block_size: int,
    recurrence_layer_slots: list[int] | None,
    recurrence_query_scale: float,
    recurrence_max_tokens: int,
    recurrence_io_dtype: str,
    recurrence_verify_reference: bool,
) -> None:
    """Enable an explicitly unsupported private-ANE research configuration.

    The default service never calls this function.  ``mlp`` enables oMLX's
    fixed-shape MLP kernels.  ``quality-gated`` additionally loads a measured
    stable-block GDN profile.  It may cover all 18 recurrence layers or a
    calibrated subset, but compilation alone is never treated as proof of
    numerical correctness.
    """
    if mode == "none":
        return
    if sequence_length < 64 or sequence_length % 64:
        raise ValueError("private ANE sequence length must be a multiple of 64")
    if not 0.0 < mlp_fraction <= 1.0:
        raise ValueError("private ANE MLP fraction must be within (0, 1]")
    if mlp_max_layers < 1 or mlp_max_layers > 24:
        raise ValueError("private ANE MLP max layers must be within 1..24")
    if recurrence_block_size not in {2, 4, 8}:
        raise ValueError("private ANE recurrence block size must be 2, 4, or 8")
    if recurrence_max_tokens < 64 or recurrence_max_tokens % 64:
        raise ValueError("private ANE recurrence max tokens must be a multiple of 64")
    if recurrence_io_dtype not in {"fp16", "fp32"}:
        raise ValueError("private ANE recurrence I/O dtype must be fp16 or fp32")

    try:
        from omlx.custom_kernels.qwen35_prefill import fast
        from omlx.patches.qwen35_ane_prefill import (
            ane_prefill_transient_bytes,
            enable_qwen35_ane_prefill,
            qwen35_ane_prefill_status,
        )
    except ImportError as exc:
        raise RuntimeError(
            "private ANE mode requires the isolated .venv-ane oMLX runtime"
        ) from exc

    enabled_layers = enable_qwen35_ane_prefill(
        engine.language,
        sequence_length=sequence_length,
        fraction=mlp_fraction,
        variant=mlp_variant,
        max_layers=mlp_max_layers,
        gdn=False,
        dual_ane=False,
        tail_padding_min_tokens=private_ane_tail_threshold(sequence_length),
    )
    # MLX modules default to training=True, including newly installed wrappers.
    # The untouched GDN layers must retain their Metal inference kernel path.
    inference_modules(engine.language)
    status = qwen35_ane_prefill_status(engine.language)
    if not status["configured"] or not enabled_layers:
        raise RuntimeError(f"oMLX did not configure any ANE MLP layers: {status}")
    fast.qwen35_ane_profile_reset()
    fast.qwen35_ane_profile_set_enabled(True)

    runtime: dict[str, Any] = {
        "scope": "unsupported-private-ane-research-runtime",
        "mode": mode,
        "sequence_length": sequence_length,
        "kernel_parameters": {
            "tail_padding_min_tokens": private_ane_tail_threshold(sequence_length),
            "mlp_fraction": mlp_fraction,
            "mlp_variant": mlp_variant,
            "mlp_max_layers": mlp_max_layers,
            "recurrence_block_size": recurrence_block_size,
            "recurrence_layer_slots": recurrence_layer_slots,
            "recurrence_query_scale": recurrence_query_scale,
            "recurrence_max_tokens": recurrence_max_tokens,
            "recurrence_io_dtype": recurrence_io_dtype,
            "recurrence_verify_reference": recurrence_verify_reference,
        },
        "mlp": status,
        "mlp_transient_surface_bytes": ane_prefill_transient_bytes(
            engine.language
        ),
        "fast_profile": fast,
        "recurrence": None,
        "recurrence_profile": None,
    }
    if mode == "quality-gated":
        if recurrence_profile_path is None:
            raise ValueError(
                "quality-gated private ANE mode requires a recurrence profile"
            )
        if experiment_root is None and (bridge_path is None or mil_path is None):
            raise ValueError(
                "quality-gated mode requires experiment root or explicit bridge and MIL"
            )
        profile = json.loads(
            recurrence_profile_path.resolve().read_text(encoding="utf-8")
        )
        settings = profile.get("settings") or {}
        comparison = profile.get("comparison") or {}
        slots = settings.get("recurrence_layer_slots")
        if slots is None:
            measured_slots = ((profile.get("model") or {}).get("recurrence") or {}).get(
                "enabled_layer_slots"
            )
            maximum = settings.get("recurrence_max_layers")
            if isinstance(measured_slots, list):
                slots = measured_slots
            elif isinstance(maximum, int) and 0 < maximum <= 18:
                slots = list(range(maximum))
        if recurrence_layer_slots is not None:
            if recurrence_layer_slots != slots:
                raise RuntimeError(
                    "requested recurrence layer slots do not match the verified profile"
                )
            slots = recurrence_layer_slots
        cosine = float(comparison.get("vector_cosine", 0.0))
        gdn_decoder_layers = [
            layer for layer in range(24) if (layer + 1) % 4 != 0
        ]
        decoder_layers = (
            [gdn_decoder_layers[slot] for slot in slots]
            if isinstance(slots, list)
            and all(isinstance(slot, int) and 0 <= slot < 18 for slot in slots)
            else None
        )
        minimum = float(
            (profile.get("quality_gate") or {}).get("minimum_vector_cosine", 1.0)
        )
        reference_gate = (profile.get("quality_gate") or {}).get("reference")
        recurrence_result = (profile.get("model") or {}).get("recurrence") or {}
        algorithm = recurrence_result.get("algorithm")
        solve_block_size = recurrence_result.get("solve_block_size")
        measured_query_scale = float(recurrence_result.get("query_scale", 0.0))
        measured_max_tokens = int(recurrence_result.get("max_tokens", 0))
        measured_io_dtype = str(recurrence_result.get("io_dtype", ""))
        if (
            not isinstance(slots, list)
            or not slots
            or any(not isinstance(slot, int) or slot < 0 or slot >= 18 for slot in slots)
            or settings.get("mlp") is not True
            or settings.get("recurrence_ane") is not True
            or (profile.get("quality_gate") or {}).get("passed") is not True
            or not isinstance(reference_gate, dict)
            or reference_gate.get("passed") is not True
            or cosine < minimum
            or algorithm != "block-forward-substitution-v1"
            or solve_block_size not in {2, 4, 8}
            or int(settings.get("sequence_length", 0)) != sequence_length
            or float(settings.get("fraction", 0.0)) != mlp_fraction
            or int(settings.get("max_layers", 0)) != mlp_max_layers
            or mlp_variant != 8
            or solve_block_size != recurrence_block_size
            or measured_query_scale != recurrence_query_scale
            or measured_max_tokens != recurrence_max_tokens
            or measured_io_dtype != recurrence_io_dtype
        ):
            raise RuntimeError(
                "combined MLP+recurrence profile did not pass its quality gate"
            )
        profile_package = Path(str(profile.get("package") or "")).resolve()
        profile_fingerprint = str(profile.get("package_fingerprint") or "")
        if profile_fingerprint != engine.package_fingerprint:
            raise RuntimeError(
                "recurrence profile fingerprint does not match the loaded package"
            )

        root = experiment_root.resolve() if experiment_root is not None else None
        internal_kernel_root = Path(__file__).resolve().with_name("private_ane")
        for import_root in (root, internal_kernel_root):
            if import_root is not None and str(import_root) not in sys.path:
                sys.path.insert(0, str(import_root))
        from wemm_mlx_recurrence_backend import PrivateANEGatedDeltaPrefill
        from mlx_vlm.models.qwen3_5.gated_delta import (
            register_qwen3_5_gated_delta_prefill_backend,
        )

        recurrence = PrivateANEGatedDeltaPrefill(
            bridge_path=bridge_path.resolve() if bridge_path is not None else (
                root / ".cache/ane-private-runtime/bridge/libane_bridge.dylib"
            ),
            mil_path=mil_path.resolve() if mil_path is not None else (
                root / "results/real_g_safe_c64_specialized.mil"
            ),
            query_scale=recurrence_query_scale,
            max_tokens=recurrence_max_tokens,
            layer_slots=slots,
            solve_block_size=recurrence_block_size,
            verify_reference=recurrence_verify_reference,
            io_dtype=recurrence_io_dtype,
        )
        register_qwen3_5_gated_delta_prefill_backend(recurrence)
        runtime["recurrence"] = recurrence
        runtime["recurrence_profile"] = {
            "path": str(recurrence_profile_path.resolve()),
            "package": str(profile_package),
            "minimum_vector_cosine": minimum,
            "measured_vector_cosine": cosine,
            "slots": slots,
            "decoder_layers": decoder_layers,
            "algorithm": algorithm,
            "solve_block_size": recurrence_block_size,
            "query_scale": recurrence_query_scale,
            "max_tokens": recurrence_max_tokens,
            "io_dtype": recurrence_io_dtype,
            "reference_gate": reference_gate,
        }

    engine._private_ane_runtime = runtime
    engine.backend += f"+private-ane-{mode}"


def private_ane_health(engine: Any) -> dict[str, Any] | None:
    runtime = getattr(engine, "_private_ane_runtime", None)
    if runtime is None:
        return None
    recurrence = runtime["recurrence"]
    return {
        "scope": runtime["scope"],
        "mode": runtime["mode"],
        "sequence_length": runtime["sequence_length"],
        "kernel_parameters": runtime["kernel_parameters"],
        "mlp": runtime["mlp"],
        "mlp_transient_surface_bytes": runtime["mlp_transient_surface_bytes"],
        "native_profile": runtime["fast_profile"].qwen35_ane_profile_snapshot(),
        "recurrence_profile": runtime["recurrence_profile"],
        "recurrence": recurrence.profile() if recurrence is not None else None,
        "video_gpu_projection": engine._video_projection.profile() if hasattr(engine, "_video_projection") else None,
        "video_pipeline": getattr(engine, "video_pipeline", 1),
    }


def close_private_ane_runtime(engine: Any) -> None:
    runtime = getattr(engine, "_private_ane_runtime", None)
    if runtime is None:
        return
    recurrence = runtime.get("recurrence")
    if recurrence is not None:
        from mlx_vlm.models.qwen3_5.gated_delta import (
            register_qwen3_5_gated_delta_prefill_backend,
        )

        register_qwen3_5_gated_delta_prefill_backend(None)
        recurrence.close()


@dataclass
class ServiceState:
    engine: Any
    default_dimension: int
    max_request_bytes: int
    started_at: float
    auth_token: str = ""

    def embedding_space(self, dimension: int) -> str:
        template = getattr(self.engine, "embedding_space_template", "")
        if template:
            return str(template).format(dimension=dimension)
        return f"{model_id(dimension)}-{dimension}-wemm-indexed-v1"

    def models(self) -> list[dict[str, Any]]:
        ordered = [self.default_dimension] + [
            value for value in OFFICIAL_DIMENSIONS if value != self.default_dimension
        ]
        return [
            {
                "id": model_id(dimension),
                "object": "model",
                "owned_by": "indexed-apple",
                "dimension": dimension,
                "embedding_space": self.embedding_space(dimension),
                "max_model_len": 8192,
                "modalities": ["text", "image"],
            }
            for dimension in ordered
        ]


def handler_class(state: ServiceState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "IndexedAppleEmbedding/0.1"

        def _headers(self, status: int, length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header(
                "Access-Control-Allow-Headers",
                "Content-Type, Authorization, X-Indexed-Token",
            )
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def _json(self, status: int, value: object) -> None:
            payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
            self._headers(status, len(payload))
            self.wfile.write(payload)

        def _error(self, error: Exception) -> None:
            status = error.status if isinstance(error, RequestError) else 500
            if status >= 500:
                print(f"[apple-embedding] request failed: {error!r}", file=sys.stderr)
            self._json(
                status,
                {
                    "error": {
                        "message": str(error),
                        "type": "invalid_request_error" if status < 500 else "server_error",
                    }
                },
            )

        def _body(self) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length", "")
            try:
                length = int(raw_length)
            except ValueError as error:
                raise RequestError("Content-Length 无效") from error
            if length < 1:
                raise RequestError("请求体为空")
            if length > state.max_request_bytes:
                raise RequestError("请求体超过 Apple helper 限制", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            raw = self.rfile.read(length)
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise RequestError("请求体不是有效 JSON") from error
            if not isinstance(value, dict):
                raise RequestError("请求体必须是 JSON 对象")
            return value

        def _authorize(self) -> None:
            expected = state.auth_token
            if not expected:
                return
            authorization = self.headers.get("Authorization", "")
            bearer = authorization[7:] if authorization.startswith("Bearer ") else ""
            alternate = self.headers.get("X-Indexed-Token", "")
            if not (
                hmac.compare_digest(bearer, expected)
                or hmac.compare_digest(alternate, expected)
            ):
                raise RequestError("Unauthorized", HTTPStatus.UNAUTHORIZED)

        def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self._headers(HTTPStatus.NO_CONTENT, 0)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            try:
                self._authorize()
                path = urlparse(self.path).path
                if path == "/health":
                    backend = getattr(state.engine, "backend", "coreml-ane+mps")
                    self._json(
                        HTTPStatus.OK,
                        {
                            "status": "ready",
                            "backend": backend,
                            "uptime_seconds": time.time() - state.started_at,
                            "load_seconds": state.engine.load_seconds,
                            "warmup_seconds": state.engine.warmup_seconds,
                            "allocated_bytes": getattr(
                                state.engine,
                                "allocated_bytes",
                                state.engine.mps_allocated_bytes,
                            ),
                            "mps_allocated_bytes": state.engine.mps_allocated_bytes,
                            "package_fingerprint": getattr(
                                state.engine, "package_fingerprint", None
                            ),
                            "private_ane": private_ane_health(state.engine),
                            "language_training": getattr(getattr(state.engine, "language", None), "training", None),
                            "input_contract": "apple-single-embedding-token-v2",
                        },
                    )
                    return
                if path == "/v1/models":
                    self._json(HTTPStatus.OK, {"object": "list", "data": state.models()})
                    return
                raise RequestError("Not found", HTTPStatus.NOT_FOUND)
            except Exception as error:
                self._error(error)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            try:
                self._authorize()
                if urlparse(self.path).path != "/v1/embeddings":
                    raise RequestError("Not found", HTTPStatus.NOT_FOUND)
                body = self._body()
                dimension = dimension_for_model(body.get("model"), state.default_dimension)
                messages = body.get("messages")
                if messages is None and isinstance(body.get("input"), str):
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": body["input"]}],
                        }
                    ]
                result = state.engine.embed(messages, dimension=dimension)
                backend = getattr(state.engine, "backend", "coreml-ane+mps")
                self._json(
                    HTTPStatus.OK,
                    {
                        "object": "list",
                        "model": model_id(dimension),
                        "data": [
                            {
                                "object": "embedding",
                                "index": 0,
                                "embedding": result.vector,
                            }
                        ],
                        "usage": {"prompt_tokens": result.prompt_tokens, "total_tokens": result.prompt_tokens},
                        "indexed": {
                            "backend": backend,
                            "modality": result.modality,
                            "dimension": dimension,
                            "embedding_space": state.embedding_space(dimension),
                            "timings_ms": result.timings_ms,
                        },
                    },
                )
            except Exception as error:
                self._error(error)

        def log_message(self, template: str, *args: object) -> None:
            print(
                f"[apple-embedding] {self.address_string()} {template % args}",
                file=sys.stderr,
            )

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mlx", "pytorch"), default="mlx")
    parser.add_argument("--package", type=Path, help="Converted WeMM Apple package")
    parser.add_argument("--model", type=Path, help="Original WeMM-Embedding-2B directory")
    parser.add_argument("--coreml-model", type=Path, help="WeMM vision .mlpackage or compiled model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--default-dimension", type=int, default=256, choices=OFFICIAL_DIMENSIONS)
    parser.add_argument("--compute-unit", choices=("cpu_ane", "cpu_gpu", "all"), default="cpu_ane")
    parser.add_argument("--execution-mode", choices=("a", "b", "c", "d"))
    parser.add_argument("--max-request-mb", type=int, default=64)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--allow-non-loopback", action="store_true")
    parser.add_argument(
        "--private-ane-mode",
        choices=("none", "mlp", "quality-gated"),
        default="none",
        help="unsupported research mode using private Apple ANE APIs",
    )
    parser.add_argument("--private-ane-sequence-length", type=int, default=2112)
    parser.add_argument("--video-down-projection", choices=("q8", "fp16"), default="q8")
    parser.add_argument("--video-pipeline", choices=(1, 2), type=int, default=2)
    parser.add_argument("--private-ane-mlp-fraction", type=float, default=0.75)
    parser.add_argument("--private-ane-mlp-variant", type=int, default=8)
    parser.add_argument("--private-ane-mlp-max-layers", type=int, default=24)
    parser.add_argument("--private-ane-experiment-root", type=Path)
    parser.add_argument("--private-ane-recurrence-profile", type=Path)
    parser.add_argument("--private-ane-bridge", type=Path)
    parser.add_argument("--private-ane-mil", type=Path)
    parser.add_argument(
        "--private-ane-recurrence-block-size", type=int, choices=(2, 4, 8), default=8
    )
    parser.add_argument("--private-ane-recurrence-layer-slots")
    parser.add_argument("--private-ane-recurrence-query-scale", type=float, default=4096.0)
    parser.add_argument("--private-ane-recurrence-max-tokens", type=int, default=8192)
    parser.add_argument(
        "--private-ane-recurrence-io-dtype",
        choices=("fp16", "fp32"),
        default="fp16",
    )
    parser.add_argument("--private-ane-recurrence-verify-reference", action="store_true")
    parser.add_argument(
        "--auth-token",
        default=os.environ.get("INDEXED_APPLE_EMBEDDING_AUTH_TOKEN", ""),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.execution_mode:
        try:
            args.compute_unit, args.private_ane_mode = execution_settings(
                args.execution_mode
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
    if not args.allow_non_loopback and args.host not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("拒绝监听非 loopback 地址；研究环境请显式传 --allow-non-loopback")
    if args.max_request_mb < 1:
        raise SystemExit("--max-request-mb 必须大于零")
    if args.private_ane_mode != "none" and args.backend != "mlx":
        raise SystemExit("private ANE mode only supports --backend mlx")

    print("[apple-embedding] loading WeMM and Core ML models...", file=sys.stderr)
    if args.backend == "mlx":
        if args.package is None:
            raise SystemExit("--backend mlx 需要 --package")
        engine = MLXWeMMEngine(
            package_path=args.package,
            image_size=args.image_size,
            compute_unit=args.compute_unit,
            warmup=not args.skip_warmup,
        )
        enable_private_ane_runtime(
            engine,
            mode=args.private_ane_mode,
            sequence_length=args.private_ane_sequence_length,
            mlp_fraction=args.private_ane_mlp_fraction,
            mlp_variant=args.private_ane_mlp_variant,
            mlp_max_layers=args.private_ane_mlp_max_layers,
            experiment_root=args.private_ane_experiment_root,
            recurrence_profile_path=args.private_ane_recurrence_profile,
            bridge_path=args.private_ane_bridge,
            mil_path=args.private_ane_mil,
            recurrence_block_size=args.private_ane_recurrence_block_size,
            recurrence_layer_slots=parse_layer_slots(
                args.private_ane_recurrence_layer_slots
            ),
            recurrence_query_scale=args.private_ane_recurrence_query_scale,
            recurrence_max_tokens=args.private_ane_recurrence_max_tokens,
            recurrence_io_dtype=args.private_ane_recurrence_io_dtype,
            recurrence_verify_reference=args.private_ane_recurrence_verify_reference,
        )
        if args.private_ane_mode != "none":
            from video_projection import install_video_projection
            engine._video_projection = install_video_projection(engine.language, args.video_down_projection)
            engine._pipeline_gate = threading.BoundedSemaphore(args.video_pipeline)
            engine.video_pipeline = args.video_pipeline
            if args.video_pipeline == 2:
                from ane_lane import prioritize_recurrence
                prioritize_recurrence(engine)
            engine.allocated_bytes = int(engine._mx.get_active_memory())
            engine.mps_allocated_bytes = engine.allocated_bytes
    else:
        if args.model is None or args.coreml_model is None:
            raise SystemExit("--backend pytorch 需要 --model 和 --coreml-model")
        engine = AppleWeMMEngine(
            model_path=args.model.resolve(),
            coreml_model_path=args.coreml_model.resolve(),
            image_size=args.image_size,
            compute_unit=args.compute_unit,
            warmup=not args.skip_warmup,
        )
    state = ServiceState(
        engine=engine,
        default_dimension=args.default_dimension,
        max_request_bytes=args.max_request_mb * 1024 * 1024,
        started_at=time.time(),
        auth_token=args.auth_token,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler_class(state))
    server.daemon_threads = True

    def stop(_signal: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    ready = {
        "status": "ready",
        "url": f"http://{args.host}:{server.server_port}",
        "default_model": model_id(args.default_dimension),
        "backend": getattr(engine, "backend", "coreml-ane+mps"),
        "execution_mode": args.execution_mode,
        "load_seconds": engine.load_seconds,
        "package_fingerprint": getattr(engine, "package_fingerprint", None),
        "default_embedding_space": state.embedding_space(args.default_dimension),
    }
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        close_private_ane_runtime(engine)


if __name__ == "__main__":
    main()
