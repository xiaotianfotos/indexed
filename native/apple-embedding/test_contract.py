from __future__ import annotations

import base64
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from PIL import Image

from service import (
    OFFICIAL_DIMENSIONS,
    EmbeddingResult,
    MLXWeMMEngine,
    RequestError,
    ServiceState,
    dimension_for_model,
    execution_settings,
    handler_class,
    model_id,
    normalize_messages,
    parse_layer_slots,
    private_ane_tail_threshold,
    inference_modules,
    tokenize_embedding_prompt,
)


class FakeEngine:
    load_seconds = 0.1
    warmup_seconds = 0.05
    allocated_bytes = 123
    mps_allocated_bytes = 123
    package_fingerprint = "test-fingerprint"
    embedding_space_template = "test-q8-{dimension}-fingerprint"

    def embed(self, messages: object, *, dimension: int) -> EmbeddingResult:
        normalized, image_count = normalize_messages(messages, image_size=448)
        if not normalized:
            raise AssertionError("messages were not normalized")
        return EmbeddingResult(
            vector=[1.0] + [0.0] * (dimension - 1),
            timings_ms={"preprocess": 1.0, "vision": float(image_count), "language": 2.0, "total": 3.0},
            modality="image" if image_count else "text",
        )


def image_data_uri() -> str:
    image = Image.new("RGB", (2, 2), (255, 0, 0))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


class ContractTests(unittest.TestCase):
    def test_pipeline_overlaps_vision_but_never_two_language_calls(self) -> None:
        import numpy as np
        for width in (1, 2):
            engine = object.__new__(MLXWeMMEngine)
            engine._image_size = 448
            engine._pipeline_gate = threading.BoundedSemaphore(width)
            engine._vision_lock = threading.Lock()
            engine._lock = threading.Lock()
            engine._video_projection = SimpleNamespace(video_active=False)
            engine._mx = SimpleNamespace(array=np.array, bfloat16=np.float32, float32=np.float32,
                                         linalg=np.linalg, eval=lambda _: None)
            active = {"language": 0, "peak": 0, "overlap": False}
            entered = threading.Event()
            def predict(_):
                active["overlap"] |= active["language"] > 0
                time.sleep(0.01)
                return {"image_embeds": np.ones((1, 4), dtype=np.float32)}
            def hidden(*_):
                self.assertTrue(engine._video_projection.video_active)
                active["language"] += 1
                active["peak"] = max(active["peak"], active["language"])
                entered.set()
                time.sleep(0.08)
                active["language"] -= 1
                return np.ones((1, 1, 64), dtype=np.float32)
            engine.vision = SimpleNamespace(predict=predict)
            engine._inputs = lambda *_: (np.ones((1, 3)), [np.ones(1)], None, np.ones(1), 1, "video")
            engine._hidden = hidden
            with ThreadPoolExecutor(2) as pool:
                first = pool.submit(engine.embed, [{"role": "user", "content": "test"}], dimension=64)
                self.assertTrue(entered.wait(2))
                second = pool.submit(engine.embed, [{"role": "user", "content": "test"}], dimension=64)
                self.assertEqual(first.result().vector, second.result().vector)
            self.assertEqual(active["peak"], 1)
            self.assertEqual(active["overlap"], width == 2)

    def test_inference_mode_is_explicit_for_every_module_tree(self) -> None:
        class Module:
            training = True
            def eval(self) -> None:
                self.training = False
        visual, text = Module(), Module()
        inference_modules(visual, text)
        self.assertFalse(visual.training)
        self.assertFalse(text.training)

    def test_tokenizer_does_not_append_a_second_embedding_token(self) -> None:
        def tokenizer(prompt: str, **kwargs: object) -> object:
            self.assertTrue(prompt.endswith("<embedding>"))
            self.assertIs(kwargs.get("add_special_tokens"), False)
            return {"input_ids": [1, 2, 248088]}
        self.assertEqual(tokenize_embedding_prompt(tokenizer, "example<embedding>"), {"input_ids": [1, 2, 248088]})

    def test_private_ane_text_uses_the_patched_language_tree(self) -> None:
        calls: list[str] = []

        class Language:
            def get_rope_index(self, *_args: object, **_kwargs: object) -> tuple[str, None]:
                return "positions", None

            def model(self, _ids: object, **kwargs: object) -> str:
                calls.append(f"private:{kwargs['position_ids']}")
                return "private"

        class TextLanguage:
            def model(self, _ids: object) -> str:
                calls.append("public")
                return "public"

        engine = object.__new__(MLXWeMMEngine)
        engine._mx = object()
        engine.language = Language()
        engine.text_language = TextLanguage()
        engine._private_ane_runtime = {"mode": "quality-gated", "sequence_length": 2112}
        class Input:
            shape = (1, 206)
        self.assertEqual(engine._hidden(Input(), None, None), "public")
        Input.shape = (1, 2112)
        self.assertEqual(engine._hidden(Input(), None, None), "private")
        self.assertEqual(calls, ["public", "private:positions"])

    def test_private_ane_short_input_padding_is_bounded(self) -> None:
        self.assertEqual(private_ane_tail_threshold(2112), 1056)
        self.assertEqual(private_ane_tail_threshold(256), 128)
        self.assertLess(206, private_ane_tail_threshold(2112))
        self.assertGreaterEqual(206, private_ane_tail_threshold(256))

    def test_execution_modes_and_kernel_slot_parser(self) -> None:
        self.assertEqual(execution_settings("a"), ("cpu_gpu", "none"))
        self.assertEqual(execution_settings("b"), ("cpu_ane", "none"))
        self.assertEqual(execution_settings("c"), ("cpu_ane", "mlp"))
        self.assertEqual(execution_settings("d"), ("cpu_ane", "quality-gated"))
        with self.assertRaisesRegex(ValueError, "E mode is retired"):
            execution_settings("e")
        self.assertEqual(parse_layer_slots("4,2,4,0"), [0, 2, 4])
        self.assertIsNone(parse_layer_slots(""))
        with self.assertRaises(ValueError):
            parse_layer_slots("18")

    def test_model_ids_encode_official_dimensions(self) -> None:
        self.assertEqual(dimension_for_model(model_id(256), 64), 256)
        self.assertEqual(dimension_for_model("", 128), 128)
        with self.assertRaises(RequestError):
            dimension_for_model("wemm-embedding-2b-apple-12", 256)

    def test_normalizes_text_and_one_image(self) -> None:
        messages, count = normalize_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_uri()}},
                        {"type": "text", "text": "Represent this image."},
                    ],
                }
            ],
            image_size=448,
        )
        self.assertEqual(count, 1)
        self.assertEqual(messages[0]["content"][0]["type"], "image")
        self.assertEqual(messages[0]["content"][1]["text"], "Represent this image.")

    def test_normalizes_ordered_video_frames(self) -> None:
        uri = image_data_uri()
        messages, count = normalize_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_frames",
                            "frames": [
                                {"image_url": {"url": uri}, "timestamp": 0.0},
                                {"image_url": {"url": uri}, "timestamp": 0.5},
                            ],
                        },
                        {"type": "text", "text": "Represent this video."},
                    ],
                }
            ],
            image_size=448,
        )
        self.assertEqual(count, 0)
        frames = messages[0]["content"][0]["frames"]
        self.assertEqual(len(frames), 2)
        self.assertEqual([frame["timestamp"] for frame in frames], [0.0, 0.5])

    def test_rejects_multiple_visual_inputs(self) -> None:
        uri = image_data_uri()
        with self.assertRaises(RequestError):
            normalize_messages(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": uri}},
                            {
                                "type": "video_frames",
                                "frames": [{"image_url": {"url": uri}, "timestamp": 0.0}],
                            },
                        ],
                    }
                ],
                image_size=448,
            )

    def test_http_contract_matches_indexed(self) -> None:
        state = ServiceState(FakeEngine(), 256, 1024 * 1024, 0.0)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class(state))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            models = json.load(urllib.request.urlopen(f"{base}/v1/models"))
            self.assertEqual(models["data"][0]["id"], model_id(256))
            self.assertEqual(
                models["data"][0]["embedding_space"],
                "test-q8-256-fingerprint",
            )
            self.assertEqual(
                {entry["dimension"] for entry in models["data"]},
                set(OFFICIAL_DIMENSIONS),
            )
            request = urllib.request.Request(
                f"{base}/v1/embeddings",
                method="POST",
                headers={"Content-Type": "application/json"},
                data=json.dumps(
                    {
                        "model": model_id(64),
                        "messages": [
                            {"role": "user", "content": [{"type": "text", "text": "hello"}]}
                        ],
                    }
                ).encode(),
            )
            response = json.load(urllib.request.urlopen(request))
            self.assertEqual(response["model"], model_id(64))
            self.assertEqual(len(response["data"][0]["embedding"]), 64)
            self.assertEqual(response["indexed"]["modality"], "text")
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)

    def test_http_contract_requires_managed_token_when_configured(self) -> None:
        state = ServiceState(FakeEngine(), 256, 1024 * 1024, 0.0, "test-token")
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class(state))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with self.assertRaises(urllib.error.HTTPError) as unauthorized:
                urllib.request.urlopen(f"{base}/health")
            self.assertEqual(unauthorized.exception.code, 401)
            request = urllib.request.Request(
                f"{base}/health", headers={"Authorization": "Bearer test-token"}
            )
            self.assertEqual(json.load(urllib.request.urlopen(request))["status"], "ready")
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
