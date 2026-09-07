#!/usr/bin/env python3
"""Experimental MLX-to-private-ANE bridge for WeMM's GDN recurrence.

This module is intentionally outside the public Indexed runtime.  It uses an
undocumented Apple in-memory compiler interface and is therefore suitable for
research builds only.  Unsupported calls return ``None`` so mlx-vlm keeps its
normal Metal/MLX behavior.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from inmemory_batched_graph_runner import shared_constants
from inmemory_partition_runner import (
    ANEBridge,
    MATRIX_BYTES_F16,
    MATRIX_BYTES_F32,
    SIZE,
    parse_graph,
)
from inmemory_segmented_runner import add_query_scaling, make_segments
from probe_inmemory_split_surfaces import DYNAMIC_NAMES, split_surface_mil
from stable_block_solve import block_solve_constants, stabilize_gdn_operations


class PrivateANEGatedDeltaPrefill:
    """Run fixed-shape Qwen3.5 Gated DeltaNet prefill recurrence on ANE."""

    batch = 16
    chunk_size = 64
    decoder_layers = tuple(
        layer for layer in range(24) if (layer + 1) % 4 != 0
    )

    def __init__(
        self,
        *,
        bridge_path: Path,
        mil_path: Path,
        query_scale: float = 4096.0,
        max_tokens: int = 256,
        enabled_layer_slots: int = 18,
        total_layer_slots: int = 18,
        layer_slots: Iterable[int] | None = None,
        solve_block_size: int = 8,
        verify_reference: bool = False,
        io_dtype: str = "fp16",
    ) -> None:
        if max_tokens < self.chunk_size or max_tokens % self.chunk_size:
            raise ValueError("max_tokens must be a positive multiple of 64")
        self.query_scale = float(query_scale)
        self.max_tokens = int(max_tokens)
        self.solve_block_size = int(solve_block_size)
        self.verify_reference = bool(verify_reference)
        if io_dtype not in {"fp16", "fp32"}:
            raise ValueError("io_dtype must be fp16 or fp32")
        self.io_dtype = io_dtype
        self.surface_dtype = np.float16 if io_dtype == "fp16" else np.float32
        matrix_bytes = MATRIX_BYTES_F16 if io_dtype == "fp16" else MATRIX_BYTES_F32
        self.total_layer_slots = int(total_layer_slots)
        enabled_layer_slots = int(enabled_layer_slots)
        if not 0 <= enabled_layer_slots <= self.total_layer_slots:
            raise ValueError("enabled_layer_slots must be within total_layer_slots")
        self.set_layer_slots(
            range(enabled_layer_slots) if layer_slots is None else layer_slots
        )
        self._lock = threading.Lock()
        self._fallbacks: Counter[str] = Counter()
        self._calls = 0
        self._ane_evaluations = 0
        self._runtime_ms = 0.0
        self._dynamic_bytes_written = 0
        self._output_bytes_read = 0
        self._verification: list[dict[str, Any]] = []

        stable_operations = stabilize_gdn_operations(
            parse_graph(Path(mil_path)),
            active_tokens=self.chunk_size,
            block_size=self.solve_block_size,
        )
        operations = add_query_scaling(
            stable_operations, self.query_scale
        )
        segments = make_segments(
            operations, len(operations), connected_only=False
        )
        if len(segments) != 1:
            raise RuntimeError(f"expected one GDN segment, got {len(segments)}")
        self.segment = segments[0]
        self.dynamic_names = [
            name for name in self.segment.inputs if name in DYNAMIC_NAMES
        ]
        self.shared_names = [
            name for name in self.segment.inputs if name not in DYNAMIC_NAMES
        ]
        if "state" not in self.dynamic_names:
            raise RuntimeError("GDN graph has no dynamic state input")
        if not {"output", "final_state"}.issubset(self.segment.outputs):
            raise RuntimeError(
                f"unexpected GDN outputs: {self.segment.outputs}"
            )

        self.dynamic_bytes = (
            len(self.dynamic_names) * self.batch * matrix_bytes
        )
        self.shared_bytes = len(self.shared_names) * matrix_bytes
        self.output_bytes = (
            len(self.segment.outputs) * self.batch * matrix_bytes
        )
        self.handle_name = "wemm_gdn_recurrence_h16"
        self.bridge = ANEBridge(Path(bridge_path))
        compile_started = time.perf_counter()
        self.bridge.compile_surfaces(
            self.handle_name,
            split_surface_mil(
                self.segment,
                self.batch,
                self.dynamic_names,
                self.shared_names,
                io_dtype=self.io_dtype,
            ),
            [self.dynamic_bytes, self.shared_bytes],
            [self.output_bytes],
            input_dtype=self.surface_dtype,
            output_dtype=self.surface_dtype,
        )
        self.compile_seconds = time.perf_counter() - compile_started
        # Calls are serialized by ``_lock``. Reuse the large staging arrays
        # instead of allocating and stacking them for every 64-token chunk.
        self._dynamic_surface = np.empty(
            (len(self.dynamic_names), self.batch, SIZE, SIZE),
            dtype=self.surface_dtype,
        )
        self._output_surface = np.empty(
            (SIZE, SIZE * len(self.segment.outputs) * self.batch),
            dtype=self.surface_dtype,
        )
        self.bridge.write_input(self.handle_name, 1, self._make_shared_surface())

    def _make_shared_surface(self) -> np.ndarray:
        values = {
            **shared_constants(self.batch),
            **{
                name: value[None]
                for name, value in block_solve_constants(
                    active_tokens=self.chunk_size,
                    block_size=self.solve_block_size,
                ).items()
            },
        }
        values["query_inverse_scale"] = np.full(
            (self.batch, SIZE, SIZE),
            1.0 / self.query_scale,
            dtype=np.float32,
        )
        try:
            return np.stack(
                [values[name][0] for name in self.shared_names], axis=0
            )[:, None].astype(self.surface_dtype)
        except KeyError as exc:
            raise RuntimeError(f"unknown shared GDN input: {exc.args[0]}") from exc

    def _fallback(self, reason: str) -> None:
        self._fallbacks[reason] += 1

    def __call__(
        self,
        q: Any,
        k: Any,
        v: Any,
        g: Any,
        beta: Any,
        state: Any,
        mask: Any | None,
        backend_context: Any | None = None,
    ) -> tuple[Any, Any] | None:
        import mlx.core as mx

        self._calls += 1
        decoder_layer = getattr(
            backend_context, "_qwen3_5_decoder_layer_idx", None
        )
        if decoder_layer in self.decoder_layers:
            layer_slot = self.decoder_layers.index(decoder_layer)
        else:
            # Direct oracle tests do not have a decoder module.  The fallback
            # counter remains deterministic as long as they execute complete
            # 18-call passes, while the real service always uses layer IDs.
            layer_slot = (self._calls - 1) % self.total_layer_slots
        if layer_slot not in self.layer_slots:
            self._fallback("layer_filter")
            return None
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or g.ndim != 3:
            self._fallback("rank")
            return None
        B, tokens, key_heads, key_dim = map(int, q.shape)
        value_heads, value_dim = map(int, v.shape[-2:])
        if (
            B != 1
            or tokens <= 1
            or tokens > self.max_tokens
            or key_heads != self.batch
            or value_heads != self.batch
            or key_dim != SIZE
            or value_dim != SIZE
        ):
            self._fallback("shape")
            return None
        if tuple(map(int, state.shape)) != (1, self.batch, SIZE, SIZE):
            self._fallback("state_shape")
            return None
        if tuple(map(int, beta.shape)) != (1, tokens, self.batch):
            self._fallback("beta_shape")
            return None
        if mask is not None:
            mx.eval(mask)
            mask_np = np.asarray(mask)
            if mask_np.size != tokens or not bool(np.all(mask_np)):
                self._fallback("mask")
                return None

        started = time.perf_counter()
        mx.eval(q, k, v, g, beta, state)
        # MLX bfloat16 does not expose a NumPy-compatible PEP 3118 format.
        # Round tensors straight to the IOSurface type before crossing the
        # process boundary. They were rounded again during packing previously.
        transfer_dtype = mx.float16 if self.io_dtype == "fp16" else mx.float32
        q_np = np.asarray(q.astype(transfer_dtype))[0]
        k_np = np.asarray(k.astype(transfer_dtype))[0]
        v_np = np.asarray(v.astype(transfer_dtype))[0]
        # Keep decay in FP32 until log() has been evaluated.
        g_np = np.asarray(g.astype(mx.float32))[0]
        beta_np = np.asarray(beta.astype(transfer_dtype))[0]
        state_np = np.asarray(state.astype(transfer_dtype))[0].transpose(0, 2, 1)
        # The MIL graph consumes log decay.  Match mlx-vlm's chunked path by
        # clipping underflowed scalar decay before taking the logarithm.
        log_g_np = np.log(np.clip(g_np, 1.0e-6, 1.0))

        output_np = np.empty(
            (tokens, self.batch, SIZE), dtype=self.surface_dtype
        )
        verification_parts: list[dict[str, Any]] = []
        with self._lock:
            for start in range(0, tokens, self.chunk_size):
                stop = min(start + self.chunk_size, tokens)
                valid = stop - start
                dynamic = self._pack_dynamic_input(
                    q_np[start:stop],
                    k_np[start:stop],
                    v_np[start:stop],
                    log_g_np[start:stop],
                    beta_np[start:stop],
                    state_np,
                )
                self.bridge.write_input(self.handle_name, 0, dynamic)
                raw = self.bridge.evaluate_preloaded(
                    self.handle_name,
                    len(self.segment.outputs) * self.batch,
                    result=self._output_surface,
                ).reshape(
                    len(self.segment.outputs), self.batch, SIZE, SIZE
                )
                values = dict(zip(self.segment.outputs, raw))
                if self.verify_reference:
                    verification_parts.append(
                        self._verify_chunk(
                            q_np[start:stop],
                            k_np[start:stop],
                            v_np[start:stop],
                            log_g_np[start:stop],
                            beta_np[start:stop],
                            state_np,
                            values["output"][:, :valid],
                            values["final_state"],
                            layer_slot=layer_slot,
                            decoder_layer=decoder_layer,
                            start=start,
                            stop=stop,
                        )
                    )
                output_np[start:stop] = np.transpose(
                    values["output"][:, :valid], (1, 0, 2)
                )
                state_np = np.ascontiguousarray(values["final_state"])
                self._ane_evaluations += 1
                self._dynamic_bytes_written += self.dynamic_bytes
                self._output_bytes_read += self.output_bytes

        self._verification.extend(verification_parts)

        output_np = output_np[None]
        next_state_np = state_np.transpose(0, 2, 1)[None]
        output = mx.array(output_np).astype(v.dtype)
        next_state = mx.array(next_state_np).astype(mx.float32)
        self._runtime_ms += (time.perf_counter() - started) * 1000.0
        return output, next_state

    def _verify_chunk(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        log_g: np.ndarray,
        beta: np.ndarray,
        initial_state: np.ndarray,
        actual_output: np.ndarray,
        actual_state: np.ndarray,
        *,
        layer_slot: int,
        decoder_layer: int | None,
        start: int,
        stop: int,
    ) -> dict[str, Any]:
        from inmemory_partition_runner import metrics
        from prepare_wemm_head import sequential_reference

        def fp16(value: np.ndarray) -> np.ndarray:
            return value.astype(np.float16).astype(np.float32)

        reference_outputs = []
        reference_states = []
        for head in range(self.batch):
            output, final_state = sequential_reference(
                fp16(q[:, head]),
                fp16(k[:, head]),
                fp16(v[:, head]),
                fp16(log_g[:, head]),
                fp16(beta[:, head]),
                fp16(initial_state[head]),
            )
            reference_outputs.append(output)
            reference_states.append(final_state)
        expected_output = np.stack(reference_outputs)
        expected_state = np.stack(reference_states)
        return {
            "layer_slot": layer_slot,
            "decoder_layer": decoder_layer,
            "token_range": [start, stop],
            "output": metrics(actual_output, expected_output),
            "final_state": metrics(actual_state, expected_state),
        }

    def _pack_dynamic_input(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        log_g: np.ndarray,
        beta: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        valid = q.shape[0]
        dynamic = self._dynamic_surface
        dynamic.fill(0)
        diagonal = np.arange(SIZE)
        q_heads = np.transpose(q, (1, 0, 2))
        k_heads = np.transpose(k, (1, 0, 2))
        v_heads = np.transpose(v, (1, 0, 2))
        beta_heads = beta.T
        log_g_heads = log_g.T
        for index, name in enumerate(self.dynamic_names):
            target = dynamic[index]
            if name == "q":
                target[:, :valid] = q_heads
                target *= self.query_scale
            elif name == "k":
                target[:, :valid] = k_heads
            elif name == "kt":
                target[:, :, :valid] = np.transpose(k_heads, (0, 2, 1))
            elif name == "v":
                target[:, :valid] = v_heads
            elif name == "beta":
                target[:, :valid, :] = beta_heads[:, :, None]
            elif name == "log_decay_diagonal":
                target[:, diagonal[:valid], diagonal[:valid]] = log_g_heads
            elif name == "state":
                target[:] = state
            else:
                raise RuntimeError(f"unknown dynamic GDN input: {name}")
        return dynamic

    def profile(self) -> dict[str, Any]:
        return {
            "backend": "apple-private-inmemory-ane-gdn-recurrence",
            "io_dtype": self.io_dtype,
            "calls": self._calls,
            "ane_evaluations": self._ane_evaluations,
            "fallbacks": dict(self._fallbacks),
            "runtime_ms": self._runtime_ms,
            "compile_seconds": self.compile_seconds,
            "max_tokens": self.max_tokens,
            "enabled_layer_slots": sorted(self.layer_slots),
            "total_layer_slots": self.total_layer_slots,
            "query_scale": self.query_scale,
            "algorithm": "block-forward-substitution-v1",
            "active_chunk_size": self.chunk_size,
            "solve_block_size": self.solve_block_size,
            "blocks_per_chunk": self.chunk_size // self.solve_block_size,
            "ane_operation_count": len(self.segment.operations),
            "dynamic_surface_bytes_per_chunk": self.dynamic_bytes,
            "shared_surface_bytes_written_once": self.shared_bytes,
            "output_surface_bytes_per_chunk": self.output_bytes,
            "dynamic_bytes_written": self._dynamic_bytes_written,
            "output_bytes_read": self._output_bytes_read,
            "staging_buffer_reuse": True,
            "mlx_transfer_dtype": self.io_dtype,
            "reference_verification": self._verification_summary(),
        }

    def _verification_summary(self) -> dict[str, Any] | None:
        if not self._verification:
            return None
        by_layer: dict[int, list[dict[str, Any]]] = {}
        for row in self._verification:
            by_layer.setdefault(int(row["layer_slot"]), []).append(row)
        layers = []
        for slot, rows in sorted(by_layer.items()):
            layers.append(
                {
                    "layer_slot": slot,
                    "decoder_layer": rows[0]["decoder_layer"],
                    "chunks": len(rows),
                    "output_min_cosine": min(
                        float(row["output"]["cosine"]) for row in rows
                    ),
                    "output_max_relative_l2": max(
                        float(row["output"]["relative_l2"]) for row in rows
                    ),
                    "state_min_cosine": min(
                        float(row["final_state"]["cosine"]) for row in rows
                    ),
                    "state_max_relative_l2": max(
                        float(row["final_state"]["relative_l2"]) for row in rows
                    ),
                    "nonfinite": sum(
                        int(row[kind]["nonfinite"])
                        for row in rows
                        for kind in ("output", "final_state")
                    ),
                }
            )
        return {
            "enabled": True,
            "chunks": len(self._verification),
            "layers": layers,
            "minimum_output_cosine": min(
                row["output_min_cosine"] for row in layers
            ),
            "minimum_state_cosine": min(
                row["state_min_cosine"] for row in layers
            ),
            "maximum_output_relative_l2": max(
                row["output_max_relative_l2"] for row in layers
            ),
            "maximum_state_relative_l2": max(
                row["state_max_relative_l2"] for row in layers
            ),
            "nonfinite": sum(row["nonfinite"] for row in layers),
        }

    def reset_profile(self) -> None:
        self._fallbacks.clear()
        self._calls = 0
        self._ane_evaluations = 0
        self._runtime_ms = 0.0
        self._dynamic_bytes_written = 0
        self._output_bytes_read = 0
        self._verification.clear()

    def set_layer_slots(self, slots: Iterable[int]) -> None:
        selected = frozenset(int(slot) for slot in slots)
        if any(slot < 0 or slot >= self.total_layer_slots for slot in selected):
            raise ValueError("layer slot is outside total_layer_slots")
        self.layer_slots = selected

    def close(self) -> None:
        if getattr(self, "bridge", None) is not None:
            self.bridge.close()
            self.bridge = None
