#!/usr/bin/env python3
"""Run real WeMM GDN heads as one batched private in-memory ANE graph."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

from inmemory_partition_runner import (
    ANEBridge,
    MATRIX_BYTES_F32,
    SIZE,
    metrics,
    parse_graph,
)
from inmemory_segmented_runner import Operation, Segment, add_query_scaling, make_segments
from prepare_wemm_head import sequential_reference


def batched_segment_mil(segment: Segment, batch: int) -> str:
    input_count = len(segment.inputs)
    output_count = len(segment.outputs)
    lines = [
        "program(1.3)",
        '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]',
        "{",
        f"  func main<ios18>(tensor<fp32, [{input_count},{batch},128,128]> packed) {{",
        '    string to16 = const()[name=string("to16"), val=string("fp16")];',
        f'    tensor<fp16, [{input_count},{batch},128,128]> packed16 = cast(dtype=to16, x=packed)[name=string("cast_in")];',
        f'    tensor<int32, [4]> slice_shape = const()[name=string("slice_shape"), val=tensor<int32, [4]>([1,{batch},128,128])];',
        f'    tensor<int32, [4]> matrix_shape = const()[name=string("matrix_shape"), val=tensor<int32, [4]>([{batch},1,128,128])];',
        f'    tensor<int32, [4]> live_shape = const()[name=string("live_shape"), val=tensor<int32, [4]>([1,{batch},128,128])];',
        '    bool f = const()[name=string("f"), val=bool(false)];',
    ]
    aliases: dict[str, str] = {}
    for index, name in enumerate(segment.inputs):
        alias = f"input_{index}"
        aliases[name] = alias
        lines.extend(
            [
                f'    tensor<int32, [4]> begin_{index} = const()[name=string("begin_{index}"), val=tensor<int32, [4]>([{index},0,0,0])];',
                f'    tensor<fp16, [1,{batch},128,128]> raw_{index} = slice_by_size(x=packed16, begin=begin_{index}, size=slice_shape)[name=string("slice_{index}")];',
                f'    tensor<fp16, [{batch},1,128,128]> {alias} = reshape(shape=matrix_shape, x=raw_{index})[name=string("reshape_{index}")];',
            ]
        )
    for local_index, (output, operation, left, right) in enumerate(segment.operations):
        output_alias = f"value_{local_index}"
        left_alias = aliases[left]
        right_alias = aliases[right] if right is not None else None
        if operation == "matmul":
            expression = (
                f"matmul(transpose_x=f, transpose_y=f, "
                f"x={left_alias}, y={right_alias})"
            )
        elif operation == "exp":
            expression = f"exp(x={left_alias})"
        else:
            expression = f"{operation}(x={left_alias}, y={right_alias})"
        lines.append(
            f'    tensor<fp16, [{batch},1,128,128]> {output_alias} = {expression}[name=string("op_{local_index}")];'
        )
        aliases[output] = output_alias
    for index, output in enumerate(segment.outputs):
        lines.append(
            f'    tensor<fp16, [1,{batch},128,128]> live_{index} = reshape(shape=live_shape, x={aliases[output]})[name=string("live_{index}")];'
        )
    if output_count == 1:
        lines.append(
            f'    tensor<fp16, [1,{batch},128,128]> combined = reshape(shape=live_shape, x=live_0)[name=string("single_output")];'
        )
    else:
        values = ",".join(f"live_{index}" for index in range(output_count))
        lines.extend(
            [
                '    int32 concat_axis = const()[name=string("concat_axis"), val=int32(0)];',
                '    bool concat_interleave = const()[name=string("concat_interleave"), val=bool(false)];',
                f'    tensor<fp16, [{output_count},{batch},128,128]> combined = concat(axis=concat_axis, interleave=concat_interleave, values=({values}))[name=string("pack_outputs")];',
            ]
        )
    lines.extend(
        [
            '    string to32 = const()[name=string("to32"), val=string("fp32")];',
            f'    tensor<fp32, [{output_count},{batch},128,128]> result = cast(dtype=to32, x=combined)[name=string("cast_out")];',
            "  } -> (result);",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def normalize_all(query: np.ndarray, key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query = query.astype(np.float32, copy=True)
    key = key.astype(np.float32, copy=True)
    query /= np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), 1.0e-6)
    key /= np.maximum(np.linalg.norm(key, axis=-1, keepdims=True), 1.0e-6)
    query *= 1.0 / math.sqrt(SIZE)
    return query, key


def shared_constants(batch: int) -> dict[str, np.ndarray]:
    row, column = np.indices((SIZE, SIZE))
    constants = {
        "negative_strict_lower": np.where(column < row, -1.0, 0.0),
        "lower_inclusive": np.where(column <= row, 1.0, 0.0),
        "upper_inclusive": np.where(column >= row, 1.0, 0.0),
        "identity": np.eye(SIZE),
        "ones": np.ones((SIZE, SIZE)),
        "last_row_selector": np.broadcast_to(np.eye(SIZE)[-1], (SIZE, SIZE)),
        "negative_ones": -np.ones((SIZE, SIZE)),
    }
    return {
        name: np.broadcast_to(value.astype(np.float32), (batch, SIZE, SIZE)).copy()
        for name, value in constants.items()
    }


def prepare_chunks(
    oracle_path: Path, batch: int, query_scale: float
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    with np.load(oracle_path) as oracle:
        query = np.asarray(oracle["recurrent_query"], dtype=np.float32)[0, :, :batch]
        key = np.asarray(oracle["recurrent_key"], dtype=np.float32)[0, :, :batch]
        value = np.asarray(oracle["recurrent_value"], dtype=np.float32)[0, :, :batch]
        log_decay = np.asarray(oracle["recurrent_g"], dtype=np.float32)[0, :, :batch]
        beta = np.asarray(oracle["recurrent_beta"], dtype=np.float32)[0, :, :batch]
        captured_output = np.asarray(oracle["recurrent_output"], dtype=np.float32)[
            0, :, :batch
        ]
        captured_state = np.asarray(
            oracle["recurrent_final_state"], dtype=np.float32
        )[0, :batch]
    query, key = normalize_all(query, key)
    constants = shared_constants(batch)
    inverse_scale = np.full(
        (batch, SIZE, SIZE), 1.0 / query_scale, dtype=np.float32
    )
    chunks = []
    for start in range(0, query.shape[0], 64):
        stop = min(start + 64, query.shape[0])
        valid = stop - start
        q = np.zeros((batch, SIZE, SIZE), dtype=np.float16)
        k = np.zeros_like(q)
        v = np.zeros_like(q)
        b = np.zeros((batch, SIZE), dtype=np.float16)
        g = np.zeros_like(b)
        q[:, :valid] = np.transpose(query[start:stop], (1, 0, 2)).astype(np.float16)
        k[:, :valid] = np.transpose(key[start:stop], (1, 0, 2)).astype(np.float16)
        v[:, :valid] = np.transpose(value[start:stop], (1, 0, 2)).astype(np.float16)
        b[:, :valid] = np.transpose(beta[start:stop], (1, 0)).astype(np.float16)
        g[:, :valid] = np.transpose(log_decay[start:stop], (1, 0)).astype(np.float16)
        external = {
            **constants,
            "q": q.astype(np.float32) * query_scale,
            "k": k.astype(np.float32),
            "kt": np.transpose(k.astype(np.float32), (0, 2, 1)).copy(),
            "v": v.astype(np.float32),
            "beta": np.broadcast_to(b[:, :, None], (batch, SIZE, SIZE)).copy().astype(np.float32),
            "log_decay_diagonal": np.stack(
                [np.diag(head.astype(np.float32)) for head in g]
            ),
            "state": np.zeros((batch, SIZE, SIZE), dtype=np.float32),
            "query_inverse_scale": inverse_scale,
        }
        chunks.append(
            {
                "start": start,
                "stop": stop,
                "valid": valid,
                "external": external,
                "q": q.astype(np.float32),
                "k": k.astype(np.float32),
                "v": v.astype(np.float32),
                "beta_vector": b.astype(np.float32),
                "g_vector": g.astype(np.float32),
            }
        )
    return chunks, captured_output, captured_state


def pack_chunks(chunks: list[dict[str, object]], inputs: list[str]) -> None:
    for chunk in chunks:
        external = chunk["external"]
        chunk["packed"] = np.stack([external[name] for name in inputs], axis=0)


def run_ane_chain(
    bridge: ANEBridge,
    handle_name: str,
    segment: Segment,
    chunks: list[dict[str, object]],
    batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros((batch, SIZE, SIZE), dtype=np.float32)
    output_parts = []
    state_input = segment.inputs.index("state")
    for chunk in chunks:
        packed = chunk["packed"]
        packed[state_input] = state
        raw = bridge.evaluate(
            handle_name, packed, len(segment.outputs) * batch
        ).reshape(len(segment.outputs), batch, SIZE, SIZE)
        values = dict(zip(segment.outputs, raw))
        valid = int(chunk["valid"])
        output_parts.append(np.transpose(values["output"][:, :valid], (1, 0, 2)))
        state = np.ascontiguousarray(values["final_state"])
    return np.concatenate(output_parts, axis=0), state


def run_cpu_chain(
    chunks: list[dict[str, object]], batch: int
) -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros((batch, SIZE, SIZE), dtype=np.float32)
    output_parts = []
    for chunk in chunks:
        valid = int(chunk["valid"])
        head_outputs = []
        next_states = []
        for head in range(batch):
            output, final_state = sequential_reference(
                chunk["q"][head],
                chunk["k"][head],
                chunk["v"][head],
                chunk["g_vector"][head],
                chunk["beta_vector"][head],
                state[head],
            )
            head_outputs.append(output[:valid])
            next_states.append(final_state.astype(np.float16).astype(np.float32))
        output_parts.append(np.transpose(np.stack(head_outputs), (1, 0, 2)))
        state = np.stack(next_states)
    return np.concatenate(output_parts, axis=0), state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", required=True, type=Path)
    parser.add_argument("--mil", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--query-scale", type=float, default=4096.0)
    parser.add_argument("--benchmark-runs", type=int, default=20)
    args = parser.parse_args()

    source_operations = parse_graph(args.mil)
    operations = add_query_scaling(source_operations, args.query_scale)
    segment = make_segments(operations, len(operations), connected_only=False)[0]
    rows = []
    for batch in args.batches:
        chunks, captured_output, captured_state = prepare_chunks(
            args.oracle, batch, args.query_scale
        )
        pack_chunks(chunks, segment.inputs)
        input_bytes = len(segment.inputs) * batch * MATRIX_BYTES_F32
        output_bytes = len(segment.outputs) * batch * MATRIX_BYTES_F32
        bridge = ANEBridge(args.bridge)
        handle_name = f"gdn_batch_{batch}"
        try:
            started = time.perf_counter_ns()
            try:
                bridge.compile(
                    handle_name,
                    batched_segment_mil(segment, batch),
                    input_bytes,
                    output_bytes,
                )
            except RuntimeError as error:
                rows.append(
                    {
                        "heads": batch,
                        "compiled": False,
                        "error": str(error),
                        "input_bytes": input_bytes,
                        "output_bytes": output_bytes,
                    }
                )
                continue
            compile_ms = (time.perf_counter_ns() - started) / 1_000_000
            actual_output, actual_state = run_ane_chain(
                bridge, handle_name, segment, chunks, batch
            )
            samples = []
            for _ in range(args.benchmark_runs):
                started = time.perf_counter_ns()
                run_ane_chain(bridge, handle_name, segment, chunks, batch)
                samples.append((time.perf_counter_ns() - started) / 1_000_000)
        finally:
            bridge.close()

        cpu_output, cpu_state = run_cpu_chain(chunks, batch)
        output_metrics = metrics(actual_output, captured_output)
        state_metrics = metrics(actual_state, captured_state)
        ordered = sorted(samples)
        rows.append(
            {
                "heads": batch,
                "compiled": True,
                "compile_ms": compile_ms,
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
                "ane_vs_captured_we_mm_model": {
                    "output": output_metrics,
                    "final_state": state_metrics,
                },
                "ane_vs_cpu_fp16_boundary_chain": {
                    "output": metrics(actual_output, cpu_output),
                    "final_state": metrics(actual_state, cpu_state),
                },
                "benchmark": {
                    "runs": len(samples),
                    "median_ms": statistics.median(samples),
                    "p95_ms": ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)],
                    "min_ms": min(samples),
                    "max_ms": max(samples),
                    "scope": "all four chunks; packed buffers prebuilt; state surface update included",
                },
                "valid": all(
                    item["nonfinite"] == 0
                    and item["relative_l2"] <= 0.05
                    and item["cosine"] >= 0.999
                    for item in (output_metrics, state_metrics)
                ),
            }
        )

    single = next((row for row in rows if row["heads"] == 1 and row["compiled"]), None)
    if single:
        single_ms = single["benchmark"]["median_ms"]
        for row in rows:
            if row["compiled"]:
                row["speedup_vs_serial_single_head"] = (
                    single_ms * row["heads"] / row["benchmark"]["median_ms"]
                )
    payload = {
        "schema_version": 1,
        "experiment": "real WeMM 56-op GDN batched-head ANE graph",
        "route": "apple_private_inmemory_single_graph",
        "executes_research_generated_hwx": False,
        "source_operation_count": len(source_operations),
        "compiled_operation_count": len(operations),
        "query_scale": args.query_scale,
        "tokens": 206,
        "chunks": 4,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
