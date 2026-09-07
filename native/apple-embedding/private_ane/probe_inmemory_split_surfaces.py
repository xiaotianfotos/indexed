#!/usr/bin/env python3
"""Probe batched GDN with dynamic and shared inputs on separate ANE surfaces."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

from inmemory_batched_graph_runner import prepare_chunks, run_cpu_chain
from inmemory_partition_runner import (
    ANEBridge,
    MATRIX_BYTES_F32,
    SIZE,
    metrics,
    parse_graph,
)
from inmemory_segmented_runner import Segment, add_query_scaling, make_segments


DYNAMIC_NAMES = {
    "q",
    "k",
    "kt",
    "v",
    "beta",
    "log_decay_diagonal",
    "state",
}


def split_surface_mil(
    segment: Segment,
    batch: int,
    dynamic_names: list[str],
    shared_names: list[str],
    *,
    io_dtype: str = "fp32",
) -> str:
    if io_dtype not in {"fp16", "fp32"}:
        raise ValueError("io_dtype must be fp16 or fp32")
    output_count = len(segment.outputs)
    lines = [
        "program(1.3)",
        '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]',
        "{",
        f"  func main<ios18>(tensor<{io_dtype}, [{len(dynamic_names)},{batch},128,128]> dynamic, tensor<{io_dtype}, [{len(shared_names)},1,128,128]> shared) {{",
        f'    tensor<int32, [4]> dynamic_slice_shape = const()[name=string("dynamic_slice_shape"), val=tensor<int32, [4]>([1,{batch},128,128])];',
        '    tensor<int32, [4]> shared_slice_shape = const()[name=string("shared_slice_shape"), val=tensor<int32, [4]>([1,1,128,128])];',
        f'    tensor<int32, [4]> dynamic_matrix_shape = const()[name=string("dynamic_matrix_shape"), val=tensor<int32, [4]>([{batch},1,128,128])];',
        '    tensor<int32, [4]> shared_matrix_shape = const()[name=string("shared_matrix_shape"), val=tensor<int32, [4]>([1,1,128,128])];',
        f'    tensor<int32, [4]> live_shape = const()[name=string("live_shape"), val=tensor<int32, [4]>([1,{batch},128,128])];',
        '    bool f = const()[name=string("f"), val=bool(false)];',
    ]
    if io_dtype == "fp32":
        lines.extend(
            [
                '    string to16 = const()[name=string("to16"), val=string("fp16")];',
                f'    tensor<fp16, [{len(dynamic_names)},{batch},128,128]> dynamic16 = cast(dtype=to16, x=dynamic)[name=string("cast_dynamic")];',
                f'    tensor<fp16, [{len(shared_names)},1,128,128]> shared16 = cast(dtype=to16, x=shared)[name=string("cast_shared")];',
            ]
        )
    input_suffix = "16" if io_dtype == "fp32" else ""
    aliases: dict[str, str] = {}
    for kind, names in (("dynamic", dynamic_names), ("shared", shared_names)):
        size_name = f"{kind}_slice_shape"
        shape_name = f"{kind}_matrix_shape"
        batch_dim = batch if kind == "dynamic" else 1
        for index, name in enumerate(names):
            alias = f"{kind}_input_{index}"
            aliases[name] = alias
            lines.extend(
                [
                    f'    tensor<int32, [4]> {kind}_begin_{index} = const()[name=string("{kind}_begin_{index}"), val=tensor<int32, [4]>([{index},0,0,0])];',
                    f'    tensor<fp16, [1,{batch_dim},128,128]> {kind}_raw_{index} = slice_by_size(x={kind}{input_suffix}, begin={kind}_begin_{index}, size={size_name})[name=string("{kind}_slice_{index}")];',
                    f'    tensor<fp16, [{batch_dim},1,128,128]> {alias} = reshape(shape={shape_name}, x={kind}_raw_{index})[name=string("{kind}_reshape_{index}")];',
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
    values = ",".join(f"live_{index}" for index in range(output_count))
    lines.extend(
        [
            '    int32 concat_axis = const()[name=string("concat_axis"), val=int32(0)];',
            '    bool concat_interleave = const()[name=string("concat_interleave"), val=bool(false)];',
            f'    tensor<fp16, [{output_count},{batch},128,128]> combined = concat(axis=concat_axis, interleave=concat_interleave, values=({values}))[name=string("pack_outputs")];',
            *(
                [
                    '    string to32 = const()[name=string("to32"), val=string("fp32")];',
                    f'    tensor<fp32, [{output_count},{batch},128,128]> result = cast(dtype=to32, x=combined)[name=string("cast_out")];',
                ]
                if io_dtype == "fp32"
                else []
            ),
            f"  }} -> ({'result' if io_dtype == 'fp32' else 'combined'});",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def run_chain(
    bridge: ANEBridge,
    name: str,
    segment: Segment,
    chunks: list[dict[str, object]],
    dynamic_names: list[str],
    batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros((batch, SIZE, SIZE), dtype=np.float32)
    output_parts = []
    state_index = dynamic_names.index("state")
    for chunk in chunks:
        dynamic = chunk["dynamic"]
        dynamic[state_index] = state
        bridge.write_input(name, 0, dynamic)
        raw = bridge.evaluate_preloaded(
            name, len(segment.outputs) * batch
        ).reshape(len(segment.outputs), batch, SIZE, SIZE)
        values = dict(zip(segment.outputs, raw))
        valid = int(chunk["valid"])
        output_parts.append(np.transpose(values["output"][:, :valid], (1, 0, 2)))
        state = np.ascontiguousarray(values["final_state"])
    return np.concatenate(output_parts, axis=0), state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", required=True, type=Path)
    parser.add_argument("--mil", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--query-scale", type=float, default=4096.0)
    parser.add_argument("--benchmark-runs", type=int, default=30)
    args = parser.parse_args()

    operations = add_query_scaling(parse_graph(args.mil), args.query_scale)
    segment = make_segments(operations, len(operations), connected_only=False)[0]
    dynamic_names = [name for name in segment.inputs if name in DYNAMIC_NAMES]
    shared_names = [name for name in segment.inputs if name not in DYNAMIC_NAMES]
    rows = []
    for batch in args.batches:
        chunks, captured_output, captured_state = prepare_chunks(
            args.oracle, batch, args.query_scale
        )
        for chunk in chunks:
            external = chunk["external"]
            chunk["dynamic"] = np.stack(
                [external[name] for name in dynamic_names], axis=0
            )
        shared = np.stack(
            [chunks[0]["external"][name][0] for name in shared_names], axis=0
        )[:, None]
        dynamic_bytes = len(dynamic_names) * batch * MATRIX_BYTES_F32
        shared_bytes = len(shared_names) * MATRIX_BYTES_F32
        output_bytes = len(segment.outputs) * batch * MATRIX_BYTES_F32
        name = f"gdn_split_{batch}"
        bridge = ANEBridge(args.bridge)
        try:
            started = time.perf_counter_ns()
            try:
                bridge.compile_surfaces(
                    name,
                    split_surface_mil(
                        segment, batch, dynamic_names, shared_names
                    ),
                    [dynamic_bytes, shared_bytes],
                    [output_bytes],
                )
            except RuntimeError as error:
                rows.append(
                    {
                        "heads": batch,
                        "compiled": False,
                        "error": str(error),
                        "dynamic_bytes_per_chunk": dynamic_bytes,
                        "shared_bytes_written_once": shared_bytes,
                    }
                )
                continue
            compile_ms = (time.perf_counter_ns() - started) / 1_000_000
            bridge.write_input(name, 1, shared)
            actual_output, actual_state = run_chain(
                bridge, name, segment, chunks, dynamic_names, batch
            )
            samples = []
            for _ in range(args.benchmark_runs):
                started = time.perf_counter_ns()
                run_chain(bridge, name, segment, chunks, dynamic_names, batch)
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
                "dynamic_surface_count": 1,
                "shared_surface_count": 1,
                "dynamic_bytes_per_chunk": dynamic_bytes,
                "shared_bytes_written_once": shared_bytes,
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
                    "scope": "four chunks; one dynamic write per chunk; shared surface preloaded once",
                },
                "valid": all(
                    item["nonfinite"] == 0
                    and item["relative_l2"] <= 0.05
                    and item["cosine"] >= 0.999
                    for item in (output_metrics, state_metrics)
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "experiment": "batched GDN split dynamic/shared ANE surfaces",
        "route": "apple_private_inmemory_single_graph_two_input_surfaces",
        "dynamic_names": dynamic_names,
        "shared_names": shared_names,
        "query_scale": args.query_scale,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
