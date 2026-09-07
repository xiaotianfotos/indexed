#!/usr/bin/env python3
"""Execute the real C128 GDN graph as fused in-memory ANE segments."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inmemory_partition_runner import (
    ANEBridge,
    MATRIX_BYTES_F32,
    SIZE,
    execute_graph_fp16_emulation,
    load_inputs,
    metrics,
    parse_graph,
)


Operation = tuple[str, str, str, str | None]


@dataclass
class Segment:
    index: int
    start: int
    end: int
    inputs: list[str]
    outputs: list[str]
    operations: list[Operation]

    @property
    def name(self) -> str:
        return f"segment_{self.index}_{self.start}_{self.end}"


def add_query_scaling(
    operations: list[Operation], scale: float
) -> list[Operation]:
    """Scale q before the graph and rescale its final output inside MIL.

    The two q-dependent branches are linear in q.  A power-of-two scale lifts
    their tiny FP16 matmul products away from the ANE's low-magnitude error
    floor without changing the represented function.  The final multiply is
    deliberately part of the ANE graph, not a hidden CPU correction.
    """
    if scale == 1.0:
        return list(operations)
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("query scale must be finite and positive")
    scaled: list[Operation] = []
    found_output = False
    for output, operation, left, right in operations:
        if output == "output":
            scaled.append(("scaled_output", operation, left, right))
            found_output = True
        else:
            scaled.append((output, operation, left, right))
    if not found_output:
        raise RuntimeError("graph has no output operation to rescale")
    scaled.append(("output", "mul", "scaled_output", "query_inverse_scale"))
    return scaled


def make_segments(
    operations: list[Operation], width: int, connected_only: bool
) -> list[Segment]:
    consumers: dict[str, list[int]] = {}
    for index, (_, _, left, right) in enumerate(operations):
        consumers.setdefault(left, []).append(index)
        if right is not None:
            consumers.setdefault(right, []).append(index)
    final_names = {"output", "final_state"}
    segments = []
    start = 0
    segment_index = 0
    while start < len(operations):
        end = start
        local_producers: set[str] = set()
        while end < len(operations) and end - start < width:
            output, _, left, right = operations[end]
            dependencies = {left} | ({right} if right is not None else set())
            if connected_only and end > start and not dependencies & local_producers:
                break
            local_producers.add(output)
            end += 1
        chunk = operations[start:end]
        produced = {output for output, _, _, _ in chunk}
        inputs: list[str] = []
        for _, _, left, right in chunk:
            for name in (left, right):
                if name is not None and name not in produced and name not in inputs:
                    inputs.append(name)
        outputs = [
            output
            for output, _, _, _ in chunk
            if output in final_names
            or any(use >= end for use in consumers.get(output, []))
        ]
        if not outputs:
            raise RuntimeError(f"segment {start}:{end} has no live output")
        segments.append(
            Segment(segment_index, start, end, inputs, outputs, chunk)
        )
        start = end
        segment_index += 1
    return segments


def segment_mil(segment: Segment) -> str:
    input_width = len(segment.inputs) * SIZE
    output_width = len(segment.outputs) * SIZE
    lines = [
        "program(1.3)",
        '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]',
        "{",
        f"  func main<ios18>(tensor<fp32, [1,128,1,{input_width}]> packed) {{",
        '    string to16 = const()[name=string("to16"), val=string("fp16")];',
        f'    tensor<fp16, [1,128,1,{input_width}]> packed16 = cast(dtype=to16, x=packed)[name=string("cast_in")];',
        '    tensor<int32, [4]> slice_size = const()[name=string("slice_size"), val=tensor<int32, [4]>([1,128,1,128])];',
        '    tensor<int32, [4]> matrix_shape = const()[name=string("matrix_shape"), val=tensor<int32, [4]>([1,1,128,128])];',
        '    tensor<int32, [4]> io_shape = const()[name=string("io_shape"), val=tensor<int32, [4]>([1,128,1,128])];',
        '    bool f = const()[name=string("f"), val=bool(false)];',
    ]
    # Keep Apple MIL identifiers deliberately boring.  Names such as ``state``
    # and ``state_read`` describe the GDN graph well, but the private compiler
    # has undocumented parser/name-resolution rules.  Map every external and
    # intermediate value to a neutral SSA identifier so a segment's ability to
    # compile depends on its dataflow, not on names inherited from the source
    # model.
    aliases: dict[str, str] = {}
    for index, name in enumerate(segment.inputs):
        alias = f"input_{index}"
        aliases[name] = alias
        lines.extend(
            [
                f'    tensor<int32, [4]> begin_{index} = const()[name=string("begin_{index}"), val=tensor<int32, [4]>([0,0,0,{index * SIZE}])];',
                f'    tensor<fp16, [1,128,1,128]> raw_{index} = slice_by_size(x=packed16, begin=begin_{index}, size=slice_size)[name=string("slice_{index}")];',
                f'    tensor<fp16, [1,1,128,128]> {alias} = reshape(shape=matrix_shape, x=raw_{index})[name=string("reshape_{index}")];',
            ]
        )
    for local_index, (output, operation, left, right) in enumerate(
        segment.operations
    ):
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
            f'    tensor<fp16, [1,1,128,128]> {output_alias} = {expression}[name=string("op_{local_index}")];'
        )
        aliases[output] = output_alias
    for index, output in enumerate(segment.outputs):
        lines.append(
            f'    tensor<fp16, [1,128,1,128]> live_{index} = reshape(shape=io_shape, x={aliases[output]})[name=string("live_{index}")];'
        )
    if len(segment.outputs) == 1:
        lines.append(
            '    tensor<fp16, [1,128,1,128]> combined = reshape(shape=io_shape, x=live_0)[name=string("single_output")];'
        )
    else:
        values = ",".join(f"live_{index}" for index in range(len(segment.outputs)))
        lines.extend(
            [
                '    int32 concat_axis = const()[name=string("concat_axis"), val=int32(3)];',
                '    bool concat_interleave = const()[name=string("concat_interleave"), val=bool(false)];',
                f'    tensor<fp16, [1,128,1,{output_width}]> combined = concat(axis=concat_axis, interleave=concat_interleave, values=({values}))[name=string("pack_outputs")];',
            ]
        )
    lines.extend(
        [
            '    string to32 = const()[name=string("to32"), val=string("fp32")];',
            f'    tensor<fp32, [1,128,1,{output_width}]> result = cast(dtype=to32, x=combined)[name=string("cast_out")];',
            "  } -> (result);",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def execute(
    bridge: ANEBridge,
    segments: list[Segment],
    external: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    values = dict(external)
    for segment in segments:
        packed = np.concatenate([values[name] for name in segment.inputs], axis=1)
        result = bridge.evaluate(segment.name, packed, len(segment.outputs))
        for index, name in enumerate(segment.outputs):
            values[name] = np.ascontiguousarray(
                result[:, index * SIZE : (index + 1) * SIZE]
            )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", required=True, type=Path)
    parser.add_argument("--mil", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ops-per-segment", required=True, type=int)
    parser.add_argument("--connected-only", action="store_true")
    parser.add_argument("--query-scale", type=float, default=1.0)
    parser.add_argument("--benchmark-runs", type=int, default=10)
    args = parser.parse_args()

    source_operations = parse_graph(args.mil)
    operations = add_query_scaling(source_operations, args.query_scale)
    segments = make_segments(
        operations, args.ops_per_segment, args.connected_only
    )
    external = load_inputs(args.data_dir)
    if args.query_scale != 1.0:
        external["q"] = np.ascontiguousarray(external["q"] * args.query_scale)
        external["query_inverse_scale"] = np.full(
            (SIZE, SIZE), 1.0 / args.query_scale, dtype=np.float32
        )
    expected_output = np.fromfile(
        args.data_dir / "expected_output.f32", dtype=np.float32
    ).reshape(SIZE, SIZE)
    expected_state = np.fromfile(
        args.data_dir / "expected_final_state.f32", dtype=np.float32
    ).reshape(SIZE, SIZE)

    bridge = ANEBridge(args.bridge)
    compile_ms = []
    try:
        for segment in segments:
            generated = segment_mil(segment)
            started = time.perf_counter_ns()
            bridge.compile(
                segment.name,
                generated,
                len(segment.inputs) * MATRIX_BYTES_F32,
                len(segment.outputs) * MATRIX_BYTES_F32,
            )
            compile_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        values = execute(bridge, segments, external)
        samples_ms = []
        for _ in range(args.benchmark_runs):
            started = time.perf_counter_ns()
            execute(bridge, segments, external)
            samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    finally:
        bridge.close()

    output_metrics = metrics(values["output"], expected_output)
    state_metrics = metrics(values["final_state"], expected_state)
    fp16_emulated = execute_graph_fp16_emulation(operations, external)
    fp16_emulation_agreement = {
        "output": metrics(values["output"], fp16_emulated["output"]),
        "final_state": metrics(
            values["final_state"], fp16_emulated["final_state"]
        ),
    }
    ordered = sorted(samples_ms)
    valid = all(
        item["nonfinite"] == 0
        and item["max_abs_error"] <= 0.01
        and item["relative_l2"] <= 0.05
        for item in (output_metrics, state_metrics)
    )
    payload = {
        "schema_version": 1,
        "route": "apple_private_inmemory_segmented",
        "executes_research_generated_hwx": False,
        "ops_per_segment_limit": args.ops_per_segment,
        "segmentation_strategy": (
            "direct-dataflow-connected" if args.connected_only else "fixed-width"
        ),
        "operation_count": len(operations),
        "source_operation_count": len(source_operations),
        "query_scaling": {
            "scale": args.query_scale,
            "inverse_scale_is_ane_graph_operation": args.query_scale != 1.0,
            "mathematical_contract": "both q branches are linear in q",
        },
        "segment_count": len(segments),
        "segments": [
            {
                "start": segment.start,
                "end": segment.end,
                "operation_count": len(segment.operations),
                "input_surface_matrices": len(segment.inputs),
                "output_surface_matrices": len(segment.outputs),
            }
            for segment in segments
        ],
        "compile_ms_total": sum(compile_ms),
        "output": output_metrics,
        "final_state": state_metrics,
        "ane_vs_cpu_fp16_mil_emulation": fp16_emulation_agreement,
        "benchmark": {
            "runs": len(samples_ms),
            "median_ms": statistics.median(samples_ms),
            "p95_ms": ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)],
            "min_ms": min(samples_ms),
            "max_ms": max(samples_ms),
            "includes_cpu_segment_pack_and_synchronous_dispatches": True,
        },
        "valid": valid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
