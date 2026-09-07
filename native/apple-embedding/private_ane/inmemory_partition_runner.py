#!/usr/bin/env python3
"""Run a partitioned C128 DeltaNet graph through Apple's in-memory ANE path.

This is an alternate hardware evidence path for systems that protect the raw
aned cache.  It executes the same MIL operation DAG, but asks Apple's private
compiler to build three reusable single-input kernels (binary matmul, binary
ALU, unary exp).  It intentionally does not claim to execute mil-hwxc's HWX.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import re
import statistics
import time
from pathlib import Path

import numpy as np


SIZE = 128
MATRIX_ELEMENTS = SIZE * SIZE
MATRIX_BYTES_F16 = MATRIX_ELEMENTS * 2
MATRIX_BYTES_F32 = MATRIX_ELEMENTS * 4
PACKED_BYTES_F32 = SIZE * SIZE * 2 * 4
OPERATION = re.compile(
    r"tensor<fp16,\s*\[[^]]+\]>\s+(\w+)\s*=\s*"
    r"(matmul|mul|add|exp)\((.*?)\)\[name",
    re.DOTALL,
)
ARGUMENT = re.compile(r"\b([xy])\s*=\s*(\w+)")


def mil_header(input_shape: str) -> str:
    return f'''program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(tensor<fp32, [{input_shape}]> packed) {{
'''


def binary_mil(operation: str) -> str:
    if operation not in {"matmul", "mul", "add"}:
        raise ValueError(operation)
    common = f'''        string to16 = const()[name=string("to16"), val=string("fp16")];
        tensor<fp16, [1, 128, 1, 256]> packed16 = cast(dtype=to16, x=packed)[name=string("cast_in")];
        tensor<int32, [4]> begin_a = const()[name=string("begin_a"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [4]> begin_b = const()[name=string("begin_b"), val=tensor<int32, [4]>([0,0,0,128])];
        tensor<int32, [4]> slice_size = const()[name=string("slice_size"), val=tensor<int32, [4]>([1,128,1,128])];
        tensor<fp16, [1,128,1,128]> a_raw = slice_by_size(x=packed16, begin=begin_a, size=slice_size)[name=string("slice_a")];
        tensor<fp16, [1,128,1,128]> b_raw = slice_by_size(x=packed16, begin=begin_b, size=slice_size)[name=string("slice_b")];
        tensor<int32, [4]> matrix_shape = const()[name=string("matrix_shape"), val=tensor<int32, [4]>([1,1,128,128])];
        tensor<fp16, [1,1,128,128]> a = reshape(shape=matrix_shape, x=a_raw)[name=string("reshape_a")];
        tensor<fp16, [1,1,128,128]> b = reshape(shape=matrix_shape, x=b_raw)[name=string("reshape_b")];
'''
    if operation == "matmul":
        compute = '''        bool f = const()[name=string("f"), val=bool(false)];
        tensor<fp16, [1,1,128,128]> result = matmul(transpose_x=f, transpose_y=f, x=a, y=b)[name=string("compute")];
'''
    else:
        compute = (
            f'        tensor<fp16, [1,1,128,128]> result = {operation}'
            '(x=a, y=b)[name=string("compute")];\n'
        )
    footer = '''        tensor<int32, [4]> output_shape = const()[name=string("output_shape"), val=tensor<int32, [4]>([1,128,1,128])];
        tensor<fp16, [1,128,1,128]> result_raw = reshape(shape=output_shape, x=result)[name=string("reshape_out")];
        string to32 = const()[name=string("to32"), val=string("fp32")];
        tensor<fp32, [1,128,1,128]> output = cast(dtype=to32, x=result_raw)[name=string("cast_out")];
    } -> (output);
}
'''
    return mil_header("1, 128, 1, 256") + common + compute + footer


def unary_exp_mil() -> str:
    return mil_header("1, 128, 1, 128") + '''        string to16 = const()[name=string("to16"), val=string("fp16")];
        tensor<fp16, [1,128,1,128]> input16 = cast(dtype=to16, x=packed)[name=string("cast_in")];
        tensor<int32, [4]> matrix_shape = const()[name=string("matrix_shape"), val=tensor<int32, [4]>([1,1,128,128])];
        tensor<fp16, [1,1,128,128]> matrix = reshape(shape=matrix_shape, x=input16)[name=string("reshape_in")];
        tensor<fp16, [1,1,128,128]> result = exp(x=matrix)[name=string("compute")];
        tensor<int32, [4]> output_shape = const()[name=string("output_shape"), val=tensor<int32, [4]>([1,128,1,128])];
        tensor<fp16, [1,128,1,128]> result_raw = reshape(shape=output_shape, x=result)[name=string("reshape_out")];
        string to32 = const()[name=string("to32"), val=string("fp32")];
        tensor<fp32, [1,128,1,128]> output = cast(dtype=to32, x=result_raw)[name=string("cast_out")];
    } -> (output);
}
'''


class ANEBridge:
    def __init__(self, library: Path):
        self.lib = ctypes.CDLL(str(library))
        self.lib.ane_bridge_init.restype = ctypes.c_int
        self.lib.ane_bridge_compile.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.ane_bridge_compile.restype = ctypes.c_void_p
        self.lib.ane_bridge_eval.argtypes = [ctypes.c_void_p]
        self.lib.ane_bridge_eval.restype = ctypes.c_bool
        self.lib.ane_bridge_write_input.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.ane_bridge_read_output.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.ane_bridge_free.argtypes = [ctypes.c_void_p]
        if self.lib.ane_bridge_init() != 0:
            raise RuntimeError("cannot initialize private ANE bridge")
        self.handles: dict[str, int] = {}
        self.input_sizes: dict[str, list[int]] = {}
        self.output_sizes: dict[str, list[int]] = {}
        self.input_dtypes: dict[str, np.dtype] = {}
        self.output_dtypes: dict[str, np.dtype] = {}

    def compile(
        self,
        name: str,
        mil: str,
        input_bytes: int,
        output_bytes: int = MATRIX_BYTES_F32,
    ) -> None:
        self.compile_surfaces(name, mil, [input_bytes], [output_bytes])

    def compile_surfaces(
        self,
        name: str,
        mil: str,
        input_bytes: list[int],
        output_bytes: list[int],
        *,
        input_dtype: object = np.float32,
        output_dtype: object = np.float32,
    ) -> None:
        encoded = mil.encode()
        inputs = (ctypes.c_size_t * len(input_bytes))(*input_bytes)
        outputs = (ctypes.c_size_t * len(output_bytes))(*output_bytes)
        handle = self.lib.ane_bridge_compile(
            encoded,
            len(encoded),
            None,
            0,
            len(input_bytes),
            inputs,
            len(output_bytes),
            outputs,
        )
        if not handle:
            raise RuntimeError(f"ANE in-memory compile/load failed for {name}")
        self.handles[name] = handle
        self.input_sizes[name] = list(input_bytes)
        self.output_sizes[name] = list(output_bytes)
        self.input_dtypes[name] = np.dtype(input_dtype)
        self.output_dtypes[name] = np.dtype(output_dtype)

    def write_input(self, name: str, index: int, value: np.ndarray) -> None:
        source = np.ascontiguousarray(value, dtype=self.input_dtypes[name])
        expected_bytes = self.input_sizes[name][index]
        if source.nbytes != expected_bytes:
            raise ValueError(
                f"input {index} for {name} has {source.nbytes} bytes, "
                f"expected {expected_bytes}"
            )
        self.lib.ane_bridge_write_input(
            self.handles[name],
            index,
            source.ctypes.data_as(ctypes.c_void_p),
            source.nbytes,
        )

    def evaluate_preloaded(
        self,
        name: str,
        output_matrices: int = 1,
        result: np.ndarray | None = None,
    ) -> np.ndarray:
        expected_shape = (SIZE, SIZE * output_matrices)
        expected_dtype = self.output_dtypes[name]
        if result is None:
            result = np.empty(expected_shape, dtype=expected_dtype)
        elif (
            result.shape != expected_shape
            or result.dtype != expected_dtype
            or not result.flags.c_contiguous
        ):
            raise ValueError(
                f"output buffer for {name} must be contiguous "
                f"{expected_shape} {expected_dtype}"
            )
        handle = self.handles[name]
        if not self.lib.ane_bridge_eval(handle):
            raise RuntimeError(f"ANE evaluation failed for {name}")
        self.lib.ane_bridge_read_output(
            handle, 0, result.ctypes.data_as(ctypes.c_void_p), result.nbytes
        )
        return result

    def evaluate(
        self, name: str, value: np.ndarray, output_matrices: int = 1
    ) -> np.ndarray:
        source = np.ascontiguousarray(value, dtype=np.float32)
        self.write_input(name, 0, source)
        return self.evaluate_preloaded(name, output_matrices)

    def close(self) -> None:
        for handle in self.handles.values():
            self.lib.ane_bridge_free(handle)
        self.handles.clear()
        self.input_sizes.clear()
        self.output_sizes.clear()
        self.input_dtypes.clear()
        self.output_dtypes.clear()


def load_inputs(directory: Path) -> dict[str, np.ndarray]:
    read_matrix = lambda name: np.fromfile(
        directory / name, dtype=np.float16, count=MATRIX_ELEMENTS
    ).astype(np.float32).reshape(SIZE, SIZE)
    k = read_matrix("k.f16")
    beta = np.fromfile(directory / "beta.f16", dtype=np.float16, count=SIZE)
    decay = np.fromfile(
        directory / "log_decay.f16", dtype=np.float16, count=SIZE
    )
    row, column = np.indices((SIZE, SIZE))
    return {
        "q": read_matrix("q.f16"),
        "k": k,
        "kt": np.ascontiguousarray(k.T),
        "v": read_matrix("v.f16"),
        "beta": np.broadcast_to(beta.astype(np.float32)[:, None], (SIZE, SIZE)).copy(),
        "log_decay_diagonal": np.diag(decay.astype(np.float32)),
        "state": read_matrix("state.f16"),
        "negative_strict_lower": np.where(column < row, -1.0, 0.0).astype(np.float32),
        "lower_inclusive": np.where(column <= row, 1.0, 0.0).astype(np.float32),
        "upper_inclusive": np.where(column >= row, 1.0, 0.0).astype(np.float32),
        "identity": np.eye(SIZE, dtype=np.float32),
        "ones": np.ones((SIZE, SIZE), dtype=np.float32),
        "last_row_selector": np.broadcast_to(
            np.eye(SIZE, dtype=np.float32)[-1], (SIZE, SIZE)
        ).copy(),
        "negative_ones": -np.ones((SIZE, SIZE), dtype=np.float32),
    }


def parse_graph(path: Path) -> list[tuple[str, str, str, str | None]]:
    operations = []
    for output, operation, arguments in OPERATION.findall(path.read_text()):
        operands = dict(ARGUMENT.findall(arguments))
        operations.append((output, operation, operands["x"], operands.get("y")))
    if not operations:
        raise RuntimeError(f"no operations parsed from {path}")
    return operations


def execute_graph(
    bridge: ANEBridge,
    operations: list[tuple[str, str, str, str | None]],
    external: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    values = dict(external)
    for output, operation, left_name, right_name in operations:
        left = values[left_name]
        if operation == "exp":
            values[output] = bridge.evaluate("exp", left)
        else:
            right = values[right_name]
            packed = np.concatenate((left, right), axis=1)
            kernel = "matmul" if operation == "matmul" else operation
            values[output] = bridge.evaluate(kernel, packed)
    return values


def execute_graph_fp16_emulation(
    operations: list[tuple[str, str, str, str | None]],
    external: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Emulate the declared MIL tensor precision at every operation boundary.

    This is intentionally different from the FP32 sequential recurrence used
    as the model-quality reference.  Every external input is cast to FP16 and
    every op produces an FP16 tensor, matching the generated MIL types.  It
    lets us distinguish expected reduced-precision drift from a bad ANE graph.
    """
    values = {
        name: np.asarray(value, dtype=np.float16)
        for name, value in external.items()
    }
    with np.errstate(over="ignore", invalid="ignore"):
        for output, operation, left_name, right_name in operations:
            left = values[left_name]
            if operation == "exp":
                result = np.exp(left)
            else:
                right = values[right_name]
                if operation == "matmul":
                    result = left @ right
                elif operation == "mul":
                    result = left * right
                elif operation == "add":
                    result = left + right
                else:
                    raise ValueError(operation)
            values[output] = np.asarray(result, dtype=np.float16)
    return {name: value.astype(np.float32) for name, value in values.items()}


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    actual64 = actual.astype(np.float64).reshape(-1)
    expected64 = expected.astype(np.float64).reshape(-1)
    difference = actual64 - expected64
    expected_norm = float(np.linalg.norm(expected64))
    actual_norm = float(np.linalg.norm(actual64))
    cosine_denominator = max(expected_norm * actual_norm, 1e-30)
    return {
        "nonfinite": int(np.count_nonzero(~np.isfinite(actual))),
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "relative_l2": float(
            np.linalg.norm(difference) / max(expected_norm, 1e-30)
        ),
        "cosine": float(actual64 @ expected64 / cosine_denominator),
        "actual_l2_norm": actual_norm,
        "expected_l2_norm": expected_norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", required=True, type=Path)
    parser.add_argument("--mil", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmark-runs", type=int, default=10)
    args = parser.parse_args()

    operations = parse_graph(args.mil)
    external = load_inputs(args.data_dir)
    expected_output = np.fromfile(
        args.data_dir / "expected_output.f32", dtype=np.float32
    ).reshape(SIZE, SIZE)
    expected_state = np.fromfile(
        args.data_dir / "expected_final_state.f32", dtype=np.float32
    ).reshape(SIZE, SIZE)

    bridge = ANEBridge(args.bridge)
    try:
        bridge.compile("matmul", binary_mil("matmul"), PACKED_BYTES_F32)
        bridge.compile("mul", binary_mil("mul"), PACKED_BYTES_F32)
        bridge.compile("add", binary_mil("add"), PACKED_BYTES_F32)
        bridge.compile("exp", unary_exp_mil(), MATRIX_BYTES_F32)
        values = execute_graph(bridge, operations, external)
        samples_ms = []
        for _ in range(args.benchmark_runs):
            started = time.perf_counter_ns()
            execute_graph(bridge, operations, external)
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
    per_operation_agreement = []
    for index, (output, operation, left, right) in enumerate(operations):
        per_operation_agreement.append(
            {
                "index": index,
                "output": output,
                "operation": operation,
                "inputs": [name for name in (left, right) if name is not None],
                **metrics(values[output], fp16_emulated[output]),
            }
        )
    worst_relative = sorted(
        per_operation_agreement,
        key=lambda item: float(item["relative_l2"]),
        reverse=True,
    )[:8]
    valid = all(
        item["nonfinite"] == 0
        and item["max_abs_error"] <= 0.01
        and item["relative_l2"] <= 0.05
        for item in (output_metrics, state_metrics)
    )
    operation_counts = {
        name: sum(operation == name for _, operation, _, _ in operations)
        for name in ("matmul", "mul", "add", "exp")
    }
    ordered = sorted(samples_ms)
    payload = {
        "schema_version": 1,
        "route": "apple_private_inmemory_partitioned",
        "executes_research_generated_hwx": False,
        "graph": str(args.mil),
        "operation_count": len(operations),
        "operation_counts": operation_counts,
        "reusable_kernel_count": 4,
        "input_policy": "one packed fp32 IOSurface per kernel; cast to fp16 on ANE",
        "intermediate_policy": "synchronous ANE output readback and repack between operations",
        "output": output_metrics,
        "final_state": state_metrics,
        "ane_vs_cpu_fp16_mil_emulation": fp16_emulation_agreement,
        "ane_vs_cpu_fp16_per_operation": per_operation_agreement,
        "worst_fp16_emulation_relative_l2_operations": worst_relative,
        "benchmark": {
            "runs": len(samples_ms),
            "median_ms": statistics.median(samples_ms),
            "p95_ms": ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)],
            "min_ms": min(samples_ms),
            "max_ms": max(samples_ms),
            "includes_cpu_pack_and_55_synchronous_dispatches": True,
        },
        "valid": valid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
