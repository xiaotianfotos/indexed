#!/usr/bin/env python3
"""Numerically stable block forward solve for the private ANE GDN graph.

The original graph forms a unit-lower-triangular inverse with a repeated-
squaring Neumann product.  That is algebraically exact, but its ``R**8`` and
later intermediates can overflow FP16.  This module rewrites only that solve:

* invert independent, small diagonal blocks with the same short Neumann form;
* propagate completed blocks in causal order with block forward substitution.

All operations remain dense matmul/ALU operations in the ANE graph.  No part
of the recurrence is moved to CPU by this rewrite.
"""

from __future__ import annotations

import math
import re
from typing import TypeAlias

import numpy as np


SIZE = 128
Operation: TypeAlias = tuple[str, str, str, str | None]

_UNSTABLE_INVERSE_VALUE = re.compile(
    r"^(?:inverse(?:_\d+)?|power_\d+(?:_copy)?|factor_\d+|recurrence_copy)$"
)


def validate_geometry(active_tokens: int, block_size: int) -> None:
    if active_tokens <= 0 or active_tokens > SIZE:
        raise ValueError(f"active_tokens must be in [1, {SIZE}]")
    if block_size < 2 or block_size > active_tokens:
        raise ValueError("block_size must be between 2 and active_tokens")
    if block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two")
    if active_tokens % block_size:
        raise ValueError("active_tokens must be divisible by block_size")


def block_solve_constants(
    *, active_tokens: int = 64, block_size: int = 8
) -> dict[str, np.ndarray]:
    """Return masks consumed by :func:`stabilize_gdn_operations`."""

    validate_geometry(active_tokens, block_size)
    block_count = active_tokens // block_size
    rows, columns = np.indices((SIZE, SIZE))
    active = (rows < active_tokens) & (columns < active_tokens)
    row_blocks = rows // block_size
    column_blocks = columns // block_size
    constants = {
        "solve_block_diagonal": (
            active & (row_blocks == column_blocks)
        ).astype(np.float32),
        "solve_block_strict_lower": (
            active & (row_blocks > column_blocks)
        ).astype(np.float32),
    }
    for block in range(block_count):
        start = block * block_size
        stop = start + block_size
        mask = np.zeros((SIZE, SIZE), dtype=np.float32)
        mask[start:stop, :] = 1.0
        constants[f"solve_rows_{block}"] = mask
    return constants


def _stable_solve_operations(
    *, active_tokens: int, block_size: int
) -> list[Operation]:
    """Build the replacement for ``inverse @ right_hand_side``."""

    validate_geometry(active_tokens, block_size)
    block_count = active_tokens // block_size
    stages = int(math.log2(block_size))
    operations: list[Operation] = [
        (
            "solve_local_recurrence",
            "mul",
            "recurrence",
            "solve_block_diagonal",
        ),
        (
            "solve_cross_recurrence",
            "mul",
            "recurrence",
            "solve_block_strict_lower",
        ),
        (
            "solve_local_inverse_0",
            "add",
            "identity",
            "solve_local_recurrence",
        ),
    ]

    power = "solve_local_recurrence"
    inverse = "solve_local_inverse_0"
    for stage in range(1, stages):
        power_copy = f"solve_local_power_{stage - 1}_copy"
        next_power = f"solve_local_power_{stage}"
        factor = f"solve_local_factor_{stage}"
        next_inverse = f"solve_local_inverse_{stage}"
        operations.extend(
            [
                (power_copy, "mul", power, "ones"),
                (next_power, "matmul", power, power_copy),
                (factor, "add", "identity", next_power),
                (next_inverse, "matmul", inverse, factor),
            ]
        )
        power = next_power
        inverse = next_inverse

    first_solution = (
        "update_values" if block_count == 1 else "solve_solution_0"
    )
    operations.extend(
        [
            ("solve_local_rhs", "matmul", inverse, "right_hand_side"),
            (
                "solve_block_transfer",
                "matmul",
                inverse,
                "solve_cross_recurrence",
            ),
            (
                first_solution,
                "mul",
                "solve_local_rhs",
                "solve_rows_0",
            ),
        ]
    )
    solution = first_solution
    for block in range(1, block_count):
        propagated = f"solve_propagated_{block}"
        candidate = f"solve_candidate_{block}"
        selected = f"solve_selected_{block}"
        next_solution = (
            "update_values"
            if block == block_count - 1
            else f"solve_solution_{block}"
        )
        operations.extend(
            [
                (
                    propagated,
                    "matmul",
                    "solve_block_transfer",
                    solution,
                ),
                (candidate, "add", "solve_local_rhs", propagated),
                (selected, "mul", candidate, f"solve_rows_{block}"),
                (next_solution, "add", solution, selected),
            ]
        )
        solution = next_solution
    return operations


def stabilize_gdn_operations(
    operations: list[Operation], *, active_tokens: int = 64, block_size: int = 8
) -> list[Operation]:
    """Replace the unstable full Neumann inverse with a block forward solve."""

    validate_geometry(active_tokens, block_size)
    update_indexes = [
        index
        for index, (output, operation, _left, right) in enumerate(operations)
        if output == "update_values"
        and operation == "matmul"
        and right == "right_hand_side"
    ]
    if len(update_indexes) != 1:
        raise RuntimeError("expected one inverse/right_hand_side update operation")
    update_index = update_indexes[0]
    inverse_source = operations[update_index][2]
    if not inverse_source.startswith("inverse"):
        raise RuntimeError(f"unexpected inverse source: {inverse_source}")

    rewritten: list[Operation] = []
    removed = 0
    inserted = False
    for index, operation in enumerate(operations):
        output = operation[0]
        if _UNSTABLE_INVERSE_VALUE.fullmatch(output):
            removed += 1
            continue
        if index == update_index:
            rewritten.extend(
                _stable_solve_operations(
                    active_tokens=active_tokens, block_size=block_size
                )
            )
            inserted = True
            continue
        rewritten.append(operation)
    if not inserted or removed == 0:
        raise RuntimeError("GDN inverse rewrite did not replace the expected graph")
    produced = {output for output, _op, _left, _right in rewritten}
    if "update_values" not in produced:
        raise RuntimeError("stable solve did not produce update_values")
    return rewritten
