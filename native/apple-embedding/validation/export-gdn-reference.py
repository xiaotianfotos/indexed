#!/usr/bin/env python3
"""Optional maintainer export of the frozen D graph; never a product build step."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

kernel_root = Path(__file__).resolve().parent.parent / "private_ane"
sys.path.insert(0, str(kernel_root))
from inmemory_partition_runner import parse_graph
from inmemory_segmented_runner import add_query_scaling, make_segments
from stable_block_solve import stabilize_gdn_operations, block_solve_constants
from probe_inmemory_split_surfaces import DYNAMIC_NAMES, split_surface_mil
from inmemory_batched_graph_runner import shared_constants


def export(source: Path) -> dict:
    operations = add_query_scaling(
        stabilize_gdn_operations(parse_graph(source), active_tokens=64, block_size=8), 4096
    )
    segments = make_segments(operations, len(operations), connected_only=False)
    if len(segments) != 1:
        raise ValueError("Expected one complete recurrence graph")
    segment = segments[0]
    dynamic = [name for name in segment.inputs if name in DYNAMIC_NAMES]
    shared = [name for name in segment.inputs if name not in DYNAMIC_NAMES]
    values = {
        **shared_constants(16),
        **{name: value[None] for name, value in block_solve_constants(active_tokens=64, block_size=8).items()},
        "query_inverse_scale": np.full((16, 128, 128), 1 / 4096, dtype=np.float32),
    }
    constants = np.stack([values[name][0] for name in shared], axis=0)[:, None].astype(np.float16)
    mil = split_surface_mil(segment, 16, dynamic, shared, io_dtype="fp16")
    digest = lambda value: hashlib.sha256(value).hexdigest()
    return {
        "version": 1, "sourceSHA256": digest(source.read_bytes()),
        "dynamicNames": dynamic, "sharedNames": shared, "outputNames": segment.outputs,
        "operations": segment.operations, "milSHA256": digest(mil.encode()),
        "sharedSHA256": digest(constants.tobytes()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mil", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.mil)
    # Never overwrite a frozen reference or read a user configuration/database.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
