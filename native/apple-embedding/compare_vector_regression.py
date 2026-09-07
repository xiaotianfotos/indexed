#!/usr/bin/env python3
"""Compare two saved WeMM vector sets row-by-row without mixing vector spaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summary(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    if baseline.shape != candidate.shape or baseline.ndim != 2:
        raise ValueError(f"vector shape mismatch: {baseline.shape} vs {candidate.shape}")
    left = baseline / np.linalg.norm(baseline, axis=1, keepdims=True)
    right = candidate / np.linalg.norm(candidate, axis=1, keepdims=True)
    cosine = np.sum(left * right, axis=1)
    return {
        "rows": int(baseline.shape[0]),
        "dimension": int(baseline.shape[1]),
        "cosine_min": float(cosine.min()),
        "cosine_mean": float(cosine.mean()),
        "cosine_median": float(np.median(cosine)),
        "cosine_p05": float(np.quantile(cosine, 0.05)),
        "max_absolute_error": float(np.max(np.abs(baseline - candidate))),
    }


def main() -> None:
    args = parse_args()
    baseline = np.load(args.baseline.resolve())
    candidate = np.load(args.candidate.resolve())
    for key in ("image_ids", "query_ids"):
        if not np.array_equal(baseline[key], candidate[key]):
            raise ValueError(f"{key} mismatch")
    report = {
        "schema_version": 1,
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "groups": {
            key: summary(baseline[key].astype(np.float32), candidate[key].astype(np.float32))
            for key in ("corpus", "text", "native_mixed")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
