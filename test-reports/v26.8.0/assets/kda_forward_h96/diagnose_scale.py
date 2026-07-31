#!/usr/bin/env python3
"""Diagnose a constant scale mismatch between two sampled tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--expect", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    real = np.load(args.real).astype(np.float64)
    expect = np.load(args.expect).astype(np.float64)
    valid = np.abs(expect) > 1e-6
    ratio = real[valid] / expect[valid]
    result = {
        "sample_count": int(real.size),
        "ratio_count": int(valid.sum()),
        "actual_over_expected_quantiles": {
            "p01": float(np.quantile(ratio, 0.01)),
            "p50": float(np.quantile(ratio, 0.50)),
            "p99": float(np.quantile(ratio, 0.99)),
        },
        "least_squares_scale": float(np.dot(real, expect) / np.dot(expect, expect)),
        "relative_l2": float(
            np.linalg.norm(real - expect) / np.linalg.norm(expect)
        ),
        "relative_l2_after_dividing_actual_by_2": float(
            np.linalg.norm(real / 2 - expect) / np.linalg.norm(expect)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
