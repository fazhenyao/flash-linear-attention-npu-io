#!/usr/bin/env python3
"""Summarize KDA H96 msopprof BasicInfo CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


CORE_STAGES = (
    "KdaGateCumsum",
    "ChunkKdaFwdPrepare",
    "ChunkKdaFwdPostWu",
    "ChunkGatedDeltaRuleFwdH",
    "ChunkKdaFwdFinalize",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument("--platform", required=True, choices=("A2", "A5"))
    parser.add_argument("--input-cast-count", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def classify(name: str, instance: int, input_cast_count: int) -> str | None:
    for stage in CORE_STAGES:
        if name.startswith(stage):
            return f"core/{stage}"
    if name.startswith("Transpose_"):
        return "layout_conversion"
    if name.startswith("Cast_b271"):
        return "beta_sigmoid" if instance == 2 else "qk_l2norm"
    if name.startswith("Sigmoid_"):
        return "beta_sigmoid"
    if name.startswith(("Mul_", "ReduceSum_", "Add_", "Rsqrt_")):
        return "qk_l2norm"
    if name.startswith("Cast_9f288") and instance >= input_cast_count:
        return "qk_l2norm"
    return None


def main() -> None:
    args = parse_args()
    tasks = []
    for path in sorted(args.profile_dir.rglob("OpBasicInfo_*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            row = next(csv.DictReader(handle))
        name = row["Op Name"]
        instance = int(path.parent.name)
        category = classify(name, instance, args.input_cast_count)
        if category is None:
            continue
        tasks.append(
            {
                "category": category,
                "op_name": name,
                "instance": instance,
                "duration_us": float(row["Task Duration(us)"]),
                "current_freq_mhz": int(row["Current Freq"]),
                "rated_freq_mhz": int(row["Rated Freq"]),
            }
        )

    sums = defaultdict(float)
    for task in tasks:
        sums[task["category"]] += task["duration_us"]

    core_us = sum(sums[f"core/{stage}"] for stage in CORE_STAGES)
    preprocessing_us = sums["qk_l2norm"] + sums["beta_sigmoid"]
    public_total_us = core_us + preprocessing_us + sums["layout_conversion"]
    result = {
        "platform": args.platform,
        "unit": "us",
        "summary": {
            "qk_l2norm": sums["qk_l2norm"],
            "beta_sigmoid": sums["beta_sigmoid"],
            "layout_conversion": sums["layout_conversion"],
            **{stage: sums[f"core/{stage}"] for stage in CORE_STAGES},
            "kda_core_total": core_us,
            "preprocessing_total": preprocessing_us,
            "public_semantics_total": public_total_us,
        },
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
