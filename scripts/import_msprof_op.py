#!/usr/bin/env python3
"""Import msopprof (OPPROF_*) output into performance-data.json."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PERF_PATHS = [ROOT / "data" / "performance-data.json", ROOT / "docs" / "performance-data.json"]
BJ_TZ = timezone(timedelta(hours=8))

PIPE_NAME_MAP = {
    "mte1": ("mte1", "mte1_ratio", "aic_mte1"),
    "mte2": ("mte2", "mte2_ratio", "aic_mte2", "aiv_mte2"),
    "mte3": ("mte3", "mte3_ratio", "aiv_mte3"),
    "cube": ("cube", "mac", "aic_mac", "cube_util"),
    "vector": ("vector", "vec", "aiv_vec", "vector_util"),
}


def load_prof_gdr_module():
    module_path = ROOT / "scripts" / "import_prof_gdr.py"
    spec = importlib.util.spec_from_file_location("import_prof_gdr", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def now_iso() -> str:
    return datetime.now(BJ_TZ).replace(microsecond=0).isoformat()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    return list(csv.DictReader(text.splitlines()))


def to_float(value: str | None) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(str(value).strip().replace("%", ""))
    except ValueError:
        return None


def parse_opprof_timestamp(prof_dir: Path) -> tuple[str, str]:
    match = re.search(r"(\d{8})(\d{6})", prof_dir.name, re.IGNORECASE)
    if match:
        dt = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S").replace(tzinfo=BJ_TZ)
        return dt.date().isoformat(), dt.isoformat()
    return datetime.now(BJ_TZ).date().isoformat(), now_iso()


RATIO_PRECISION = 4


def round_ratio(value: float | None) -> float:
    if value is None:
        return 0.0
    return round(value, RATIO_PRECISION)


def normalize_ratio(value: float | None) -> float:
    if value is None:
        return 0.0
    if value > 1:
        return round_ratio(value / 100)
    return round_ratio(value)


def match_pipe_key(name: str) -> str | None:
    lowered = name.lower().replace(" ", "").replace("_", "")
    for key, aliases in PIPE_NAME_MAP.items():
        for alias in aliases:
            if alias.replace("_", "") in lowered or lowered in alias.replace("_", ""):
                return key
    return None


def pipe_time_stats_from_rows(pipe_rows: list[dict[str, Any]], pipe_key: str) -> dict[str, float]:
    times = [
        float(row["time_us"])
        for row in pipe_rows
        if row.get("time_us") is not None and match_pipe_key(str(row.get("pipe", ""))) == pipe_key
    ]
    if not times:
        return {}
    return {
        f"{pipe_key}_time_avg_us": round(sum(times) / len(times), 2),
        f"{pipe_key}_time_min_us": round(min(times), 2),
        f"{pipe_key}_time_max_us": round(max(times), 2),
    }


def is_wide_pipe_format(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    keys = {str(key).lower() for key in rows[0]}
    return "aiv_vec_ratio" in keys and "block_id" in keys


def weighted_avg(items: list[tuple[float, float | None]]) -> float:
    picked = [(weight, value) for weight, value in items if value is not None]
    if not picked:
        return 0.0
    if all(weight > 0 for weight, _ in picked):
        denom = sum(weight for weight, _ in picked)
        return round_ratio(sum(weight * value for weight, value in picked) / denom)
    return round_ratio(sum(value for _, value in picked) / len(picked))


def pipe_time_stats_wide(rows: list[dict[str, str]], pipe: str) -> dict[str, float]:
    if pipe == "mte1":
        col = "aic_mte1_time(us)"
        picked = [row for row in rows if "cube" in str(row.get("sub_block_id", "")).lower()]
    elif pipe == "mte2":
        col = "aic_mte2_time(us)"
        picked = [row for row in rows if "cube" in str(row.get("sub_block_id", "")).lower()]
        aiv_col = "aiv_mte2_time(us)"
        aiv_rows = [row for row in rows if "vector" in str(row.get("sub_block_id", "")).lower()]
        aiv_times = [to_float(row.get(aiv_col)) for row in aiv_rows if to_float(row.get(aiv_col)) is not None]
        times = [to_float(row.get(col)) for row in picked if to_float(row.get(col)) is not None]
        if aiv_times:
            times.extend(aiv_times)
        if not times:
            return {}
        return {
            "mte2_time_avg_us": round(sum(times) / len(times), 2),
            "mte2_time_min_us": round(min(times), 2),
            "mte2_time_max_us": round(max(times), 2),
        }
    elif pipe == "mte3":
        col = "aiv_mte3_time(us)"
        picked = [row for row in rows if "vector" in str(row.get("sub_block_id", "")).lower()]
    elif pipe == "cube":
        col = "aic_cube_time(us)"
        picked = [row for row in rows if "cube" in str(row.get("sub_block_id", "")).lower()]
    elif pipe == "vector":
        col = "aiv_vec_time(us)"
        picked = [row for row in rows if "vector" in str(row.get("sub_block_id", "")).lower()]
    else:
        return {}
    times = [to_float(row.get(col)) for row in picked if to_float(row.get(col)) is not None]
    if not times:
        return {}
    return {
        f"{pipe}_time_avg_us": round(sum(times) / len(times), 2),
        f"{pipe}_time_min_us": round(min(times), 2),
        f"{pipe}_time_max_us": round(max(times), 2),
    }


PIPE_UTIL_FROM_TASK_DURATION = (
    ("mte1_ratio", "mte1_time_max_us"),
    ("mte2_ratio", "mte2_time_max_us"),
    ("mte3_ratio", "mte3_time_max_us"),
    ("cube_util", "cube_time_max_us"),
    ("vector_util", "vector_time_max_us"),
)


def apply_pipeline_util_from_task_duration(
    pipe_metrics: dict[str, Any],
    task_duration_us: float | None,
) -> None:
    """流水线时间占比 = max(pipe_time) / OpBasicInfo Task Duration。"""
    if task_duration_us is None or task_duration_us <= 0:
        return
    for ratio_key, time_key in PIPE_UTIL_FROM_TASK_DURATION:
        time_max = pipe_metrics.get(time_key)
        if time_max is None:
            continue
        pipe_metrics[ratio_key] = round_ratio(float(time_max) / task_duration_us)
    mte2 = float(pipe_metrics.get("mte2_ratio") or 0)
    mte3 = float(pipe_metrics.get("mte3_ratio") or 0)
    bottleneck_candidates = {
        "Cube": float(pipe_metrics.get("cube_util") or 0),
        "HBM": mte2,
        "Vector": float(pipe_metrics.get("vector_util") or 0),
        "MTE3": mte3,
        "MTE1": float(pipe_metrics.get("mte1_ratio") or 0),
    }
    pipe_metrics["bottleneck"] = max(bottleneck_candidates, key=bottleneck_candidates.get)


def parse_wide_pipe_utilization(rows: list[dict[str, str]]) -> dict[str, Any]:
    cube_items: list[dict[str, float | None]] = []
    vector_items: list[dict[str, float | None]] = []
    for row in rows:
        sub_block = str(row.get("sub_block_id", "")).lower()
        if "cube" in sub_block:
            cube_items.append(
                {
                    "weight": to_float(row.get("aic_time(us)")) or 0.0,
                    "cube": to_float(row.get("aic_cube_ratio")) or to_float(row.get("aic_mac_ratio")),
                    "mte1": to_float(row.get("aic_mte1_ratio")),
                    "mte2": to_float(row.get("aic_mte2_ratio")),
                    "mte3": to_float(row.get("aic_mte3_ratio")),
                }
            )
        elif "vector" in sub_block:
            vector_items.append(
                {
                    "weight": to_float(row.get("aiv_time(us)")) or 0.0,
                    "vector": to_float(row.get("aiv_vec_ratio")),
                    "mte2": to_float(row.get("aiv_mte2_ratio")),
                    "mte3": to_float(row.get("aiv_mte3_ratio")),
                }
            )

    def wavg(items: list[dict[str, float | None]], key: str) -> float:
        return weighted_avg([(float(item["weight"]), item.get(key)) for item in items])

    cube = wavg(cube_items, "cube")
    vector = wavg(vector_items, "vector")
    mte1 = wavg(cube_items, "mte1")
    mte2 = round_ratio(max(wavg(cube_items, "mte2"), wavg(vector_items, "mte2")))
    mte3 = wavg(vector_items, "mte3")
    bottleneck_candidates = {
        "Cube": cube,
        "HBM": mte2,
        "Vector": vector,
        "MTE3": mte3,
        "MTE1": mte1,
    }
    payload = {
        "pipe_rows": [],
        "mte1_ratio": mte1,
        "mte2_ratio": mte2,
        "mte3_ratio": mte3,
        "cube_util": cube,
        "vector_util": vector,
        "bottleneck": max(bottleneck_candidates, key=bottleneck_candidates.get),
    }
    for pipe_key in ("mte1", "mte2", "mte3", "cube", "vector"):
        payload.update(pipe_time_stats_wide(rows, pipe_key))
    return payload


def parse_pipe_utilization(path: Path) -> dict[str, Any]:
    rows = read_csv_rows(path)
    if is_wide_pipe_format(rows):
        return parse_wide_pipe_utilization(rows)
    ratios: dict[str, list[float]] = {key: [] for key in PIPE_NAME_MAP}
    pipe_rows: list[dict[str, Any]] = []

    for row in rows:
        label = (
            row.get("Pipe")
            or row.get("Pipe Type")
            or row.get("Name")
            or row.get("Metric")
            or row.get("Instruction Type")
            or ""
        )
        ratio_raw = (
            row.get("Ratio")
            or row.get("Ratio(%)")
            or row.get("Time Ratio")
            or row.get("time ratio")
            or row.get("aicore_time_ratio")
        )
        time_raw = row.get("Time(us)") or row.get("Time (us)") or row.get("aicore_time(us)")
        ratio = normalize_ratio(to_float(ratio_raw))
        if not label and not ratio_raw:
            for col, val in row.items():
                pipe_key = match_pipe_key(str(col))
                if pipe_key:
                    ratios[pipe_key].append(normalize_ratio(to_float(val)))
            continue
        pipe_key = match_pipe_key(str(label))
        if pipe_key:
            ratios[pipe_key].append(ratio)
        pipe_rows.append(
            {
                "pipe": str(label),
                "time_us": to_float(time_raw),
                "ratio": ratio,
            }
        )

    def avg(values: list[float]) -> float:
        return round_ratio(sum(values) / len(values)) if values else 0.0

    merged = {key: avg(values) for key, values in ratios.items()}
    mte2 = max(merged.get("mte2", 0), 0)
    cube = merged.get("cube", 0)
    vector = merged.get("vector", 0)
    bottleneck_candidates = {
        "Cube": cube,
        "HBM": mte2,
        "Vector": vector,
        "MTE3": merged.get("mte3", 0),
        "MTE1": merged.get("mte1", 0),
    }
    bottleneck = max(bottleneck_candidates, key=bottleneck_candidates.get)
    payload = {
        "pipe_rows": pipe_rows,
        "mte1_ratio": merged.get("mte1", 0),
        "mte2_ratio": mte2,
        "mte3_ratio": merged.get("mte3", 0),
        "cube_util": cube,
        "vector_util": vector,
        "bottleneck": bottleneck,
    }
    for pipe_key in ("mte1", "mte2", "mte3", "cube", "vector"):
        payload.update(pipe_time_stats_from_rows(pipe_rows, pipe_key))
    return payload


def find_csv_by_stem(directory: Path, stem: str) -> Path | None:
    exact = directory / f"{stem}.csv"
    if exact.exists():
        return exact
    matches = sorted(directory.glob(f"{stem}_*.csv"))
    return matches[-1] if matches else None


def list_kernel_bundle_dirs(prof_dir: Path) -> list[Path]:
    bundles: list[Path] = []
    for child in sorted(prof_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if child.name == "dump":
            continue
        if find_csv_by_stem(child, "PipeUtilization"):
            bundles.append(child)
            continue
        if any(sub.is_dir() and sub.name.isdigit() for sub in child.iterdir()):
            bundles.append(child)
    return bundles


def list_invocation_run_dirs(kernel_dir: Path) -> list[Path]:
    """Numeric subdirs (0, 1, ...) are separate invocations of the same kernel in one script run."""
    numeric_dirs = sorted(
        (sub for sub in kernel_dir.iterdir() if sub.is_dir() and sub.name.isdigit()),
        key=lambda item: int(item.name),
    )
    if numeric_dirs:
        return [item for item in numeric_dirs if find_csv_by_stem(item, "PipeUtilization")]
    if find_csv_by_stem(kernel_dir, "PipeUtilization"):
        return [kernel_dir]
    return []


WEIGHTED_OPERATOR_METRICS = (
    "mbu",
    "mfu",
    "mte1_ratio",
    "mte2_ratio",
    "mte3_ratio",
    "cube_util",
    "vector_util",
    "mem_util",
)

PIPE_TIME_STAT_FIELDS = tuple(
    f"{pipe}_time_{stat}_us"
    for pipe in ("mte1", "mte2", "mte3", "cube", "vector")
    for stat in ("avg", "min", "max")
)


def merge_operator_invocations(invocations: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not invocations:
        return None
    if len(invocations) == 1:
        merged = dict(invocations[0])
        merged["call_count"] = 1
        return merged

    total_time = sum(float(item.get("time_ms") or 0) for item in invocations)
    total_duration_us = sum(float(item.get("duration_us") or 0) for item in invocations if item.get("duration_us") is not None)
    heaviest = max(invocations, key=lambda item: float(item.get("time_ms") or 0))
    merged = dict(heaviest)
    merged["time_ms"] = total_time
    if total_duration_us > 0:
        merged["duration_us"] = total_duration_us
    merged["call_count"] = len(invocations)
    for key in WEIGHTED_OPERATOR_METRICS:
        weighted: list[tuple[float, float]] = []
        for item in invocations:
            value = item.get(key)
            weight = float(item.get("time_ms") or 0)
            if value is None:
                continue
            weighted.append((weight, float(value)))
        if not weighted:
            continue
        if total_time > 0 and all(weight > 0 for weight, _ in weighted):
            merged[key] = round(sum(weight * value for weight, value in weighted) / total_time, 4)
        else:
            merged[key] = round(sum(value for _, value in weighted) / len(weighted), 4)
    for pipe in ("mte1", "mte2", "mte3", "cube", "vector"):
        avg_key = f"{pipe}_time_avg_us"
        min_key = f"{pipe}_time_min_us"
        max_key = f"{pipe}_time_max_us"
        avg_weighted: list[tuple[float, float]] = []
        mins: list[float] = []
        maxs: list[float] = []
        for item in invocations:
            weight = float(item.get("time_ms") or 0)
            avg = item.get(avg_key)
            min_val = item.get(min_key)
            max_val = item.get(max_key)
            if avg is not None:
                avg_weighted.append((weight, float(avg)))
            if min_val is not None:
                mins.append(float(min_val))
            if max_val is not None:
                maxs.append(float(max_val))
        if avg_weighted:
            if total_time > 0 and all(weight > 0 for weight, _ in avg_weighted):
                merged[avg_key] = round(sum(weight * value for weight, value in avg_weighted) / total_time, 2)
            else:
                merged[avg_key] = round(sum(value for _, value in avg_weighted) / len(avg_weighted), 2)
        if mins:
            merged[min_key] = round(min(mins), 2)
        if maxs:
            merged[max_key] = round(max(maxs), 2)
    merged["bottleneck"] = heaviest.get("bottleneck")
    merged["rated_freq_mhz"] = heaviest.get("rated_freq_mhz")
    return merged


def is_bundled_multi_kernel_opprof(prof_dir: Path) -> bool:
    if find_csv_by_stem(prof_dir, "PipeUtilization"):
        return False
    return bool(list_kernel_bundle_dirs(prof_dir))


def load_cube_theoretical_flops_module():
    import importlib.util

    path = ROOT / "scripts" / "cube_theoretical_flops.py"
    spec = importlib.util.spec_from_file_location("cube_theoretical_flops", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_mfu(
    operator_id: str,
    attributes: dict[str, Any] | None,
    *,
    task_duration_us: float | None,
    block_dim: int | None,
    freq_mhz: float | None,
    chip: str | None = None,
) -> float | None:
    return load_cube_theoretical_flops_module().compute_mfu(
        operator_id,
        attributes,
        task_duration_us=task_duration_us,
        block_dim=block_dim,
        freq_mhz=freq_mhz,
        chip=chip,
    )


def merge_case_attributes(
    data: dict[str, Any],
    case_id: str,
    incoming: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(incoming or {})
    for case in data.get("cases", []):
        if case.get("id") != case_id:
            continue
        base = dict(case.get("attributes") or {})
        for key, value in (incoming or {}).items():
            if value not in (None, ""):
                base[key] = value
        return base
    return merged


def load_hbm_theoretical_bytes_module():
    import importlib.util

    path = ROOT / "scripts" / "hbm_theoretical_bytes.py"
    spec = importlib.util.spec_from_file_location("hbm_theoretical_bytes", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_mbu(
    operator_id: str,
    attributes: dict[str, Any] | None,
    *,
    task_duration_us: float | None,
    chip: str | None = None,
) -> float | None:
    return load_hbm_theoretical_bytes_module().compute_mbu(
        operator_id,
        attributes,
        task_duration_us=task_duration_us,
        chip=chip,
    )


def operator_task_duration_us(operator: dict[str, Any]) -> float | None:
    return load_cube_theoretical_flops_module().resolve_task_duration_us(operator)


def apply_operators_mfu(
    operators: list[dict[str, Any]],
    attributes: dict[str, Any] | None,
    *,
    chip: str | None = None,
) -> None:
    for operator in operators:
        operator_id = str(operator.get("operator_id") or "")
        task_duration_us = operator_task_duration_us(operator)
        operator["mfu"] = compute_mfu(
            operator_id,
            attributes,
            task_duration_us=task_duration_us,
            block_dim=operator.get("block_dim"),
            freq_mhz=operator.get("rated_freq_mhz"),
            chip=chip,
        )
        mbu = compute_mbu(
            operator_id,
            attributes,
            task_duration_us=task_duration_us,
            chip=chip,
        )
        operator["mbu"] = mbu
        operator["mem_util"] = mbu


def _parse_op_basic_row(row: dict[str, str]) -> dict[str, Any]:
    kernel = (
        row.get("Kernel Name")
        or row.get("Op Name")
        or row.get("Operator Name")
        or row.get("kernel_name")
        or ""
    )
    duration_us = to_float(
        row.get("Task Duration(us)")
        or row.get("aicore_time(us)")
        or row.get("Duration(us)")
        or row.get("Time(us)")
    )
    block_dim = to_float(row.get("Block Dim") or row.get("block_dim") or row.get("BlockDim"))
    rated_freq_mhz = to_float(row.get("Current Freq") or row.get("Rated Freq"))
    time_ms = round((duration_us or 0) / 1000, 3)
    if not str(kernel).strip() and time_ms <= 0:
        return {}
    return {
        "kernel_name": str(kernel).strip(),
        "block_dim": int(block_dim) if block_dim is not None else None,
        "time_ms": time_ms,
        "duration_us": duration_us,
        "rated_freq_mhz": rated_freq_mhz,
    }


def parse_op_basic_info_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    parsed = [_parse_op_basic_row(row) for row in rows]
    return [item for item in parsed if item]


def parse_op_basic_info(path: Path) -> dict[str, Any]:
    rows = parse_op_basic_info_rows(path)
    return rows[0] if rows else {}


def resolve_operator_id(kernel_name: str, fallback: str | None = None) -> str:
    if fallback:
        return fallback
    lowered = kernel_name.lower()
    gdr = load_prof_gdr_module()
    for op_type, operator_id in gdr.OP_TYPE_MAP.items():
        if op_type.lower() in lowered or operator_id in lowered:
            return operator_id
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return slug or "unknown_kernel"


def format_kernel_suffix(basic_rows: list[dict[str, Any]], kernel_name: str | None = None) -> str:
    if kernel_name:
        names = [part.strip() for part in re.split(r"[|,;\n]+", kernel_name) if part.strip()]
        if len(names) > 1:
            return f" · {'|'.join(names)}"
        return f" · {kernel_name}"
    if len(basic_rows) == 1 and basic_rows[0].get("kernel_name"):
        return f" · {basic_rows[0]['kernel_name']}"
    if len(basic_rows) > 1:
        return f" · {len(basic_rows)} kernels"
    return ""


def build_operator_record(
    operator_id: str,
    *,
    kernel_name: str,
    time_ms: float,
    block_dim: int | None,
    pipe_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "operator_id": operator_id,
        "time_ms": time_ms,
        "duration_us": None,
        "share_pct": 0.0,
        "mbu": pipe_metrics.get("mbu"),
        "mfu": pipe_metrics.get("mfu"),
        "bottleneck": pipe_metrics["bottleneck"],
        "mte1_ratio": pipe_metrics["mte1_ratio"],
        "mte2_ratio": pipe_metrics["mte2_ratio"],
        "mte3_ratio": pipe_metrics["mte3_ratio"],
        "mte1_time_avg_us": pipe_metrics.get("mte1_time_avg_us"),
        "mte1_time_min_us": pipe_metrics.get("mte1_time_min_us"),
        "mte1_time_max_us": pipe_metrics.get("mte1_time_max_us"),
        "mte2_time_avg_us": pipe_metrics.get("mte2_time_avg_us"),
        "mte2_time_min_us": pipe_metrics.get("mte2_time_min_us"),
        "mte2_time_max_us": pipe_metrics.get("mte2_time_max_us"),
        "mte3_time_avg_us": pipe_metrics.get("mte3_time_avg_us"),
        "mte3_time_min_us": pipe_metrics.get("mte3_time_min_us"),
        "mte3_time_max_us": pipe_metrics.get("mte3_time_max_us"),
        "cube_util": pipe_metrics["cube_util"],
        "vector_util": pipe_metrics["vector_util"],
        "cube_time_avg_us": pipe_metrics.get("cube_time_avg_us"),
        "cube_time_min_us": pipe_metrics.get("cube_time_min_us"),
        "cube_time_max_us": pipe_metrics.get("cube_time_max_us"),
        "vector_time_avg_us": pipe_metrics.get("vector_time_avg_us"),
        "vector_time_min_us": pipe_metrics.get("vector_time_min_us"),
        "vector_time_max_us": pipe_metrics.get("vector_time_max_us"),
        "mem_util": pipe_metrics.get("mbu"),
        "kernel_name": kernel_name,
        "block_dim": block_dim,
        "rated_freq_mhz": pipe_metrics.get("rated_freq_mhz"),
        "pipe_utilization": pipe_metrics["pipe_rows"],
    }


def split_kernel_name_filter(kernel_name: str | None) -> list[str]:
    if not kernel_name:
        return []
    return [part.strip() for part in re.split(r"[|,;\n]+", kernel_name) if part.strip()]


def kernel_name_override(kernel_name: str | None) -> str | None:
    names = split_kernel_name_filter(kernel_name)
    return names[0] if len(names) == 1 else None


def build_operators_from_basic_rows(
    basic_rows: list[dict[str, Any]],
    pipe_metrics: dict[str, Any],
    *,
    kernel_name: str | None = None,
    operator_id: str | None = None,
) -> tuple[list[dict[str, Any]], float]:
    single_kernel = kernel_name_override(kernel_name)
    if not basic_rows:
        fallback_kernel = single_kernel or kernel_name or ""
        resolved_operator = resolve_operator_id(fallback_kernel, operator_id)
        return (
            [
                build_operator_record(
                    resolved_operator,
                    kernel_name=fallback_kernel,
                    time_ms=0.0,
                    block_dim=None,
                    pipe_metrics=pipe_metrics,
                )
            ],
            0.0,
        )

    single_row = len(basic_rows) == 1
    buckets: dict[str, dict[str, Any]] = {}
    for row in basic_rows:
        resolved_kernel = (single_kernel or row.get("kernel_name") or "").strip()
        resolved_operator = resolve_operator_id(
            resolved_kernel,
            operator_id if single_row else None,
        )
        bucket = buckets.setdefault(
            resolved_operator,
            build_operator_record(
                resolved_operator,
                kernel_name=resolved_kernel,
                time_ms=0.0,
                block_dim=row.get("block_dim"),
                pipe_metrics=pipe_metrics,
            ),
        )
        bucket["time_ms"] += float(row.get("time_ms") or 0)
        if row.get("duration_us") is not None:
            bucket["duration_us"] = float(bucket.get("duration_us") or 0) + float(row["duration_us"])
        if row.get("block_dim") is not None:
            bucket["block_dim"] = row["block_dim"]
        if resolved_kernel:
            bucket["kernel_name"] = resolved_kernel

    operators = sorted(buckets.values(), key=lambda item: item["time_ms"], reverse=True)
    total_ms = sum(float(item["time_ms"] or 0) for item in operators)
    for item in operators:
        item["time_ms"] = round(float(item["time_ms"] or 0), 3)
        item["share_pct"] = round((item["time_ms"] / total_ms * 100) if total_ms else 0, 2)
    return operators, round(total_ms, 3)


def load_operator_from_run_dir(run_dir: Path, kernel_name: str) -> dict[str, Any] | None:
    pipe_path = find_csv_by_stem(run_dir, "PipeUtilization")
    if pipe_path is None:
        return None
    pipe_metrics = parse_pipe_utilization(pipe_path)
    basic_path = find_csv_by_stem(run_dir, "OpBasicInfo")
    basic_rows = parse_op_basic_info_rows(basic_path) if basic_path else []
    if basic_rows:
        basic = basic_rows[0]
        apply_pipeline_util_from_task_duration(pipe_metrics, basic.get("duration_us"))
        pipe_metrics["rated_freq_mhz"] = basic.get("rated_freq_mhz")
    else:
        pipe_metrics["rated_freq_mhz"] = None
    if not basic_rows:
        basic_rows = [{"kernel_name": kernel_name, "block_dim": None, "time_ms": 0.0}]
    operators, _ = build_operators_from_basic_rows(
        basic_rows,
        pipe_metrics,
        kernel_name=kernel_name,
    )
    if not operators:
        return None
    operator = operators[0]
    if not operator.get("kernel_name"):
        operator["kernel_name"] = kernel_name
    operator["call_count"] = 1
    return operator


def import_bundled_msprof_op(
    prof_dir: Path,
    *,
    kernel_name: str | None = None,
    operator_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    requested = split_kernel_name_filter(kernel_name)
    operators: list[dict[str, Any]] = []
    basic_rows: list[dict[str, Any]] = []
    kernel_dirs = list_kernel_bundle_dirs(prof_dir)
    for kernel_dir in kernel_dirs:
        bundle_kernel = kernel_dir.name
        if requested and not any(
            token.lower() in bundle_kernel.lower() for token in requested
        ):
            continue
        invocation_ops = [
            op
            for run_dir in list_invocation_run_dirs(kernel_dir)
            if (op := load_operator_from_run_dir(run_dir, bundle_kernel)) is not None
        ]
        operator = merge_operator_invocations(invocation_ops)
        if operator is None:
            continue
        if operator_id and len(kernel_dirs) == 1:
            operator["operator_id"] = resolve_operator_id(bundle_kernel, operator_id)
        operators.append(operator)
        basic_rows.append(
            {
                "kernel_name": bundle_kernel,
                "block_dim": operator.get("block_dim"),
                "time_ms": operator.get("time_ms") or 0.0,
            }
        )

    total_ms = sum(float(item.get("time_ms") or 0) for item in operators)
    for item in operators:
        item["time_ms"] = round(float(item.get("time_ms") or 0), 3)
        item["share_pct"] = round((item["time_ms"] / total_ms * 100) if total_ms else 0, 2)
    return operators, basic_rows, round(total_ms, 3)


def build_case_from_attributes(
    attributes: dict[str, Any],
    prof_dir: Path,
    *,
    kernel_name: str | None = None,
    basic_rows: list[dict[str, Any]] | None = None,
    prof_tool: str = "msprof_op",
) -> dict[str, Any]:
    gdr = load_prof_gdr_module()
    attrs = attributes or {}
    snapshot_date, created_at = parse_opprof_timestamp(prof_dir)
    time_slug_match = re.search(r"(\d{8})(\d{6})", prof_dir.name, re.IGNORECASE)
    time_slug = (
        f"{time_slug_match.group(1)}t{time_slug_match.group(2)}"
        if time_slug_match
        else re.sub(r"[^a-z0-9]+", "", prof_dir.name.lower())[-16:]
    )
    time_label = str(created_at).replace("T", " ").replace("+08:00", "")[:16]
    batch = int(attrs.get("batch") or 1)
    query_heads = int(attrs.get("query_heads") or 32)
    value_heads = int(attrs.get("value_heads") or query_heads)
    tokens = int(attrs.get("tokens") or 4087)
    key_dim = int(attrs.get("key_dim") or 128)
    value_dim = int(attrs.get("value_dim") or key_dim)
    chunk_size = int(attrs.get("chunk_size") or 64)
    case_id = f"case-gdn-t{tokens}-k{key_dim}-v{value_dim}-c{chunk_size}-{time_slug}"
    kernel_suffix = format_kernel_suffix(basic_rows or [], kernel_name)
    tool_label = "msopprof" if prof_tool == "msprof_op" else "msopprof sim"
    label = (
        f"B={batch} QH={query_heads} VH={value_heads} T={tokens} "
        f"K={key_dim} V={value_dim} chunk={chunk_size} {gdr.normalize_layout(attrs.get('layout'), varlen=attrs.get('varlen'))} @ {time_label}{kernel_suffix} [{tool_label}]"
    )
    return {
        "id": case_id,
        "label": label,
        "category": gdr.normalize_layout(attrs.get("layout"), varlen=attrs.get("varlen")).lower(),
        "attributes": gdr.build_case_attributes(
            batch=batch,
            query_heads=query_heads,
            value_heads=value_heads,
            tokens=tokens,
            key_dim=key_dim,
            value_dim=value_dim,
            chunk_size=chunk_size,
            dtype=str(attrs.get("dtype") or "bf16"),
            layout=attrs.get("layout"),
            varlen=attrs.get("varlen"),
            mean_len=int(attrs.get("mean_len") or 1024),
            cu_seqlens=str(attrs.get("cu_seqlens") or ""),
            scale=attrs.get("scale"),
        ),
        "position": 0,
        "active": True,
    }


def import_msprof_op(
    prof_dir: Path,
    model_id: str,
    chip: str,
    *,
    attributes: dict[str, Any] | None = None,
    kernel_name: str | None = None,
    operator_id: str | None = None,
    prof_tool: str = "msprof_op",
    device_id: int = 2,
) -> dict[str, Any]:
    prof_dir = prof_dir.resolve()
    if not prof_dir.exists():
        raise FileNotFoundError(prof_dir)

    if is_bundled_multi_kernel_opprof(prof_dir):
        operators, basic_rows, total_time_ms = import_bundled_msprof_op(
            prof_dir,
            kernel_name=kernel_name,
            operator_id=operator_id,
        )
        if not operators:
            raise FileNotFoundError(f"No kernel bundles with PipeUtilization.csv under {prof_dir}")
    else:
        pipe_path = find_csv_by_stem(prof_dir, "PipeUtilization")
        if pipe_path is None:
            raise FileNotFoundError(f"Missing PipeUtilization.csv under {prof_dir}")
        pipe_metrics = parse_pipe_utilization(pipe_path)
        basic_path = find_csv_by_stem(prof_dir, "OpBasicInfo")
        basic_rows = parse_op_basic_info_rows(basic_path) if basic_path else []
        if basic_rows:
            basic = basic_rows[0]
            apply_pipeline_util_from_task_duration(pipe_metrics, basic.get("duration_us"))
            pipe_metrics["rated_freq_mhz"] = basic.get("rated_freq_mhz")
        else:
            pipe_metrics["rated_freq_mhz"] = None
        operators, total_time_ms = build_operators_from_basic_rows(
            basic_rows,
            pipe_metrics,
            kernel_name=kernel_name,
            operator_id=operator_id,
        )
    if kernel_name and len(split_kernel_name_filter(kernel_name)) > 1:
        resolved_kernel = "|".join(split_kernel_name_filter(kernel_name))
    else:
        resolved_kernel = kernel_name_override(kernel_name) or (
            operators[0].get("kernel_name") if len(operators) == 1 else ""
        )
    case = build_case_from_attributes(
        attributes or {},
        prof_dir,
        kernel_name=resolved_kernel or None,
        basic_rows=basic_rows,
        prof_tool=prof_tool,
    )
    gdr = load_prof_gdr_module()
    data = gdr.load_perf_data()
    case_attrs = merge_case_attributes(data, case["id"], case.get("attributes"))
    case["attributes"] = case_attrs
    apply_operators_mfu(operators, case_attrs, chip=chip)
    snapshot_date, created_at = parse_opprof_timestamp(prof_dir)
    if len(operators) > 1:
        snapshot_label = f"{snapshot_date} {len(operators)} kernels"
    elif operators:
        snapshot_label = f"{snapshot_date} {operators[0]['operator_id'] or 'op'}"
    else:
        snapshot_label = f"{snapshot_date} op"

    snapshot = {
        "id": f"snap-{prof_dir.name.lower()}",
        "label": snapshot_label,
        "snapshot_date": snapshot_date,
        "branch": "main",
        "model_id": model_id,
        "case_id": case["id"],
        "chip": chip,
        "total_time_ms": total_time_ms,
        "status": "done",
        "created_at": created_at,
        "prof_source": prof_dir.name,
        "prof_tool": prof_tool,
        "kernel_name": resolved_kernel,
        "device_id": device_id,
        "operators": operators,
    }

    gdr.upsert_case(data, case)
    snapshots = [item for item in data.get("snapshots", []) if item.get("id") != snapshot["id"]]
    snapshots.insert(0, snapshot)
    data["snapshots"] = snapshots
    if not any(item.get("id") == model_id for item in data["models"]):
        data["models"].append({"id": model_id, "label": model_id.upper(), "position": 0, "active": True})
    data["version"] = now_iso()
    for path in PERF_PATHS:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import msopprof OPPROF_* output into performance-data.json")
    parser.add_argument("prof_dir", type=Path, nargs="?", help="OPPROF directory or prof_op root with --all")
    parser.add_argument("--all", action="store_true", help="Import every OPPROF_* under prof_dir (default: data/prof_op)")
    parser.add_argument("--model", default="gdn")
    parser.add_argument("--chip", default="A2", choices=["A2", "A3"])
    parser.add_argument("--kernel-name", default="")
    parser.add_argument("--operator-id", default="")
    parser.add_argument("--prof-tool", default="msprof_op", choices=["msprof_op", "msprof_op_sim"])
    parser.add_argument("--attributes-json", type=Path, help="Optional trigger attributes JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    attributes: dict[str, Any] = {}
    if args.attributes_json and args.attributes_json.exists():
        attributes = json.loads(args.attributes_json.read_text(encoding="utf-8"))

    if args.all:
        root = (args.prof_dir or ROOT / "data" / "prof_op").resolve()
        prof_dirs = sorted(p for p in root.glob("OPPROF_*") if p.is_dir())
        if not prof_dirs:
            raise SystemExit(f"No OPPROF_* directories under {root}")
    else:
        if not args.prof_dir:
            raise SystemExit("prof_dir is required unless --all is set")
        prof_dirs = [args.prof_dir]

    data = None
    for prof_dir in prof_dirs:
        data = import_msprof_op(
            prof_dir,
            args.model,
            args.chip,
            attributes=attributes,
            kernel_name=args.kernel_name or None,
            operator_id=args.operator_id or None,
            prof_tool=args.prof_tool,
        )
        snapshot = next(item for item in data["snapshots"] if item.get("prof_source") == prof_dir.name)
        print(f"Imported {prof_dir.name}")
        op_count = len(snapshot["operators"])
        if op_count == 1:
            print(f"  operator: {snapshot['operators'][0]['operator_id']}")
        else:
            print(f"  operators: {op_count}")
        print(f"  total: {snapshot['total_time_ms']} ms")
    print(f"Done: {len(data['snapshots'])} snapshots, {len(data['cases'])} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
