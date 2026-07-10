#!/usr/bin/env python3
"""Import msprof op (OPPROF_*) output into performance-data.json."""

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
    mbu = round_ratio(max(mte2, mte3))
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
        "mbu": mbu,
        "bottleneck": max(bottleneck_candidates, key=bottleneck_candidates.get),
    }
    for pipe_key in ("mte1", "mte2", "mte3"):
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
    mbu = max(mte2, merged.get("mte3", 0))
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
        "mbu": mbu,
        "bottleneck": bottleneck,
    }
    for pipe_key in ("mte1", "mte2", "mte3"):
        payload.update(pipe_time_stats_from_rows(pipe_rows, pipe_key))
    return payload


def parse_mfu_from_arithmetic(prof_dir: Path) -> float | None:
    """Estimate MFU from ArithmeticUtilization.csv when cube_utilization(%) is unavailable."""
    path = prof_dir / "ArithmeticUtilization.csv"
    if not path.exists():
        return None
    rows = read_csv_rows(path)
    weighted: list[tuple[float, float]] = []
    throughputs: list[float] = []
    for row in rows:
        if "cube" not in str(row.get("sub_block_id", "")).lower():
            continue
        flops = to_float(row.get("aic_cube_fops"))
        ratio = to_float(row.get("aic_cube_ratio"))
        time_us = to_float(row.get("aic_time(us)"))
        if flops is None or ratio is None or time_us is None or ratio <= 0 or time_us <= 0:
            continue
        active_us = ratio * time_us
        if active_us <= 0:
            continue
        throughput = flops / active_us
        throughputs.append(throughput)
        weighted.append((time_us, throughput))
    if not throughputs:
        return None
    peak = max(throughputs)
    denom = sum(weight for weight, _ in weighted) or 1
    return round_ratio(sum(weight * (throughput / peak) for weight, throughput in weighted) / denom)


def parse_op_basic_info(path: Path) -> dict[str, Any]:
    rows = read_csv_rows(path)
    if not rows:
        return {}
    row = rows[0]
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
    return {
        "kernel_name": str(kernel).strip(),
        "block_dim": int(block_dim) if block_dim is not None else None,
        "time_ms": round((duration_us or 0) / 1000, 3),
    }


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


def build_case_from_attributes(
    attributes: dict[str, Any],
    prof_dir: Path,
    *,
    kernel_name: str | None = None,
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
    kernel_suffix = f" · {kernel_name}" if kernel_name else ""
    tool_label = "msprof op" if prof_tool == "msprof_op" else "msprof op sim"
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

    pipe_path = prof_dir / "PipeUtilization.csv"
    basic_path = prof_dir / "OpBasicInfo.csv"
    if not pipe_path.exists():
        raise FileNotFoundError(f"Missing PipeUtilization.csv under {prof_dir}")

    pipe_metrics = parse_pipe_utilization(pipe_path)
    pipe_metrics["mfu"] = parse_mfu_from_arithmetic(prof_dir)
    basic = parse_op_basic_info(basic_path) if basic_path.exists() else {}
    resolved_kernel = kernel_name or basic.get("kernel_name") or ""
    resolved_operator = resolve_operator_id(resolved_kernel, operator_id)
    time_ms = basic.get("time_ms") or 0.0
    case = build_case_from_attributes(attributes or {}, prof_dir, kernel_name=resolved_kernel or None, prof_tool=prof_tool)
    snapshot_date, created_at = parse_opprof_timestamp(prof_dir)

    operator = {
        "operator_id": resolved_operator,
        "time_ms": time_ms,
        "share_pct": 100.0,
        "mbu": pipe_metrics["mbu"],
        "mfu": pipe_metrics["mfu"],
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
        "mem_util": pipe_metrics["mbu"],
        "kernel_name": resolved_kernel,
        "block_dim": basic.get("block_dim"),
        "pipe_utilization": pipe_metrics["pipe_rows"],
    }

    snapshot = {
        "id": f"snap-{prof_dir.name.lower()}",
        "label": f"{snapshot_date} {resolved_operator or 'op'}",
        "snapshot_date": snapshot_date,
        "branch": "main",
        "model_id": model_id,
        "case_id": case["id"],
        "chip": chip,
        "total_time_ms": time_ms,
        "status": "done",
        "created_at": created_at,
        "prof_source": prof_dir.name,
        "prof_tool": prof_tool,
        "kernel_name": resolved_kernel,
        "device_id": device_id,
        "operators": [operator],
    }

    gdr = load_prof_gdr_module()
    data = gdr.load_perf_data()
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
    parser = argparse.ArgumentParser(description="Import msprof op OPPROF_* output into performance-data.json")
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
        print(f"  operator: {snapshot['operators'][0]['operator_id']}")
        print(f"  total: {snapshot['total_time_ms']} ms")
    print(f"Done: {len(data['snapshots'])} snapshots, {len(data['cases'])} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
