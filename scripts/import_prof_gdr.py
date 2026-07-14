#!/usr/bin/env python3
"""Import MindStudio GDR profiler output into performance-data.json."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PERF_PATHS = [ROOT / "data" / "performance-data.json", ROOT / "docs" / "performance-data.json"]
BJ_TZ = timezone(timedelta(hours=8))
RATIO_PRECISION = 4


def load_cube_theoretical_flops_module():
    import importlib.util

    path = ROOT / "scripts" / "cube_theoretical_flops.py"
    spec = importlib.util.spec_from_file_location("cube_theoretical_flops", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_hbm_theoretical_bytes_module():
    import importlib.util

    path = ROOT / "scripts" / "hbm_theoretical_bytes.py"
    spec = importlib.util.spec_from_file_location("hbm_theoretical_bytes", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def round_ratio(value: float | None) -> float:
    if value is None:
        return 0.0
    return round(value, RATIO_PRECISION)

# OP Type / kernel name -> project operator id
OP_TYPE_MAP = {
    "ChunkGatedDeltaRuleFwdH": "chunk_gated_delta_rule_fwd_h",
    "ChunkFwdO": "chunk_fwd_o",
    "RecomputeWUFwd": "recompute_wu_fwd",
    "ChunkBwdDvLocal": "chunk_bwd_dv_local",
    "ChunkBwdDqkwg": "chunk_bwd_dqkwg",
    "ChunkGatedDeltaRuleBwdDhu": "chunk_gated_delta_rule_bwd_dhu",
    "PrepareWyReprBwdFull": "prepare_wy_repr_bwd_full",
    "PrepareWyReprBwdDa": "prepare_wy_repr_bwd_da",
    "solve_tril_16x16_kernel_paral_v3": "solve_tril",
    "merge_16x16_to_32x32_inverse_kernel": "solve_tril",
    "merge_32x32_to_64x64_inverse_kernel": "solve_tril",
    "chunk_scaled_dot_kkt_fwd_kernel": "chunk_scaled_dot_kkt_fwd",
    "l2norm_fwd_kernel": "l2norm_fwd",
    "l2norm_bwd_kernel": "l2norm_bwd",
    "chunk_local_cumsum_scalar_kernel": "chunk_local_cumsum",
}

CORE_GDN_OPS = {
    "chunk_gated_delta_rule_fwd_h",
    "chunk_fwd_o",
    "recompute_wu_fwd",
    "chunk_bwd_dv_local",
    "chunk_bwd_dqkwg",
    "chunk_gated_delta_rule_bwd_dhu",
    "prepare_wy_repr_bwd_full",
    "prepare_wy_repr_bwd_da",
    "solve_tril",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import prof_gdr snapshot(s) into performance-data.json")
    parser.add_argument("prof_dir", type=Path, nargs="?", help="Profiler result directory, or prof_gdr root with --all")
    parser.add_argument("--all", action="store_true", help="Import every PROF_* directory under prof_dir (default: data/prof_gdr)")
    parser.add_argument("--model", default="gdn", help="Model id, default gdn")
    parser.add_argument("--chip", default="A2", choices=["A2", "A3"], help="Chip type")
    parser.add_argument("--replace-mock", action="store_true", help="Remove previous prof snapshots before import")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(BJ_TZ).replace(microsecond=0).isoformat()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    return list(csv.DictReader(text.splitlines()))


def find_output_file(prof_dir: Path, prefix: str) -> Path:
    output_dir = prof_dir / "mindstudio_profiler_output"
    matches = sorted(output_dir.glob(f"{prefix}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"Missing {prefix}_*.csv under {output_dir}")
    return matches[-1]


def parse_prof_timestamp(prof_dir: Path) -> tuple[str, str]:
    match = re.search(r"(\d{8})(\d{6})", prof_dir.name)
    if match:
        dt = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S").replace(tzinfo=BJ_TZ)
        return dt.date().isoformat(), dt.isoformat()
    return datetime.now(BJ_TZ).date().isoformat(), now_iso()


def normalize_layout(layout: str | None = None, *, varlen: bool | None = None) -> str:
    if layout:
        normalized = str(layout).strip().upper()
        if normalized in {"TND", "BSND"}:
            return normalized
    if varlen is False:
        return "BSND"
    return "TND"


def build_case_attributes(
    *,
    batch: int = 1,
    query_heads: int = 32,
    value_heads: int = 32,
    tokens: int = 4087,
    key_dim: int = 128,
    value_dim: int = 128,
    chunk_size: int = 64,
    dtype: str = "bf16",
    layout: str | None = None,
    varlen: bool | None = None,
    mean_len: int = 1024,
    cu_seqlens: str = "",
    scale: float | None = None,
) -> dict[str, Any]:
    if scale is None:
        scale = round(key_dim ** -0.5, 6)
    resolved_layout = normalize_layout(layout, varlen=varlen)
    return {
        "batch": batch,
        "query_heads": query_heads,
        "value_heads": value_heads,
        "tokens": tokens,
        "key_dim": key_dim,
        "value_dim": value_dim,
        "chunk_size": chunk_size,
        "scale": scale,
        "dtype": dtype,
        "mean_len": mean_len,
        "cu_seqlens": cu_seqlens,
        "layout": resolved_layout,
    }


def parse_shape_parts(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    return [part.strip().strip('"') for part in text.split(";") if part.strip().strip('"')]


def parse_dtype_parts(raw: str) -> list[str]:
    return [part.strip() for part in (raw or "").split(";") if part.strip()]


def parse_batch_from_shapes(shapes: str) -> int | None:
    for match in re.finditer(r"(\d+),(\d+),(\d+),(\d+)", shapes):
        batch, heads, tokens, dim = (int(match.group(i)) for i in range(1, 5))
        if heads <= 128 and tokens >= 256 and dim in {16, 32, 64, 128}:
            return batch
    return None


def core_gdn_row_is_varlen(row: dict[str, str]) -> bool | None:
    op_type = row.get("OP Type", "")
    if op_type not in OP_TYPE_MAP or OP_TYPE_MAP[op_type] not in CORE_GDN_OPS:
        return None
    shapes = parse_shape_parts(row.get("Input Shapes", ""))
    dtypes = parse_dtype_parts(row.get("Input Data Types", ""))
    if not shapes or not dtypes:
        return None
    if not any(dtype.upper() == "INT64" for dtype in dtypes):
        return False
    for idx, dtype in enumerate(dtypes):
        if dtype.upper() != "INT64" or idx >= len(shapes):
            continue
        part = shapes[idx]
        if re.fullmatch(r"\d+", part):
            return True
        if "," in part:
            nums = [int(item) for item in part.split(",")]
            if nums and max(nums) <= 4096:
                return True
    return True


def infer_varlen_from_summary(rows: list[dict[str, str]]) -> bool:
    saw_varlen = False
    saw_fixed = False
    for row in rows:
        verdict = core_gdn_row_is_varlen(row)
        if verdict is True:
            saw_varlen = True
        elif verdict is False:
            saw_fixed = True
    if saw_varlen:
        return True
    if saw_fixed:
        return False
    return True


def parse_chunk_from_shapes(shapes: str) -> int | None:
    """Parse chunk_size from B,H,chunk,K,V tensors such as 1,32,64,128,128."""
    for match in re.finditer(r"(\d+),(\d+),(\d+),(\d+),(\d+)", shapes):
        _, heads, chunk_candidate, key_dim, value_dim = (int(match.group(i)) for i in range(1, 6))
        if (
            heads <= 128
            and chunk_candidate in {16, 32, 64, 128}
            and key_dim in {16, 32, 64, 128}
            and value_dim in {16, 32, 64, 128}
        ):
            return chunk_candidate
    return None


def parse_qkv_dims_from_shapes(shapes: str) -> tuple[int | None, int | None, int | None, int | None]:
    """Parse query_heads, tokens, key_dim, value_dim from ordered B,H,T,D tensors."""
    matches: list[tuple[int, int, int]] = []
    for match in re.finditer(r"(\d+),(\d+),(\d+),(\d+)", shapes):
        batch, heads, tokens, dim = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
        if batch >= 1 and heads <= 128 and tokens >= 256 and dim in {16, 32, 64, 128}:
            matches.append((heads, tokens, dim))
    if not matches:
        return None, None, None, None
    query_heads, tokens, key_dim = matches[0]
    dims = [item[2] for item in matches]
    if len(dims) >= 3:
        value_dim = dims[2]
    elif len(dims) >= 2 and dims[1] != key_dim:
        value_dim = dims[1]
    else:
        value_dim = key_dim
    return query_heads, tokens, key_dim, value_dim


def infer_case_from_summary(rows: list[dict[str, str]], prof_dir: Path) -> dict[str, Any]:
    batch = 1
    tokens = None
    query_heads = None
    value_heads = None
    key_dim = None
    value_dim = None
    chunk_size = None
    dtype = "bf16"

    preferred_ops = (
        "ChunkGatedDeltaRuleFwdH",
        "ChunkFwdO",
        "RecomputeWUFwd",
        "ChunkBwdDqkwg",
        "ChunkGatedDeltaRuleBwdDhu",
    )
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            preferred_ops.index(row.get("OP Type", ""))
            if row.get("OP Type", "") in preferred_ops
            else len(preferred_ops)
        ),
    )

    for row in ordered_rows:
        op_type = row.get("OP Type", "")
        if op_type not in OP_TYPE_MAP or OP_TYPE_MAP[op_type] not in CORE_GDN_OPS:
            continue
        shapes = row.get("Input Shapes", "") or ""
        dtypes = row.get("Input Data Types", "") or ""
        if "BF16" in dtypes.upper():
            dtype = "bf16"
        elif "FLOAT" in dtypes.upper() and dtype != "bf16":
            dtype = "fp32"

        parsed_batch = parse_batch_from_shapes(shapes)
        if parsed_batch is not None:
            batch = parsed_batch

        qh, tk, kd, vd = parse_qkv_dims_from_shapes(shapes)
        if qh is not None:
            query_heads = query_heads or qh
            tokens = tokens or tk
            key_dim = key_dim or kd
            value_dim = value_dim or vd

        chunk_from_shapes = parse_chunk_from_shapes(shapes)
        if chunk_from_shapes is not None:
            chunk_size = chunk_size or chunk_from_shapes

    if key_dim is None:
        for row in rows:
            op_type = row.get("OP Type", "")
            if op_type not in OP_TYPE_MAP:
                continue
            shapes = row.get("Input Shapes", "") or ""
            qh, tk, kd, vd = parse_qkv_dims_from_shapes(shapes)
            if kd is None:
                continue
            query_heads = query_heads or qh
            tokens = tokens or tk
            key_dim = kd
            value_dim = vd
            break

    tokens = tokens or 4087
    query_heads = query_heads or 32
    value_heads = value_heads or query_heads
    key_dim = key_dim or 128
    value_dim = value_dim or key_dim
    chunk_size = chunk_size or 64
    varlen = infer_varlen_from_summary(rows)
    layout = normalize_layout(varlen=varlen)

    snapshot_date, created_at = parse_prof_timestamp(prof_dir)
    time_slug_match = re.search(r"(\d{8})(\d{6})", prof_dir.name)
    time_slug = (
        f"{time_slug_match.group(1)}t{time_slug_match.group(2)}"
        if time_slug_match
        else re.sub(r"[^a-z0-9]+", "", prof_dir.name.lower())[-16:]
    )
    time_label = str(created_at).replace("T", " ").replace("+08:00", "")[:16]

    case_id = f"case-gdn-t{tokens}-k{key_dim}-v{value_dim}-c{chunk_size}-{time_slug}"
    label = (
        f"B={batch} QH={query_heads} VH={value_heads} T={tokens} "
        f"K={key_dim} V={value_dim} chunk={chunk_size} {layout} @ {time_label}"
    )
    return {
        "id": case_id,
        "label": label,
        "category": layout.lower(),
        "attributes": build_case_attributes(
            batch=batch,
            query_heads=query_heads,
            value_heads=value_heads,
            tokens=tokens,
            key_dim=key_dim,
            value_dim=value_dim,
            chunk_size=chunk_size,
            dtype=dtype,
            layout=layout,
            varlen=varlen,
        ),
        "position": 0,
        "active": True,
    }


def to_float(value: str | None) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def pick_bottleneck(row: dict[str, str]) -> str:
    candidates = {
        "Cube": to_float(row.get("aic_mac_ratio")) or 0,
        "HBM": to_float(row.get("aic_mte2_ratio")) or 0,
        "L2": to_float(row.get("aiv_mte2_ratio")) or 0,
        "Vector": to_float(row.get("aiv_vec_ratio")) or 0,
        "MTE3": to_float(row.get("aiv_mte3_ratio")) or 0,
    }
    return max(candidates, key=candidates.get)


def row_pipe_time_us(row: dict[str, str], pipe: str) -> float | None:
    if pipe == "mte1":
        return to_float(row.get("aic_mte1_time(us)"))
    if pipe == "mte2":
        aic = to_float(row.get("aic_mte2_time(us)"))
        aiv = to_float(row.get("aiv_mte2_time(us)"))
        if aic is None and aiv is None:
            return None
        return max(aic or 0, aiv or 0)
    if pipe == "mte3":
        return to_float(row.get("aiv_mte3_time(us)"))
    if pipe == "cube":
        return to_float(row.get("aic_mac_time(us)"))
    if pipe == "vector":
        return to_float(row.get("aiv_vec_time(us)"))
    return None


def pipe_time_stats(metrics: list[dict[str, Any]], pipe: str) -> dict[str, float]:
    field = f"{pipe}_time_us"
    picked = [item for item in metrics if item.get(field) is not None]
    if not picked:
        return {}
    values = [float(item[field]) for item in picked]
    denom = sum(item["time_ms"] for item in picked) or 1
    avg = sum(float(item[field]) * item["time_ms"] for item in picked) / denom
    return {
        f"{pipe}_time_avg_us": round(avg, 2),
        f"{pipe}_time_min_us": round(min(values), 2),
        f"{pipe}_time_max_us": round(max(values), 2),
    }


def metric_from_summary_row(row: dict[str, str]) -> dict[str, Any]:
    duration = to_float(row.get("Task Duration(us)")) or to_float(row.get("aicore_time(us)")) or 0
    block_dim = to_float(row.get("Block Num") or row.get("block_dim") or row.get("BlockDim"))
    cube_util_pct = to_float(row.get("cube_utilization(%)"))
    aic_mac = to_float(row.get("aic_mac_ratio"))
    aic_mte1 = to_float(row.get("aic_mte1_ratio"))
    aic_mte2 = to_float(row.get("aic_mte2_ratio"))
    aiv_mte2 = to_float(row.get("aiv_mte2_ratio"))
    aiv_mte3 = to_float(row.get("aiv_mte3_ratio"))
    aiv_vec = to_float(row.get("aiv_vec_ratio"))
    mfu = (cube_util_pct / 100) if cube_util_pct is not None else None
    mte2 = max(aic_mte2 or 0, aiv_mte2 or 0)
    payload = {
        "time_ms": round(duration / 1000, 3),
        "duration_us": duration if duration > 0 else None,
        "mbu": None,
        "mfu": round_ratio(mfu) if mfu is not None else None,
        "bottleneck": pick_bottleneck(row),
        "mte1_ratio": round_ratio(aic_mte1 or 0),
        "mte2_ratio": round_ratio(mte2),
        "mte3_ratio": round_ratio(aiv_mte3 or 0),
        "cube_util": round_ratio(aic_mac or 0),
        "vector_util": round_ratio(aiv_vec or 0),
        "mem_util": None,
        "block_dim": int(block_dim) if block_dim is not None else None,
        "core_type": row.get("Task Type") or row.get("Core Type") or "",
        "count": 1,
    }
    for pipe in ("mte1", "mte2", "mte3", "cube", "vector"):
        time_us = row_pipe_time_us(row, pipe)
        if time_us is not None:
            payload[f"{pipe}_time_us"] = round(time_us, 2)
    return payload


def aggregate_summary_metrics(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        op_type = row.get("OP Type", "")
        op_name = row.get("Op Name", "")
        key = OP_TYPE_MAP.get(op_type) or OP_TYPE_MAP.get(op_name)
        if not key:
            continue
        grouped.setdefault(key, []).append(metric_from_summary_row(row))

    result: dict[str, dict[str, Any]] = {}
    for operator_id, metrics in grouped.items():
        total_ms = sum(item["time_ms"] for item in metrics)
        weight = [item["time_ms"] for item in metrics]
        denom = sum(weight) or 1

        def wavg(field: str) -> float | None:
            picked = [item for item in metrics if item.get(field) is not None]
            if not picked:
                return None
            denom = sum(item["time_ms"] for item in picked) or 1
            return round_ratio(sum(float(item[field]) * item["time_ms"] for item in picked) / denom)

        bottlenecks = {}
        for item in metrics:
            bottlenecks[item["bottleneck"]] = bottlenecks.get(item["bottleneck"], 0) + item["time_ms"]
        result[operator_id] = {
            "time_ms": round(total_ms, 3),
            "duration_us": sum(float(item.get("duration_us") or 0) for item in metrics) or None,
            "mbu": wavg("mbu"),
            "mfu": wavg("mfu"),
            "bottleneck": max(bottlenecks, key=bottlenecks.get),
            "mte1_ratio": wavg("mte1_ratio"),
            "mte2_ratio": wavg("mte2_ratio"),
            "mte3_ratio": wavg("mte3_ratio"),
            "cube_util": wavg("cube_util"),
            "vector_util": wavg("vector_util"),
            "mem_util": wavg("mem_util"),
            "count": len(metrics),
            "block_dim": max((item.get("block_dim") or 0) for item in metrics) or None,
        }
        for pipe in ("mte1", "mte2", "mte3", "cube", "vector"):
            result[operator_id].update(pipe_time_stats(metrics, pipe))
    return result


def aggregate_statistic_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], float]:
    operators: dict[str, dict[str, Any]] = {}
    other_time = 0.0
    for row in rows:
        op_type = row.get("OP Type", "")
        op_name = row.get("Op Name", "")
        operator_id = OP_TYPE_MAP.get(op_type) or OP_TYPE_MAP.get(op_name)
        total_us = to_float(row.get("Total Time(us)")) or 0
        ratio = to_float(row.get("Ratio(%)")) or 0
        if not operator_id:
            other_time += total_us
            continue
        bucket = operators.setdefault(
            operator_id,
            {"operator_id": operator_id, "time_ms": 0.0, "share_pct": 0.0, "op_type": op_type},
        )
        bucket["time_ms"] += total_us / 1000
        bucket["share_pct"] += ratio
    if other_time > 0:
        operators["other_runtime"] = {
            "operator_id": "other_runtime",
            "time_ms": round(other_time / 1000, 3),
            "share_pct": 0.0,
            "op_type": "Other",
        }
    ordered = sorted(operators.values(), key=lambda item: item["time_ms"], reverse=True)
    total_ms = sum(item["time_ms"] for item in ordered)
    for item in ordered:
        item["time_ms"] = round(item["time_ms"], 3)
        item["share_pct"] = round((item["time_ms"] / total_ms * 100) if total_ms else 0, 2)
    return ordered, round(total_ms, 3)


def load_perf_data() -> dict[str, Any]:
    for path in PERF_PATHS:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8-sig"))
    return {
        "version": now_iso(),
        "models": [{"id": "gdn", "label": "GDN", "position": 0, "active": True}],
        "cases": [],
        "snapshots": [],
        "runs": [],
    }


def upsert_case(data: dict[str, Any], case: dict[str, Any]) -> None:
    cases = data.setdefault("cases", [])
    for index, existing in enumerate(cases):
        if existing.get("id") == case["id"]:
            cases[index] = case
            return
    case["position"] = len(cases)
    cases.append(case)


def build_snapshot(
    prof_dir: Path,
    model_id: str,
    chip: str,
    case: dict[str, Any],
    statistic_ops: list[dict[str, Any]],
    summary_metrics: dict[str, dict[str, Any]],
    total_ms: float,
    *,
    device_id: int = 2,
) -> dict[str, Any]:
    snapshot_date, created_at = parse_prof_timestamp(prof_dir)
    operators = []
    for item in statistic_ops:
        operator_id = item["operator_id"]
        detail = summary_metrics.get(operator_id, {})
        operators.append(
            {
                "operator_id": operator_id,
                "time_ms": item["time_ms"],
                "duration_us": detail.get("duration_us"),
                "share_pct": item["share_pct"],
                "mbu": detail.get("mbu"),
                "mfu": detail.get("mfu"),
                "bottleneck": detail.get("bottleneck", "—"),
                "mte1_ratio": detail.get("mte1_ratio", 0),
                "mte2_ratio": detail.get("mte2_ratio", 0),
                "mte3_ratio": detail.get("mte3_ratio", 0),
                "mte1_time_avg_us": detail.get("mte1_time_avg_us"),
                "mte1_time_min_us": detail.get("mte1_time_min_us"),
                "mte1_time_max_us": detail.get("mte1_time_max_us"),
                "mte2_time_avg_us": detail.get("mte2_time_avg_us"),
                "mte2_time_min_us": detail.get("mte2_time_min_us"),
                "mte2_time_max_us": detail.get("mte2_time_max_us"),
                "mte3_time_avg_us": detail.get("mte3_time_avg_us"),
                "mte3_time_min_us": detail.get("mte3_time_min_us"),
                "mte3_time_max_us": detail.get("mte3_time_max_us"),
                "cube_util": detail.get("cube_util", 0),
                "vector_util": detail.get("vector_util", 0),
                "cube_time_avg_us": detail.get("cube_time_avg_us"),
                "cube_time_min_us": detail.get("cube_time_min_us"),
                "cube_time_max_us": detail.get("cube_time_max_us"),
                "vector_time_avg_us": detail.get("vector_time_avg_us"),
                "vector_time_min_us": detail.get("vector_time_min_us"),
                "vector_time_max_us": detail.get("vector_time_max_us"),
                "mem_util": detail.get("mem_util"),
                "block_dim": detail.get("block_dim"),
                "op_type": item.get("op_type", ""),
                "call_count": detail.get("count", 0),
            }
        )
    return {
        "id": f"snap-{prof_dir.name.lower()}",
        "label": f"{snapshot_date} 主线",
        "snapshot_date": snapshot_date,
        "branch": "main",
        "model_id": model_id,
        "case_id": case["id"],
        "chip": chip,
        "total_time_ms": total_ms,
        "status": "done",
        "created_at": created_at,
        "prof_source": prof_dir.name,
        "device_id": device_id,
        "operators": operators,
    }


def import_prof(
    prof_dir: Path,
    model_id: str,
    chip: str,
    replace_mock: bool = False,
    *,
    device_id: int = 2,
) -> dict[str, Any]:
    prof_dir = prof_dir.resolve()
    if not prof_dir.exists():
        raise FileNotFoundError(prof_dir)

    statistic_rows = read_csv_rows(find_output_file(prof_dir, "op_statistic"))
    summary_rows = read_csv_rows(find_output_file(prof_dir, "op_summary"))
    case = infer_case_from_summary(summary_rows, prof_dir)
    statistic_ops, total_ms = aggregate_statistic_rows(statistic_rows)
    summary_metrics = aggregate_summary_metrics(summary_rows)
    snapshot = build_snapshot(
        prof_dir, model_id, chip, case, statistic_ops, summary_metrics, total_ms, device_id=device_id,
    )
    cube_mod = load_cube_theoretical_flops_module()
    hbm_mod = load_hbm_theoretical_bytes_module()
    attrs = case.get("attributes") or {}
    for operator in snapshot.get("operators", []):
        operator_id = str(operator.get("operator_id") or "")
        task_duration_us = cube_mod.resolve_task_duration_us(operator)
        analytical_mfu = cube_mod.compute_mfu(
            operator_id,
            attrs,
            task_duration_us=task_duration_us,
            block_dim=operator.get("block_dim"),
            chip=chip,
        )
        if analytical_mfu is not None:
            operator["mfu"] = analytical_mfu
        analytical_mbu = hbm_mod.compute_mbu(
            operator_id,
            attrs,
            task_duration_us=task_duration_us,
            chip=chip,
        )
        if analytical_mbu is not None:
            operator["mbu"] = analytical_mbu
            operator["mem_util"] = analytical_mbu

    data = load_perf_data()
    upsert_case(data, case)
    snapshots = [item for item in data.get("snapshots", []) if item.get("prof_source")]
    if replace_mock:
        snapshots = []
    snapshots = [item for item in snapshots if item.get("id") != snapshot["id"]]
    snapshots.insert(0, snapshot)
    data["snapshots"] = snapshots
    if not any(item.get("id") == model_id for item in data["models"]):
        data["models"].append({"id": model_id, "label": model_id.upper(), "position": 0, "active": True})
    runs = [item for item in data.get("runs", []) if item.get("created_by") == "import_prof_gdr"]
    run = {
        "id": f"run-{prof_dir.name.lower()}",
        "case_id": case["id"],
        "model_id": model_id,
        "chip": chip,
        "device": str(snapshot.get("device_id", device_id)),
        "attributes": case.get("attributes", {}),
        "status": "done",
        "snapshot_id": snapshot["id"],
        "created_by": "import_prof_gdr",
        "created_at": snapshot["created_at"],
        "finished_at": now_iso(),
        "message": f"导入 MindStudio Prof：{prof_dir.name}",
        "snapshot": snapshot,
    }
    runs = [item for item in runs if item.get("id") != run["id"]]
    runs.insert(0, run)
    data["runs"] = runs
    referenced_cases = {item["case_id"] for item in data["snapshots"]}
    data["cases"] = [item for item in data.get("cases", []) if item.get("id") in referenced_cases]
    data["version"] = now_iso()

    for path in PERF_PATHS:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def main() -> int:
    args = parse_args()
    prof_dirs: list[Path]
    if args.all:
        root = (args.prof_dir or ROOT / "data" / "prof_gdr").resolve()
        prof_dirs = sorted(p for p in root.glob("PROF_*") if p.is_dir())
        if not prof_dirs:
            raise SystemExit(f"No PROF_* directories under {root}")
    else:
        if not args.prof_dir:
            raise SystemExit("prof_dir is required unless --all is set")
        prof_dirs = [args.prof_dir]

    data = None
    for index, prof_dir in enumerate(prof_dirs):
        data = import_prof(prof_dir, args.model, args.chip, replace_mock=args.replace_mock and index == 0)
        snapshot = next(item for item in data["snapshots"] if item.get("prof_source") == prof_dir.name)
        case = next(c for c in data["cases"] if c["id"] == snapshot["case_id"])
        print(f"Imported {prof_dir.name}")
        print(f"  case: {snapshot['case_id']} ({case['label']})")
        print(f"  total: {snapshot['total_time_ms']} ms, operators: {len(snapshot['operators'])}")
        print("  top3: ", end="")
        for op in snapshot["operators"][:3]:
            print(f"{op['operator_id']} {op['share_pct']}%", end=" | ")
        print()
    print(f"Done: {len(data['snapshots'])} snapshots, {len(data['cases'])} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
