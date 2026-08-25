#!/usr/bin/env python3
"""MFU for the supported GDN/KDA fused operators."""

from __future__ import annotations

from typing import Any

RATIO_PRECISION = 4
FUSED_FORWARD_OPERATORS = {"chunk_gdr_fwd", "chunk_kda_fwd"}
FUSED_BACKWARD_OPERATORS = {"chunk_gdr_bwd", "chunk_kda_bwd"}
FUSED_OPERATORS = FUSED_FORWARD_OPERATORS | FUSED_BACKWARD_OPERATORS
MFU_DEVICE_PROFILES = {
    "A2_A3": {"work_divisor": 1024.0 * 1024.0, "peak_tflops": 384.0},
    "950DT": {"work_divisor": 1024.0 * 1024.0, "peak_tflops": 432.0},
    "950PR": {"work_divisor": 1000.0 * 1000.0, "peak_tflops": 378.0},
}


def round_ratio(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, RATIO_PRECISION)


def resolve_mfu_device_profile(
    chip: str | None = None,
    soc_version: str | None = None,
) -> dict[str, float]:
    """Resolve A2/A3 and 950 variants; legacy unknown calls use 950DT."""

    def compact(value: str | None) -> str:
        return str(value or "").upper().replace("_", "").replace("-", "").replace(" ", "")

    soc = compact(soc_version)
    normalized_chip = compact(chip)
    if "950PR" in soc:
        return MFU_DEVICE_PROFILES["950PR"]
    if "950DT" in soc:
        return MFU_DEVICE_PROFILES["950DT"]
    if normalized_chip in {"A2", "A3"} or "910B" in soc or "91093" in soc:
        return MFU_DEVICE_PROFILES["A2_A3"]
    if "950PR" in normalized_chip:
        return MFU_DEVICE_PROFILES["950PR"]
    return MFU_DEVICE_PROFILES["950DT"]


def fused_mfu_flops(operator_id: str, attributes: dict[str, Any] | None) -> float | None:
    """Return the requested fused-op work before device-specific scaling."""
    if not attributes or operator_id not in FUSED_OPERATORS:
        return None
    try:
        batch = max(int(attributes.get("batch") or 1), 1)
        heads = max(int(
            attributes.get("heads")
            or attributes.get("query_heads")
            or attributes.get("value_heads")
            or 0
        ), 1)
        tokens = max(int(attributes.get("tokens") or 0), 0)
        key_dim = max(int(attributes.get("key_dim") or 0), 0)
        value_dim = max(int(attributes.get("value_dim") or 0), 0)
        chunk_size = max(int(attributes.get("chunk_size") or 0), 0)
    except (TypeError, ValueError):
        return None
    if not all((tokens, key_dim, value_dim, chunk_size)):
        return None
    work = 2.0 * batch * heads * tokens * (
        3 * key_dim * value_dim
        + 3 * chunk_size * key_dim
        + 2 * chunk_size * value_dim
    )
    if operator_id in FUSED_BACKWARD_OPERATORS:
        work *= 2
    return work


def resolve_task_duration_us(operator: dict[str, Any]) -> float | None:
    """优先用 OpBasicInfo / op_summary 的 Task Duration(us)，否则回退 time_ms×1000。"""
    duration = operator.get("duration_us")
    if duration is not None:
        duration = float(duration)
        if duration > 0:
            return duration
    time_ms = float(operator.get("time_ms") or 0)
    if time_ms > 0:
        return time_ms * 1000
    return None


def compute_mfu(
    operator_id: str,
    attributes: dict[str, Any] | None,
    *,
    task_duration_us: float | None,
    block_dim: int | None = None,
    freq_mhz: float | None = None,
    chip: str | None = None,
    soc_version: str | None = None,
) -> float | None:
    """MFU for the four fused operators, using the selected device model."""
    if task_duration_us is None or task_duration_us <= 0:
        return None
    flops = fused_mfu_flops(operator_id, attributes)
    if flops is None or flops <= 0:
        return None
    profile = resolve_mfu_device_profile(chip, soc_version)
    scaled_work = flops / profile["work_divisor"]
    return round_ratio(scaled_work / task_duration_us / profile["peak_tflops"])
