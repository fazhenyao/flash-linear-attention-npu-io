#!/usr/bin/env python3
"""Analytical HBM traffic for GDN operators (theoretical bytes in + out)."""

from __future__ import annotations

from typing import Any

RATIO_PRECISION = 4

# HBM 标称带宽 (TB/s) -> bytes/us = TB/s * 10^6
CHIP_HBM_BANDWIDTH_TBPS = {"A2": 1.6, "A3": 1.4}

MEMORY_MODEL_OPERATORS = {
    "chunk_bwd_dv_local",
    "chunk_fwd_o",
    "chunk_gated_delta_rule_fwd_h",
    "recompute_wu_fwd",
    "chunk_bwd_dqkwg",
    "chunk_gated_delta_rule_bwd_dhu",
    "prepare_wy_repr_bwd_da",
    "prepare_wy_repr_bwd_full",
}


def round_ratio(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, RATIO_PRECISION)


def hbm_bytes_per_us(chip: str | None) -> float:
    tbps = CHIP_HBM_BANDWIDTH_TBPS.get((chip or "A2").upper(), CHIP_HBM_BANDWIDTH_TBPS["A2"])
    return tbps * 1_000_000.0


def element_bytes(attributes: dict[str, Any]) -> int:
    dtype = str(attributes.get("dtype") or "bf16").lower()
    if dtype in {"fp32", "float", "float32"}:
        return 4
    return 2


def normalize_dims(attributes: dict[str, Any]) -> tuple[int, int, int, int, int, int, int]:
    batch = max(int(attributes.get("batch") or 1), 1)
    hk = max(int(attributes.get("query_heads") or 32), 1)
    hv = max(int(attributes.get("value_heads") or hk), 1)
    tokens = max(int(attributes.get("tokens") or 0), 0)
    k_dim = max(int(attributes.get("key_dim") or 128), 1)
    v_dim = max(int(attributes.get("value_dim") or k_dim), 1)
    chunk_size = max(int(attributes.get("chunk_size") or 64), 1)
    return batch, hk, hv, tokens, k_dim, v_dim, chunk_size


def theoretical_memory_bytes(operator_id: str, attributes: dict[str, Any] | None) -> float | None:
    """理论搬入+搬出量（bytes），按算子 I/O 张量元素数估算。"""
    if not attributes or operator_id not in MEMORY_MODEL_OPERATORS:
        return None

    b, hk, hv, t, k, v, c = normalize_dims(attributes)
    if t <= 0:
        return None
    e = element_bytes(attributes)

    if operator_id == "chunk_bwd_dv_local":
        elements = 2 * b * hk * t * k + b * hv * t * v + b * hv * t + b * hv * t * c + b * hv * t * v
    elif operator_id == "chunk_fwd_o":
        elements = 2 * b * hk * t * k + b * hv * t * k + 2 * b * hv * t * v + b * hv * t * c
    elif operator_id == "chunk_gated_delta_rule_fwd_h":
        elements = 2 * b * hv * t * k + 2 * b * hv * t * v + b * hv * t + 2 * b * hv * k * v
    elif operator_id == "recompute_wu_fwd":
        elements = b * hv * t * c + 2 * b * hv * t * v + 2 * b * hv * t * k
    elif operator_id == "chunk_bwd_dqkwg":
        elements = (
            2 * b * hk * t * k
            + 3 * b * hv * t * v
            + 2 * b * hv * t * k
            + 3 * b * hv * t
            + b * hv * k * v
        )
    elif operator_id == "chunk_gated_delta_rule_bwd_dhu":
        elements = 2 * b * hk * t * k + b * hv * t * k + 2 * b * hv * t * v + 2 * b * hv * t + 2 * b * hv * k * v
    elif operator_id == "prepare_wy_repr_bwd_da":
        elements = (
            b * hk * t * k
            + 2 * b * hv * t * v
            + 2 * b * hv * t
            + 2 * b * hv * t * c
            + b * hv * t * k
        )
    elif operator_id == "prepare_wy_repr_bwd_full":
        elements = 2 * b * hk * t * k + 2 * b * hv * t * v + 2 * b * hv * t * c + 2 * b * hv * t + b * hv * t * v
    else:
        return None

    return float(elements * e)


def compute_mbu(
    operator_id: str,
    attributes: dict[str, Any] | None,
    *,
    task_duration_us: float | None,
    chip: str | None = None,
) -> float | None:
    """MBU = 理论访存耗时 / Task Duration(us)。"""
    if task_duration_us is None or task_duration_us <= 0:
        return None
    bytes_total = theoretical_memory_bytes(operator_id, attributes)
    if bytes_total is None or bytes_total <= 0:
        return None
    bandwidth = hbm_bytes_per_us(chip)
    theoretical_memory_time_us = bytes_total / bandwidth
    return round_ratio(theoretical_memory_time_us / task_duration_us)
