#!/usr/bin/env python3
"""Analytical Cube FLOPs for GDN operators (matmul-only, M*N*K*2)."""

from __future__ import annotations

import re
from typing import Any

RATIO_PRECISION = 4
CUBE_MAC_SHAPE = 4096  # DAV_2201 FP16 Cube M×K×N
CHIP_RATED_FREQ_MHZ = {"A2": 1800.0, "A3": 1650.0}

# Operators whose Cube work is dominated by GEMM tiles we can model from case attrs.
CUBE_GEMM_OPERATORS = {
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


def matmul_flops(m: int, k: int, n: int) -> float:
    return float(m * k * n * 2)


def cube_hardware_flops_per_us(freq_mhz: float, block_dim: int) -> float:
    per_core = CUBE_MAC_SHAPE * freq_mhz * 2
    return per_core * max(int(block_dim), 1)


def chunk_sizes_for_tokens(tokens: int, chunk_size: int) -> list[int]:
    if tokens <= 0 or chunk_size <= 0:
        return []
    full = tokens // chunk_size
    rem = tokens % chunk_size
    sizes = [chunk_size] * full
    if rem:
        sizes.append(rem)
    return sizes


def parse_cu_seqlens(raw: Any) -> list[int] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    values = [int(part.strip()) for part in re.split(r"[,;\s]+", text) if part.strip()]
    if len(values) < 2:
        return None
    return [max(0, values[i + 1] - values[i]) for i in range(len(values) - 1)]


def sequence_lengths(attributes: dict[str, Any]) -> list[int]:
    batch = max(int(attributes.get("batch") or 1), 1)
    cu_lengths = parse_cu_seqlens(attributes.get("cu_seqlens"))
    if cu_lengths:
        return cu_lengths
    tokens = max(int(attributes.get("tokens") or 0), 0)
    if tokens <= 0:
        return []
    return [tokens] * batch


def iter_chunk_sizes(attributes: dict[str, Any]) -> list[int]:
    chunk_size = max(int(attributes.get("chunk_size") or 64), 1)
    sizes: list[int] = []
    for seq_len in sequence_lengths(attributes):
        sizes.extend(chunk_sizes_for_tokens(seq_len, chunk_size))
    return sizes


def normalize_dims(attributes: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    batch = max(int(attributes.get("batch") or 1), 1)
    hk = max(int(attributes.get("query_heads") or 32), 1)
    hv = max(int(attributes.get("value_heads") or hk), 1)
    k_dim = max(int(attributes.get("key_dim") or 128), 1)
    v_dim = max(int(attributes.get("value_dim") or k_dim), 1)
    chunk_size = max(int(attributes.get("chunk_size") or 64), 1)
    return batch, hk, hv, k_dim, v_dim, chunk_size


def per_chunk_wy_da_flops(c: int, k_dim: int, v_dim: int) -> float:
    # dw @ kbg.T, du @ vb.T, DA4 @ A.T, DA5.T @ A
    return (
        matmul_flops(c, k_dim, c)
        + matmul_flops(c, v_dim, c)
        + 2 * matmul_flops(c, c, c)
    )


def per_chunk_wy_full_flops(c: int, k_dim: int, v_dim: int) -> float:
    # dA^T@K, A^T@dw, dA@Kbeta, A^T@du, K@K^T
    return 4 * matmul_flops(c, k_dim, c) + matmul_flops(c, v_dim, c)


def theoretical_cube_flops(operator_id: str, attributes: dict[str, Any] | None) -> float | None:
    if not attributes or operator_id not in CUBE_GEMM_OPERATORS:
        return None

    _batch, hk, hv, k_dim, v_dim, _chunk_size = normalize_dims(attributes)
    chunk_sizes = iter_chunk_sizes(attributes)
    if not chunk_sizes:
        return None

    if operator_id == "chunk_bwd_dv_local":
        per_tasks = sum(matmul_flops(c, k_dim, c) + matmul_flops(c, c, v_dim) for c in chunk_sizes)
        return hk * per_tasks

    if operator_id == "chunk_fwd_o":
        per_tasks = sum(
            matmul_flops(c, k_dim, c) + matmul_flops(c, k_dim, v_dim) + matmul_flops(c, c, v_dim)
            for c in chunk_sizes
        )
        return hv * per_tasks

    if operator_id == "chunk_gated_delta_rule_fwd_h":
        per_tasks = sum(2 * matmul_flops(c, k_dim, v_dim) for c in chunk_sizes)
        return hv * per_tasks

    if operator_id == "recompute_wu_fwd":
        return sum(matmul_flops(c, c, v_dim) + matmul_flops(c, c, k_dim) for c in chunk_sizes)

    if operator_id == "chunk_bwd_dqkwg":
        per_tasks = 0.0
        for c in chunk_sizes:
            per_tasks += (
                matmul_flops(c, v_dim, k_dim)
                + matmul_flops(c, k_dim, c)
                + matmul_flops(c, v_dim, c)
                + matmul_flops(c, v_dim, k_dim)
                + matmul_flops(c, v_dim, k_dim)
                + matmul_flops(c, c, k_dim)
                + matmul_flops(c, c, k_dim)
            )
        return hv * per_tasks

    if operator_id == "chunk_gated_delta_rule_bwd_dhu":
        per_tasks = 0.0
        n = len(chunk_sizes)
        for idx, c in enumerate(chunk_sizes):
            if idx < n - 1:
                per_tasks += matmul_flops(c, k_dim, v_dim)
            if idx > 0:
                per_tasks += 2 * matmul_flops(k_dim, c, v_dim)
        return hk * per_tasks

    if operator_id == "prepare_wy_repr_bwd_da":
        per_tasks = sum(per_chunk_wy_da_flops(c, k_dim, v_dim) for c in chunk_sizes)
        return hv * per_tasks

    if operator_id == "prepare_wy_repr_bwd_full":
        per_tasks = sum(per_chunk_wy_full_flops(c, k_dim, v_dim) for c in chunk_sizes)
        return hv * per_tasks

    return None


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
    block_dim: int | None,
    freq_mhz: float | None = None,
    chip: str | None = None,
) -> float | None:
    """MFU = (计算量 / Task Duration) / 标称 FLOPS。"""
    if task_duration_us is None or task_duration_us <= 0:
        return None
    if block_dim is None or block_dim <= 0:
        return None
    resolved_freq = freq_mhz
    if resolved_freq is None and chip:
        resolved_freq = CHIP_RATED_FREQ_MHZ.get(chip.upper())
    if resolved_freq is None or resolved_freq <= 0:
        resolved_freq = CHIP_RATED_FREQ_MHZ["A2"]
    theoretical_flops = theoretical_cube_flops(operator_id, attributes)
    if theoretical_flops is None or theoretical_flops <= 0:
        return None
    hardware_peak = cube_hardware_flops_per_us(resolved_freq, block_dim)
    actual_flops_per_us = theoretical_flops / task_duration_us
    return round_ratio(actual_flops_per_us / hardware_peak)
