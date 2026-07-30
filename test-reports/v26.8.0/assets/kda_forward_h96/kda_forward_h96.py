#!/usr/bin/env python3
"""Validate the dense H96 KDA forward path against an NPU Torch reference."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from pathlib import Path

import numpy as np


FLA_ORG_COMMIT = "0f0f0c97af39343855b43bbbaddcedfda5cb9d77"
TRITON_ASCEND_KERNELS_COMMIT = "4cd4b506d4153ac18ac1ca8f4c770eac9fd3fcc8"
OUTPUT_NAMES = (
    "attn_out",
    "final_state",
    "gk",
    "Aqk",
    "Akk",
    "w",
    "u",
    "qg",
    "kg",
    "v_new",
    "h",
    "initial_state",
)
FILE_NAMES = {
    "Aqk": "aqk",
    "Akk": "akk",
}


def _configure_npu_math(torch):
    matmul = getattr(torch.npu, "matmul", None)
    if matmul is not None and hasattr(matmul, "allow_hf32"):
        matmul.allow_hf32 = False


def _event(torch):
    return torch.npu.Event(enable_timing=True)


def _elapsed_ms(torch, function):
    start = _event(torch)
    end = _event(torch)
    start.record()
    result = function()
    end.record()
    torch.npu.synchronize()
    return result, float(start.elapsed_time(end))


def _make_inputs(torch, *, batch, seqlen, heads, dim, seed):
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    device = torch.device("npu")
    shape = (batch, seqlen, heads, dim)
    return {
        "q": torch.randn(shape, device=device, dtype=torch.bfloat16),
        "k": torch.randn(shape, device=device, dtype=torch.bfloat16),
        "v": torch.randn(shape, device=device, dtype=torch.bfloat16),
        "beta": torch.randn(
            (batch, seqlen, heads),
            device=device,
            dtype=torch.bfloat16,
        ),
        "g": torch.randn(shape, device=device, dtype=torch.float32),
        "A_log": torch.rand((heads,), device=device, dtype=torch.float32),
        "dt_bias": torch.rand(
            (heads * dim,),
            device=device,
            dtype=torch.float32,
        ),
    }


def _load_ascendc_kda():
    from fla_npu.ops.ascendc import chunk_kda_fwd

    return chunk_kda_fwd


def _run_dut(
    torch,
    inputs,
    *,
    chunk_size,
    lower_bound,
    capture_intermediates,
    mstx_range,
):
    del capture_intermediates
    chunk_kda_fwd = _load_ascendc_kda()
    range_id = None
    try:
        if mstx_range:
            range_id = torch.npu.mstx.range_start(
                mstx_range,
                torch.npu.current_stream(),
            )
        # The pinned model wrapper predates beta fusion and its Triton L2Norm
        # exceeds the NPU grid limit for H96. Apply the same FLA formulas here.
        q = _l2norm(torch, inputs["q"])
        k = _l2norm(torch, inputs["k"])
        beta = torch.sigmoid(inputs["beta"].float())
        outputs = list(
            chunk_kda_fwd(
                q,
                k,
                inputs["v"],
                inputs["g"],
                beta,
                q.shape[-1] ** -0.5,
                chunk_size,
                layout="BSND",
                initial_state=None,
                output_final_state=True,
                cu_seqlens=None,
                chunk_indices=None,
                safe_gate=True,
                lower_bound=lower_bound,
                use_gate_in_kernel=True,
                A_log=inputs["A_log"],
                dt_bias=inputs["dt_bias"],
                disable_recompute=True,
                return_intermediate_states=False,
                state_v_first=True,
            )
        )
    finally:
        if range_id is not None:
            torch.npu.mstx.range_end(range_id)

    torch.npu.synchronize()
    if len(outputs) != len(OUTPUT_NAMES):
        raise RuntimeError(
            f"Expected {len(OUTPUT_NAMES)} outputs, received {len(outputs)}."
        )
    for index in range(2, 10):
        outputs[index] = outputs[index].permute(0, 2, 1, 3).contiguous()
    return tuple(outputs)


def _l2norm(torch, value, eps=1e-6):
    value_fp32 = value.float()
    rstd = torch.rsqrt(
        torch.sum(value_fp32 * value_fp32, dim=-1, keepdim=True) + eps
    )
    return (value_fp32 * rstd).to(value.dtype)


def _chunk_gate_cumsum(
    torch,
    g,
    A_log,
    dt_bias,
    *,
    chunk_size,
    lower_bound,
):
    batch, seqlen, heads, dim = g.shape
    if seqlen % chunk_size:
        raise ValueError("This dense validation case requires full chunks.")
    gate = lower_bound * torch.sigmoid(
        torch.exp(A_log).view(1, 1, heads, 1)
        * (g.float() + dt_bias.view(1, 1, heads, dim))
    )
    num_chunks = seqlen // chunk_size
    return (
        gate.view(batch, num_chunks, chunk_size, heads, dim)
        .cumsum(dim=2)
        .mul_(1.0 / math.log(2.0))
        .view(batch, seqlen, heads, dim)
    )


def _head_chunk_view(value, *, chunk_size):
    batch, seqlen, heads = value.shape[:3]
    tail = value.shape[3:]
    num_chunks = seqlen // chunk_size
    shaped = value.view(batch, num_chunks, chunk_size, heads, *tail)
    order = (0, 3, 1, 2, *range(4, shaped.dim()))
    return shaped.permute(order)


def _store_head_block(destination, source, h0, h1, *, chunk_size):
    view = _head_chunk_view(destination, chunk_size=chunk_size)
    view[:, h0:h1].copy_(source)


def _compute_intra_block(
    torch,
    q,
    k,
    v,
    gk,
    beta,
    *,
    chunk_size,
    scale,
    sub_chunk=16,
):
    dtype = q.dtype
    batch, head_block, num_chunks, _, dim = q.shape
    value_dim = v.shape[-1]
    causal = torch.tril(
        torch.ones(
            (chunk_size, chunk_size),
            dtype=torch.bool,
            device=q.device,
        )
    )
    strict = torch.tril(causal, diagonal=-1)
    aqk_fp32 = torch.zeros(
        (batch, head_block, num_chunks, chunk_size, chunk_size),
        dtype=torch.float32,
        device=q.device,
    )
    lower = torch.zeros_like(aqk_fp32)

    num_sub_chunks = chunk_size // sub_chunk
    for query_block in range(num_sub_chunks):
        qi = slice(query_block * sub_chunk, (query_block + 1) * sub_chunk)
        anchor = gk[..., query_block * sub_chunk, :]
        q_scaled = q[..., qi, :].float() * torch.exp2(
            gk[..., qi, :] - anchor[..., None, :]
        )
        kq_scaled = k[..., qi, :].float() * torch.exp2(
            gk[..., qi, :] - anchor[..., None, :]
        )
        beta_query = beta[..., qi].float()
        for key_block in range(query_block + 1):
            kj = slice(key_block * sub_chunk, (key_block + 1) * sub_chunk)
            k_scaled = k[..., kj, :].float() * torch.exp2(
                anchor[..., None, :] - gk[..., kj, :]
            )
            score = torch.matmul(q_scaled, k_scaled.transpose(-1, -2))
            k_score = torch.matmul(
                kq_scaled,
                k_scaled.transpose(-1, -2),
            ) * beta_query[..., None]
            if query_block == key_block:
                local_causal = torch.tril(
                    torch.ones(
                        (sub_chunk, sub_chunk),
                        dtype=torch.bool,
                        device=q.device,
                    )
                )
                score = torch.where(local_causal, score, 0.0)
                k_score = torch.where(
                    torch.tril(local_causal, diagonal=-1),
                    k_score,
                    0.0,
                )
            aqk_fp32[..., qi, kj] = score * float(scale)
            lower[..., qi, kj] = k_score

    aqk_fp32 = torch.where(causal, aqk_fp32, 0.0)
    lower = torch.where(strict, lower, 0.0)
    eye = torch.eye(
        chunk_size,
        device=q.device,
        dtype=torch.float32,
    )
    lhs = lower + eye
    lhs_flat = lhs.reshape(-1, chunk_size, chunk_size)
    rhs = eye.expand(lhs_flat.shape[0], chunk_size, chunk_size)
    inverse = torch.linalg.solve_triangular(
        lhs_flat,
        rhs,
        upper=False,
    ).reshape_as(lhs)

    aqk = aqk_fp32.to(dtype)
    akk = inverse.to(dtype)
    inverse_quantized = akk.float()
    beta_fp32 = beta.float()
    exp_g = torch.exp2(gk)
    w_seed = k.float() * beta_fp32[..., None] * exp_g
    u_seed = v.float() * beta_fp32[..., None]
    w = torch.matmul(inverse_quantized, w_seed).to(dtype)
    u = torch.matmul(inverse_quantized, u_seed).to(dtype)
    qg = (q.float() * exp_g).to(dtype)
    last_gate = gk[..., -1, :]
    kg = (
        k.float()
        * torch.exp2(last_gate[..., None, :] - gk)
    ).to(dtype)
    return aqk, akk, w, u, qg, kg


def _reference(torch, inputs, *, chunk_size, lower_bound, head_block):
    q = _l2norm(torch, inputs["q"])
    k = _l2norm(torch, inputs["k"])
    v = inputs["v"]
    beta = torch.sigmoid(inputs["beta"].float())
    gk = _chunk_gate_cumsum(
        torch,
        inputs["g"],
        inputs["A_log"],
        inputs["dt_bias"],
        chunk_size=chunk_size,
        lower_bound=lower_bound,
    )
    batch, seqlen, heads, dim = q.shape
    value_dim = v.shape[-1]
    num_chunks = seqlen // chunk_size
    scale = dim ** -0.5

    aqk = torch.zeros(
        (batch, seqlen, heads, chunk_size),
        device=q.device,
        dtype=q.dtype,
    )
    akk = torch.zeros_like(aqk)
    w = torch.empty_like(q)
    u = torch.empty_like(v)
    qg = torch.empty_like(q)
    kg = torch.empty_like(k)

    qh = _head_chunk_view(q, chunk_size=chunk_size)
    kh = _head_chunk_view(k, chunk_size=chunk_size)
    vh = _head_chunk_view(v, chunk_size=chunk_size)
    gh = _head_chunk_view(gk, chunk_size=chunk_size)
    bh = _head_chunk_view(beta[..., None], chunk_size=chunk_size)[..., 0]

    for h0 in range(0, heads, head_block):
        h1 = min(h0 + head_block, heads)
        block = _compute_intra_block(
            torch,
            qh[:, h0:h1],
            kh[:, h0:h1],
            vh[:, h0:h1],
            gh[:, h0:h1],
            bh[:, h0:h1],
            chunk_size=chunk_size,
            scale=scale,
        )
        for destination, value in zip((aqk, akk, w, u, qg, kg), block):
            _store_head_block(
                destination,
                value,
                h0,
                h1,
                chunk_size=chunk_size,
            )

    aqkh = _head_chunk_view(aqk, chunk_size=chunk_size)
    wh = _head_chunk_view(w, chunk_size=chunk_size)
    uh = _head_chunk_view(u, chunk_size=chunk_size)
    kgh = _head_chunk_view(kg, chunk_size=chunk_size)
    state = torch.zeros(
        (batch, heads, dim, value_dim),
        device=q.device,
        dtype=torch.float32,
    )
    attn_out = torch.empty_like(v)
    v_new = torch.empty_like(v)
    h = torch.empty(
        (batch, num_chunks, heads, value_dim, dim),
        device=q.device,
        dtype=q.dtype,
    )
    oh = _head_chunk_view(attn_out, chunk_size=chunk_size)
    vnh = _head_chunk_view(v_new, chunk_size=chunk_size)
    h_head = h.permute(0, 2, 1, 3, 4)

    for chunk in range(num_chunks):
        state_quantized = state.to(q.dtype).float()
        h_head[:, :, chunk].copy_(
            state_quantized.transpose(-1, -2).to(q.dtype)
        )
        v_new_fp32 = uh[:, :, chunk].float() - torch.matmul(
            wh[:, :, chunk].float(),
            state_quantized,
        )
        v_new_stored = v_new_fp32.to(v.dtype)
        vnh[:, :, chunk].copy_(v_new_stored)
        state = (
            state
            * torch.exp2(gh[:, :, chunk, -1]).unsqueeze(-1)
            + torch.matmul(
                kgh[:, :, chunk].float().transpose(-1, -2),
                v_new_fp32,
            )
        )
        qg_math = qh[:, :, chunk].float() * torch.exp2(
            gh[:, :, chunk]
        )
        inter = torch.matmul(qg_math, state_quantized) * float(scale)
        local = torch.matmul(
            aqkh[:, :, chunk].float(),
            v_new_stored.float(),
        )
        oh[:, :, chunk].copy_((inter + local).to(v.dtype))

    return {
        "attn_out": attn_out,
        "final_state": state.transpose(-1, -2).contiguous(),
        "gk": gk,
        "Aqk": aqk,
        "Akk": akk,
        "w": w,
        "u": u,
        "qg": qg,
        "kg": kg,
        "v_new": v_new,
        "h": h,
        "initial_state": None,
    }


def _metrics(torch, actual, expected, *, block_elements):
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(
            f"Shape mismatch: actual={tuple(actual.shape)}, "
            f"expected={tuple(expected.shape)}"
        )
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    count = actual_flat.numel()
    sums = {
        "abs": 0.0,
        "diff2": 0.0,
        "actual2": 0.0,
        "expected2": 0.0,
        "dot": 0.0,
    }
    max_abs = 0.0
    nonfinite_actual = 0
    nonfinite_expected = 0
    for begin in range(0, count, block_elements):
        end = min(begin + block_elements, count)
        actual_block = actual_flat[begin:end].float()
        expected_block = expected_flat[begin:end].float()
        nonfinite_actual += int((~torch.isfinite(actual_block)).sum().item())
        nonfinite_expected += int((~torch.isfinite(expected_block)).sum().item())
        diff = actual_block - expected_block
        abs_diff = diff.abs()
        max_abs = max(max_abs, float(abs_diff.max().item()))
        sums["abs"] += float(abs_diff.sum().item())
        sums["diff2"] += float((diff * diff).sum().item())
        sums["actual2"] += float((actual_block * actual_block).sum().item())
        sums["expected2"] += float((expected_block * expected_block).sum().item())
        sums["dot"] += float((actual_block * expected_block).sum().item())

    relative_l2 = math.sqrt(sums["diff2"]) / max(
        math.sqrt(sums["expected2"]),
        1e-30,
    )
    cosine = sums["dot"] / max(
        math.sqrt(sums["actual2"] * sums["expected2"]),
        1e-30,
    )
    return {
        "shape": list(actual.shape),
        "dtype_actual": str(actual.dtype).removeprefix("torch."),
        "dtype_expected": str(expected.dtype).removeprefix("torch."),
        "elements": count,
        "nonfinite_actual": nonfinite_actual,
        "nonfinite_expected": nonfinite_expected,
        "max_abs_diff": max_abs,
        "mean_abs_diff": sums["abs"] / count,
        "relative_l2": relative_l2,
        "cosine_similarity": cosine,
    }


def _save_samples(torch, output_dir, name, actual, expected, sample_count):
    count = actual.numel()
    selected = min(sample_count, count)
    indices = (
        torch.arange(selected, device=actual.device, dtype=torch.int64)
        * count
        // selected
    )
    real = actual.reshape(-1)[indices].float().cpu().numpy()
    golden = expected.reshape(-1)[indices].float().cpu().numpy()
    file_name = FILE_NAMES.get(name, name)
    real_dir = output_dir / "samples" / "real"
    expect_dir = output_dir / "samples" / "expect"
    real_dir.mkdir(parents=True, exist_ok=True)
    expect_dir.mkdir(parents=True, exist_ok=True)
    np.save(real_dir / f"{file_name}.npy", real)
    np.save(expect_dir / f"{file_name}.npy", golden)
    return {
        "sample_count": selected,
        "sampling": "uniform_flat_index",
        "real": f"samples/real/{file_name}.npy",
        "expect": f"samples/expect/{file_name}.npy",
    }


def _environment(torch, platform):
    result = {
        "platform": platform,
        "device": torch.npu.get_device_name(0),
        "torch": torch.__version__,
        "triton_ascend_kernels_commit": TRITON_ASCEND_KERNELS_COMMIT,
        "fla_org_reference_commit": FLA_ORG_COMMIT,
    }
    for package in ("torch-npu", "triton-ascend", "flash-linear-attention-npu"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _case(args):
    return {
        "B": args.batch,
        "S": args.sequence_length,
        "H": args.heads,
        "K": args.dim,
        "V": args.dim,
        "dtype_qkv": "bfloat16",
        "dtype_beta": "bfloat16",
        "dtype_g": "float32",
        "dtype_A_log": "float32",
        "dtype_dt_bias": "float32",
        "chunk_size": args.chunk_size,
        "initial_state": None,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "allow_neg_eigval": False,
        "safe_gate": True,
        "lower_bound": args.lower_bound,
        "state_v_first": True,
        "cu_seqlens": None,
        "disable_recompute": True,
    }


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_accuracy(torch, args, inputs):
    output_dir = Path(args.output_dir).resolve()
    torch.npu.reset_peak_memory_stats()
    dut, dut_ms = _elapsed_ms(
        torch,
        lambda: _run_dut(
            torch,
            inputs,
            chunk_size=args.chunk_size,
            lower_bound=args.lower_bound,
            capture_intermediates=True,
            mstx_range=None,
        ),
    )
    reference, reference_ms = _elapsed_ms(
        torch,
        lambda: _reference(
            torch,
            inputs,
            chunk_size=args.chunk_size,
            lower_bound=args.lower_bound,
            head_block=args.head_block,
        ),
    )
    dut_map = dict(zip(OUTPUT_NAMES, dut))
    metrics = {}
    samples = {}
    failures = {}
    for name in OUTPUT_NAMES:
        actual = dut_map[name]
        expected = reference[name]
        if actual is None or expected is None:
            if actual is not expected:
                failures[name] = "optional output mismatch"
            metrics[name] = {
                "actual": None if actual is None else "tensor",
                "expected": None if expected is None else "tensor",
                "status": "PASS" if actual is expected else "FAIL",
            }
            continue
        values = _metrics(
            torch,
            actual,
            expected,
            block_elements=args.metric_block_elements,
        )
        passed = (
            values["nonfinite_actual"] == 0
            and values["nonfinite_expected"] == 0
            and values["relative_l2"] <= args.max_relative_l2
            and values["cosine_similarity"] >= args.min_cosine
        )
        values["status"] = "PASS" if passed else "FAIL"
        metrics[name] = values
        samples[name] = _save_samples(
            torch,
            output_dir,
            name,
            actual,
            expected,
            args.sample_count,
        )
        if not passed:
            failures[name] = {
                "relative_l2": values["relative_l2"],
                "cosine_similarity": values["cosine_similarity"],
            }

    result = {
        "status": "PASS" if not failures else "FAIL",
        "environment": _environment(torch, args.platform),
        "case": _case(args),
        "seed": args.seed,
        "input_distribution": {
            "q_k_v_g_beta": "standard normal on NPU",
            "A_log_dt_bias": "uniform [0, 1) on NPU",
        },
        "reference": {
            "implementation": "independent batched PyTorch small-op KDA on NPU",
            "fla_org_commit": FLA_ORG_COMMIT,
            "l2norm_eps": 1e-6,
            "gate_scale": "1 / ln(2)",
            "state_accumulation_dtype": "float32",
            "state_output_layout": "[B,H,V,K]",
        },
        "thresholds": {
            "max_relative_l2": args.max_relative_l2,
            "min_cosine_similarity": args.min_cosine,
            "nonfinite": 0,
        },
        "timing_note": "NPU event timings are diagnostic only; use msopprof for performance conclusions.",
        "diagnostic_event_ms": {
            "dut_top_level": dut_ms,
            "torch_reference": reference_ms,
        },
        "peak_allocated_bytes": int(torch.npu.max_memory_allocated()),
        "metrics": metrics,
        "samples": samples,
        "failures": failures,
    }
    _write_json(output_dir / "accuracy_metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise AssertionError(f"Accuracy failures: {json.dumps(failures)}")


def run_profile(torch, args, inputs):
    torch.npu.reset_peak_memory_stats()

    def target():
        return _run_dut(
            torch,
            inputs,
            chunk_size=args.chunk_size,
            lower_bound=args.lower_bound,
            capture_intermediates=False,
            mstx_range=args.mstx_range,
        )

    _, elapsed_ms = _elapsed_ms(torch, target)
    result = {
        "status": "PASS",
        "environment": _environment(torch, args.platform),
        "case": _case(args),
        "seed": args.seed,
        "mstx_range": args.mstx_range,
        "diagnostic_event_ms": elapsed_ms,
        "peak_allocated_bytes": int(torch.npu.max_memory_allocated()),
        "note": "Use the enclosing msopprof BasicInfo range for the reported device time.",
    }
    if args.output_dir:
        _write_json(Path(args.output_dir).resolve() / "profile_run.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("accuracy", "profile"), required=True)
    parser.add_argument("--platform", choices=("A2", "A5"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=18432)
    parser.add_argument("--heads", type=int, default=96)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--lower-bound", type=float, default=-5.0)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--head-block", type=int, default=4)
    parser.add_argument("--sample-count", type=int, default=200000)
    parser.add_argument("--metric-block-elements", type=int, default=4_000_000)
    parser.add_argument("--max-relative-l2", type=float, default=1e-2)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--mstx-range", default="KDA_H96_FULL")
    args = parser.parse_args()

    if args.sequence_length % args.chunk_size:
        raise ValueError("sequence length must be divisible by chunk size")
    if args.dim != 128:
        raise ValueError("this archived validation script is fixed to K=V=128")

    import torch
    import torch_npu  # noqa: F401

    _configure_npu_math(torch)
    # Register the packaged custom OPP before the first NPU allocation. The
    # Ascend runtime does not reliably discover a newly added OPP after device
    # initialization.
    _load_ascendc_kda()
    inputs = _make_inputs(
        torch,
        batch=args.batch,
        seqlen=args.sequence_length,
        heads=args.heads,
        dim=args.dim,
        seed=args.seed,
    )
    torch.npu.synchronize()
    started = time.time()
    if args.mode == "accuracy":
        run_accuracy(torch, args, inputs)
    else:
        run_profile(torch, args, inputs)
    print(f"HOST_ELAPSED_SECONDS={time.time() - started:.3f}")


if __name__ == "__main__":
    main()
