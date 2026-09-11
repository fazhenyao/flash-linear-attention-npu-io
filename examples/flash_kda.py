# Copyright (c) 2026 Tianjin University, Ltd.

"""Train-time chunk KDA and complete KDA mixer examples on Ascend NPU."""

from __future__ import annotations

import argparse
import math
import os

# Register fla_npu's packaged OPP before torch_npu initializes the NPU runtime.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
        description="Run fused Ascend C KDA forward/backward or a complete KDA mixer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=("Fused training requires equal Q/K/value heads, K=V=128, "
                "chunk_size=64 and safe_gate=True. Initial/final state gradients "
                "are unsupported."),
    )
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32,
                        help="Legacy alias for --query-heads and --value-heads.")
    parser.add_argument("--query-heads", type=int, default=None)
    parser.add_argument("--value-heads", type=int, default=None)
    parser.add_argument("--tokens", type=int, default=65536)
    parser.add_argument("--key-dim", type=int, choices=(128,), default=128)
    parser.add_argument("--value-dim", type=int, choices=(128,), default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--scale", type=float, default=None,
                        help="Forward/backward score scale in both core and model mode; None means K**-0.5.")
    parser.add_argument("--chunk-size", type=int, choices=(64,), default=64,
                        help="Shared forward/backward chunk_size; fused backward supports only 64.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qk-l2norm", dest="qk_l2norm", action="store_true", default=True)
    parser.add_argument("--no-qk-l2norm", dest="qk_l2norm", action="store_false",
                        help="Disable Q/K normalization in the core smoke example.")
    parser.add_argument("--varlen", dest="varlen", action="store_true", default=True)
    parser.add_argument("--no-varlen", dest="varlen", action="store_false",
                        help="Use dense inputs when no explicit cu_seqlens are supplied.")
    parser.add_argument(
        "--cu-seqlens",
        default="",
        help="Comma-separated variable-length offsets, for example 0,64,128.",
    )
    parser.add_argument(
        "--mean-len",
        type=int,
        default=1024,
        help="Approximate sequence length when --varlen is used without explicit offsets.",
    )
    parser.add_argument("--lower-bound", type=float, default=-5.0)
    parser.add_argument("--demo-model", action="store_true")
    args = parser.parse_args()
    # Follow the GDR example: explicit dimensions take precedence over aliases.
    args.query_heads = args.heads if args.query_heads is None else args.query_heads
    args.value_heads = args.heads if args.value_heads is None else args.value_heads
    args.hidden_size = args.query_heads * args.key_dim
    # Explicit sequence boundaries select packed input even with --no-varlen.
    if args.cu_seqlens.strip():
        args.varlen = True
    return args


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch": args.batch,
        "tokens": args.tokens,
        "hidden_size": args.hidden_size,
        "query_heads": args.query_heads,
        "value_heads": args.value_heads,
        "key_dim": args.key_dim,
        "value_dim": args.value_dim,
        "mean_len": args.mean_len,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if args.query_heads != args.value_heads:
        raise ValueError(
            "fused KDA backward requires query_heads == "
            f"value_heads, got {args.query_heads} and {args.value_heads}"
        )
    if args.key_dim != 128 or args.value_dim != 128:
        raise ValueError("fused KDA backward requires key_dim == value_dim == 128")
    if args.query_heads > 128:
        raise ValueError("fused KDA forward requires at most 128 heads")
    if args.scale is not None and not math.isfinite(args.scale):
        raise ValueError("scale must be finite")
    if args.varlen and args.batch != 1:
        raise ValueError("variable-length input requires batch=1; use --no-varlen for dense B > 1")
    if args.demo_model and not args.qk_l2norm:
        raise ValueError("demo-model uses its built-in Q/K L2Norm; --no-qk-l2norm is core-only")
    if not -5.0 <= args.lower_bound < 0.0:
        raise ValueError("lower_bound must be in [-5, 0) when safe_gate is enabled")


def _parse_cu_seqlens(value: str, total_tokens: int) -> list[int] | None:
    if not value.strip():
        return None
    offsets = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(offsets) < 2:
        raise ValueError("cu_seqlens must contain at least two offsets")
    if offsets[0] != 0 or offsets[-1] != total_tokens:
        raise ValueError(
            f"cu_seqlens must start at 0 and end at tokens={total_tokens}, got {offsets}"
        )
    if any(left >= right for left, right in zip(offsets, offsets[1:])):
        raise ValueError(f"cu_seqlens must be strictly increasing, got {offsets}")
    return offsets


def _build_cu_seqlens(args: argparse.Namespace) -> list[int] | None:
    offsets = _parse_cu_seqlens(args.cu_seqlens, args.tokens)
    if offsets is not None or not args.varlen:
        return offsets
    offsets = list(range(0, args.tokens, args.mean_len))
    if not offsets or offsets[0] != 0:
        offsets.insert(0, 0)
    if offsets[-1] != args.tokens:
        offsets.append(args.tokens)
    return offsets


def _head_major(tensor: torch.Tensor, *, varlen: bool) -> torch.Tensor:
    # BSND/BSH -> BNSD/BHS, or packed NTD/NT with no batch dimension.
    tensor = tensor.transpose(1, 2).contiguous()
    return tensor.squeeze(0) if varlen else tensor


def _l2norm_fwd(x):
    x32 = x.float()
    rstd = torch.rsqrt(x32.square().sum(-1, keepdim=True) + 1e-6)
    return (x32 * rstd).to(x.dtype), rstd


def _l2norm_bwd(x, rstd, grad):
    # Use the original input to retain the FP32 normalization Jacobian, even
    # though the fused attention consumes normalized Q/K in BF16 or FP16.
    normalized = x.float() * rstd
    grad = grad.float()
    return rstd * (grad - normalized * (normalized * grad).sum(-1, keepdim=True))


class AscendCChunkKDAFunction(torch.autograd.Function):
    """Bind the fused Ascend C forward and backward using saved head-major tensors."""

    @staticmethod
    def forward(ctx, q, k, v, raw_gate, beta, A_log, dt_bias, scale,
                cu_seqlens_host, lower_bound, use_gate_in_kernel,
                use_qk_l2norm_in_kernel, use_beta_sigmoid_in_kernel, allow_neg_eigval,
                chunk_size, safe_gate, qkv_head_major):
        from fla_npu.ops.ascendc import chunk_kda_fwd

        ctx.set_materialize_grads(False)
        # The model passes convolution QKV directly in BNSD; the public adapter
        # still accepts BSND. Gate/beta projections use BSND/BSH in both paths.
        ctx.qkv_head_major = qkv_head_major
        if qkv_head_major:
            q, k, v = (tensor.contiguous() for tensor in (q, k, v))
        else:
            q, k, v = (_head_major(tensor, varlen=False) for tensor in (q, k, v))
        raw_gate, beta = (_head_major(tensor, varlen=False) for tensor in (raw_gate, beta))
        q_input, k_input, beta_input = q, k, beta
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = _l2norm_fwd(q)
            k, k_rstd = _l2norm_fwd(k)
        if use_beta_sigmoid_in_kernel:
            beta = torch.sigmoid(beta.float()) * (2.0 if allow_neg_eigval else 1.0)
        # Both fused kernels require BF16/FP32 gate and beta, including FP16 models.
        gate = raw_gate.float() if raw_gate.dtype == torch.float16 else raw_gate
        beta = beta.to(torch.float32 if beta_input.dtype == torch.float16 else beta_input.dtype)
        (out, _final_state, gk, Aqk, Akk, w, _u, qg, kg, v_new, h,
         _initial_state) = chunk_kda_fwd(
            q, k, v, gate, beta, scale, chunk_size,
            layout="BNSD", initial_state=None, output_final_state=False,
            cu_seqlens=cu_seqlens_host, chunk_indices=None,
            safe_gate=safe_gate, lower_bound=lower_bound, use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias.reshape(-1).contiguous() if dt_bias is not None else None,
            disable_recompute=True,
            return_intermediate_states=False, state_v_first=False,
        )
        # These eight tensors come from this forward invocation, not from
        # model projections or a separate reconstruction in backward.
        saved = (gk, Aqk, Akk, w, qg, kg, v_new, h)
        if any(tensor is None for tensor in saved):
            raise RuntimeError("chunk_kda_fwd did not return all fused backward intermediates")
        ctx.save_for_backward(q, k, v, gate, beta, A_log, dt_bias,
                              q_input, k_input, q_rstd, k_rstd, beta_input, *saved)
        ctx.gate_dtype = raw_gate.dtype
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        ctx.safe_gate = safe_gate
        ctx.cu_seqlens_host = cu_seqlens_host
        ctx.lower_bound = lower_bound
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        ctx.allow_neg_eigval = allow_neg_eigval
        return out.to(q.dtype), None

    @staticmethod
    def backward(ctx, do, dht):
        from fla_npu.ops.ascendc import chunk_kda_bwd

        if dht is not None:
            raise RuntimeError("Fused AscendC KDA backward does not support final-state gradients.")
        (q, k, v, raw_gate, beta, A_log, dt_bias, q_input, k_input,
         q_rstd, k_rstd, beta_input, gk, Aqk, Akk, w, qg, kg, v_new, h) = ctx.saved_tensors
        # q/k are the normalized forward inputs; beta is post-sigmoid.
        # do is supplied by autograd from the downstream norm/projection/loss.
        saved = (gk, Aqk, Akk, w, qg, kg, v_new, h)
        varlen = ctx.cu_seqlens_host is not None
        # Saved tensors already use head-major layout. Packed backward only
        # removes the singleton batch, including h: [1,Nc,H,K,V] -> [Nc,H,K,V].
        if varlen:
            saved = [tensor.squeeze(0) for tensor in saved]
        gk_head, Aqk_head, Akk_head, w_head, qg_head, kg_head, v_new_head, h_head = saved
        q_head, k_head, v_head, beta_head, gate_head = (
            tensor.squeeze(0) if varlen else tensor
            for tensor in (q, k, v, beta, raw_gate)
        )
        # Forward always returns BSND, even when its inputs are BNSD.
        # Only the downstream gradient needs a layout conversion here.
        do_head = _head_major(do, varlen=varlen)
        dq, dk, dv, dbeta, dg, _dh0, dA, dbias = chunk_kda_bwd(
            q_head, k_head, v_head, beta_head,
            gk_head, Aqk_head, Akk_head, w_head, qg_head, kg_head, v_new_head, h_head,
            do_head, ctx.scale,
            raw_g=gate_head if ctx.use_gate_in_kernel else None, A_log=A_log,
            dt_bias=(dt_bias.reshape(q.shape[1], q.shape[3]).contiguous()
                     if dt_bias is not None else None),
            initial_state=None, dht=None, cu_seqlens=ctx.cu_seqlens_host,
            chunk_indices=None, chunk_size=ctx.chunk_size, safe_gate=ctx.safe_gate,
            lower_bound=ctx.lower_bound, use_gate_in_kernel=ctx.use_gate_in_kernel,
            disable_recompute=True, use_exp2=True, state_v_first=False,
        )
        if varlen:
            dq, dk, dv, dg, dbeta = (grad.unsqueeze(0) for grad in (dq, dk, dv, dg, dbeta))
        # Match example_1.py: consume FP32 dq/dk/dbeta before the final casts.
        if ctx.use_qk_l2norm_in_kernel:
            dq = _l2norm_bwd(q_input, q_rstd, dq)
            dk = _l2norm_bwd(k_input, k_rstd, dk)
        if ctx.use_beta_sigmoid_in_kernel:
            sigmoid = torch.sigmoid(beta_input.float())
            dbeta = dbeta.float() * sigmoid * (1.0 - sigmoid)
            if ctx.allow_neg_eigval:
                dbeta = dbeta * 2.0
        # Cast while contiguous/head-major: casting a transposed view on NPU
        # can materialize the view and then restore its strides via transposes.
        dq, dk, dv = dq.to(q_input.dtype), dk.to(k_input.dtype), dv.to(v.dtype)
        dg, dbeta = dg.to(ctx.gate_dtype), dbeta.to(beta_input.dtype)
        # Convolution receives BNSD gradients directly. Only the public BSND
        # adapter restores QKV views; gate/beta return to their projections.
        if not ctx.qkv_head_major:
            dq, dk, dv = (grad.transpose(1, 2) for grad in (dq, dk, dv))
        dg, dbeta = dg.transpose(1, 2), dbeta.transpose(1, 2)
        return (dq, dk, dv, dg, dbeta,
                dA.to(A_log.dtype) if dA is not None else None,
                dbias.reshape_as(dt_bias).to(dt_bias.dtype) if dbias is not None else None,
                None, None, None, None, None, None, None, None, None, None)


@torch.compiler.disable
def chunk_kda(q, k, v, g, beta, *, A_log=None, dt_bias=None, scale=None,
              use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
              use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
              cu_seqlens=None, cu_seqlens_cpu=None, lower_bound=-5.0,
              chunk_size=64, safe_gate=True):
    """BSND/BSH training adapter; return BSND output and no final state.

    Input layout and preprocessing options match example_1.py. The demo model
    uses the private implementation to pass convolution QKV directly in BNSD.
    """
    return _chunk_kda_impl(
        q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias, scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval, cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu, lower_bound=lower_bound,
        chunk_size=chunk_size, safe_gate=safe_gate,
    )


@torch.compiler.disable
def _chunk_kda_impl(q, k, v, g, beta, *, A_log=None, dt_bias=None, scale=None,
              use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
              use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
              cu_seqlens=None, cu_seqlens_cpu=None, lower_bound=-5.0,
              chunk_size=64, safe_gate=True, _qkv_head_major=False):
    """Training subset of example_1.py; return (out, None) with fused backward.

    QKV use BNSD on the private model path; otherwise BSND. Gate/beta always
    use BSND/BSH and output uses BSND, matching the original caller API.
    The Function reuses BNSD/BHS inputs for both fused calls. QKV gradients
    match the selected internal path; gate/beta gradients use BSND/BSH views.
    With use_gate_in_kernel=True, g is the raw gate
    and A_log (FP32 [H]) is required; dt_bias (FP32 [H*K] or [H,K]) is optional.
    Otherwise g is the precomputed log-space decay. The L2Norm and beta sigmoid
    switches follow the reference API and use FP32 tensor math in the custom
    autograd Function, including backward before the final dtype casts. Beta is
    already activated unless use_beta_sigmoid_in_kernel=True; allow_neg_eigval
    then scales sigmoid to (0,2). Preprocessing switches default to False.

    Both forward and backward use K=V=128, equal heads, chunk_size=64,
    safe gate, zero initial state and saved intermediates (no recomputation).
    """
    if chunk_size != 64 or not safe_gate:
        raise ValueError("fused KDA backward requires chunk_size=64 and safe_gate=True")
    if q.ndim != 4 or q.shape != k.shape or v.shape != q.shape or q.shape[-1] != 128:
        raise ValueError("fused KDA requires rank-4 q/k/v of equal shape with K=V=128")
    gate_shape = (q.shape[0], q.shape[2], q.shape[1], q.shape[3]) if _qkv_head_major else q.shape
    if g.shape != gate_shape or beta.shape != gate_shape[:-1]:
        raise ValueError("g must use BSND and beta BSH matching QKV's batch/token/head dimensions")
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("allow_neg_eigval=True requires use_beta_sigmoid_in_kernel=True")
    if use_gate_in_kernel:
        if A_log is None:
            raise ValueError("A_log is required when use_gate_in_kernel=True")
        if A_log.dtype != torch.float32 or (dt_bias is not None and dt_bias.dtype != torch.float32):
            raise TypeError("A_log and dt_bias must use float32")
        heads, key_dim = gate_shape[-2:]
        if A_log.shape != (heads,):
            raise ValueError("A_log must have shape [H]")
        if dt_bias is not None and dt_bias.shape not in ((heads * key_dim,), (heads, key_dim)):
            raise ValueError("dt_bias must have shape [H*K] or [H,K]")
        if lower_bound is None or not -5.0 <= lower_bound < 0.0:
            raise ValueError("lower_bound must be in [-5, 0) for the fused safe gate")
    else:
        A_log, dt_bias = None, None

    metadata = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    if metadata is not None:
        if q.shape[0] != 1:
            raise ValueError("packed KDA input requires batch=1")
        if isinstance(metadata, torch.Tensor):
            metadata = metadata.detach().cpu().reshape(-1).tolist()
        metadata = tuple(int(value) for value in metadata)
    return AscendCChunkKDAFunction.apply(
        q, k, v, g, beta, A_log, dt_bias,
        q.shape[-1] ** -0.5 if scale is None else float(scale), metadata, lower_bound,
        use_gate_in_kernel, use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel, allow_neg_eigval, chunk_size, safe_gate, _qkv_head_major,
    )


class AscendCCausalConv1dFunction(torch.autograd.Function):
    """Training-mode causal conv with explicit Ascend C backward binding."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        head_num: int,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        from fla_npu.ops.ascendc import causal_conv1d

        op_weight = weight.transpose(0, 1).contiguous()
        width, feature_dim = op_weight.shape
        cu_list = None if cu_seqlens is None else cu_seqlens.detach().cpu().tolist()
        is_varlen = cu_list is not None
        op_x = x.reshape(-1, feature_dim).contiguous() if is_varlen else x.contiguous()
        sequence_count = len(cu_list) - 1 if cu_list is not None else int(x.shape[0])
        conv_states = torch.zeros(
            sequence_count,
            width - 1,
            feature_dim,
            dtype=x.dtype,
            device=x.device,
        )
        preactivation = causal_conv1d(
            op_x,
            op_weight,
            bias=bias,
            conv_states=conv_states,
            query_start_loc=cu_list,
            activation_mode=0,
            pad_slot_id=-1,
            run_mode=0,
            head_num=head_num,
        )
        if is_varlen:
            preactivation = preactivation.unsqueeze(0)

        ctx.save_for_backward(x, op_weight, preactivation)
        ctx.has_bias = bias is not None
        ctx.is_varlen = is_varlen
        ctx.query_start_loc = cu_list
        if is_varlen:
            ctx.input_layout = "TND" if head_num == 0 else "NTD"
        else:
            ctx.input_layout = "BSH" if head_num == 0 else "BNSD"
        return F.silu(preactivation)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        from fla_npu.ops.ascendc import causal_conv1d_bwd

        x, op_weight, preactivation = ctx.saved_tensors
        op_x = x.reshape(-1, x.shape[-1]).contiguous() if ctx.is_varlen else x.contiguous()
        op_grad = grad.squeeze(0).contiguous() if ctx.is_varlen else grad.contiguous()
        op_y = (
            preactivation.squeeze(0).contiguous()
            if ctx.is_varlen
            else preactivation.contiguous()
        )
        dx, dw, db, _ = causal_conv1d_bwd(
            x=op_x,
            y=op_y,
            weight=op_weight,
            dy=op_grad,
            initial_state=None,
            dht=None,
            query_start_loc=ctx.query_start_loc,
            activation=1,
            input_layout=ctx.input_layout,
        )
        return (
            dx.reshape_as(x),
            dw.transpose(0, 1).contiguous(),
            db if ctx.has_bias else None,
            None,
            None,
        )


def causal_conv1d_train(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    head_num: int,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    return AscendCCausalConv1dFunction.apply(
        x,
        weight,
        bias,
        head_num,
        cu_seqlens,
    )


class SigmoidGatedRMSNorm(nn.Module):
    def __init__(self, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        normalized = x.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + self.eps
        )
        normalized = normalized * self.weight.float()
        return (normalized * torch.sigmoid(gate.float())).to(input_dtype)


class DemoKimiDeltaAttention(nn.Module):
    """Complete train-time KDA mixer without the surrounding Transformer block."""

    def __init__(
        self,
        hidden_size: int,
        *,
        heads: int,
        key_dim: int,
        value_dim: int,
        use_short_conv: bool,
        conv_kernel: int,
        conv_bias: bool,
        lower_bound: float,
        allow_neg_eigval: bool,
        scale: float | None = None,
        chunk_size: int = 64,
        safe_gate: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.use_short_conv = use_short_conv
        self.conv_kernel_size = conv_kernel
        self.lower_bound = lower_bound
        self.allow_neg_eigval = allow_neg_eigval
        self.scale = key_dim**-0.5 if scale is None else float(scale)
        self.chunk_size = chunk_size
        self.safe_gate = safe_gate

        if key_dim != 128 or value_dim != 128:
            raise ValueError("fused KDA training requires key_dim == value_dim == 128")

        key_size = heads * key_dim
        value_size = heads * value_dim
        gate_size = heads * key_dim
        conv_size = 2 * key_size + value_size
        self.key_size = key_size
        self.value_size = value_size

        self.in_proj_qkv = nn.Linear(hidden_size, conv_size, bias=False)

        if use_short_conv:
            # Match DemoGatedDeltaNet: Conv1d owns the depthwise parameters;
            # execution still goes through the Ascend C training operators.
            self.conv1d = nn.Conv1d(
                in_channels=conv_size,
                out_channels=conv_size,
                kernel_size=self.conv_kernel_size,
                groups=conv_size,
                padding=self.conv_kernel_size - 1,
                bias=conv_bias,
            )

        self.in_proj_a = nn.Sequential(
            nn.Linear(hidden_size, value_dim, bias=False),
            nn.Linear(value_dim, gate_size, bias=False),
        )
        self.in_proj_b = nn.Linear(hidden_size, heads, bias=False)
        self.in_proj_z = nn.Sequential(
            nn.Linear(hidden_size, value_dim, bias=False),
            nn.Linear(value_dim, value_size, bias=True),
        )

        self.A_log = nn.Parameter(torch.zeros(heads, dtype=torch.float32))
        dt = torch.exp(
            torch.rand(gate_size, dtype=torch.float32)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        self.norm = SigmoidGatedRMSNorm(value_dim)
        self.out_proj = nn.Linear(value_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
        cu_seqlens_cpu: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, tokens, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(
            batch, tokens, self.heads, self.value_dim
        )
        # Match MindSpeed-MM Kimi-K3: the model supplies FP32 raw beta logits;
        # the KDA Function activates them and saves that same beta for backward.
        b = self.in_proj_b(hidden_states).float()
        a = self.in_proj_a(hidden_states)

        if self.use_short_conv:
            mixed_qkv = causal_conv1d_train(
                mixed_qkv,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                head_num=3 * self.heads,
                cu_seqlens=cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens,
            )
        else:
            mixed_qkv = _head_major(
                F.silu(mixed_qkv).reshape(batch, tokens, 3 * self.heads, self.key_dim),
                varlen=False,
            )

        query, key, value = torch.split(
            mixed_qkv,
            self.heads,
            dim=1,
        )
        # QKV stay BNSD from convolution through KDA forward/backward.
        # Projection branches keep their natural BSND/BSH layout.
        raw_gate = a.reshape(batch, tokens, self.heads, self.key_dim)
        core_out, _ = _chunk_kda_impl(
            _qkv_head_major=True,
            q=query,
            k=key,
            v=value,
            g=raw_gate,
            beta=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.scale,
            chunk_size=self.chunk_size,
            safe_gate=self.safe_gate,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            lower_bound=self.lower_bound,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=self.allow_neg_eigval,
        )
        core_out = self.norm(core_out, z)
        output = self.out_proj(core_out.reshape(batch, tokens, -1))
        return output


def _move_model_parameters(
    model: DemoKimiDeltaAttention,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    model.to(device=device)
    for name, parameter in model.named_parameters():
        if name not in {"A_log", "dt_bias"}:
            parameter.data = parameter.data.to(dtype=dtype)


def _print_grad(name: str, tensor: torch.Tensor) -> None:
    grad = tensor.grad
    if grad is None or not bool(torch.isfinite(grad.float()).all().item()):
        raise RuntimeError(f"{name} did not receive a finite gradient")
    print(
        f"{name}.grad:",
        "finite=True",
        f"norm={float(grad.float().norm().item()):.6g}",
    )


def _run_core(
    args: argparse.Namespace,
    *,
    device: torch.device,
    dtype: torch.dtype,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
) -> None:
    shape_k = (args.batch, args.tokens, args.query_heads, args.key_dim)
    shape_v = (args.batch, args.tokens, args.value_heads, args.value_dim)
    q = (torch.randn(shape_k, dtype=dtype, device=device) * 0.02).requires_grad_(True)
    k = (torch.randn(shape_k, dtype=dtype, device=device) * 0.02).requires_grad_(True)
    v = (torch.randn(shape_v, dtype=dtype, device=device) * 0.02).requires_grad_(True)
    raw_gate = (
        torch.randn(shape_k, dtype=dtype, device=device) * 0.02
    ).requires_grad_(True)
    raw_beta = torch.randn(
        args.batch,
        args.tokens,
        args.value_heads,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    A_log = torch.zeros(
        args.value_heads,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    dt_bias = torch.zeros(
        args.value_heads * args.key_dim,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    out, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=raw_gate,
        beta=raw_beta.float(),
        A_log=A_log,
        dt_bias=dt_bias,
        scale=args.scale if args.scale is not None else args.key_dim**-0.5,
        chunk_size=args.chunk_size,
        safe_gate=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        lower_bound=args.lower_bound,
        use_qk_l2norm_in_kernel=args.qk_l2norm,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=False,
    )
    # Exercise a nonzero VJP even in FP16: the tiny smoke inputs combined with
    # a mean-square loss can underflow d_out before it reaches the fused kernel.
    upstream_grad = torch.randn_like(out, dtype=torch.float32)
    loss = (out.float() * upstream_grad).sum()
    loss.backward()
    torch.npu.synchronize()

    print("forward:", tuple(out.shape), out.dtype, f"loss={float(loss.item()):.6g}")
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("raw_gate", raw_gate),
        ("raw_beta", raw_beta),
        ("A_log", A_log),
        ("dt_bias", dt_bias),
    ):
        _print_grad(name, tensor)


def _run_model(
    args: argparse.Namespace,
    *,
    device: torch.device,
    dtype: torch.dtype,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
) -> None:
    model = DemoKimiDeltaAttention(
        args.hidden_size,
        heads=args.query_heads,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        use_short_conv=True,
        conv_kernel=4,
        conv_bias=False,
        lower_bound=args.lower_bound,
        allow_neg_eigval=False,
        scale=args.scale,
        chunk_size=args.chunk_size,
        safe_gate=True,
    )
    _move_model_parameters(model, device=device, dtype=dtype)
    model.train()
    hidden_states = torch.randn(
        args.batch,
        args.tokens,
        args.hidden_size,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    output = model(
        hidden_states,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    loss = output.float().square().mean()
    loss.backward()
    torch.npu.synchronize()

    print(
        "model forward:",
        tuple(output.shape),
        output.dtype,
        f"loss={float(loss.item()):.6g}",
    )
    _print_grad("hidden_states", hidden_states)
    missing = []
    nonfinite = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(parameter.grad.float()).all().item()):
            nonfinite.append(name)
    print(
        "parameter gradients:",
        f"total={sum(1 for _ in model.parameters())}",
        f"missing={missing}",
        f"nonfinite={nonfinite}",
    )
    if missing or nonfinite:
        raise RuntimeError(
            "complete KDA mixer did not produce finite gradients for all parameters"
        )


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    import fla_npu.ops.ascendc  # noqa: F401
    import torch_npu

    device = torch.device(f"npu:{args.device}")
    torch_npu.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    cu_values = _build_cu_seqlens(args)
    cu_seqlens = (
        None
        if cu_values is None
        else torch.tensor(cu_values, dtype=torch.int64, device=device)
    )
    cu_seqlens_cpu = (
        None if cu_values is None else torch.tensor(cu_values, dtype=torch.int64)
    )
    sequence_count = args.batch if cu_values is None else len(cu_values) - 1

    print(
        "config:",
        f"mode={'model' if args.demo_model else 'core'}",
        f"B={args.batch}",
        f"T={args.tokens}",
        f"H={args.query_heads}",
        f"K={args.key_dim}",
        f"V={args.value_dim}",
        f"dtype={args.dtype}",
        f"varlen={args.varlen}",
        f"sequences={sequence_count}",
        f"scale={args.scale if args.scale is not None else args.key_dim**-0.5}",
        f"chunk_size={args.chunk_size}",
        "safe_gate=True",
        f"lower_bound={args.lower_bound}",
        f"qk_l2norm={args.qk_l2norm}",
        "backend=AscendC fused forward/backward",
    )
    if args.demo_model:
        _run_model(
            args,
            device=device,
            dtype=dtype,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
    else:
        _run_core(
            args,
            device=device,
            dtype=dtype,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )


if __name__ == "__main__":
    main()
