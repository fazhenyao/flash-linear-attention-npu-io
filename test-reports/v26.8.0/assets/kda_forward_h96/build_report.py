#!/usr/bin/env python3
"""Build the archived KDA H96 validation report from machine-readable data."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT = ROOT.parents[1] / "2026-07-31_kda_forward_accuracy_validation.md"
OUTPUTS = (
    ("attn_out", "attn_out"),
    ("final_state", "final_state"),
    ("gk", "gk"),
    ("Aqk", "aqk"),
    ("Akk", "akk"),
    ("w", "w"),
    ("u", "u"),
    ("qg", "qg"),
    ("kg", "kg"),
    ("v_new", "v_new"),
    ("h", "h"),
)


def load(platform: str, file_name: str) -> dict:
    path = ROOT / "results" / platform / file_name
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: float) -> str:
    return f"{value:.8g}"


def ms(value_us: float) -> str:
    return f"{value_us / 1000:.3f}"


def main() -> None:
    accuracy = {platform: load(platform, "accuracy_metrics.json") for platform in ("a2", "a5")}
    performance = {
        platform: load(platform, "performance_summary.json")
        for platform in ("a2", "a5")
    }
    scale = load("a5", "attn_out_scale_diagnostic.json")

    lines = [
        "# v26.8.0 KDA 正向精度与性能测试报告",
        "",
        "> **结论：A2 通过；A5 不通过。** A2 的 11 个张量输出全部满足精度标准。"
        "A5 仅 `attn_out` 失败，最小二乘倍率为 1.99859，属于稳定的近 2 倍结构性错误；"
        "其余 10 个张量输出全部通过。",
        "",
        "## 1. 归档信息",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        "| 测试日期 | 2026-07-31 |",
        "| 报告版本 | v26.8.0 |",
        "| 实现仓 PR | [flashserve/flash-linear-attention-npu#228]"
        "(https://github.com/flashserve/flash-linear-attention-npu/pull/228) |",
        "| 被测提交 | [`c25b0bca`](https://github.com/flashserve/flash-linear-attention-npu/commit/"
        "c25b0bca159d50c8b03ee0be443e6a6e54fcb07f) |",
        "| FLA 语义基线 | [`0f0f0c97`](https://github.com/fla-org/flash-linear-attention/commit/"
        "0f0f0c97af39343855b43bbbaddcedfda5cb9d77) |",
        "| 模型适配参考 | [`triton-ascend-kernels@4cd4b506`](https://gitcode.com/Ascend/"
        "triton-ascend-kernels/blob/4cd4b506d4153ac18ac1ca8f4c770eac9fd3fcc8/src/"
        "triton_ascend_kernels/attention/fla/kda/cumsum_kda.py) |",
        "| 硬件范围 | A2、A5 |",
        "| 性能口径 | `msopprof BasicInfo` 的目标 device task duration 之和 |",
        "",
        "性能不使用 Python wall time。精度表统计完整张量的全部元素；CT 图使用固定种子从每个"
        "张量均匀抽样 200,000 个扁平索引，抽样只用于可视化，不参与精度判定。",
        "",
        "## 2. 测试用例与参数映射",
        "",
        "| 参数 | 本次取值 |",
        "|---|---|",
        "| q / k / v | `[1, 18432, 96, 128]`, BF16 |",
        "| beta | `[1, 18432, 96]`, BF16 raw logits |",
        "| g | `[1, 18432, 96, 128]`, FP32 raw gate input |",
        "| A_log | `[96]`, FP32 |",
        "| dt_bias | `[12288]`, FP32，按 `[96,128]` 解释 |",
        "| initial_state | `None` |",
        "| chunk_size | 64 |",
        "| output_final_state | `True` |",
        "| use_qk_l2norm_in_kernel | `True` |",
        "| use_gate_in_kernel | `True` |",
        "| use_beta_sigmoid_in_kernel | `True` |",
        "| safe_gate / lower_bound | `True` / `-5.0` |",
        "| transpose_state_layout | `True`，映射为 `state_v_first=True` |",
        "| cu_seqlens | `None`，dense BSND |",
        "| skip_recompute | `True`，映射为 `disable_recompute=True` |",
        "| scale | `1 / sqrt(128)` |",
        "| seed | `20260731` |",
        "",
        "用户给出的 `initial_final_state` 按功能语义映射为 `initial_state`。当前稳定 L2 接口不直接"
        "暴露 Q/K L2Norm 和 beta sigmoid 两个模型层布尔开关，因此测试适配层先执行",
        "",
        "$$\\hat q = q / \\sqrt{\\sum_d q_d^2 + 10^{-6}},\\qquad"
        "\\hat k = k / \\sqrt{\\sum_d k_d^2 + 10^{-6}},\\qquad"
        "\\hat\\beta = \\operatorname{sigmoid}(\\beta).$$",
        "",
        "raw `g/A_log/dt_bias` 仍交给 `chunk_kda_fwd` 的 `use_gate_in_kernel=True` 路径。"
        "`safe_gate=True` 对应",
        "",
        "$$g_{safe}=lower\\_bound\\cdot\\operatorname{sigmoid}"
        "\\left(\\exp(A_{log})\\cdot(g+dt\\_bias)\\right),\\qquad "
        "g_k=\\operatorname{cumsum}_{chunk}(g_{safe})/\\ln 2.$$",
        "",
        "固定版本的模型适配脚本尚未实现 beta 融合开关，而且其 Triton L2Norm 在 H96 长序列上"
        "会生成超过设备上限的 grid。因此本报告使用 NPU 上的 PyTorch 小算子构造完全相同的"
        "FLA 顶层语义，再调用 `fla_npu.ops.ascendc.chunk_kda_fwd`；没有缩小 shape、收窄"
        "输入范围或更改精度阈值。",
        "",
        "输入均在 NPU 上生成：`q/k/v/g/beta` 服从标准正态分布，`A_log/dt_bias` 服从"
        "`[0,1)` 均匀分布。",
        "",
        "## 3. 精度方法",
        "",
        "Golden 是按固定 FLA 提交语义实现的独立 batched PyTorch 小算子，并在 NPU 上以 FP32"
        "执行 gate、chunk 累计、矩阵求解和 state 累积。DUT 与 Golden 使用同一组输入。"
        "通过标准如下：",
        "",
        "- 两端 non-finite 数量均为 0；",
        "- 相对 L2 `<= 1e-2`；",
        "- 余弦相似度 `>= 0.9999`。",
        "",
        "### 3.1 全量输出指标",
        "",
        "| 输出 | shape / dtype | A2 max abs | A2 rel L2 | A2 cosine | A2 | "
        "A5 max abs | A5 rel L2 | A5 cosine | A5 |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]

    for display_name, metric_name in OUTPUTS:
        a2 = accuracy["a2"]["metrics"][display_name]
        a5 = accuracy["a5"]["metrics"][display_name]
        shape = "x".join(str(value) for value in a2["shape"])
        lines.append(
            f"| `{display_name}` | `{shape}` / `{a2['dtype_actual']}` | "
            f"{number(a2['max_abs_diff'])} | {number(a2['relative_l2'])} | "
            f"{number(a2['cosine_similarity'])} | **{a2['status']}** | "
            f"{number(a5['max_abs_diff'])} | {number(a5['relative_l2'])} | "
            f"{number(a5['cosine_similarity'])} | **{a5['status']}** |"
        )

    lines.extend(
        [
            "",
            "`initial_state` 在 DUT 和 Golden 两端均为 `None`，可选输出语义一致，因此记为"
            "PASS，但没有张量可生成 CT 图。所有实际张量输出的 non-finite 数量均为 0。",
            "",
            "机器可读结果："
            "[A2 accuracy_metrics.json](assets/kda_forward_h96/results/a2/accuracy_metrics.json)、"
            "[A5 accuracy_metrics.json](assets/kda_forward_h96/results/a5/accuracy_metrics.json)。",
            "",
            "### 3.2 A5 `attn_out` 结构性错误",
            "",
            f"- 20 万均匀样本最小二乘倍率：`{scale['least_squares_scale']:.8f}`；",
            f"- `actual / expected` 中位数："
            f"`{scale['actual_over_expected_quantiles']['p50']:.8f}`；",
            f"- 全张量相对 L2："
            f"`{accuracy['a5']['metrics']['attn_out']['relative_l2']:.8f}`；",
            f"- 全张量余弦相似度："
            f"`{accuracy['a5']['metrics']['attn_out']['cosine_similarity']:.8f}`。",
            "",
            "A5 CT 点云沿约 `y=2x` 分布。`final_state` 及 Finalize 之前公开保存的全部中间输出"
            "均通过，因此问题已收敛到 A5 `chunk_size=64` 满 chunk 的 "
            "`ChunkKdaFwdFinalize` 融合输出计算/写回路径；现有数据不能进一步证明是哪一条"
            "Cube 累加或写回指令造成倍率错误。",
            "",
            "### 3.3 CT 可视化",
            "",
            "图中 Test/Real/NPU 表示 PR #228 AscendC 输出，Golden/Expect/CPU 表示独立"
            "FLA 语义标杆。每张图均匀抽样 200,000 点。",
            "",
        ]
    )

    for display_name, file_name in OUTPUTS:
        lines.extend(
            [
                f"<details><summary><strong>{display_name}</strong></summary>",
                "",
                "| A2 | A5 |",
                "|---|---|",
                f"| ![A2 {display_name}](assets/kda_forward_h96/results/a2/ct_viz/"
                f"{file_name}/{file_name}_Standard.png) | "
                f"![A5 {display_name}](assets/kda_forward_h96/results/a5/ct_viz/"
                f"{file_name}/{file_name}_Standard.png) |",
                "",
                "</details>",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. 性能结果",
            "",
            "下表由一次 `msopprof BasicInfo` 采集的目标 kernel 数据汇总。随机输入生成 kernel "
            "全部排除。`KDA 核心合计` 是五个 AscendC 阶段之和；`完整公开语义合计` 还包含"
            "Q/K L2Norm、beta sigmoid 和 L2 边界所需布局转换。",
            "",
            "| 阶段 | A2 (ms) | A5 (ms) | A5 相对 A2 |",
            "|---|---:|---:|---:|",
        ]
    )

    perf_rows = (
        ("Q/K L2Norm", "qk_l2norm"),
        ("beta sigmoid", "beta_sigmoid"),
        ("布局转换", "layout_conversion"),
        ("KdaGateCumsum", "KdaGateCumsum"),
        ("ChunkKdaFwdPrepare", "ChunkKdaFwdPrepare"),
        ("ChunkKdaFwdPostWu", "ChunkKdaFwdPostWu"),
        ("ChunkGatedDeltaRuleFwdH", "ChunkGatedDeltaRuleFwdH"),
        ("ChunkKdaFwdFinalize", "ChunkKdaFwdFinalize"),
        ("**KDA 核心合计**", "kda_core_total"),
        ("**完整公开语义合计**", "public_semantics_total"),
    )
    for display_name, key in perf_rows:
        a2 = performance["a2"]["summary"][key]
        a5 = performance["a5"]["summary"][key]
        lines.append(
            f"| {display_name} | {ms(a2)} | {ms(a5)} | {a2 / a5:.3f}x |"
        )

    lines.extend(
        [
            "",
            "阶段占比显示两平台的核心主耗时均为 `ChunkKdaFwdPrepare`，其次是"
            "`KdaGateCumsum`。A2 的两者分别占 KDA 核心 42.73% 和 19.74%；A5 分别占"
            "38.47% 和 25.47%。完整公开语义中，预处理和布局转换占 A2 37.12%、A5 38.91%，"
            "说明性能优化不能只看五个 L0 kernel。",
            "",
            "A5 profiler 在完成全部目标 task 采集和 BasicInfo 解析后，于工具重放的末尾同步阶段"
            "报告设备错误；本报告只使用已经成功落盘并逐项解析的目标 task duration，不使用该次"
            "运行的 NPU event 或 host wall time。机器可读汇总："
            "[A2 performance_summary.json](assets/kda_forward_h96/results/a2/performance_summary.json)、"
            "[A5 performance_summary.json](assets/kda_forward_h96/results/a5/performance_summary.json)。",
            "",
            "## 5. 复现方法",
            "",
            "### 5.1 精度与采样",
            "",
            "```bash",
            "python assets/kda_forward_h96/kda_forward_h96.py \\",
            "  --mode accuracy \\",
            "  --platform A2 \\",
            "  --output-dir output/a2 \\",
            "  --batch 1 \\",
            "  --sequence-length 18432 \\",
            "  --heads 96 \\",
            "  --dim 128 \\",
            "  --chunk-size 64 \\",
            "  --lower-bound -5.0 \\",
            "  --seed 20260731",
            "```",
            "",
            "A5 将 `--platform A2` 改为 `--platform A5`。脚本会生成全量"
            "`accuracy_metrics.json` 和每个输出的 20 万点 `real/expect` NPY。",
            "",
            "### 5.2 CT viz",
            "",
            "```bash",
            "for name in attn_out final_state gk aqk akk w u qg kg v_new h; do",
            "  ct viz \\",
            "    output/a2/samples/real/${name}.npy \\",
            "    output/a2/samples/expect/${name}.npy \\",
            "    --out_dir output/a2/ct_viz/${name} \\",
            "    --name ${name} \\",
            "    --spatial \\",
            "    -sc 200000",
            "done",
            "```",
            "",
            "### 5.3 msopprof",
            "",
            "```bash",
            "msopprof \\",
            "  --application=\"python assets/kda_forward_h96/kda_forward_h96.py "
            "--mode profile --platform A2 --output-dir output/a2-profile "
            "--batch 1 --sequence-length 18432 --heads 96 --dim 128 --chunk-size 64 "
            "--lower-bound -5.0 --seed 20260731\" \\",
            "  --output=profile/a2 \\",
            "  --aic-metrics=BasicInfo \\",
            "  --launch-count=5000 \\",
            "  --warm-up=0 \\",
            "  --replay-mode=application \\",
            "  --kill=off",
            "",
            "python assets/kda_forward_h96/summarize_msopprof.py \\",
            "  profile/a2 \\",
            "  --platform A2 \\",
            "  --input-cast-count 4 \\",
            "  --output output/a2/performance_summary.json",
            "```",
            "",
            "A5 使用 `--platform A5 --input-cast-count 0`。`input-cast-count` 只用于排除"
            "PyTorch 随机输入生成阶段的 BF16 Cast，不影响目标算子分类。",
            "",
            "## 6. 最终判定",
            "",
            "| 平台 | 精度 | 性能数据 | 本用例准入 |",
            "|---|---|---|---|",
            "| A2 | 11/11 张量输出通过 | 完整公开语义 80.286 ms；KDA 核心 50.484 ms | **通过** |",
            "| A5 | 10/11 张量输出通过；`attn_out` 近 2 倍 | 完整公开语义 53.390 ms；KDA 核心 32.617 ms | **不通过** |",
            "",
            "A5 必须先修复 `ChunkKdaFwdFinalize` 的满 chunk 融合输出路径，再以同一输入、"
            "同一阈值重跑全量精度、CT viz 和 `msopprof`。在此之前，PR #228 的 A5 H96"
            "本用例不能给出精度通过结论。",
            "",
            "## 7. 归档文件",
            "",
            "- [主测试脚本](assets/kda_forward_h96/kda_forward_h96.py)",
            "- [msopprof 汇总脚本](assets/kda_forward_h96/summarize_msopprof.py)",
            "- [倍率诊断脚本](assets/kda_forward_h96/diagnose_scale.py)",
            "- [A5 attn_out 倍率诊断]"
            "(assets/kda_forward_h96/results/a5/attn_out_scale_diagnostic.json)",
        ]
    )

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for _, file_name in OUTPUTS:
        for platform in ("a2", "a5"):
            image = (
                ROOT
                / "results"
                / platform
                / "ct_viz"
                / file_name
                / f"{file_name}_Standard.png"
            )
            if not image.is_file():
                raise FileNotFoundError(image)
    print(REPORT)


if __name__ == "__main__":
    main()
