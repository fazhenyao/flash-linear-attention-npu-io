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
    accuracy = {
        "a2": load("a2", "accuracy_metrics.json"),
        "a5": load("a5", "accuracy_metrics_fixed.json"),
    }
    a5_before_accuracy = load("a5", "accuracy_metrics.json")
    performance = {
        "a2": load("a2", "performance_summary.json"),
        "a5": load("a5", "performance_summary_fixed.json"),
    }
    a5_before_performance = load("a5", "performance_summary.json")
    scale_before = load("a5", "attn_out_scale_diagnostic.json")
    scale_fixed = load("a5", "attn_out_scale_diagnostic_fixed.json")

    lines = [
        "# v26.8.0 KDA 正向精度与性能测试报告",
        "",
        "> **结论：A2、A5 均通过。** A5 `ChunkKdaFwdFinalize` 的满 chunk 融合路径曾使"
        "`attn_out` 稳定放大近 2 倍；改为两个独立 FP32 Cube 结果写入 workspace、再由"
        "AIV 相加后，A2/A5 的 11 个张量输出全部满足精度标准。",
        "",
        "## 1. 归档信息",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        "| 测试日期 | 2026-07-31 |",
        "| 报告版本 | v26.8.0 |",
        "| 实现仓 PR | [flashserve/flash-linear-attention-npu#228]"
        "(https://github.com/flashserve/flash-linear-attention-npu/pull/228) |",
        "| PR #228 基线提交 | [`c25b0bca`](https://github.com/flashserve/flash-linear-attention-npu/commit/"
        "c25b0bca159d50c8b03ee0be443e6a6e54fcb07f) |",
        "| A5 修复提交 | [`8503735a`](https://github.com/weinachuan/flash-linear-attention-npu/"
        "commit/8503735a0768574f73321bf8af3c398ae59be0d9) |",
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
            "[A5 修复前 accuracy_metrics.json]"
            "(assets/kda_forward_h96/results/a5/accuracy_metrics.json)、"
            "[A5 修复后 accuracy_metrics_fixed.json]"
            "(assets/kda_forward_h96/results/a5/accuracy_metrics_fixed.json)。",
            "",
            "### 3.2 A5 `attn_out` 问题定位与修复",
            "",
            "| 指标 | 修复前 | 修复后 |",
            "|---|---:|---:|",
            f"| 20 万样本最小二乘倍率 | {scale_before['least_squares_scale']:.8f} | "
            f"{scale_fixed['least_squares_scale']:.8f} |",
            f"| `actual / expected` 中位数 | "
            f"{scale_before['actual_over_expected_quantiles']['p50']:.8f} | "
            f"{scale_fixed['actual_over_expected_quantiles']['p50']:.8f} |",
            f"| 全张量相对 L2 | "
            f"{a5_before_accuracy['metrics']['attn_out']['relative_l2']:.8f} | "
            f"{accuracy['a5']['metrics']['attn_out']['relative_l2']:.8f} |",
            f"| 全张量余弦相似度 | "
            f"{a5_before_accuracy['metrics']['attn_out']['cosine_similarity']:.8f} | "
            f"{accuracy['a5']['metrics']['attn_out']['cosine_similarity']:.8f} |",
            "",
            "修复前第 0 个 chunk 没有历史 state 项，输出仍接近 `2 * local`；将第二个 MMAD "
            "改为覆盖 L0C 后倍率恢复为 1，证明问题来自 A5 手写融合路径的 L0C 跨 MMAD "
            "累加，而不是 gate、state、布局转换或最终 BF16 Cast。最终实现仍将四个输入提前"
            "搬入 L1，但两个 MMAD 分别以 FP32 写入 state/local workspace，再由 AIV 做 FP32 "
            "加法。这样避开错误累加，同时保留输入 staging。",
            "",
            "修复后完整 H96 张量 `attn_out` 的相对 L2 为 "
            f"`{accuracy['a5']['metrics']['attn_out']['relative_l2']:.8f}`，余弦相似度为 "
            f"`{accuracy['a5']['metrics']['attn_out']['cosine_similarity']:.8f}`，"
            "11 个张量输出均 PASS。",
            "",
            "### 3.3 CT 可视化",
            "",
            "图中 Test/Real/NPU 表示 AscendC 输出，Golden/Expect/CPU 表示独立 FLA 语义"
            "标杆。每张图均匀抽样 200,000 点。`attn_out` 保留 A5 修复前后对比；其余 A5 "
            "列使用修复后的最终二进制结果。",
            "",
        ]
    )

    for display_name, file_name in OUTPUTS:
        if display_name == "attn_out":
            lines.extend(
                [
                    "<details><summary><strong>attn_out</strong></summary>",
                    "",
                    "| A2 | A5 修复前 | A5 修复后 |",
                    "|---|---|---|",
                    "| ![A2 attn_out](assets/kda_forward_h96/results/a2/ct_viz/"
                    "attn_out/attn_out_Standard.png) | "
                    "![A5 修复前 attn_out](assets/kda_forward_h96/results/a5/ct_viz/"
                    "attn_out/attn_out_Standard.png) | "
                    "![A5 修复后 attn_out](assets/kda_forward_h96/results/a5/ct_viz_fixed/"
                    "attn_out/attn_out_Standard.png) |",
                    "",
                    "</details>",
                    "",
                ]
            )
            continue
        lines.extend(
            [
                f"<details><summary><strong>{display_name}</strong></summary>",
                "",
                "| A2 | A5 |",
                "|---|---|",
                f"| ![A2 {display_name}](assets/kda_forward_h96/results/a2/ct_viz/"
                f"{file_name}/{file_name}_Standard.png) | "
                f"![A5 {display_name}](assets/kda_forward_h96/results/a5/ct_viz_fixed/"
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
            "| 阶段 | A2 (ms) | A5 修复前 (ms) | A5 修复后 (ms) | 修复后 A5 相对 A2 |",
            "|---|---:|---:|---:|---:|",
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
        a5_before = a5_before_performance["summary"][key]
        lines.append(
            f"| {display_name} | {ms(a2)} | {ms(a5_before)} | {ms(a5)} | "
            f"{a2 / a5:.3f}x |"
        )

    lines.extend(
        [
            "",
            "修复后 A5 `ChunkKdaFwdFinalize` 为 3.890 ms，较错误融合路径的 3.286 ms 增加 "
            "0.605 ms（18.41%）；KDA 核心由 32.617 ms 增至 33.243 ms（1.92%），完整公开"
            "语义由 53.390 ms 增至 53.840 ms（0.84%）。增加的主要工作是第二次 Fixpipe "
            "写入以及 AIV 对两个 FP32 workspace 的读取和相加。主耗时仍为 "
            "`ChunkKdaFwdPrepare`，其次为 `KdaGateCumsum`。",
            "",
            "修复后的 A5 `msopprof` 正常退出并完成 BasicInfo 解析。机器可读汇总："
            "[A2 performance_summary.json](assets/kda_forward_h96/results/a2/performance_summary.json)、"
            "[A5 修复前 performance_summary.json]"
            "(assets/kda_forward_h96/results/a5/performance_summary.json)、"
            "[A5 修复后 performance_summary_fixed.json]"
            "(assets/kda_forward_h96/results/a5/performance_summary_fixed.json)。",
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
            "| A5 | 11/11 张量输出通过 | 完整公开语义 53.840 ms；KDA 核心 33.243 ms | **通过** |",
            "",
            "A5 已使用同一输入、同一阈值完成全量精度、CT viz 和 `msopprof` 闭环；"
            "`attn_out` 近 2 倍问题消失，本 H96 正向用例满足 A2/A5 准入。",
            "",
            "## 7. 归档文件",
            "",
            "- [主测试脚本](assets/kda_forward_h96/kda_forward_h96.py)",
            "- [msopprof 汇总脚本](assets/kda_forward_h96/summarize_msopprof.py)",
            "- [倍率诊断脚本](assets/kda_forward_h96/diagnose_scale.py)",
            "- [A5 修复前 attn_out 倍率诊断]"
            "(assets/kda_forward_h96/results/a5/attn_out_scale_diagnostic.json)",
            "- [A5 修复后 attn_out 倍率诊断]"
            "(assets/kda_forward_h96/results/a5/attn_out_scale_diagnostic_fixed.json)",
            "- [A5 修复后精度结果]"
            "(assets/kda_forward_h96/results/a5/accuracy_metrics_fixed.json)",
            "- [A5 修复后性能结果]"
            "(assets/kda_forward_h96/results/a5/performance_summary_fixed.json)",
        ]
    )

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for _, file_name in OUTPUTS:
        for platform, directory in (("a2", "ct_viz"), ("a5", "ct_viz_fixed")):
            image = ROOT / "results" / platform / directory / file_name / f"{file_name}_Standard.png"
            if not image.is_file():
                raise FileNotFoundError(image)
    print(REPORT)


if __name__ == "__main__":
    main()
