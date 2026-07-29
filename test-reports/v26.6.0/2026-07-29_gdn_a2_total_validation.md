# v26.6.0 A2 GDN 总体验证报告

## 1. 归档信息

| 项目 | 内容 |
| --- | --- |
| 测试日期 | 2026-07-29 |
| 软件版本 | v26.6.0 |
| 实现仓 PR | [flashserve/flash-linear-attention-npu#249](https://github.com/flashserve/flash-linear-attention-npu/pull/249) |
| 被测提交 | [`9aa9b324`](https://github.com/weinachuan/flash-linear-attention-npu/commit/9aa9b324455efb33419fe682fe26d95f72423082) |
| 主要算子 | AscendC `solve_tri` |
| 端到端范围 | causal-conv 正反向、GDR 正反向、checkpoint recompute、纯异步多层串联 |
| 硬件范围 | A2 |

本报告使用 v26.6.0 最新源码本地构建的 wheel，不使用 release 中的预编译
wheel。报告中的性能结论来自 `msopprof BasicInfo`，不使用 Python wall
time。

## 2. 总结论

AscendC `solve_tri` 的 A2 64×64 FP32 kernel 完成稀疏块合并优化后：

| heads | 原 AscendC | 优化后 AscendC | 时延降低 | 加速比 |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 5581.43 us | 4442.20 us | 20.41% | 1.256x |
| 16 | 11285.32 us | 8871.40 us | 21.39% | 1.272x |

- 优化后的 AscendC 核心比 fla-org Triton-Ascend 对照实现快约 1.205x。
- 每个参与 AIC 的 FP32 workspace 从 192 KiB 降到 64 KiB，减少 66.7%。
- 所有中间矩阵乘保持原生 FP32，明确关闭 HF32。
- 单算子测试 15/15 通过。
- 原问题 `T=32768, H=16, BT=64` 单层正反向通过。
- 100 层 × 20 step 纯异步压力通过，输出、输入梯度、grad norm 和全部
  参数梯度逐 step 二进制一致。
- 四卡 TP/SP/CP、Qwen3.5/Qwen-Next/GVA、BF16/FP16 泛化用例通过。
- TND 上三角与短尾无效区保持为 0。

## 3. 环境与测试口径

| 项目 | 版本或配置 |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.10.0 |
| torch-npu | 2.10.0.post2 |
| triton-ascend | 3.2.1 |
| fla-npu | v26.6.0 源码构建 |
| 主测试 dtype | BF16 |
| 补充 dtype | FP16 |
| chunk size | 64 |
| 原问题 shape | `T=32768, H=16, K=V=128` |
| varlen metadata | 原始 64 段 `cu_seqlens` |

测试未设置 launch blocking、关闭 task queue 或其他强串行环境变量。
100 层压力脚本在层内、层间和 step 提交主体中不调用
`torch.npu.synchronize()`；全部 step 提交后才统一回读 finite 和确定性
结果。

## 4. 优化内容

### 4.1 原实现瓶颈

原 64×64 路径在一个 tile 内执行 10 次完整 `64×64×64` FP32 GEMM：

- 6 次用于四个 16×16 MCH 基块递推；
- 4 次用于 16→32 和 32→64 两级 MXR 合并。

MXR 辅助矩阵的大部分区域为零，但原实现仍执行完整 64×64 GEMM。

### 4.2 最终算法

最终方案保留 A2 上性能更好的连续 MCH GEMM，只优化两级 MXR 合并：

1. 16→32：将两组独立块打包，每阶段执行一次
   `32×16 @ 16×32` FP32 GEMM。
2. 32→64：每阶段只执行一次 `32×32 @ 32×32` FP32 GEMM。
3. AIV 只把有效下三角合并块写回运行中的逆矩阵。
4. workspace 使用四个连续 64×64 FP32 槽，分别保存运行逆矩阵、幂矩阵、
   GEMM 临时结果和 `-A`。

两级合并的 GEMM MAC 数从 `4 × 64³` 降为
`2 × (32×32×16) + 2 × 32³`，减少 90.625%；整个 64×64 solve 的
GEMM MAC 数减少 36.25%。

AIC/AIV ready/free 双向握手和 tile workspace 复用边界保持不变，没有
引入额外 host 同步或 device-to-host 回读。

### 4.3 未采用方案

以下方案在 A2、`T=32768, H=8, BT=64` 上性能更差，因此未采用：

| 方案 | SolveTri |
| --- | ---: |
| 仅 workspace stride 128→64 | 5513.63 us |
| MCH 拆为四次 16×16 GEMM | 6448.24 us |
| GM 拼块后执行 MCH | 7484.97 us |
| L1 直接拼块后执行 MCH | 6317.57 us |

结果表明 A2 上多个小 GEMM 的调用、拼块和搬运开销会抵消计算量收益；
稀疏优化应集中在两级 MXR 合并。

## 5. 适配层调用

GDR 适配脚本通过稳定入口调用：

```python
from fla_npu.ops.ascendc import npu_solve_tri
```

实际调用语义：

- KKT 输出 FP32 `A`；
- 适配层按模型 dtype 将 `A` 转为 BF16/FP16；
- AscendC kernel 内部提升为 FP32；
- 最终结果转换回输入 dtype；
- packed varlen 使用 TND；
- dense 输入保留 `[B,S,N,D]` 并使用 BSND。

`causal_conv1d` 和 `causal_conv1d_bwd` 由独立 autograd wrapper 调用，
正向启用 NTD 输出。层内和层间没有额外 device-to-host 同步。

## 6. 单算子与精度验证

### 6.1 单算子

`solve_tri` 共 15 个单算子用例，15/15 通过，覆盖：

- chunk size 16、32、64、128；
- FP16/BF16；
- MCH 大数值中间结果；
- BSND/BHTD/TND；
- 非 16 对齐尾块；
- TND 上三角和短尾无效区清零。

### 6.2 真实 KKT 输入

输入由 BF16 `k/g/beta` 经真实 `chunk_local_cumsum` 和
`chunk_scaled_dot_kkt_fwd(output_dtype=torch.float32)` 生成。AscendC
输出与 fla-org Triton-Ascend 参考结果对比如下：

| shape | 两端 finite | 最大绝对误差 | 相对 L2 |
| --- | --- | ---: | ---: |
| `T=32768, H=8, BT=64` | 是 | 0.0009765625 | 4.6058543e-5 |
| `T=32768, H=16, BT=64` | 是 | 0.0009765625 | 4.6225377e-5 |

## 7. 性能验证

只统计 SolveTri 核心，不包含 KKT、FP32→BF16 cast 和适配层无效区清零。
H=8 捕获 21 次、H=16 捕获 13 次，取全部 Task Duration 平均值。

| shape | fla-org Triton-Ascend | 原 AscendC | 优化后 AscendC |
| --- | ---: | ---: | ---: |
| `T=32768, H=8, BT=64` | 5350.82 us | 5581.43 us | 4442.20 us |
| `T=32768, H=16, BT=64` | 10692.24 us | 11285.32 us | 8871.40 us |

相对 fla-org，优化后 H=8/H=16 时延分别降低 16.98%/17.03%。

## 8. 原问题与压力验证

### 8.1 单层原问题

| 项目 | 结果 |
| --- | --- |
| 配置 | 1 层、2 step、BF16、关闭 checkpoint |
| shape | `T=32768, H=16, K=V=128, BT=64` |
| forward/backward | finite |
| grad norm | 两个 step 均为 `4.1270857` |
| 二进制确定性 | output、input grad、grad norm、全部参数梯度一致 |
| 峰值显存 | 4429.14 MiB |

### 8.2 100 层 × 20 step

| 项目 | 结果 |
| --- | --- |
| activation checkpoint | 启用 |
| causal-conv 正反向 | 启用，正向 NTD 输出 |
| GDR 正反向 | 启用 |
| step 主体冗余同步 | 无 |
| finite | 20/20 step 通过 |
| grad norm | 20 个 step 均为 `40.915497` |
| 二进制确定性 | output、input grad、grad norm、全部参数梯度逐 step 一致 |
| 峰值显存 | 21.219 GiB |
| 总结果 | PASS |

## 9. 四卡泛化验证

每个 case 对相同输入运行两次。确定性使用 `torch.equal` 比较输出和全部
训练梯度。shape 为单 rank 的
`T / key_heads / value_heads / key_dim / value_dim`。

| 场景 | 单 rank shape | dtype | finite | 确定性 | 峰值显存 |
| --- | --- | --- | --- | --- | ---: |
| Qwen3.5-4B TP4 训练 | 32768 / 4 / 8 / 128 / 128 | BF16 | 通过 | 通过 | 1694.09 MiB |
| Qwen3.5-4B SP4 训练 | 8192 / 16 / 32 / 128 / 128 | BF16 | 通过 | 通过 | 1692.53 MiB |
| Qwen3.5-35B-A3B TP4 训练 | 32768 / 4 / 8 / 128 / 128 | BF16 | 通过 | 通过 | 1694.09 MiB |
| Qwen3.5-35B-A3B CP4 训练 | 8192 / 16 / 32 / 128 / 128 | BF16 | 通过 | 通过 | 1692.55 MiB |
| Qwen3-Next TP4 prefill | 32768 / 4 / 8 / 128 / 128 | BF16 | 通过 | 通过 | 971.07 MiB |
| Qwen3-Next CP4 prefill | 8192 / 16 / 32 / 128 / 128 | BF16 | 通过 | 通过 | 969.53 MiB |
| GVA V=256 TP4 训练 | 16384 / 4 / 8 / 128 / 256 | BF16 | 通过 | 通过 | 1334.06 MiB |
| Qwen3.5-4B TP4 训练 | 32768 / 4 / 8 / 128 / 128 | FP16 | 通过 | 通过 | 1694.09 MiB |

本轮按模型不使用 `initial_state` 的前提执行。模型权重、optimizer state、
MoE 和 dense attention 不在适配层显存统计内。

## 10. 执行命令

### 10.1 源码构建

```bash
FLA_NPU_SOC=ascend910b \
python -m pip wheel --no-build-isolation --no-deps . -w dist

python -m pip install --force-reinstall --no-deps \
  dist/flash_linear_attention_npu-*.whl
```

### 10.2 单算子

```bash
python fla/ops/ascendc/gdn/recurrent_gdn/solve_tri/test/test.py
```

### 10.3 单层与 100 层压力

```bash
python examples/flash_gated_delta_rule_100layer_adapter_stress.py \
  --layers 1 --steps 2 --no-checkpoint --replay-step-inputs --md5

python examples/flash_gated_delta_rule_100layer_adapter_stress.py \
  --layers 100 --steps 20 --replay-step-inputs
```

### 10.4 四卡模型用例

```bash
python -m torch.distributed.run --nproc_per_node 4 \
  examples/chunk_gated_delta_rule_model_matrix.py \
  --case qwen35_35b_tp4_train \
  --determinism-runs 2
```

### 10.5 精度对照

```bash
export FLA_ORG_ROOT=/path/to/flash-linear-attention

python examples/solve_tri_backend_benchmark.py \
  --backend compare \
  --tokens 32768 \
  --heads 8 \
  --chunk-size 64 \
  --dtype bf16 \
  --a-dtype fp32 \
  --input-source kkt \
  --repeats 1
```

## 11. 未覆盖范围

- 本轮硬件闭环范围为 A2。
- A3/A5 未做本轮硬件实测，不将 A2 结果外推为其他平台结论。
- 本轮未构建 sanitizer 专用对象，因此内存部分仅报告单算子边界、
  无效区检查和端到端峰值显存，不作为 sanitizer 结论。
- 不覆盖模型 `initial_state` 路径。
