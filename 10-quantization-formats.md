# 量化格式详解

## 概述

LLM Compressor 支持多种量化数据格式，覆盖从 4-bit 到 8-bit 的整数和浮点格式。选择合适的格式需要考虑：硬件支持、精度需求、压缩比和推理性能。

## 整数格式

### INT8（8-bit 整数）

**范围**：
- 有符号：[-128, 127]
- 无符号：[0, 255]

**量化公式**：
$$q = \text{clamp}(\text{round}(x / s) + z, -128, 127)$$

**特点**：
- 所有现代 GPU 都支持 INT8 GEMM
- 对称量化常用于权重，非对称用于激活
- Per-channel 精度接近 FP16（损失 < 0.5%）

**在 LLM Compressor 中的 Scheme**：

| Scheme | 权重 | 激活 | 策略 |
|--------|------|------|------|
| `W8A8` | INT8 对称 per-channel | INT8 对称 per-token dynamic | 需校准 |
| `W8A16` | INT8 对称 per-channel | FP16 | 无需校准 |
| `W8A8_ASYM` | INT8 非对称 per-channel | INT8 非对称 per-token | 需校准 |

### INT4（4-bit 整数）

**范围**：
- 有符号：[-8, 7]
- 无符号：[0, 15]

**量化公式**（per-group, group_size=128）：
$$s_g = \frac{\max(|W_g|)}{7}, \quad q_i = \text{clamp}(\text{round}(w_i / s_g), -8, 7)$$

**特点**：
- 仅 16 个离散级别，精度有限
- 通常配合 per-group 策略（group_size=128）
- 需要 GPTQ/AWQ 等算法保证精度
- 约 3.5× 压缩（含 FP16 scale 存储开销）

**在 LLM Compressor 中的 Scheme**：

| Scheme | 权重 | 策略 | 典型算法 |
|--------|------|------|---------|
| `W4A16` | INT4 对称 per-group(128) | Weight-only | GPTQ/AWQ |
| `W4A16_ASYM` | INT4 非对称 per-group(128) | Weight-only | GPTQ/AWQ |
| `W4A8` | INT4 per-group + INT8 activation | Mixed | GPTQ |

## 浮点格式

### FP8 E4M3

**位分配**：1 sign + 4 exponent + 3 mantissa

**位级布局图**：

```
┌───┬───────────┬─────────┐
│ S │  E3 E2 E1 E0  │ M2 M1 M0 │
└───┴───────────┴─────────┘
 1b      4 bits       3 bits     = 8 bits total

S  = 符号位 (0=正, 1=负)
E  = 指数位 (偏移量 bias = 7, 实际指数 = E_val - 7)
M  = 尾数位 (隐含前导 1, 即 1.M2M1M0)
```

**编码规则**：
- 指数范围：E ∈ [1, 14]（0 和 15 为特殊值）
- E=0: 次正规数（denormal），值 = (-1)^S × 2^(-6) × 0.M2M1M0
- E=15: NaN（FP8 E4M3 没有 Inf，所有 E=15 均为 NaN）
- 正常值：value = (-1)^S × 2^(E-7) × 1.M2M1M0

**数值编码示例**：

```
示例 1：编码值 0.125
  0.125 = 2^(-3) × 1.0
  S=0, E = -3+7 = 4 (0100), M = 000
  二进制: 0 0100 000 = 0x20

示例 2：编码值 -1.5
  -1.5 = -1 × 2^0 × 1.5 = -1 × 2^0 × 1.100_2
  S=1, E = 0+7 = 7 (0111), M = 100
  二进制: 1 0111 100 = 0xBC

示例 3：编码值 448 (最大正常值)
  448 = 2^7 × 1.75 = 2^7 × 1.111_2
  S=0, E = 7+7 = 14 (1110), M = 111
  二进制: 0 1110 111 = 0x77 (即 FP8_E4M3_MAX)

示例 4：编码值 0.001953125 (最小正正常值)
  = 2^(-6) × 1.0
  S=0, E = -6+7 = 1 (0001), M = 000
  二进制: 0 0001 000 = 0x08
```

**范围**：±448（比 INT8 的 ±127 大得多）

**精度**：约 3.5 位有效十进制数字

**特点**：
- 对数间距：0 附近密集，远离 0 稀疏
- 天然适合神经网络权重分布
- Hopper (H100) GPU 原生 FP8 GEMM 支持
- 量化精度通常优于 INT8

### FP8 E5M2

**位分配**：1 sign + 5 exponent + 2 mantissa

**位级布局图**：

```
┌───┬──────────────┬──────┐
│ S │ E4 E3 E2 E1 E0 │ M1 M0 │
└───┴──────────────┴──────┘
 1b       5 bits       2 bits   = 8 bits total

S  = 符号位
E  = 指数位 (偏移量 bias = 15, 实际指数 = E_val - 15)
M  = 尾数位 (隐含前导 1, 即 1.M1M0)
```

**范围**：±57344（范围极大）

**精度**：约 2.5 位有效数字

**典型用途**：梯度（训练时），激活中的极端值

**FP8 E4M3 vs E5M2 对比**：

```
               E4M3                    E5M2
最大值:        448                     57344
最小正正常值:  2^(-6) = 0.015625      2^(-14) ≈ 6.1e-5
精度级别:      8 (尾数 3bit)           4 (尾数 2bit)
典型用途:      权重 + 激活             梯度
```

### FP8_DYNAMIC Scheme

```
权重：FP8 E4M3, per-channel 静态量化
激活：FP8 E4M3, per-token 动态量化（推理时计算 scale）
```

**量化过程**：

```python
# 权重量化（离线，静态）
weight_scale = max(|W[row,:]|) / 448.0  # per-channel
W_q = cast_to_fp8(W / weight_scale)

# 激活量化（在线，动态 per-token）
input_scale = max(|X[token,:]|) / 448.0  # per-token
X_q = cast_to_fp8(X / input_scale)
```

**数值编码完整示例（权重量化）**：

```
假设某 channel 的权重为: [0.32, -1.05, 2.88, -0.15, ...]
max(|W|) = 2.88
weight_scale = 2.88 / 448.0 = 0.006429

量化第一个元素 0.32:
  scaled = 0.32 / 0.006429 = 49.77
  FP8 E4M3 最近表示: 48.0 (= 2^5 × 1.5 → 0 1100 100)
  反量化: 48.0 × 0.006429 = 0.3086
  量化误差: |0.32 - 0.3086| = 0.0114

量化第二个元素 -1.05:
  scaled = -1.05 / 0.006429 = -163.3
  FP8 E4M3 最近表示: -160.0 (= -1 × 2^7 × 1.25 → 1 1110 010)
  反量化: -160.0 × 0.006429 = -1.0286
  量化误差: |(-1.05) - (-1.0286)| = 0.0214
```

**优势**：
- 无需校准数据（权重直接从值计算 scale）
- 激活动态量化适应任何输入
- 精度极高（通常 perplexity 增加 < 0.1）

### FP8_BLOCK Scheme

```
权重：FP8 E4M3, block-wise 量化（128×128 块）
激活：FP8 E4M3, block-wise 动态量化
```

**Block 量化**：将权重矩阵切分为 128×128 的块，每块独立量化：

$$s_{block} = \frac{\max(|W_{block}|)}{448.0}$$

**对比 FP8_DYNAMIC**：
- FP8_BLOCK 精度更高（更细粒度的 scale）
- FP8_BLOCK 兼容 DeepGEMM 内核（DeepSeek 风格）
- FP8_DYNAMIC 推理更简单

## NVIDIA 专有格式

### NVFP4（NVIDIA Blackwell FP4）

**位分配**：1 sign + 2 exponent + 1 mantissa（共 4-bit）

**位级布局图**：

```
┌───┬────────┬────┐
│ S │ E1  E0 │ M0 │
└───┴────────┴────┘
 1b   2 bits  1b   = 4 bits total

S  = 符号位 (0=正, 1=负)
E  = 指数位 (偏移量 bias = 1, 实际指数 = E_val - 1)
M  = 尾数位 (隐含前导 1, 即 1.M0)
```

**可表示的所有值（共 16 个）**：

```
E=0, M=0: ±0
E=0, M=1: ±0.5   (次正规: 0.1_2 × 2^0 = 0.5)
E=1, M=0: ±1.0   (1.0_2 × 2^0)
E=1, M=1: ±1.5   (1.1_2 × 2^0)
E=2, M=0: ±2.0   (1.0_2 × 2^1)
E=2, M=1: ±3.0   (1.1_2 × 2^1)
E=3, M=0: ±4.0   (1.0_2 × 2^2)
E=3, M=1: ±6.0   (1.1_2 × 2^2)

fp4_max = 6.0
```

**量化结构：两级缩放（Two-Level Scaling）**

```
Global Scale (FP32)           # 全局缩放因子
    └── Per-Group Scale (FP8)  # 每 16 个元素的局部缩放
        └── FP4 Data           # 4-bit 浮点数据
```

**两级缩放的详细机制**：

NVFP4 使用两级缩放结构来克服 4-bit 浮点数极其有限的动态范围（仅能表示 0~6 的几个值）。其核心思想是：

1. **Global Scale（全局缩放，FP32）**：一个标量，作用于整个权重张量。它负责将权重的整体范围映射到 local scale 可以处理的范围内。Global scale 的存在使得 local scale 可以用较低精度（FP8）存储而不溢出。

2. **Local Scale（局部缩放，FP8 E4M3）**：每 16 个元素共享一个局部缩放因子。它精细调整每个小组内部的值范围，确保映射到 FP4 的有限表示值（0, 0.5, 1, 1.5, 2, 3, 4, 6）时误差最小。

**为什么需要两级而非一级缩放？**

```
问题：如果只用一个 global scale（FP32），直接映射到 FP4:
  - FP4 最大值 = 6.0
  - 假设权重范围 [-0.5, 0.5]，global_scale = 0.5/6 = 0.0833
  - 但不同区域的权重密度不同，单一 scale 无法适应局部变化
  - 结果：某些区域量化误差极大

解决：两级缩放
  - global_scale 粗粒度调整整体范围
  - local_scale (per-16-elements) 细粒度适应局部分布
  - 内存开销：每 16 个 FP4 元素 + 1 个 FP8 scale = 4×16 + 8 = 72 bits
    → 平均每元素 4.5 bits（开销仅 12.5%）
```

**量化公式**：
$$\text{local\_scale}_g = \frac{\max(|W_g|)}{\text{global\_scale} \cdot \text{fp4\_max}}$$
$$W_{q,i} = \text{cast\_to\_fp4}(W_i / (\text{global\_scale} \cdot \text{local\_scale}_g))$$

**反量化公式**：
$$W_i \approx W_{q,i} \times \text{global\_scale} \times \text{local\_scale}_g$$

**完整数值编码示例**：

```
假设权重张量的 16 个元素（一个 group）:
W_group = [0.12, -0.35, 0.08, 0.45, -0.22, 0.31, -0.18, 0.05,
           0.28, -0.41, 0.15, -0.09, 0.33, -0.27, 0.19, -0.38]

步骤 1: 确定 global_scale（对整个张量计算一次）
  假设整个张量的统计分析得出 global_scale = 0.1（FP32）

步骤 2: 计算 local_scale（每 16 个元素一个）
  max(|W_group|) = 0.45
  local_scale = max(|W_group|) / (global_scale × fp4_max)
               = 0.45 / (0.1 × 6.0)
               = 0.75
  存储为 FP8 E4M3: 0.75 = 2^(-1) × 1.5 → 0 0110 100

步骤 3: 量化各元素到 FP4
  effective_scale = global_scale × local_scale = 0.1 × 0.75 = 0.075

  W[0] = 0.12:  0.12/0.075 = 1.6  → round to FP4: 1.5 (E=1,M=1)
  W[1] = -0.35: -0.35/0.075 = -4.67 → round to FP4: -4.0 (E=3,M=0)
  W[2] = 0.08:  0.08/0.075 = 1.07  → round to FP4: 1.0 (E=1,M=0)
  W[3] = 0.45:  0.45/0.075 = 6.0   → round to FP4: 6.0 (E=3,M=1)
  ...

步骤 4: 反量化验证
  W[0]_dequant = 1.5 × 0.075 = 0.1125  (原值 0.12, 误差 0.0075)
  W[1]_dequant = -4.0 × 0.075 = -0.30   (原值 -0.35, 误差 0.05)
  W[3]_dequant = 6.0 × 0.075 = 0.45     (原值 0.45, 误差 0)
```

**内存布局（实际存储格式）**：

```
对于 N 个元素的权重张量:
┌──────────────────────────────────────────────────┐
│ Global Scale: 1 × FP32 = 4 bytes                 │
├──────────────────────────────────────────────────┤
│ Local Scales: (N/16) × FP8 = N/16 bytes          │
├──────────────────────────────────────────────────┤
│ FP4 Data: N × 4bits = N/2 bytes                  │
└──────────────────────────────────────────────────┘

总存储: 4 + N/16 + N/2 bytes
每元素平均: ≈ 4.5 bits (当 N 很大时)
```

**特点**：
- group_size = 16（非常细粒度）
- global_scale 需要从校准数据确定（对于激活量化）
- 仅 NVIDIA Blackwell (SM100+) GPU 支持
- 4× 压缩比 + 硬件加速

**在 LLM Compressor 中的 Scheme**：

| Scheme | 权重 | 激活 | 说明 |
|--------|------|------|------|
| `NVFP4` | NVFP4 per-group(16) | NVFP4 dynamic | W4A4 全量化 |
| `NVFP4A16` | NVFP4 per-group(16) | FP16 | 仅权重量化 |

## OCP Microscaling 格式

### MXFP4（Microscaling FP4）

OCP (Open Compute Project) 定义的标准微缩放格式：

**位级布局图**：

```
MXFP4 元素格式 (E2M1):
┌───┬────────┬────┐
│ S │ E1  E0 │ M0 │
└───┴────────┴────┘
 1b   2 bits  1b   = 4 bits total

与 NVFP4 的 FP4 元素格式相同，但 scaling 机制不同！
```

**结构**：
```
Shared Exponent (E8M0, 8-bit)  # 每 32 个元素共享一个指数
    └── Element Data (FP4)      # E2M1 格式
```

**两级缩放结构（与 NVFP4 的对比）**：

```
┌─────────────────────────────────────────────────────────┐
│              NVFP4                    MXFP4              │
├─────────────────────────────────────────────────────────┤
│  Global Scale (FP32)          Shared Exponent (E8M0)    │
│      │                              │                    │
│  Local Scale (FP8, per-16)    无额外 local scale        │
│      │                              │                    │
│  FP4 Data (E2M1)              FP4 Data (E2M1)           │
├─────────────────────────────────────────────────────────┤
│  group_size = 16              group_size = 32            │
│  scale 精度: FP8 (多级)       scale 精度: E8M0 (仅指数)  │
│  需要校准数据                  无需校准数据               │
│  NVIDIA 专有                   OCP 开放标准               │
└─────────────────────────────────────────────────────────┘
```

**E8M0 Shared Exponent 详解**：

```
E8M0 格式: 8 位全部用于指数, 没有符号位和尾数位
┌──────────────────────────┐
│ E7 E6 E5 E4 E3 E2 E1 E0 │
└──────────────────────────┘
         8 bits

值 = 2^(E_val - 127)  (偏移量 bias=127，同 FP32 指数)
范围: 2^(-127) ~ 2^(128)

作用: 为 32 个 FP4 元素提供一个共享的缩放基准
     每个 FP4 元素的实际值 = FP4_val × 2^(shared_exp - 127)
```

**MXFP4 完整数值编码示例**：

```
假设 32 个元素的组:
W_group = [0.12, -0.35, 0.08, ..., -0.38]  (共32个)

步骤 1: 计算 shared exponent
  max(|W_group|) = 0.45
  需要找到 E 使得: 2^(E-127) × fp4_max >= 0.45
  fp4_max = 6.0
  2^(E-127) >= 0.45/6.0 = 0.075
  E-127 >= log2(0.075) = -3.74
  E >= 123.26 → E = 124
  shared_scale = 2^(124-127) = 2^(-3) = 0.125

步骤 2: 量化各元素
  W[0] = 0.12: 0.12/0.125 = 0.96 → FP4: 1.0 (E=1,M=0)
  W[1] = -0.35: -0.35/0.125 = -2.8 → FP4: -3.0 (E=2,M=1)
  ...

步骤 3: 反量化
  W[0]_dequant = 1.0 × 0.125 = 0.125 (原值 0.12, 误差 0.005)
  W[1]_dequant = -3.0 × 0.125 = -0.375 (原值 -0.35, 误差 0.025)
```

**特点**：
- group_size = 32（OCP 标准）
- 跨平台兼容（非 NVIDIA 专有）
- NVIDIA Blackwell GPU 支持
- 无需校准数据（RTN 即可）

**在 LLM Compressor 中的 Scheme**：

| Scheme | 权重 | 激活 | 说明 |
|--------|------|------|------|
| `MXFP4` | MXFP4 per-group(32) | MXFP4 dynamic | W4A4 |
| `MXFP4A16` | MXFP4 per-group(32) | FP16 | 仅权重 |

### MXFP8（Microscaling FP8）

**结构**：
```
Shared Exponent (E8M0, 8-bit)  # 每 32 个元素共享指数
    └── Element Data (FP8 E4M3) # 8-bit 浮点数据
```

**特点**：
- 比标准 FP8 更高精度（细粒度 shared exponent）
- group_size = 32
- 无需校准数据

**在 LLM Compressor 中的 Scheme**：

| Scheme | 权重 | 激活 | 说明 |
|--------|------|------|------|
| `MXFP8` | MXFP8 per-group(32) | MXFP8 dynamic | W8A8 微缩放 |
| `MXFP8A16` | MXFP8 per-group(32) | FP16 | 仅权重 |

## 格式对比总结

| 格式 | 比特数 | 压缩比 | 精度损失 | 需要校准 | 硬件要求 |
|------|:---:|:---:|------|:---:|------|
| FP8_DYNAMIC | 8 | 2× | 极小 (< 0.1%) | 否 | Hopper+ |
| FP8_BLOCK | 8 | 2× | 极小 | 否 | Hopper+ |
| INT8 (W8A8) | 8 | 2× | 小 (0.1-0.5%) | 是 | 所有 GPU |
| MXFP8 | 8 | 2× | 极小 | 否 | Blackwell |
| INT4 (W4A16) | 4 | ~3.5× | 中 (1-3%) | 是 | 所有 GPU |
| NVFP4 | 4 | 4× | 小-中 | 是* | Blackwell |
| MXFP4 | 4 | 4× | 中 | 否 | Blackwell |

> *NVFP4 权重量化无需校准，但激活 global_scale 需要

## 硬件兼容性

| GPU 架构 | INT8 | INT4 | FP8 | NVFP4 | MXFP4/8 |
|----------|:---:|:---:|:---:|:---:|:---:|
| Ampere (A100, RTX 3090) | ✅ | ✅ | ❌ | ❌ | ❌ |
| Hopper (H100, H200) | ✅ | ✅ | ✅ | ❌ | ❌ |
| Blackwell (B100, B200) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Ada (RTX 4090, L40S) | ✅ | ✅ | ✅ | ❌ | ❌ |

## vLLM 内核支持详情

不同量化格式在 vLLM 推理引擎中由不同的 CUDA 内核提供支持：

| 格式/Scheme | vLLM 内核 | 说明 |
|------------|-----------|------|
| FP8_DYNAMIC (W8A8) | `cutlass_scaled_mm` | Cutlass FP8 GEMM，per-token 动态量化 |
| FP8_BLOCK | `deepgemm` / `cutlass_block_scaled_mm` | DeepGEMM 128×128 block 内核 |
| INT8 (W8A8) | `cutlass_scaled_mm` (int8) | Cutlass INT8 GEMM |
| INT4 (W4A16) | `marlin` / `gptq_marlin` | Marlin 4-bit dequant + FP16 GEMM 融合内核 |
| INT4 (W4A8) | `machete` | Mixed INT4 权重 + INT8 激活内核 |
| NVFP4 | `cutlass_scaled_mm_nvfp4` | Blackwell 原生 FP4 Tensor Core 内核 |
| MXFP4 | `cutlass_mx_mm` | OCP MX 格式 Cutlass 内核 |
| MXFP8 | `cutlass_mx_mm` | OCP MX 格式 Cutlass 内核 |
| AWQ (W4A16) | `marlin` / `awq_marlin` | 与 GPTQ 共用 Marlin 内核 |

**Marlin 内核详情**：
- Marlin 是 vLLM 中用于 4-bit weight-only 量化的高性能内核
- 将 dequant（INT4→FP16）与 GEMM 融合为一次内核调用
- 支持 per-group scale（group_size=128）
- 吞吐量接近 FP16 GEMM（内存带宽约束场景下）
- 变体：`gptq_marlin`（GPTQ格式）、`awq_marlin`（AWQ格式）、`marlin_24`（2:4稀疏）

**DeepGEMM 内核详情**：
- 专为 DeepSeek 风格的 block-wise FP8 量化设计
- 每个 128×128 block 拥有独立的 FP32 scale
- 支持 Hopper+ 架构的 TMA (Tensor Memory Accelerator)
- 与 DeepSeek-V3/R1 的量化方案完全兼容

**Machete 内核详情**：
- 支持 W4A8 混合精度（权重 INT4 + 激活 INT8）
- 在 Hopper 架构上利用 INT8 Tensor Core
- 激活在线量化到 INT8 后与 INT4 权重进行混合精度 GEMM

## 格式选择决策树

```
你的 GPU 是什么架构？
│
├── Blackwell (B100/B200)
│   ├── 追求最大压缩 → NVFP4 (W4A4)
│   ├── 平衡精度与压缩 → FP8_BLOCK
│   └── 标准兼容 → MXFP4/MXFP8
│
├── Hopper (H100/H200)
│   ├── 快速部署 → FP8_DYNAMIC
│   ├── DeepSeek 风格 → FP8_BLOCK
│   └── 最大压缩 → INT4 (W4A16 GPTQ)
│
├── Ada (RTX 4090/L40S)
│   ├── 8-bit → FP8_DYNAMIC
│   └── 4-bit → INT4 (W4A16 GPTQ/AWQ)
│
└── Ampere (A100/RTX 3090)
    ├── 8-bit → INT8 (W8A8 + SmoothQuant)
    └── 4-bit → INT4 (W4A16 GPTQ/AWQ)
```

## 使用示例

### FP8 Dynamic

```python
recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
oneshot(model=model, recipe=recipe)  # 无需 dataset
```

### FP8 Block

```python
recipe = QuantizationModifier(targets="Linear", scheme="FP8_BLOCK", ignore=["lm_head", "re:.*mlp.gate$"])
oneshot(model=model, recipe=recipe)
```

### NVFP4

```python
recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head", "re:.*mlp.gate$"])
oneshot(model=model, recipe=recipe, dataset=dataset, num_calibration_samples=512)
```

### MXFP4

```python
recipe = QuantizationModifier(targets="Linear", scheme="MXFP4", ignore=["lm_head"])
oneshot(model=model, recipe=recipe)  # 无需 dataset
```

### W4A16 INT4 (GPTQ)

```python
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
oneshot(model=model, recipe=recipe, dataset=dataset, num_calibration_samples=512)
```
