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

**范围**：±448（比 INT8 的 ±127 大得多）

**精度**：约 3.5 位有效十进制数字

**特点**：
- 对数间距：0 附近密集，远离 0 稀疏
- 天然适合神经网络权重分布
- Hopper (H100) GPU 原生 FP8 GEMM 支持
- 量化精度通常优于 INT8

### FP8 E5M2

**位分配**：1 sign + 5 exponent + 2 mantissa

**范围**：±57344（范围极大）

**精度**：约 2.5 位有效数字

**典型用途**：梯度（训练时），激活中的极端值

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

**量化结构**：两级缩放

```
Global Scale (FP32)           # 全局缩放因子
    └── Per-Group Scale (FP8)  # 每 16 个元素的局部缩放
        └── FP4 Data           # 4-bit 浮点数据
```

**量化公式**：
$$\text{local\_scale}_g = \frac{\max(|W_g|)}{\text{global\_scale} \cdot \text{fp4\_max}}$$
$$W_{q,i} = \text{cast\_to\_fp4}(W_i / (\text{global\_scale} \cdot \text{local\_scale}_g))$$

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

**结构**：
```
Shared Exponent (E8M0, 8-bit)  # 每 32 个元素共享一个指数
    └── Element Data (FP4)      # E2M1 或 E3M0 格式
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
