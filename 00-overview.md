# LLM Compressor 总览与快速入门

## 项目简介

`llm-compressor` 是 vLLM 项目提供的大语言模型后训练压缩库，专注于将 LLM 量化为可在 vLLM 高效推理的格式。项目核心特性：

- **全面的量化算法**：RTN、GPTQ、AWQ、AutoRound、SmoothQuant、SpinQuant、QuIP
- **丰富的量化格式**：INT4/INT8、FP8（Dynamic/Block）、NVFP4、MXFP4/MXFP8
- **多维度量化**：权重、激活、KV Cache、Attention 量化
- **大模型支持**：DDP 分布式、Sequential Onloading、磁盘卸载，支持 675B+ 参数模型
- **无缝集成**：与 Hugging Face 生态无缝衔接，输出 `compressed-tensors` 格式直接兼容 vLLM

## 安装

```bash
pip install llmcompressor
```

从源码安装（获取最新特性）：

```bash
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor
pip install -e .
```

## 30 秒快速上手

以下示例将 Llama-3 模型量化为 FP8 格式（无需校准数据）：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# 加载模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", dtype="auto")

# 配置量化：FP8 动态量化（无需校准数据）
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

# 一键量化
oneshot(model=model, recipe=recipe)

# 保存量化模型
model.save_pretrained("Llama-3-8B-FP8")
```

## 支持的量化精度

| 类型 | 格式 | 说明 |
|------|------|------|
| 权重+激活量化 | W8A8 (INT8/FP8) | 8位权重 + 8位激活 |
| 权重+激活量化 | W4A4 (NVFP4/MXFP4) | 4位权重 + 4位激活 |
| 权重+激活量化 | W4A8 (FP8) | 4位权重 + 8位激活 |
| 仅权重量化 | W4A16 (INT4) | 4位权重，激活保持 FP16 |
| 仅权重量化 | W8A16 | 8位权重，激活保持 FP16 |
| 微缩放格式 | MXFP4/MXFP8 | OCP 标准微缩放浮点 |
| KV Cache | FP8 | KV 缓存 FP8 量化 |
| Attention | FP8/NVFP4 | 注意力层量化 |

## 支持的量化算法

| 算法 | 类型 | 是否需要校准数据 | 适用场景 |
|------|------|:---:|------|
| **RTN** (Round-to-Nearest) | 最近舍入 | 否* | 快速量化，FP8 精度损失小 |
| **GPTQ** | 基于 Hessian 的权重优化 | 是 | W4A16 高精度，主流选择 |
| **AWQ** | 激活感知缩放 | 是 | W4A16 高精度，与 GPTQ 互补 |
| **AutoRound** | 梯度优化舍入 | 是 | 高精度，支持多种格式 |
| **SmoothQuant** | 激活平滑变换 | 是 | W8A8 前置处理 |
| **SpinQuant** | 旋转变换 | 否 | 降低权重不相干性 |
| **QuIP** | Hadamard 旋转 | 否 | 降低权重不相干性 |
| **IMatrix** | 重要性加权 | 是 | 增强其他算法精度 |

> *注：RTN 用于 FP8 时无需校准数据，用于 INT8/INT4 时建议使用校准数据。

## 量化方法选择指南

```
需要量化 LLM？
│
├── 追求速度，精度损失可接受？
│   └── FP8_DYNAMIC (RTN, 无需校准数据)
│
├── 需要 4-bit 压缩？
│   ├── 有 Blackwell GPU？
│   │   └── NVFP4 (W4A4)
│   ├── 需要最高精度？
│   │   └── GPTQ 或 AWQ (W4A16)
│   └── 需要快速量化？
│       └── RTN + MXFP4
│
├── 需要 W8A8 整型量化？
│   └── SmoothQuant + GPTQ (INT8)
│
└── 模型非常大（70B+）？
    ├── 单卡 → Sequential Onloading
    ├── 多卡 → DDP 分布式
    └── 无 HF 定义 → model_free_ptq
```

## 文档导航

| 文档 | 内容 |
|------|------|
| [01-项目架构](01-architecture.md) | Modifier、Recipe、Pipeline、Observer 系统解析 |
| [02-量化基础理论](02-quantization-fundamentals.md) | 量化数学基础、误差分析 |
| [03-RTN 量化](03-rtn-quantization.md) | 最近舍入量化原理与实践 |
| [04-GPTQ](04-gptq.md) | 基于 Hessian 的最优量化 |
| [05-AWQ](05-awq.md) | 激活感知权重量化 |
| [06-AutoRound](06-autoround.md) | 梯度优化舍入 |
| [07-SmoothQuant](07-smoothquant.md) | 激活平滑变换 |
| [08-旋转量化](08-rotation-quantization.md) | SpinQuant & QuIP |
| [09-IMatrix](09-imatrix.md) | 重要性加权校准 |
| [10-量化格式](10-quantization-formats.md) | FP8/NVFP4/MXFP/INT 格式详解 |
| [11-KV Cache 量化](11-kv-cache-quantization.md) | KV 缓存量化 |
| [12-高级实践](12-advanced-practice.md) | 大模型、多模态、MoE、组合配方 |
| [13-问题排查](13-troubleshooting.md) | 常见问题与最佳实践 |
