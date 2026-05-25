# KV Cache 量化

## 背景：KV Cache 的内存问题

在 LLM 推理中，KV Cache 存储每个 token 的 Key 和 Value 状态，随着序列长度增长线性增加。对于长序列场景，KV Cache 可能成为主要内存瓶颈。

### 内存占用分析

单层 KV Cache 的内存：

$$\text{Memory} = 2 \times \text{batch} \times \text{seq\_len} \times \text{num\_heads} \times \text{head\_dim} \times \text{bytes\_per\_element}$$

以 Llama-3-70B 为例（80 层，64 头，head_dim=128）：

| 序列长度 | FP16 KV Cache | FP8 KV Cache | 节省 |
|----------|:---:|:---:|:---:|
| 2K | 2.5 GB | 1.25 GB | 50% |
| 8K | 10 GB | 5 GB | 50% |
| 32K | 40 GB | 20 GB | 50% |
| 128K | 160 GB | 80 GB | 50% |

KV Cache 量化到 FP8 可以将缓存占用减半，等效于将可服务的序列长度翻倍。

## 量化原理

### 目标模块

KV Cache 量化作用于 Attention 层的 Key 和 Value 输出：

```
q_proj → Q states（Query，每次推理重新计算）
k_proj → K states → 存入 KV Cache ← 量化目标
v_proj → V states → 存入 KV Cache ← 量化目标
```

在 LLM Compressor 中，KV Cache 量化通过对 `q_proj` 和 `k_proj` 的**输出**施加量化实现（注：虽然名字是 q_proj，但这里量化的是输出激活，即 K 和 V 的缓存值）。

### 量化策略

**Per-Tensor（默认）**：
- 整个 KV Cache 共享一个 scale
- 最简单，推理开销最小
- 精度可能不足（不同头的分布差异大）

**Per-Head**：
- 每个注意力头独立量化
- 精度更高（适应各头不同的值域）
- vLLM 支持 per-head FP8 KV Cache

### 与权重量化的区别

| | 权重量化 | KV Cache 量化 |
|--|---------|--------------|
| 量化对象 | 模型权重（静态） | 中间激活（动态） |
| 量化时机 | 离线一次性 | 推理时每次存入缓存 |
| scale 计算 | 校准时确定 | 动态 或 校准时确定 |
| 存储位置 | SafeTensors 文件 | GPU 显存中的缓存 |

## 在 LLM Compressor 中的配置

### kv_cache_scheme 参数

KV Cache 量化通过 `QuantizationModifier` 的 `kv_cache_scheme` 参数配置：

```python
from compressed_tensors.quantization import QuantizationArgs

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
    kv_cache_scheme=QuantizationArgs(
        num_bits=8,
        type="float",           # FP8
        strategy="tensor",      # per-tensor
        symmetric=True,
    ),
)
```

### 配置选项

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `num_bits` | 8 | 量化位数 |
| `type` | "float" / "int" | 数据类型 |
| `strategy` | "tensor" / "head" | 量化粒度 |
| `symmetric` | True / False | 对称/非对称 |
| `dynamic` | True / False | 是否动态计算 scale |

## 使用示例

### 示例 1：FP8 KV Cache（Per-Tensor）

```python
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationArgs

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
    kv_cache_scheme=QuantizationArgs(
        num_bits=8,
        type="float",
        strategy="tensor",
        symmetric=True,
    ),
)

oneshot(
    model=model,
    recipe=recipe,
    dataset=dataset,
    num_calibration_samples=512,
    max_seq_length=2048,
)

model.save_pretrained("Llama-3-8B-FP8-KV")
```

### 示例 2：Per-Head KV Cache 量化

```python
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
    kv_cache_scheme=QuantizationArgs(
        num_bits=8,
        type="float",
        strategy="head",        # per-head 量化
        symmetric=True,
    ),
)
```

### 示例 3：权重量化 + KV Cache 量化组合

```python
from llmcompressor.modifiers.gptq import GPTQModifier

# GPTQ W4A16 + FP8 KV Cache
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
    kv_cache_scheme=QuantizationArgs(
        num_bits=8,
        type="float",
        strategy="tensor",
        symmetric=True,
    ),
)

oneshot(model=model, recipe=recipe, dataset=dataset, num_calibration_samples=512)
```

### 示例 4：YAML 配方格式

```yaml
quantization_stage:
  quant_modifiers:
    QuantizationModifier:
      targets: "Linear"
      scheme: "FP8_DYNAMIC"
      ignore: ["lm_head"]
      kv_cache_scheme:
        num_bits: 8
        type: float
        strategy: tensor
        symmetric: true
```

## vLLM 推理配置

量化后的模型在 vLLM 中使用 KV Cache 量化：

```python
from vllm import LLM

# 加载带 KV Cache 量化的模型
llm = LLM(
    model="Llama-3-8B-FP8-KV",
    kv_cache_dtype="fp8",      # 启用 FP8 KV Cache
    # 或 kv_cache_dtype="fp8_e4m3"
)
```

**注意**：vLLM 会自动从模型的 `config.json` 中读取 KV Cache 量化参数（scale 等）。

## 精度影响

KV Cache 量化对模型精度的影响通常很小：

| 方案 | Perplexity 增加 | 下游任务影响 |
|------|:---:|------|
| FP8 per-tensor | +0.01 ~ +0.05 | 几乎无影响 |
| FP8 per-head | +0.005 ~ +0.02 | 几乎无影响 |
| INT8 per-tensor | +0.05 ~ +0.1 | 极小影响 |

**原因**：
1. KV Cache 中的值通常范围有限（softmax 后的注意力权重限制了 V 的使用方式）
2. FP8 的精度对于 KV 值已经足够
3. 量化误差在多头注意力中被平均

## 最佳实践

1. **首选 FP8 per-tensor**：精度损失极小，推理开销最低
2. **长序列场景必备**：序列长度 > 8K 时，KV Cache 量化收益明显
3. **与权重量化正交**：可以同时使用 W4A16 + FP8 KV Cache
4. **Per-head 用于高要求场景**：如果 per-tensor 精度不够，升级到 per-head
5. **校准数据使用长序列**：确保校准数据包含长序列样本
