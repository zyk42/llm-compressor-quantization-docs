# RTN（Round-to-Nearest）最近舍入量化

## 算法概述

RTN 是最简单的量化方法——将每个权重值舍入到最近的量化级别。虽然简单，但对于 FP8 等高精度格式，RTN 已经能获得极佳效果。

## 算法原理

### 核心公式

RTN 的量化过程可以分解为三步：

**Step 1: 计算缩放因子**

对称量化：
$$s = \frac{\max(|W|)}{q_{max}}$$

非对称量化：
$$s = \frac{W_{max} - W_{min}}{q_{max} - q_{min}}, \quad z = q_{min} - \text{round}\left(\frac{W_{min}}{s}\right)$$

**Step 2: 量化**
$$W_q = \text{clamp}\left(\text{round}\left(\frac{W}{s}\right) + z,\ q_{min},\ q_{max}\right)$$

**Step 3: 反量化（推理时）**
$$\hat{W} = s \cdot (W_q - z)$$

### 为什么 FP8 + RTN 就足够了？

1. **FP8 E4M3 有 448 个不同值**：足以表示大多数权重分布
2. **浮点量化天然适配**：权重分布集中在 0 附近，FP8 的对数间距正好在 0 附近更密集
3. **误差极小**：FP8 的量化误差通常在 0.1% 以内

### RTN 在低比特下的局限

当降到 4-bit（仅 16 个量化级别）时：
- 舍入误差显著增大：$E[\epsilon^2] = s^2/12$ 中的 $s$ 变大
- 异常权重（outlier）的截断误差严重
- 各列误差独立累积，无补偿机制

这就是为什么 4-bit 量化通常需要 GPTQ/AWQ 等更精细的算法。

## 在 LLM Compressor 中的实现

### QuantizationModifier

RTN 量化通过 `QuantizationModifier` 实现，它是所有量化方法的基础类：

```python
# src/llmcompressor/modifiers/quantization/quantization/base.py

class QuantizationModifier(Modifier, QuantizationMixin):
    """
    标准后训练量化修改器。
    使用 Observer 收集统计量，然后计算 scale/zero_point。
    """
    
    def on_initialize(self, state, **kwargs):
        # 将量化配置应用到目标模块
        self.initialize_quantization(state.model)
    
    def on_start(self, state, **kwargs):
        # 开始校准：注册 Observer 钩子
        self.start_calibration(state.model)
    
    def on_event(self, state, event, **kwargs):
        if event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            # 观察权重，计算量化参数
            for module in quantized_modules:
                observe(module, "weight")       # 让 Observer 看到权重
                update_qparams(module, "weight") # 计算 scale/zp
    
    def on_end(self, state, **kwargs):
        # 冻结量化：移除 Observer，固定参数
        self.end_calibration(state.model)
```

### Observer 如何工作

以 MinMax Observer 为例：

```python
# src/llmcompressor/observers/min_max.py

class MemorylessMinMaxObserver(Observer):
    """只使用当前观测值的 min/max"""
    
    def update_statistics_from_observed(self, tensor):
        # 直接用当前 tensor 的 min/max
        self.min_vals = tensor.min(dim=-1).values
        self.max_vals = tensor.max(dim=-1).values
    
    def get_qparams(self):
        # 根据 min/max 计算 scale 和 zero_point
        return calculate_qparams(self.min_vals, self.max_vals, self.quantization_args)
```

### Data-Free 模式

当使用 FP8_DYNAMIC 等方案且无校准数据时：

1. 权重量化：直接从权重值计算 scale（`observe(weight)` → `update_qparams()`）
2. 激活量化：标记为 `dynamic=True`，在推理时动态计算 per-token scale
3. Pipeline 选择：自动使用 `DataFreePipeline`

## 使用示例

### 示例 1：FP8 动态量化（无需校准数据）

```python
from transformers import AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    dtype="auto",
)

# FP8 动态量化：权重静态 FP8，激活动态 per-token FP8
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

# 无需 dataset 参数 — 直接量化
oneshot(model=model, recipe=recipe)
model.save_pretrained("Llama-3-8B-FP8-Dynamic")
```

### 示例 2：FP8 Block 量化

```python
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_BLOCK",  # Block-wise FP8 (类似 DeepSeek 风格)
    ignore=["lm_head", "re:.*mlp.gate$"],  # 忽略门控层
)

oneshot(model=model, recipe=recipe)
```

### 示例 3：带校准的 W8A8 INT8 量化

```python
from datasets import load_dataset

# 准备校准数据
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

recipe = QuantizationModifier(
    targets="Linear",
    scheme="W8A8",      # INT8 权重 + INT8 激活
    ignore=["lm_head"],
)

oneshot(
    model=model,
    recipe=recipe,
    dataset=dataset,
    num_calibration_samples=512,
    max_seq_length=2048,
)
```

### 示例 4：W4A16 RTN 量化（精度较低，仅用于快速测试）

```python
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W4A16",     # 4-bit 权重，FP16 激活
    ignore=["lm_head"],
)

# 注意：4-bit RTN 精度不如 GPTQ/AWQ，生产环境建议使用后者
oneshot(model=model, recipe=recipe)
```

## 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `targets` | str/list | 量化目标层，通常为 `"Linear"` |
| `scheme` | str | 量化方案预设名（见下表） |
| `ignore` | list | 不量化的层，支持正则 `"re:..."` |
| `kv_cache_scheme` | dict | KV Cache 量化配置 |
| `observer` | str/dict | Observer 类型选择 |

### 常用 Scheme 预设

| Scheme | 权重 | 激活 | 策略 |
|--------|------|------|------|
| `FP8_DYNAMIC` | FP8 per-channel | FP8 per-token dynamic | 无需校准 |
| `FP8_BLOCK` | FP8 block (128×128) | FP8 block dynamic | 无需校准 |
| `W8A8` | INT8 per-channel | INT8 per-token dynamic | 需校准 |
| `W4A16` | INT4 per-group(128) | FP16 | 建议用 GPTQ |
| `W8A16` | INT8 per-channel | FP16 | 无需校准 |

## 适用场景

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 快速部署，精度要求高 | FP8_DYNAMIC | 几乎无损，无需校准 |
| DeepSeek 风格部署 | FP8_BLOCK | Block FP8 兼容 DeepGEMM |
| 快速验证量化效果 | W4A16 RTN | 速度最快，但精度有限 |
| 生产 4-bit 部署 | GPTQ/AWQ（非 RTN） | 4-bit RTN 精度不足 |
