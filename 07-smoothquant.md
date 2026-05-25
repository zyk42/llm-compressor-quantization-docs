# SmoothQuant 详解

## 算法概述

SmoothQuant 解决的核心问题：LLM 中的**激活存在极端离群值**（outlier channels），导致 W8A8 量化时激活量化误差很大。SmoothQuant 通过通道级等效变换，将激活的量化难度"平滑"转移到权重上，使得权重和激活都变得易于量化。

**核心思想**：激活难量化（有离群值），权重好量化（分布均匀）。通过数学等效变换，让两者的量化难度趋于平衡。

## 理论基础

### 问题：激活离群值

在 LLM 的 Transformer 层中，某些激活通道的值可能比其他通道大 100 倍以上：

```
通道 0:  [-0.2, 0.1, -0.3, 0.2, ...]    # 正常范围
通道 1:  [-0.1, 0.3, -0.2, 0.1, ...]    # 正常范围
通道 42: [-50.0, 80.0, -60.0, 70.0, ...]  # 离群值通道！
```

如果对整个激活张量使用统一的 scale，正常通道的精度会被离群通道"挤压"。

### 等效变换

对于线性层 $Y = XW^T$，引入对角缩放矩阵 $\text{diag}(s)$：

$$Y = XW^T = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)^T = \hat{X} \cdot \hat{W}^T$$

其中：
- $\hat{X} = X \cdot \text{diag}(s)^{-1}$：缩放后的激活（离群通道被缩小）
- $\hat{W} = \text{diag}(s) \cdot W$：缩放后的权重（对应通道被放大）

**数学等价性**：变换前后的输出 $Y$ 完全相同——只是将"量化难度"从激活转移到权重。

### 缩放因子的计算

SmoothQuant 的缩放因子公式：

$$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$$

其中：
- $j$：通道索引
- $\max(|X_j|)$：第 $j$ 通道激活的最大绝对值（通过校准数据统计）
- $\max(|W_j|)$：第 $j$ 列权重的最大绝对值
- $\alpha \in [0, 1]$：平滑强度参数

### α 参数的含义

| α 值 | 含义 | 效果 |
|------|------|------|
| α = 1 | 仅考虑激活范围 | 激活完全平滑，权重承担全部离群值 |
| α = 0.5 | 激活和权重平衡 | **默认推荐**，两者各承担一半 |
| α = 0 | 仅考虑权重范围 | 权重完全平滑，激活不变 |

**选择原则**：
- W8A8 场景：α = 0.5 通常最优
- 激活离群值极端：增大 α（0.7-0.9）
- 权重已经很均匀：减小 α（0.3-0.5）

### 在模型中的应用位置

SmoothQuant 通常应用于 LayerNorm 和后续线性层之间：

```
LayerNorm output → ÷ s → [smooth activation]
                              ↓
q_proj weight   → × s → [scaled weight]
k_proj weight   → × s → [scaled weight]
v_proj weight   → × s → [scaled weight]
```

这里 LayerNorm 输出的每个通道除以对应的 $s_j$，而 q/k/v_proj 的每列乘以对应的 $s_j$。

## Logarithmic Equalization（对数均衡化）变体

LLM Compressor 还支持对数均衡化方法：

$$s_j = \frac{\max(|X_j|)}{\log_2(2 + \max(|X_j|))}$$

**特点**：
- 不使用权重信息（更稳定）
- 对数函数提供自然的"压缩"效果
- 适合激活范围极端的情况

使用方式：设置 `algorithm="log_equalization"`

## Mapping 系统

与 AWQ 类似，SmoothQuant 需要定义层映射：

```python
# smooth_layer: 产出激活的层（通常是 LayerNorm）
# balance_layers: 消费该激活的层（通常是 Linear）
SmoothQuantMapping(
    smooth_layer="model.layers.0.input_layernorm",
    balance_layers=["model.layers.0.self_attn.q_proj",
                    "model.layers.0.self_attn.k_proj",
                    "model.layers.0.self_attn.v_proj"],
)
```

## 在 LLM Compressor 中的实现

### SmoothQuantModifier

```python
# src/llmcompressor/modifiers/transform/smoothquant/base.py

class SmoothQuantModifier(Modifier):
    smoothing_strength: float = 0.5      # α 参数
    mappings: list | None = None         # 层映射（None=自动推理）
    algorithm: str = "smoothquant"       # "smoothquant" 或 "log_equalization"
    num_calibration_steps: int | None = None  # 校准步数
```

### 生命周期

```python
def on_initialize(self, state):
    # 如果未指定 mappings，自动推理
    if self.mappings is None:
        self.mappings = infer_mappings(state.model)

def on_start(self, state):
    # 解析映射，注册前向钩子收集激活统计
    self._resolve_mappings(state.model)
    for mapping in self._resolved_mappings:
        # 注册钩子收集 per-channel min/max
        self.register_hook(mapping.smooth_layer, 
                          self._track_channel_stats, "forward")

def on_event(self, state, event):
    if event.type_ == SEQUENTIAL_EPOCH_END:
        # 计算缩放因子并应用
        for mapping in self._resolved_mappings:
            scales = self._compute_scales(mapping)
            self._apply_smoothing(mapping, scales)
```

### 缩放因子计算

```python
def _compute_scales(self, mapping):
    # 获取校准期间收集的激活范围
    x_max = self._activation_channel_max[mapping.smooth_name]
    
    if self.algorithm == "smoothquant":
        # 获取权重范围
        w_max = compute_weight_channel_max(mapping.balance_layers)
        # 计算缩放因子：s = x_max^α / w_max^(1-α)
        scales = x_max.pow(self.smoothing_strength) / \
                 w_max.pow(1 - self.smoothing_strength)
    
    elif self.algorithm == "log_equalization":
        # 对数均衡化：s = x_max / log2(2 + x_max)
        scales = x_max / torch.log2(2 + x_max)
    
    # 避免零值
    scales = torch.clamp(scales, min=1e-5)
    return scales
```

### DDP 分布式支持

```python
def _sync_channel_stats(self):
    """跨 DDP rank 同步激活统计"""
    for name, stats in self._activation_channel_max.items():
        # 各 rank 取全局最大值
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
```

## 使用示例

### 示例 1：SmoothQuant + GPTQ W8A8 INT8

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.gptq import GPTQModifier

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

recipe = [
    # 第一步：SmoothQuant 平滑激活
    SmoothQuantModifier(smoothing_strength=0.8),
    # 第二步：GPTQ 量化
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

oneshot(
    model=model,
    recipe=recipe,
    dataset=dataset,
    num_calibration_samples=512,
    max_seq_length=2048,
)
```

### 示例 2：调整平滑强度

```python
# 激活离群值严重时增大 α
recipe = [
    SmoothQuantModifier(smoothing_strength=0.9),  # 更激进的平滑
    QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]
```

### 示例 3：对数均衡化

```python
recipe = [
    SmoothQuantModifier(
        algorithm="log_equalization",  # 使用对数均衡化
        # 不需要 smoothing_strength 参数
    ),
    QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]
```

### 示例 4：自定义 Mapping

```python
recipe = [
    SmoothQuantModifier(
        smoothing_strength=0.5,
        mappings=[
            {
                "smooth_layer": "re:.*input_layernorm",
                "balance_layers": ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
            },
            {
                "smooth_layer": "re:.*post_attention_layernorm",
                "balance_layers": ["re:.*gate_proj", "re:.*up_proj"],
            },
        ],
    ),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]
```

## SmoothQuant 的优缺点

**优点**：
- 无需修改模型架构（纯数学等效变换）
- 启用 W8A8 量化（激活和权重都能较好量化）
- 计算简单快速（无需迭代优化）
- 可与 GPTQ、RTN 等量化器组合

**缺点**：
- 需要校准数据（收集激活统计）
- α 参数需要调优
- 对于 W4A16 场景帮助有限（主要面向 W8A8）
- 不能完全消除离群值（只是缓解）

## 适用场景

| 场景 | 是否推荐 | 原因 |
|------|:---:|------|
| W8A8 INT8 量化 | ✅ | SmoothQuant 的主要目标场景 |
| W8A8 FP8 量化 | ⚠️ | FP8 对离群值更鲁棒，收益有限 |
| W4A16 量化 | ❌ | 仅量化权重，激活平滑无意义 |
| 与 GPTQ 组合 | ✅ | 平滑后 GPTQ 精度更高 |
| 激活离群值严重的模型 | ✅ | 正是 SmoothQuant 设计解决的问题 |
