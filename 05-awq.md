# AWQ（Activation-Aware Weight Quantization）量化详解

## 算法概述

AWQ 的核心洞察：在权重矩阵中，仅有约 **1% 的通道**对模型输出至关重要（"显著通道"），而这些通道可以通过**激活统计**来识别。AWQ 通过对这些显著通道进行缩放保护，在量化前降低它们的量化误差。

**关键特性**：AWQ 本身不是量化器——它是一个**变换（Transform）**。它在量化之前修改权重，必须与 `QuantizationModifier` 或 `GPTQModifier` 配合使用。

## 理论基础

### 核心问题

对于线性层 $Y = XW^T$，如果对 $W$ 直接量化，量化误差为：

$$\Delta Y = X \cdot (W - Q(W))^T$$

其中 $Q(\cdot)$ 是量化函数。关键洞察：即使某些权重值很小，如果它们对应的输入激活通道很大，量化误差也会被放大。

### 通道重要性

AWQ 定义通道重要性为激活的平均绝对值：

$$\text{importance}_j = \frac{1}{T} \sum_{t=1}^{T} |X_{t,j}|$$

其中 $T$ 是 token 数量，$j$ 是通道索引。重要性高的通道需要保护。

### 等效缩放变换

AWQ 的核心操作是对权重施加通道缩放 $s$：

$$Y = X \cdot W^T = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)^T$$

对于 smooth_layer（产出激活的层）和 balance_layers（消费激活的层）：
- smooth_layer 的输出除以 $s$（或等效地，smooth_layer 的权重除以 $s$）
- balance_layers 的权重乘以 $s$

这样，balance_layers 中重要通道的权重被放大，量化时获得更小的相对误差。

### Grid Search 寻找最优缩放因子

AWQ 通过网格搜索确定最优的 $\alpha$ 值：

$$s_j = x_{max,j}^{\alpha}$$

其中 $x_{max,j} = \max_t |X_{t,j}|$ 是第 $j$ 个通道的最大激活值。

搜索范围：$\alpha \in \{0, 1/n_{grid}, 2/n_{grid}, ..., 1\}$（默认 $n_{grid}=20$）

对于每个 $\alpha$ 值，计算量化后的输出误差：

$$L(\alpha) = ||Q(W \cdot \text{diag}(s)) \cdot \text{diag}(s)^{-1} \cdot X^T - W \cdot X^T||_F^2$$

选择使 $L(\alpha)$ 最小的 $\alpha^*$。

### Duo-Scaling

当 `duo_scaling=True` 时，缩放因子同时考虑激活和权重：

$$s_j = \frac{x_{max,j}^{\alpha}}{w_{max,j}^{1-\alpha}}$$

其中 $w_{max,j} = \max_i |W_{i,j}|$ 是第 $j$ 列权重的最大绝对值。

**直觉**：
- 激活大的通道应该被放大（保护量化精度）
- 权重本身就大的通道不需要额外放大

当 `duo_scaling="both"` 时，前半部分 grid 用 duo_scaling，后半部分不用，取两者中更好的结果。

## Mapping 系统

AWQ 需要知道哪些层之间存在激活传递关系：

### Mapping 结构

```python
AWQMapping(
    smooth_layer="model.layers.0.self_attn_layer_norm",   # 产出激活的层
    balance_layers=["model.layers.0.q_proj",              # 消费激活的层
                    "model.layers.0.k_proj",
                    "model.layers.0.v_proj"],
)
```

**smooth_layer**：其输出将被缩放（除以 $s$），通常是 LayerNorm
**balance_layers**：其输入来自 smooth_layer，权重将被缩放（乘以 $s$）

### 自动推理 Mapping

如果用户不指定 `mappings`，AWQ 会自动从模型架构推理：

```python
# src/llmcompressor/modifiers/transform/awq/dynamic_mappings.py
def get_layer_mappings_from_model(model):
    """基于模型结构自动推理 smooth_layer ↔ balance_layers 映射"""
    # 典型映射：
    # input_layernorm → q_proj, k_proj, v_proj
    # post_attention_layernorm → gate_proj, up_proj
    # (对于某些架构) up_proj → down_proj
```

### 正则表达式匹配

支持使用正则表达式批量指定映射：

```yaml
mappings:
  - smooth_layer: "re:.*input_layernorm"
    balance_layers: ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"]
  - smooth_layer: "re:.*post_attention_layernorm"
    balance_layers: ["re:.*gate_proj", "re:.*up_proj"]
```

## 在 LLM Compressor 中的实现

### AWQModifier 生命周期

```python
class AWQModifier(Modifier):
    mappings: list[AWQMapping] | None = None
    offload_device: torch.device | None = None
    duo_scaling: bool | Literal["both"] = True
    n_grid: int = 20
    
    def on_initialize(self, state):
        # 如果用户未指定 mappings，自动推理
        if self.mappings is None:
            self.mappings = get_layer_mappings_from_model(state.model)
        # MoE 模型默认开启 CPU offload
        if is_moe_model(state.model):
            self.offload_device = torch.device("cpu")
    
    def on_start(self, state):
        # 解析正则表达式映射为实际模块引用
        self._set_resolved_mappings(state.model)
        # 注册激活缓存钩子
        self._setup_activation_cache_hooks()
    
    def on_event(self, state, event):
        if event.type_ == SEQUENTIAL_EPOCH_END:
            # 对每个 mapping 执行 AWQ 缩放
            for mapping in self._resolved_mappings:
                self._apply_smoothing(mapping)
```

### Grid Search 核心逻辑

```python
def _apply_smoothing(self, mapping):
    """对一个 mapping 执行 AWQ grid search"""
    
    # 1. 收集该 mapping 的缓存激活
    cached_activations = self._parent_args_cache[mapping.parent]
    
    # 2. 计算激活通道的最大值
    x_max = compute_channel_max(cached_activations)  # shape: [C_in]
    
    # 3. Grid search
    best_loss = float('inf')
    best_scales = None
    
    for alpha_idx in range(self.n_grid):
        alpha = alpha_idx / self.n_grid
        
        # 计算缩放因子
        if self.duo_scaling:
            scales = x_max ** alpha / w_max ** (1 - alpha)
        else:
            scales = x_max ** alpha
        
        # 模拟量化并计算误差
        scaled_weight = weight * scales.unsqueeze(0)
        q_weight = fake_quantize(scaled_weight, ...)
        output_q = (q_weight / scales.unsqueeze(0)) @ activations
        output_orig = weight @ activations
        loss = mse(output_q, output_orig)
        
        if loss < best_loss:
            best_loss = loss
            best_scales = scales
    
    # 4. 应用最优缩放
    smooth_layer.weight /= best_scales       # smooth_layer 输出除以 s
    for layer in balance_layers:
        layer.weight *= best_scales.unsqueeze(0)  # balance_layer 输入乘以 s
```

### 激活缓存与内存优化

```python
def _setup_activation_cache_hooks(self):
    """注册钩子缓存激活值"""
    for mapping in self._resolved_mappings:
        # 在 parent 模块的前向传播时缓存输入
        def hook(module, args, kwargs):
            self._parent_args_cache[module].append((args, kwargs))
        self.register_hook(mapping.parent, hook, "forward_pre")

# 内存优化：可以将缓存的激活卸载到 CPU
# offload_device=torch.device("cpu")
```

## 使用示例

### 示例 1：AWQ + W4A16 量化

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.awq import AWQModifier

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")

dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

# AWQ 变换 + 量化（两个 Modifier 组合）
recipe = [
    AWQModifier(duo_scaling=True, n_grid=20),
    QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
]

oneshot(
    model=model,
    recipe=recipe,
    dataset=dataset,
    num_calibration_samples=512,
    max_seq_length=2048,
)

model.save_pretrained("Llama-3-8B-W4A16-AWQ")
```

### 示例 2：自定义 Mapping

```python
from llmcompressor.modifiers.transform.awq.mappings import AWQMapping

recipe = [
    AWQModifier(
        mappings=[
            AWQMapping(
                smooth_layer="re:.*input_layernorm",
                balance_layers=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
            ),
            AWQMapping(
                smooth_layer="re:.*post_attention_layernorm",
                balance_layers=["re:.*gate_proj", "re:.*up_proj"],
            ),
        ],
        duo_scaling=True,
        n_grid=20,
    ),
    QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
]
```

### 示例 3：MoE 模型 AWQ

```python
# MoE 模型自动启用 CPU offload
recipe = [
    AWQModifier(
        # offload_device 会自动设为 CPU（检测到 MoE）
        duo_scaling=True,
    ),
    QuantizationModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head", "re:.*mlp.gate$"],
    ),
]

oneshot(
    model=model,
    recipe=recipe,
    dataset=dataset,
    num_calibration_samples=512,
)
```

### 示例 4：AWQ + GPTQ 组合

```python
from llmcompressor.modifiers.gptq import GPTQModifier

# AWQ 做通道缩放预处理，GPTQ 做精细量化
recipe = [
    AWQModifier(duo_scaling=True),
    GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
]
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mappings` | None（自动推理） | 层映射关系列表 |
| `duo_scaling` | True | 是否同时考虑激活和权重 |
| `n_grid` | 20 | Grid search 网格点数 |
| `offload_device` | None（MoE 自动 CPU） | 激活缓存卸载设备 |

## AWQ vs SmoothQuant

| | AWQ | SmoothQuant |
|--|-----|-------------|
| 目标场景 | W4A16 仅权重量化 | W8A8 权重+激活量化 |
| 缩放因子搜索 | Grid search（20 点） | 公式直接计算 |
| 考虑权重 | duo_scaling 可选 | 使用 α 参数平衡 |
| 是否量化器 | 否（变换） | 否（变换） |
| 典型搭配 | + QuantizationModifier/GPTQModifier | + GPTQ/QuantizationModifier |

## AWQ 的优缺点

**优点**：
- 精度高（通过保护关键通道）
- 速度快于 GPTQ（无需 Hessian 计算）
- 自动推理 Mapping（无需手动配置）
- MoE 模型友好（自动 CPU offload）

**缺点**：
- 本身不做量化，必须配合其他 Modifier
- 缓存激活占内存（大模型可能 OOM）
- Grid search 可能找不到全局最优
- 仅支持 per-channel 及以上粒度（不支持 per-tensor）
