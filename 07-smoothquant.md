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

### 等效变换的严格数学证明

为了深入理解为什么 SmoothQuant 不改变模型的数学行为，我们来严格证明这个等效变换。

**定理**：对于任意矩阵 $X \in \mathbb{R}^{m \times d}$，$W \in \mathbb{R}^{n \times d}$，以及正定对角矩阵 $S = \text{diag}(s_1, s_2, \ldots, s_d)$（其中 $s_j > 0$），有：

$$XW^T = (XS^{-1})(SW)^T$$

**证明**：

首先展开右侧：

$$(XS^{-1})(SW)^T = (XS^{-1}) \cdot W^T S^T$$

由于 $S$ 是对角矩阵，$S^T = S$，因此：

$$(XS^{-1}) \cdot W^T S = X \cdot (S^{-1} \cdot S) \cdot W^T = X \cdot I \cdot W^T = XW^T$$

更直观地看，对于输出矩阵 $Y$ 的第 $(i, k)$ 个元素：

$$Y_{ik} = \sum_{j=1}^{d} X_{ij} \cdot W_{kj}$$

变换后：

$$\hat{Y}_{ik} = \sum_{j=1}^{d} \hat{X}_{ij} \cdot \hat{W}_{kj} = \sum_{j=1}^{d} \frac{X_{ij}}{s_j} \cdot (s_j \cdot W_{kj}) = \sum_{j=1}^{d} X_{ij} \cdot W_{kj} = Y_{ik}$$

每一项中 $s_j$ 和 $1/s_j$ 完美抵消，因此输出 $Y$ 在数学上严格不变。

**关键约束**：$s_j > 0$（缩放因子必须为正），否则 $S^{-1}$ 不存在。实践中还需避免 $s_j$ 过小（导致权重过大溢出）或过大（导致激活精度不够）。

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

### 直觉理解：为什么 α = 0.5 平衡

当 $\alpha = 0.5$ 时：

$$s_j = \frac{\max(|X_j|)^{0.5}}{\max(|W_j|)^{0.5}} = \sqrt{\frac{\max(|X_j|)}{\max(|W_j|)}}$$

变换后：
- 激活通道范围：$\max(|\hat{X}_j|) = \frac{\max(|X_j|)}{s_j} = \sqrt{\max(|X_j|) \cdot \max(|W_j|)}$
- 权重通道范围：$\max(|\hat{W}_j|) = s_j \cdot \max(|W_j|) = \sqrt{\max(|X_j|) \cdot \max(|W_j|)}$

两者变为**同一个值**——激活和权重在每个通道上的最大值完全相等！这就是"平衡"的数学含义：量化难度被均匀分配。

---

## 完整数值示例

### 问题设置

假设有一个简单的线性层：输入维度 4，输出维度 4。

**权重矩阵** $W \in \mathbb{R}^{4 \times 4}$（每行对应一个输出神经元）：

```
W = [[ 0.3,  0.2, -0.1,  0.4],
     [-0.2,  0.5,  0.3, -0.1],
     [ 0.1, -0.3,  0.4,  0.2],
     [ 0.4,  0.1, -0.2,  0.3]]
```

**激活矩阵** $X \in \mathbb{R}^{2 \times 4}$（2 个 token，4 个通道），**通道 2 有严重离群值**：

```
X = [[ 0.5,  0.3, 40.0, -0.2],
     [-0.4,  0.6, 35.0,  0.1]]
```

### 第一步：计算通道统计量

**激活通道最大绝对值**（通过校准数据，对所有 token 求 max）：

```
max(|X_0|) = max(0.5, 0.4)  = 0.5
max(|X_1|) = max(0.3, 0.6)  = 0.6
max(|X_2|) = max(40.0, 35.0) = 40.0   ← 离群通道！
max(|X_3|) = max(0.2, 0.1)  = 0.2
```

**权重列最大绝对值**（对 W 的每一列取 max）：

```
max(|W_0|) = max(0.3, 0.2, 0.1, 0.4)  = 0.4
max(|W_1|) = max(0.2, 0.5, 0.3, 0.1)  = 0.5
max(|W_2|) = max(0.1, 0.3, 0.4, 0.2)  = 0.4
max(|W_3|) = max(0.4, 0.1, 0.2, 0.3)  = 0.4
```

### 第二步：计算缩放因子（α = 0.5）

$$s_j = \frac{\max(|X_j|)^{0.5}}{\max(|W_j|)^{0.5}}$$

```
s_0 = sqrt(0.5) / sqrt(0.4)  = 0.707 / 0.632 = 1.118
s_1 = sqrt(0.6) / sqrt(0.5)  = 0.775 / 0.707 = 1.095
s_2 = sqrt(40.0) / sqrt(0.4) = 6.325 / 0.632 = 10.000  ← 离群通道缩放大！
s_3 = sqrt(0.2) / sqrt(0.4)  = 0.447 / 0.632 = 0.707
```

### 第三步：应用变换

**变换后的激活** $\hat{X} = X \cdot \text{diag}(s)^{-1}$：

```
X̂ = [[ 0.5/1.118,   0.3/1.095,  40.0/10.0,  -0.2/0.707],
     [-0.4/1.118,   0.6/1.095,  35.0/10.0,   0.1/0.707]]

   = [[ 0.447,  0.274,  4.000, -0.283],
     [-0.358,  0.548,  3.500,  0.141]]
```

注意通道 2 的激活从 40.0/35.0 降低到了 4.0/3.5！

**变换后的权重** $\hat{W} = \text{diag}(s) \cdot W$（对 W 的每一列乘以对应的 $s_j$）：

```
Ŵ = [[ 0.3×1.118,  0.2×1.095, -0.1×10.0,  0.4×0.707],
     [-0.2×1.118,  0.5×1.095,  0.3×10.0, -0.1×0.707],
     [ 0.1×1.118, -0.3×1.095,  0.4×10.0,  0.2×0.707],
     [ 0.4×1.118,  0.1×1.095, -0.2×10.0,  0.3×0.707]]

   = [[ 0.335,  0.219, -1.000,  0.283],
     [-0.224,  0.548,  3.000, -0.071],
     [ 0.112, -0.329,  4.000,  0.141],
     [ 0.447,  0.110, -2.000,  0.212]]
```

### 第四步：验证等效性

原始输出 $Y = X \cdot W^T$：

```
Y[0,0] = 0.5×0.3 + 0.3×0.2 + 40.0×(-0.1) + (-0.2)×0.4 
       = 0.15 + 0.06 - 4.0 - 0.08 = -3.87

Y[0,1] = 0.5×(-0.2) + 0.3×0.5 + 40.0×0.3 + (-0.2)×(-0.1)
       = -0.1 + 0.15 + 12.0 + 0.02 = 12.07
```

变换后输出 $\hat{Y} = \hat{X} \cdot \hat{W}^T$：

```
Ŷ[0,0] = 0.447×0.335 + 0.274×0.219 + 4.0×(-1.0) + (-0.283)×0.283
       = 0.150 + 0.060 - 4.0 - 0.080 = -3.87  ✓（完全一致）

Ŷ[0,1] = 0.447×(-0.224) + 0.274×0.548 + 4.0×3.0 + (-0.283)×(-0.071)
       = -0.100 + 0.150 + 12.0 + 0.020 = 12.07  ✓（完全一致）
```

### 第五步：量化误差对比

假设使用 INT8 对称量化（范围 [-127, 127]）。

**量化前（原始激活的量化误差）**：

对 $X$ 做 per-tensor 量化：范围 = max(|X|) = 40.0，scale = 40.0/127 = 0.315

```
通道 0: X[0,0] = 0.5, 量化后 ≈ round(0.5/0.315)*0.315 = 2*0.315 = 0.630
         误差 = |0.630 - 0.5| = 0.130   （相对误差 26%！）

通道 2: X[0,2] = 40.0, 量化后 ≈ round(40.0/0.315)*0.315 = 127*0.315 = 40.005
         误差 = |40.005 - 40.0| = 0.005  （相对误差 0.01%）
```

离群通道精度很好，但正常通道精度被严重牺牲！

**量化后（平滑激活的量化误差）**：

对 $\hat{X}$ 做 per-tensor 量化：范围 = max(|$\hat{X}$|) = 4.0，scale = 4.0/127 = 0.0315

```
通道 0: X̂[0,0] = 0.447, 量化后 ≈ round(0.447/0.0315)*0.0315 = 14*0.0315 = 0.441
         误差 = |0.441 - 0.447| = 0.006  （相对误差 1.3%，改善 20 倍！）

通道 2: X̂[0,2] = 4.0, 量化后 ≈ round(4.0/0.0315)*0.0315 = 127*0.0315 = 4.0
         误差 = 0  （完美）
```

**总结**：SmoothQuant 将激活范围从 [0, 40] 压缩到 [0, 4]，使所有通道都能获得足够的量化精度。

---

## 与 LayerNorm 的融合

### 融合原理

在 Transformer 中，激活 $X$ 通常是 LayerNorm 的输出。SmoothQuant 的缩放可以直接融入 LayerNorm 的参数中，避免额外的运行时计算。

**标准 LayerNorm**：

$$\text{LayerNorm}(h) = \gamma \cdot \frac{h - \mu}{\sigma} + \beta$$

其中 $\gamma$ 是可学习的缩放向量（per-channel），$\beta$ 是偏置向量。

**融合后的 LayerNorm**：

SmoothQuant 要求输出被 $\text{diag}(s)^{-1}$ 缩放：

$$\hat{X} = \text{LayerNorm}(h) \cdot \text{diag}(s)^{-1} = \frac{\gamma}{s} \cdot \frac{h - \mu}{\sigma} + \frac{\beta}{s}$$

即直接修改 LayerNorm 参数：

$$\gamma_{\text{new}} = \gamma / s, \quad \beta_{\text{new}} = \beta / s$$

这样在推理时**完全没有额外计算**——缩放已经被"吸收"进 LayerNorm 的参数中了。

### 融合的前提条件

LayerNorm 融合能成功的前提：

1. **LayerNorm 的输出只被一组线性层消费**：例如 `input_layernorm → [q_proj, k_proj, v_proj]`
2. **缩放因子对所有消费层是相同的**：SmoothQuant 对共享同一个 LayerNorm 输出的所有线性层使用统一的 $s$（通过取所有 balance_layers 的权重列最大值的 max 来统一）

如果 LayerNorm 输出被分支到不同路径、各路径需要不同的 $s$，则无法融合。

### RMSNorm 的融合

现代 LLM（如 LLaMA）使用 RMSNorm（无偏置、无均值中心化）：

$$\text{RMSNorm}(h) = \gamma \cdot \frac{h}{\text{RMS}(h)}$$

融合方式完全类似：$\gamma_{\text{new}} = \gamma / s$

由于 RMSNorm 无 $\beta$ 项，融合更加简洁。

---

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

**典型 LLaMA/Qwen 模型的完整 Mapping**：

```
Transformer Block:
├── input_layernorm (RMSNorm)
│   └── smooth → q_proj, k_proj, v_proj     (Mapping 1)
├── Self-Attention
│   └── o_proj
├── post_attention_layernorm (RMSNorm)
│   └── smooth → gate_proj, up_proj          (Mapping 2)
└── FFN
    └── down_proj
```

每个 Transformer Block 有 2 个 Mapping：
1. `input_layernorm` → `[q_proj, k_proj, v_proj]`
2. `post_attention_layernorm` → `[gate_proj, up_proj]`

注意 `o_proj` 和 `down_proj` **不参与** SmoothQuant 变换，因为它们的输入不直接来自 LayerNorm。

## Logarithmic Equalization（对数均衡化）变体

LLM Compressor 还支持对数均衡化方法：

$$s_j = \frac{\max(|X_j|)}{\log_2(2 + \max(|X_j|))}$$

**特点**：
- 不使用权重信息（更稳定）
- 对数函数提供自然的"压缩"效果
- 适合激活范围极端的情况

使用方式：设置 `algorithm="log_equalization"`

**数值示例对比**：

```
通道 2: max(|X_2|) = 40.0

SmoothQuant (α=0.5):  s_2 = sqrt(40.0) / sqrt(0.4) = 10.0
Log Equalization:      s_2 = 40.0 / log2(2 + 40.0) = 40.0 / 5.39 = 7.42

SmoothQuant 缩放后:    X̂_2 = 40.0 / 10.0 = 4.0
Log Equalization 缩放后: X̂_2 = 40.0 / 7.42 = 5.39
```

Log Equalization 的缩放较温和，不依赖权重统计，但对于离群值极端的情况（如 max > 100），对数函数的增长缓慢性反而限制了平滑效果。

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

### 自动推理 Mapping

当 `mappings=None` 时，LLM Compressor 会自动推理 Mapping：

1. 遍历模型的所有 LayerNorm/RMSNorm 层
2. 找到每个 Norm 层的下游 Linear 层（通过计算图追踪）
3. 自动构建 `(norm_layer → [linear_layers])` 的映射关系

支持的模型架构包括 LLaMA、Mistral、Qwen、GPT-NeoX 等主流 Transformer 变体。

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

### 应用平滑的内部细节

```python
def _apply_smoothing(self, mapping, scales):
    """将缩放因子应用到 LayerNorm 和 Linear 层"""
    
    # 1. 修改 smooth_layer（LayerNorm/RMSNorm）
    smooth_module = mapping.smooth_layer
    if hasattr(smooth_module, 'weight'):
        # γ_new = γ / s
        smooth_module.weight.data /= scales
    if hasattr(smooth_module, 'bias') and smooth_module.bias is not None:
        # β_new = β / s
        smooth_module.bias.data /= scales
    
    # 2. 修改 balance_layers（Linear 层）
    for linear_module in mapping.balance_layers:
        # W_new[:, j] = s_j * W[:, j]  （对每列乘以 s_j）
        linear_module.weight.data *= scales.unsqueeze(0)  # broadcast over rows
        # 注意：Linear 的 bias 不受影响（bias 不参与输入通道的缩放）
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
- LayerNorm 融合使得推理无额外开销

**缺点**：
- 需要校准数据（收集激活统计）
- α 参数需要调优
- 对于 W4A16 场景帮助有限（主要面向 W8A8）
- 不能完全消除离群值（只是缓解）
- 假设所有 token 共享相同的离群通道模式（实际中基本成立）

## 适用场景

| 场景 | 是否推荐 | 原因 |
|------|:---:|------|
| W8A8 INT8 量化 | ✅ | SmoothQuant 的主要目标场景 |
| W8A8 FP8 量化 | ⚠️ | FP8 对离群值更鲁棒，收益有限 |
| W4A16 量化 | ❌ | 仅量化权重，激活平滑无意义 |
| 与 GPTQ 组合 | ✅ | 平滑后 GPTQ 精度更高 |
| 激活离群值严重的模型 | ✅ | 正是 SmoothQuant 设计解决的问题 |

## 常见问题与调试

### Q：如何判断模型是否需要 SmoothQuant？

运行一小批校准数据，检查激活的 per-channel max 分布：

```python
# 诊断代码
for name, module in model.named_modules():
    if isinstance(module, nn.LayerNorm):
        hook = module.register_forward_hook(
            lambda m, i, o: print(f"{name}: max={o.abs().max():.1f}, "
                                  f"channel_max_ratio={o.abs().max()/o.abs().mean():.1f}x")
        )
```

如果 `channel_max_ratio > 20x`，强烈建议使用 SmoothQuant。

### Q：SmoothQuant 和 per-channel 量化是互补还是冗余？

互补。per-channel 量化为每个通道分配独立的 scale，但对于激活量化（per-tensor 或 per-token），SmoothQuant 仍然有显著价值。即使使用 per-channel 激活量化，通道间的极端差异仍会通过有限的位宽导致精度损失。
