# AWQ（Activation-Aware Weight Quantization）量化详解

## 算法概述

AWQ 的核心洞察：在权重矩阵中，仅有约 **1% 的通道**对模型输出至关重要（"显著通道"），而这些通道可以通过**激活统计**来识别。AWQ 通过对这些显著通道进行缩放保护，在量化前降低它们的量化误差。

**关键特性**：AWQ 本身不是量化器——它是一个**变换（Transform）**。它在量化之前修改权重，必须与 `QuantizationModifier` 或 `GPTQModifier` 配合使用。

## 理论基础

### 核心问题

对于线性层 $Y = XW^T$，如果对 $W$ 直接量化，量化误差为：

$$\Delta Y = X \cdot (W - Q(W))^T$$

其中 $Q(\cdot)$ 是量化函数。关键洞察：即使某些权重值很小，如果它们对应的输入激活通道很大，量化误差也会被放大。

**详细分析**：考虑输出的第 $k$ 个元素 $Y_k = \sum_j X_j \cdot W_{k,j}$。量化后的误差为：

$$\Delta Y_k = \sum_j X_j \cdot (W_{k,j} - Q(W_{k,j}))$$

即使 $W_{k,j} - Q(W_{k,j})$ 对所有 $j$ 都相同（最坏情况约为 $s/2$，其中 $s$ 是量化 scale），通道 $j$ 对输出误差的贡献正比于 $|X_j|$。因此激活值大的通道产生的量化误差被放大。

### 通道重要性

AWQ 定义通道重要性为激活的平均绝对值：

$$\text{importance}_j = \frac{1}{T} \sum_{t=1}^{T} |X_{t,j}|$$

其中 $T$ 是 token 数量，$j$ 是通道索引。重要性高的通道需要保护。

**为什么用平均绝对值而非最大值？** 最大值容易受异常值干扰（一个极大的激活就能使某个通道看起来"重要"），而平均绝对值更稳定地反映通道在正常推理中的行为。

实际实现中，LLM Compressor 使用的是 per-channel 最大值（`x_max`）来计算缩放因子：

$$x_{max,j} = \max_t |X_{t,j}|$$

这个选择使得缩放因子有明确的物理含义：它正比于该通道激活的动态范围。

### 等效缩放变换

AWQ 的核心操作是对权重施加通道缩放 $s$：

$$Y = X \cdot W^T = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)^T$$

对于 smooth_layer（产出激活的层）和 balance_layers（消费激活的层）：
- smooth_layer 的输出除以 $s$（或等效地，smooth_layer 的权重除以 $s$）
- balance_layers 的权重乘以 $s$

这样，balance_layers 中重要通道的权重被放大，量化时获得更小的相对误差。

**等效变换的数学证明**：

设 smooth_layer 输出为 $A = f(X)$，balance_layer 的输出为 $Y = A \cdot W^T$。

引入缩放：$Y = A \cdot W^T = (A \cdot S^{-1}) \cdot (S \cdot W)^T = A' \cdot W'^T$

其中 $S = \text{diag}(s)$，$A' = A \cdot S^{-1}$，$W' = S \cdot W$。

为了使 $A' = A \cdot S^{-1}$ 成立而不改变 smooth_layer 的计算，只需将 smooth_layer 的输出权重（最后一个维度）除以 $s$：

- 如果 smooth_layer 是 LayerNorm：$\gamma_{new} = \gamma / s$（修改缩放参数）
- 如果 smooth_layer 是 Linear：$W_{smooth,new}[i,:] = W_{smooth}[i,:] / s$（修改输出维度的权重）

### Grid Search 寻找最优缩放因子

AWQ 通过网格搜索确定最优的 $\alpha$ 值：

$$s_j = x_{max,j}^{\alpha}$$

其中 $x_{max,j} = \max_t |X_{t,j}|$ 是第 $j$ 个通道的最大激活值。

搜索范围：$\alpha \in \{0, 1/n_{grid}, 2/n_{grid}, ..., 1\}$（默认 $n_{grid}=20$）

对于每个 $\alpha$ 值，计算量化后的输出误差：

$$L(\alpha) = ||Q(W \cdot \text{diag}(s)) \cdot \text{diag}(s)^{-1} \cdot X^T - W \cdot X^T||_F^2$$

选择使 $L(\alpha)$ 最小的 $\alpha^*$。

**Grid search 的直觉理解**：

- $\alpha = 0$：$s = x_{max}^0 = 1$（不做任何缩放），退化为普通量化
- $\alpha = 1$：$s = x_{max}$（完全按激活大小缩放），激活大的通道权重被放大 $x_{max}$ 倍
- 最优 $\alpha^*$ 通常在 0.3-0.7 之间，平衡"保护重要通道"与"不过度扰动其他通道"

### Duo-Scaling

当 `duo_scaling=True` 时，缩放因子同时考虑激活和权重：

$$s_j = \frac{x_{max,j}^{\alpha}}{w_{max,j}^{1-\alpha}}$$

其中 $w_{max,j} = \max_i |W_{i,j}|$ 是第 $j$ 列权重的最大绝对值。

**直觉**：
- 激活大的通道应该被放大（保护量化精度）
- 权重本身就大的通道不需要额外放大

当 `duo_scaling="both"` 时，前半部分 grid 用 duo_scaling，后半部分不用，取两者中更好的结果。

#### Duo-Scaling 的数学分析

**为什么要除以 $w_{max,j}^{1-\alpha}$？**

考虑量化误差的上界。对于 4-bit 对称量化，scale 为 $s_{quant} = \max(|W_{:,j}|) / 7$，量化误差上界为 $s_{quant} / 2$。

对于缩放后的权重 $W'_{:,j} = s_j \cdot W_{:,j}$，新的 scale 为：

$$s'_{quant,j} = \frac{\max(|s_j \cdot W_{:,j}|)}{7} = \frac{s_j \cdot w_{max,j}}{7}$$

缩放回原始空间后的等效量化误差为：

$$\text{error}_j = \frac{s'_{quant,j}}{2 \cdot s_j} = \frac{s_j \cdot w_{max,j}}{14 \cdot s_j} = \frac{w_{max,j}}{14}$$

等等——这说明对于 per-channel 量化，简单的缩放实际上不改变量化误差！但对于 per-group 或 per-tensor 量化，多个通道共享 scale，缩放确实能减小重要通道的相对误差。

实际上 AWQ 的效果来自：当多个通道共享量化参数时（per-group 量化），放大重要通道等效于让该通道在共享 scale 的计算中占据更大权重，从而使 scale 更适合保护该通道。

**Duo-scaling 的具体效果**：

设通道 $j$ 的激活大（$x_{max,j} = 10$）但权重也大（$w_{max,j} = 5$）：
- 纯 activation-based: $s_j = 10^{0.5} = 3.16$（放大 3.16 倍）
- Duo-scaling: $s_j = 10^{0.5} / 5^{0.5} = 3.16 / 2.24 = 1.41$（只放大 1.41 倍）

由于权重本身就大，它在 group 内已经占据了较大的动态范围份额，不需要额外放大太多。

## 完整数值示例：AWQ 在一个 3x4 权重矩阵上的操作

### 设定

权重矩阵（3 个输出神经元，4 个输入通道）：

$$W = \begin{pmatrix} 0.8 & -0.1 & 0.5 & 0.3 \\ -0.4 & 0.9 & -0.2 & 0.6 \\ 0.2 & -0.3 & 0.7 & -0.1 \end{pmatrix}$$

校准数据的激活统计（8 个 token 的 4 通道激活）：

| Token | Ch0 | Ch1 | Ch2 | Ch3 |
|---|---|---|---|---|
| 1 | 2.1 | 0.3 | 1.5 | 0.8 |
| 2 | 1.8 | 0.1 | 1.2 | 0.5 |
| 3 | 2.5 | 0.4 | 0.9 | 0.7 |
| 4 | 1.9 | 0.2 | 1.8 | 0.3 |
| 5 | 2.3 | 0.5 | 1.1 | 0.9 |
| 6 | 2.0 | 0.3 | 1.4 | 0.6 |
| 7 | 2.7 | 0.2 | 1.6 | 0.4 |
| 8 | 1.6 | 0.6 | 1.3 | 1.0 |

### 步骤 1：计算通道最大激活值

$$x_{max} = [2.7,\ 0.6,\ 1.8,\ 1.0]$$

**观察**：通道 0 的激活最大（2.7），是"最重要"的通道；通道 1 的激活最小（0.6），最不重要。

### 步骤 2：计算权重最大值（per-column）

$$w_{max} = [\max(|0.8|, |0.4|, |0.2|),\ \max(|0.1|, |0.9|, |0.3|),\ \max(|0.5|, |0.2|, |0.7|),\ \max(|0.3|, |0.6|, |0.1|)]$$
$$= [0.8,\ 0.9,\ 0.7,\ 0.6]$$

### 步骤 3：Grid Search（duo_scaling=True, n_grid=5 简化演示）

对 $\alpha \in \{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$ 逐一尝试：

**$\alpha = 0.0$**：$s = x_{max}^0 / w_{max}^1 = [1/0.8, 1/0.9, 1/0.7, 1/0.6] = [1.25, 1.11, 1.43, 1.67]$

**$\alpha = 0.4$**：
- $s_0 = 2.7^{0.4} / 0.8^{0.6} = 1.49 / 0.87 = 1.71$
- $s_1 = 0.6^{0.4} / 0.9^{0.6} = 0.81 / 0.94 = 0.86$
- $s_2 = 1.8^{0.4} / 0.7^{0.6} = 1.26 / 0.80 = 1.57$
- $s_3 = 1.0^{0.4} / 0.6^{0.6} = 1.00 / 0.74 = 1.35$

$$s = [1.71, 0.86, 1.57, 1.35]$$

**$\alpha = 0.8$**：
- $s_0 = 2.7^{0.8} / 0.8^{0.2} = 2.28 / 0.96 = 2.38$
- $s_1 = 0.6^{0.8} / 0.9^{0.2} = 0.66 / 0.98 = 0.67$
- $s_2 = 1.8^{0.8} / 0.7^{0.2} = 1.59 / 0.93 = 1.71$
- $s_3 = 1.0^{0.8} / 0.6^{0.2} = 1.00 / 0.90 = 1.11$

$$s = [2.38, 0.67, 1.71, 1.11]$$

### 步骤 4：对每个 $\alpha$ 计算量化误差

以 $\alpha = 0.4$，$s = [1.71, 0.86, 1.57, 1.35]$ 为例：

**缩放权重**：$W' = W \cdot \text{diag}(s)$

$$W' = \begin{pmatrix} 0.8 \times 1.71 & -0.1 \times 0.86 & 0.5 \times 1.57 & 0.3 \times 1.35 \\ -0.4 \times 1.71 & 0.9 \times 0.86 & -0.2 \times 1.57 & 0.6 \times 1.35 \\ 0.2 \times 1.71 & -0.3 \times 0.86 & 0.7 \times 1.57 & -0.1 \times 1.35 \end{pmatrix}$$

$$= \begin{pmatrix} 1.37 & -0.086 & 0.785 & 0.405 \\ -0.684 & 0.774 & -0.314 & 0.810 \\ 0.342 & -0.258 & 1.099 & -0.135 \end{pmatrix}$$

**对 $W'$ 进行 4-bit 量化**（per-row scale）：
- 第 0 行：$\max = 1.37$，$scale = 1.37/7 = 0.196$
- 第 1 行：$\max = 0.810$，$scale = 0.810/7 = 0.116$
- 第 2 行：$\max = 1.099$，$scale = 1.099/7 = 0.157$

量化后 $Q(W')$：各元素 round 到整数格点再乘以 scale...

**反缩放**：$W_{final} = Q(W') \cdot \text{diag}(s)^{-1}$

**计算误差**：用缓存的激活 $X$ 计算 $L = ||W_{final} \cdot X^T - W \cdot X^T||_F^2$

### 步骤 5：选择最优 $\alpha$

假设各 $\alpha$ 对应的 loss：

| $\alpha$ | Loss |
|---|---|
| 0.0 | 0.0082 |
| 0.2 | 0.0061 |
| 0.4 | 0.0043 |
| 0.6 | 0.0039 |
| 0.8 | 0.0051 |
| 1.0 | 0.0078 |

最优 $\alpha^* = 0.6$，对应 $s^* = [2.03, 0.76, 1.63, 1.22]$

**趋势解释**：
- $\alpha$ 太小：重要通道未得到保护，误差大
- $\alpha$ 太大：缩放过于激进，非重要通道被压缩太多，引入新误差
- 最优点在中间位置平衡两方面

### 步骤 6：应用最优缩放

**修改 smooth_layer**（如 LayerNorm）：
- 如果 smooth_layer 有 weight（$\gamma$）：$\gamma_{new} = \gamma / s^*$
- 如果有 bias（$\beta$）：bias 不变（因为 bias 在缩放之后加入）

**修改 balance_layers**（如 q_proj, k_proj, v_proj）：
- 每个 balance_layer 的权重：$W_{new}[:, j] = s^*_j \cdot W_{old}[:, j]$

## 模型内部变换的可视化

### 变换前的数据流

```
Input → LayerNorm(γ, β) → Activation A → [q_proj(W_q), k_proj(W_k), v_proj(W_v)]
                                    ↓
                          A 的第 j 通道 = γ_j * normalized + β_j
                          q = A @ W_q^T
```

### 变换后的数据流

```
Input → LayerNorm(γ/s, β) → Activation A' = A/s → [q_proj(s·W_q), k_proj(s·W_k), v_proj(s·W_v)]
                                          ↓
                          A' 的第 j 通道 = (γ_j/s_j) * normalized + β_j/s_j（被缩小）
                          但 W' 被放大：q = A' @ (s·W_q)^T = (A/s) @ (s·W_q)^T = A @ W_q^T
```

**数学等价性**：变换前后的输出完全相同（FP 精度下）。但量化时：
- 通道 0（$s_0 = 2.03$）：权重被放大 2.03 倍，量化相对误差降低
- 通道 1（$s_1 = 0.76$）：权重被缩小 0.76 倍，量化相对误差增大
- 净效果：重要通道（激活大的）的误差降低带来的收益 > 不重要通道误差增大的损失

### 为什么 AWQ 对 per-group 量化特别有效？

在 per-group 量化中，128 个通道共享一个 scale。假设一个 group 中：
- 通道 A：权重小（0.1），但激活大（5.0）
- 通道 B：权重大（2.0），但激活小（0.1）

不做 AWQ：group scale 由通道 B 的权重决定（$s = 2.0/7 = 0.286$），通道 A 的量化精度为 $0.286/2 = 0.143$，相对于其 0.1 的权重，误差高达 143%！

做 AWQ 后（假设 $s_A = 3.0$）：通道 A 权重变为 0.3，group 中的动态范围更平衡，通道 A 获得更合理的量化精度。

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

**为什么 smooth_layer 通常选择 LayerNorm？** 因为 LayerNorm 的缩放参数 $\gamma$ 天然对应通道维度，除以 $s$ 就相当于修改 $\gamma$，不需要额外操作。如果 smooth_layer 是 Linear 层，则需要修改其输出维度的权重行向量，更加复杂。

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

### Grid Search 实现细节

```python
def _grid_search_single_mapping(self, mapping, x_max, w_max, cached_acts):
    """单个 mapping 的 grid search 详细实现"""
    
    # 合并所有 balance_layers 的权重用于计算
    # （因为同一个 smooth_layer 的缩放影响所有 balance_layers）
    all_weights = torch.cat([layer.weight for layer in mapping.balance_layers], dim=0)
    # all_weights shape: [sum(C_out), C_in]
    
    # 计算原始输出作为参考
    orig_output = all_weights @ cached_acts.T  # [sum(C_out), num_tokens]
    
    best_loss = float('inf')
    best_scales = None
    
    for alpha_idx in range(self.n_grid + 1):
        alpha = alpha_idx / self.n_grid  # 0.0, 0.05, 0.10, ..., 1.0
        
        # 计算缩放因子（避免除零）
        scales = x_max.clamp(min=1e-5) ** alpha
        if self.duo_scaling:
            scales /= w_max.clamp(min=1e-5) ** (1 - alpha)
        
        # 防止缩放因子过大或过小
        scales = scales.clamp(min=1e-5)
        
        # 模拟量化
        scaled_W = all_weights * scales.unsqueeze(0)  # 放大权重
        q_scaled_W = fake_quantize(scaled_W, ...)      # 量化
        q_W = q_scaled_W / scales.unsqueeze(0)         # 缩回原空间
        
        # 计算量化后输出
        q_output = q_W @ cached_acts.T
        
        # MSE loss
        loss = ((q_output - orig_output) ** 2).sum()
        
        if loss < best_loss:
            best_loss = loss
            best_scales = scales.clone()
    
    return best_scales
```

### duo_scaling="both" 的实现

```python
if self.duo_scaling == "both":
    # 前半部分用 duo_scaling
    best_loss_duo = float('inf')
    best_scales_duo = None
    for alpha_idx in range(self.n_grid // 2):
        alpha = alpha_idx / (self.n_grid // 2)
        scales = x_max ** alpha / w_max ** (1 - alpha)
        loss = compute_loss(scales)
        if loss < best_loss_duo:
            best_loss_duo = loss
            best_scales_duo = scales
    
    # 后半部分不用 duo_scaling
    best_loss_plain = float('inf')
    best_scales_plain = None
    for alpha_idx in range(self.n_grid // 2):
        alpha = alpha_idx / (self.n_grid // 2)
        scales = x_max ** alpha
        loss = compute_loss(scales)
        if loss < best_loss_plain:
            best_loss_plain = loss
            best_scales_plain = scales
    
    # 取两者中更好的
    best_scales = best_scales_duo if best_loss_duo < best_loss_plain else best_scales_plain
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

### 应用缩放的具体操作

```python
def _apply_scales_to_model(self, mapping, best_scales):
    """将找到的最优缩放因子应用到模型参数"""
    
    s = best_scales  # [C_in]
    
    # === 修改 smooth_layer ===
    smooth_layer = mapping.smooth_layer
    
    if isinstance(smooth_layer, nn.LayerNorm):
        # LayerNorm: y = γ * (x - μ) / σ + β
        # 除以 s 后: y' = (γ/s) * (x - μ) / σ + β/s
        smooth_layer.weight.data /= s        # γ → γ/s
        if smooth_layer.bias is not None:
            smooth_layer.bias.data /= s      # β → β/s
    
    elif isinstance(smooth_layer, nn.Linear):
        # Linear: y = Wx + b
        # 除以 s 后: y' = (W/s)x + b/s  → 但这是对输出做缩放
        # 实际上是对输出维度除以 s
        smooth_layer.weight.data /= s.unsqueeze(0)  # 逐输出通道除
        if smooth_layer.bias is not None:
            smooth_layer.bias.data /= s
    
    # === 修改 balance_layers ===
    for layer in mapping.balance_layers:
        # 输入维度乘以 s
        layer.weight.data *= s.unsqueeze(0)  # [C_out, C_in] * [1, C_in]
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

### 参数调优详解

**n_grid 的选择**：

| n_grid | Grid 点数 | 搜索精度 | 耗时 |
|---|---|---|---|
| 10 | 11 个 $\alpha$ | 粗粒度，可能错过最优 | 快 |
| 20（默认） | 21 个 $\alpha$ | 精度与速度平衡 | 标准 |
| 50 | 51 个 $\alpha$ | 精细搜索，边际收益小 | 慢 |

实际经验：从 20 增加到 50 通常只提升 0.01-0.02 perplexity，不值得额外耗时。

**duo_scaling 的三种模式**：

| 值 | 行为 | 适用场景 |
|---|---|---|
| `False` | 仅用 $x_{max}^\alpha$ | 权重分布均匀的模型 |
| `True` | 用 $x_{max}^\alpha / w_{max}^{1-\alpha}$ | 大多数模型（推荐） |
| `"both"` | 两种都试，取更好的 | 不确定哪种更好时 |

## AWQ vs SmoothQuant

| | AWQ | SmoothQuant |
|--|-----|-------------|
| 目标场景 | W4A16 仅权重量化 | W8A8 权重+激活量化 |
| 缩放因子搜索 | Grid search（20 点） | 公式直接计算 |
| 考虑权重 | duo_scaling 可选 | 使用 α 参数平衡 |
| 是否量化器 | 否（变换） | 否（变换） |
| 典型搭配 | + QuantizationModifier/GPTQModifier | + GPTQ/QuantizationModifier |

### 深入对比：为什么 AWQ 用 Grid Search 而 SmoothQuant 用公式？

**SmoothQuant** 的缩放因子有闭式解：

$$s_j = \frac{x_{max,j}^\alpha}{w_{max,j}^{1-\alpha}}$$

这里 $\alpha$ 是全局超参数（通常 0.5），因为 SmoothQuant 需要同时量化权重和激活，公式直接平衡两者的量化难度。

**AWQ** 不量化激活，仅通过缩放改善权重量化。由于不同层、不同 group 的最优 $\alpha$ 不同，需要通过 grid search 对每个 mapping 找到局部最优。AWQ 的 grid search 成本不高（每次只是一个矩阵乘法 + 量化模拟），但收益显著。

## AWQ 的优缺点

**优点**：
- 精度高（通过保护关键通道）
- 速度快于 GPTQ（无需 Hessian 计算）
- 自动推理 Mapping（无需手动配置）
- MoE 模型友好（自动 CPU offload）
- 与其他量化器组合灵活（可以和 GPTQ 或 RTN 搭配）

**缺点**：
- 本身不做量化，必须配合其他 Modifier
- 缓存激活占内存（大模型可能 OOM）
- Grid search 可能找不到全局最优
- 仅支持 per-channel 及以上粒度（不支持 per-tensor）
- 对 per-tensor 量化效果有限（因为缩放不改变 per-tensor 的量化误差）

## 常见问题

### Q: AWQ 应该在 GPTQ 之前还是之后？

**答**：AWQ 必须在量化之前执行。它修改权重的数值分布，使后续的量化过程产生更小的误差。正确顺序：

```python
recipe = [
    AWQModifier(...),        # 先做变换
    GPTQModifier(...),       # 再做量化
]
```

### Q: 为什么有时 AWQ 反而降低精度？

可能原因：
1. **模型本身权重分布已经很好**：缩放引入的数值扰动大于收益
2. **校准数据与部署数据分布不同**：激活统计不准确
3. **per-tensor 量化**：如前所述，AWQ 对 per-tensor 量化效果有限
4. **LayerNorm 后的通道分布已经均匀**：LayerNorm 本身就做了归一化，进一步缩放的收益有限

### Q: n_grid=20 的计算开销是多少？

对于一个 mapping（假设 balance_layers 总共 12288 输出通道，4096 输入通道，缓存 512 tokens）：
- 每个 grid 点：矩阵乘法 $[12288, 4096] \times [4096, 512]$ + 量化模拟
- 20 个点：约 20 次矩阵乘法
- 总耗时：约 0.5-2 秒/mapping（在 A100 上）
- 整个模型（32 层 × 2 mappings/层）：约 1-2 分钟
