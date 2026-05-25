# AutoRound 量化详解

## 算法概述

AutoRound 是 Intel 提出的量化算法，通过 **SignSGD 优化器**迭代优化舍入决策和权重裁剪范围，以最小化逐层输出误差。相比 GPTQ 的一次性最优补偿，AutoRound 通过多轮梯度优化获得更精细的舍入策略。

**核心思想**：量化中的舍入不应简单取最近值，而应该通过优化找到使整体输出误差最小的舍入方向。

## 理论基础

### 从 AdaRound 到 AutoRound

**AdaRound（前身）**的核心洞察：

对于权重 $w$，量化后的值有两个候选：
- $\lfloor w/s \rfloor \cdot s$（向下取整）
- $\lceil w/s \rceil \cdot s$（向上取整）

最优舍入方向不是简单的最近取整，而是取决于该权重在网络中的作用。

**为什么最近舍入不是最优的？** 考虑一个简单例子：假设有两个权重 $w_1 = 0.6s$、$w_2 = 0.4s$（$s$ 是 scale），最近舍入给出 $\hat{w}_1 = s$、$\hat{w}_2 = 0$。但如果 $w_1$ 和 $w_2$ 的激活输入高度相关，可能 $\hat{w}_1 = s$、$\hat{w}_2 = s$（两个都向上取整）反而使输出误差更小。GPTQ 通过误差补偿部分解决了这个问题，而 AutoRound 通过直接优化舍入方向来解决。

**AutoRound 的改进**：

1. 引入可学习的舍入变量 $V$
2. 引入可学习的权重裁剪参数
3. 使用 SignSGD（签名梯度下降）进行高效优化

### 数学公式

**舍入变量参数化**：

$$V = \sigma(\theta_V) \in [0, 1]$$

其中 $\sigma$ 是 sigmoid 函数，$\theta_V$ 是可学习参数。

**量化过程**：

$$W_q = \text{clamp}\left(\lfloor W/s \rfloor + \mathbb{1}[V > 0.5],\ q_{min},\ q_{max}\right)$$

- 当 $V_i > 0.5$：向上取整
- 当 $V_i \leq 0.5$：向下取整

**权重裁剪**：

对原始权重做软裁剪，优化裁剪范围 $[\alpha, \beta]$：

$$W_{clipped} = \text{clamp}(W, \alpha, \beta)$$

$\alpha, \beta$ 也是可学习参数。

### 舍入变量 V 的详细参数化

#### 初始化

$\theta_V$ 的初始化使得初始舍入方向等价于最近取整（Round-to-Nearest, RTN）：

$$\theta_{V,i} = \sigma^{-1}(\text{frac}(w_i / s))$$

其中 $\text{frac}(x) = x - \lfloor x \rfloor$ 是小数部分。这保证了：
- 如果 $\text{frac}(w_i/s) > 0.5$（接近上界），则 $\sigma(\theta_{V,i}) > 0.5$，初始向上取整
- 如果 $\text{frac}(w_i/s) < 0.5$（接近下界），则 $\sigma(\theta_{V,i}) < 0.5$，初始向下取整

这确保了优化的起点就是 RTN，后续优化只会改善精度。

#### 训练过程中的连续松弛

在训练时，$V$ 不是离散的 $\{0, 1\}$，而是连续值 $[0, 1]$：

$$W_q = \lfloor W/s \rfloor + V$$

其中 $V = \sigma(\theta_V) \in (0, 1)$。

量化后的权重为：

$$\hat{W} = W_q \cdot s = (\lfloor W/s \rfloor + V) \cdot s$$

**这是一种"软量化"**：在训练过程中权重可以取 $\lfloor W/s \rfloor \cdot s$ 和 $\lceil W/s \rceil \cdot s$ 之间的任意值。只有在推理时才硬化为离散决策（$V > 0.5$ 取上，否则取下）。

#### 完整的量化公式（含裁剪）

```python
def soft_quantize(W, theta_V, alpha, beta, scale, zero_point, qmin, qmax):
    """AutoRound 的软量化过程"""
    
    # 1. 裁剪权重到学习到的范围
    W_clipped = torch.clamp(W, alpha, beta)
    
    # 2. 计算浮点量化索引
    W_float_idx = (W_clipped - zero_point) / scale  # 连续值
    
    # 3. 向下取整得到基础索引
    W_floor = torch.floor(W_float_idx)
    
    # 4. 加上可学习的舍入偏移
    V = torch.sigmoid(theta_V)  # [0, 1]
    W_q_idx = W_floor + V       # 连续值（训练时）
    
    # 5. 裁剪到有效范围
    W_q_idx = torch.clamp(W_q_idx, qmin, qmax)
    
    # 6. 反量化回浮点
    W_hat = W_q_idx * scale + zero_point
    
    return W_hat
```

### Straight-Through Estimator（直通估计器）

量化操作（floor + clamp）是不可导的。为了通过反向传播训练 $\theta_V$，AutoRound 使用 **Straight-Through Estimator (STE)**：

**前向传播**：正常执行量化
$$y = \text{floor}(x) + V$$

**反向传播**：假装 floor 不存在，直接传递梯度
$$\frac{\partial L}{\partial x} \approx \frac{\partial L}{\partial y}$$

具体到 $\theta_V$ 的梯度链：

$$\frac{\partial L}{\partial \theta_V} = \frac{\partial L}{\partial \hat{W}} \cdot \frac{\partial \hat{W}}{\partial V} \cdot \frac{\partial V}{\partial \theta_V}$$

其中：
- $\frac{\partial \hat{W}}{\partial V} = s$（scale，来自反量化）
- $\frac{\partial V}{\partial \theta_V} = \sigma(\theta_V)(1 - \sigma(\theta_V))$（sigmoid 导数）
- $\frac{\partial L}{\partial \hat{W}}$ 通过网络反向传播得到

**对裁剪参数 $\alpha, \beta$ 的梯度**：

$$\frac{\partial L}{\partial \alpha} = \frac{\partial L}{\partial W_{clipped}} \cdot \frac{\partial W_{clipped}}{\partial \alpha}$$

其中 $\frac{\partial W_{clipped}}{\partial \alpha} = \mathbb{1}[W < \alpha]$（只有被裁剪的权重才贡献梯度）。

同理 $\frac{\partial W_{clipped}}{\partial \beta} = \mathbb{1}[W > \beta]$。

### Block-wise 损失函数

对每个 Transformer 层（block），优化目标为：

$$\mathcal{L} = ||f(X; W + V \cdot \Delta) - f(X; W)||_2^2$$

其中：
- $f(X; W)$：层使用原始权重的输出
- $f(X; W + V \cdot \Delta)$：层使用量化权重的输出
- $\Delta$：舍入产生的增量（$\Delta_i = s$ 或 $-s$）

**更精确的表述**：

$$\mathcal{L} = \frac{1}{B \cdot T \cdot D} \sum_{b=1}^{B} \sum_{t=1}^{T} \sum_{d=1}^{D} (f(X; \hat{W})_{b,t,d} - f(X; W)_{b,t,d})^2$$

其中 $B$ 是 batch size，$T$ 是序列长度，$D$ 是输出维度。

**注意**：这里的 $f$ 是整个 Transformer 层（包括 attention + FFN + residual），不是单个 Linear 层。这是 AutoRound 的优势——它在层级别优化，考虑了层内各线性层之间的交互。

### SignSGD 优化

AutoRound 使用 SignSGD 而非 Adam/SGD：

$$\theta \leftarrow \theta - \eta \cdot \text{sign}(\nabla_\theta \mathcal{L})$$

**优势**：
- 梯度只取方向（sign），对梯度尺度不敏感
- 适合离散化搜索空间（舍入本质上是二值决策）
- 收敛速度快，通常 200 次迭代即可

#### 为什么 SignSGD 适合舍入优化？

**直觉解释**：

舍入决策本质上是二值的（向上或向下）。我们需要的不是精确的梯度大小，而是"方向"——应该让 $V$ 增大还是减小。SignSGD 恰好只利用梯度方向：

- $\text{sign}(\nabla_{\theta_V} L) > 0$：loss 随 $\theta_V$ 增大而增大 → 应减小 $\theta_V$ → 使 $V$ 更倾向 0 → 向下取整
- $\text{sign}(\nabla_{\theta_V} L) < 0$：loss 随 $\theta_V$ 增大而减小 → 应增大 $\theta_V$ → 使 $V$ 更倾向 1 → 向上取整

**与 Adam 的对比**：

| | SignSGD | Adam |
|---|---|---|
| 更新量 | 固定 $\pm \eta$ | 自适应 $\eta \cdot m / \sqrt{v}$ |
| 内存 | 无状态 | 需要 $m$（一阶矩）和 $v$（二阶矩） |
| 每步更新 | 所有参数等幅度移动 | 不同参数不同幅度 |
| 适用场景 | 二值决策（舍入方向） | 连续优化（权重训练） |

SignSGD 的内存优势很重要：对于一个 8B 模型，每层的 $\theta_V$ 矩阵与权重同大小。如果用 Adam，需要额外 2x 的优化器状态内存；SignSGD 不需要任何额外状态。

#### SignSGD 的学习率选择

学习率 $\eta$ 对收敛至关重要：

- 每步 $\theta_V$ 的变化量恒为 $\pm \eta$
- 经过 $N$ 步后，$\theta_V$ 最大可能偏移 $N \cdot \eta$
- $V = \sigma(\theta_V)$ 从 0.5 变到 0.9 需要 $\Delta\theta_V \approx 2.2$
- 若 $\eta = 0.01$，220 步可以完成一个舍入方向的翻转

auto_round 库的默认学习率选择策略：

```python
if lr is None:
    # 根据权重维度自适应选择
    lr = 1.0 / (iters * 0.8)  # 约 0.00625 for iters=200
```

### 裁剪参数的学习

#### 初始化

裁剪参数 $\alpha, \beta$ 初始化为权重的最小值和最大值：

$$\alpha_0 = \min(W), \quad \beta_0 = \max(W)$$

这保证初始时不裁剪任何权重（与不使用裁剪等价）。

#### 优化过程

裁剪参数也通过 SignSGD 优化。其梯度的物理含义：

- $\frac{\partial L}{\partial \alpha} > 0$：增大 $\alpha$（收紧下界）能减小 loss → 应该裁剪底部异常值
- $\frac{\partial L}{\partial \beta} < 0$：减小 $\beta$（收紧上界）能减小 loss → 应该裁剪顶部异常值

**裁剪为什么有效？** 在 4-bit 量化中，scale 由权重的 max/min 决定。如果有少量异常值导致 max 很大，所有正常权重的量化精度都会受损。通过学习裁剪：

- 牺牲少量异常值的精度（它们被强制裁剪到边界）
- 换取大量正常权重的精度提升（scale 变小，量化步长更细）

**示例**：假设权重分布为 $[-0.5, 0.5]$，但有一个异常值 2.0。

- 不裁剪：$scale = 2.0/7 = 0.286$，正常权重的量化精度差
- 裁剪到 $[-0.6, 0.6]$：$scale = 0.6/7 = 0.086$，正常权重精度提升 3.3 倍

### 迭代优化流程

```
for iter = 1 to iters (默认 200):
    1. 用当前 V 和裁剪参数对权重进行软量化
    2. 前向传播得到量化输出
    3. 计算与原始输出的 MSE loss
    4. 反向传播计算梯度
    5. SignSGD 更新 V 和裁剪参数
    6. 如果 loss < best_loss：保存最佳参数
    
return best_params
```

#### 详细的单次迭代

```python
def single_iteration(theta_V, alpha, beta, W_orig, X_batch, Y_ref, scale, zp, lr):
    """单次优化迭代的详细过程"""
    
    # === 前向传播 ===
    # 1. 裁剪权重
    W_clipped = torch.clamp(W_orig, alpha, beta)
    
    # 2. 软量化（STE 模式下可微分）
    V = torch.sigmoid(theta_V)
    W_floor = torch.floor((W_clipped - zp) / scale)
    W_q_idx = torch.clamp(W_floor + V, qmin, qmax)
    W_hat = W_q_idx * scale + zp
    
    # 3. 通过整个 Transformer 层前向传播
    Y_hat = transformer_layer_forward(X_batch, W_hat)
    
    # 4. 计算 loss
    loss = F.mse_loss(Y_hat, Y_ref)
    
    # === 反向传播 ===
    loss.backward()
    
    # === SignSGD 更新 ===
    with torch.no_grad():
        theta_V -= lr * torch.sign(theta_V.grad)
        alpha -= lr * torch.sign(alpha.grad)
        beta -= lr * torch.sign(beta.grad)
        
        # 清零梯度
        theta_V.grad.zero_()
        alpha.grad.zero_()
        beta.grad.zero_()
    
    return loss.item()
```

### 收敛行为分析

#### 为什么 200 次迭代通常足够？

**理论分析**：

每个权重的舍入只有两种选择（上或下），目标是找到使 loss 最小的组合。对于 $N$ 个权重，搜索空间为 $2^N$（天文数字）。但 AutoRound 利用了以下结构：

1. **初始化良好**：从 RTN 出发，大部分权重的初始舍入方向已经接近最优（约 90%+ 不需要改变）
2. **梯度信息丰富**：每次迭代的梯度告诉我们哪些权重应该翻转方向
3. **SignSGD 的快速决策**：每步固定步长，经过几十步即可让一个 $\theta_V$ 从 "偏向上" 翻转到 "偏向下"

**实验观察的收敛曲线**（典型行为）：

| 迭代次数 | Loss 相对于初始的下降 | 状态 |
|---|---|---|
| 0 | 0%（初始 = RTN 精度） | 起点 |
| 10 | ~30% | 快速下降期，大量明显错误的舍入被纠正 |
| 50 | ~60% | 主要改进完成 |
| 100 | ~80% | 精细调整阶段 |
| 150 | ~90% | 接近收敛 |
| 200 | ~95% | 充分收敛 |
| 500 | ~98% | 边际收益递减 |

**关键洞察**：前 50 次迭代贡献了大部分精度提升（约 60% 的 loss 下降）。这是因为需要"翻转方向"的权重只占少数（约 5-15%），而它们的梯度信号通常很强，几十步就能完成翻转。

#### 不同 iters 值的精度对比

| iters | 相对于 RTN 的 PPL 改善 | 相对于 200 iters 的差距 | 量化速度 |
|---|---|---|---|
| 50 | ~70% | 0.1-0.2 PPL | 4x 快 |
| 100 | ~85% | 0.05-0.1 PPL | 2x 快 |
| 200（默认） | ~95% | 基准 | 基准 |
| 500 | ~98% | 几乎可忽略 | 2.5x 慢 |

#### batch_size 对收敛的影响

每次迭代使用 `batch_size` 个样本计算梯度。更大的 batch 提供更稳定的梯度估计：

- **batch_size=4**：梯度噪声大，可能需要更多迭代收敛
- **batch_size=8**（默认）：精度与内存的良好平衡
- **batch_size=16**：梯度更准确，可能更少迭代收敛，但内存翻倍

**注意**：batch_size 不是每次迭代用同样的 batch——不同迭代会随机采样不同的 batch（从缓存的校准数据中），这提供了类似数据增强的效果。

### 数值示例：8 个权重的 AutoRound 优化过程

考虑一个小线性层，权重向量 $w = [0.73, -0.42, 0.15, 0.91, -0.28, 0.56, -0.84, 0.37]$

4-bit 对称量化，$scale = 0.91/7 = 0.13$

**RTN 量化**：

| 权重 | $w/s$ | floor | frac | RTN取整 | 量化值 | 误差 |
|---|---|---|---|---|---|---|
| 0.73 | 5.62 | 5 | 0.62 | 6 | 0.78 | -0.05 |
| -0.42 | -3.23 | -4 | 0.77 | -3 | -0.39 | -0.03 |
| 0.15 | 1.15 | 1 | 0.15 | 1 | 0.13 | +0.02 |
| 0.91 | 7.0 | 7 | 0.0 | 7 | 0.91 | 0.00 |
| -0.28 | -2.15 | -3 | 0.85 | -2 | -0.26 | -0.02 |
| 0.56 | 4.31 | 4 | 0.31 | 4 | 0.52 | +0.04 |
| -0.84 | -6.46 | -7 | 0.54 | -6 | -0.78 | -0.06 |
| 0.37 | 2.85 | 2 | 0.85 | 3 | 0.39 | -0.02 |

**初始化 $\theta_V$**：

$\theta_V = \sigma^{-1}(\text{frac}) = \sigma^{-1}([0.62, 0.77, 0.15, 0.0, 0.85, 0.31, 0.54, 0.85])$
$= [0.49, 1.21, -1.73, -\infty, 1.73, -0.80, 0.16, 1.73]$

（实践中 $-\infty$ 被截断为一个很负的值如 -10）

**假设第 1 次迭代的梯度**（基于完整层的输出误差）：

$\text{sign}(\nabla_{\theta_V} L) = [+1, +1, -1, 0, +1, -1, +1, -1]$

更新 $\theta_V \leftarrow \theta_V - 0.01 \times \text{sign}(\nabla)$：

- $\theta_{V,0}$: $0.49 - 0.01 = 0.48$ → $V_0 = \sigma(0.48) = 0.618$（略偏向上取整，不变）
- $\theta_{V,5}$: $-0.80 + 0.01 = -0.79$ → $V_5 = \sigma(-0.79) = 0.312$（更偏向下取整）
- $\theta_{V,6}$: $0.16 - 0.01 = 0.15$ → $V_6 = \sigma(0.15) = 0.537$（略偏向上取整）

**经过 200 次迭代后**，假设 $\theta_{V,6}$ 累积了足够的负梯度，从 0.16 降到 -1.5：
- $V_6 = \sigma(-1.5) = 0.18 < 0.5$ → 最终取向下取整
- 即 $\hat{w}_6 = -7 \times 0.13 = -0.91$（而非 RTN 的 -0.78）

这意味着 AutoRound "发现"对于这个权重，向下取整（虽然距离更远）在考虑整体层输出时反而更好。

## 在 LLM Compressor 中的实现

### AutoRoundModifier

```python
# src/llmcompressor/modifiers/autoround/base.py

class AutoRoundModifier(Modifier, QuantizationMixin):
    iters: int = 200                    # 优化迭代次数
    enable_torch_compile: bool = True   # 启用 torch.compile 加速
    batch_size: int = 8                 # 校准批次大小
    lr: float | None = None             # 学习率（None=自动）
    device_ids: str | None = None       # 多 GPU 设备映射
```

### 生命周期

```python
def on_initialize(self, state):
    # 1. 应用量化配置（QuantizationMixin）
    self.initialize_quantization(state.model)
    # 2. 冻结模型参数（只优化 V 和裁剪参数）
    for param in model.parameters():
        param.requires_grad = False

def on_start(self, state):
    # 注册输入捕获钩子（在 decoder layer 上）
    for layer in decoder_layers:
        self.register_hook(layer, capture_input_hook, "forward_pre")

def on_event(self, state, event):
    if event.type_ == SEQUENTIAL_EPOCH_END:
        # 逐层执行 AutoRound 优化
        for layer in decoder_layers:
            self._apply_autoround(layer, cached_inputs)

def on_end(self, state):
    # 恢复最佳参数，清理临时状态
    restore_best_params()
```

### 与 Intel auto-round 库的集成

LLM Compressor 的 AutoRound 实现依赖 Intel 的 `auto_round` 库：

```python
# 内部调用 auto_round 的优化逻辑
from auto_round import AutoRound as IntelAutoRound

def _apply_autoround(self, layer, cached_inputs):
    # 1. 包装 layer 为 AutoRound 格式
    wrapped_layer = wrap_for_autoround(layer)
    
    # 2. 暂停模型卸载（确保层在 GPU 上）
    with suspend_offloading(layer):
        # 3. 执行优化
        autoround = IntelAutoRound(
            model=wrapped_layer,
            iters=self.iters,
            batch_size=self.batch_size,
            lr=self.lr,
        )
        autoround.quantize()
    
    # 4. 提取量化参数并映射到 LLM Compressor 格式
    # auto_round 格式 → compressed-tensors 格式
    module.weight_scale = autoround.scale
    module.input_scale = autoround.act_scale
```

### 优化循环的内部实现细节

```python
class AutoRoundOptimizer:
    """Intel auto_round 库的核心优化逻辑（简化展示）"""
    
    def __init__(self, layer, quant_config, iters=200, batch_size=8, lr=None):
        self.layer = layer
        self.iters = iters
        self.batch_size = batch_size
        
        # 初始化可学习参数
        self.theta_V = {}      # 每个 Linear 层的舍入参数
        self.clip_min = {}     # 每个 Linear 层的下裁剪边界
        self.clip_max = {}     # 每个 Linear 层的上裁剪边界
        
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data
                # θ_V 初始化为 RTN 对应的值
                frac = self._compute_fractional_part(W, quant_config)
                self.theta_V[name] = torch.logit(frac).requires_grad_(True)
                # 裁剪参数初始化为 min/max
                self.clip_min[name] = W.min(dim=1).values.clone().requires_grad_(True)
                self.clip_max[name] = W.max(dim=1).values.clone().requires_grad_(True)
        
        # 选择学习率
        self.lr = lr if lr is not None else 1.0 / (iters * 0.8)
    
    def optimize(self, cached_inputs, reference_outputs):
        """执行完整的优化循环"""
        
        best_loss = float('inf')
        best_params = None
        
        for iteration in range(self.iters):
            # 随机采样一个 batch
            batch_idx = random.sample(range(len(cached_inputs)), self.batch_size)
            X_batch = cached_inputs[batch_idx]
            Y_ref = reference_outputs[batch_idx]
            
            # 前向传播（使用软量化的权重）
            self._apply_soft_quantization()
            Y_hat = self.layer(X_batch)
            
            # 计算 loss
            loss = F.mse_loss(Y_hat, Y_ref)
            
            # 反向传播
            loss.backward()
            
            # SignSGD 更新
            with torch.no_grad():
                for name in self.theta_V:
                    self.theta_V[name] -= self.lr * torch.sign(self.theta_V[name].grad)
                    self.clip_min[name] -= self.lr * torch.sign(self.clip_min[name].grad)
                    self.clip_max[name] -= self.lr * torch.sign(self.clip_max[name].grad)
                    
                    # 清零梯度
                    self.theta_V[name].grad.zero_()
                    self.clip_min[name].grad.zero_()
                    self.clip_max[name].grad.zero_()
            
            # 记录最佳
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = self._snapshot_params()
        
        # 恢复最佳参数并硬化舍入决策
        self._restore_params(best_params)
        self._harden_rounding()  # V > 0.5 → 1, V ≤ 0.5 → 0
```

### torch.compile 加速

AutoRound 支持使用 `torch.compile` 加速前向传播和反向传播：

```python
if self.enable_torch_compile:
    # 编译前向传播函数
    self._compiled_forward = torch.compile(
        self._soft_quantize_and_forward,
        mode="reduce-overhead",  # 减少 kernel launch 开销
    )
```

**加速效果**：torch.compile 通常可以将每次迭代加速 1.5-2x，主要来自：
- 算子融合（将多个小算子合并为一个 CUDA kernel）
- 减少 GPU-CPU 同步点
- 内存访问优化

对于 200 次迭代 × 多层，累积加速效果显著（整体量化时间减少 30-50%）。

### 量化参数映射

| AutoRound 参数 | LLM Compressor 参数 | 说明 |
|---|---|---|
| `scale` | `weight_scale` | 权重缩放因子 |
| `act_scale` | `input_scale` | 输入激活缩放因子 |
| `weight_global_scale` | `weight_global_scale` | 全局缩放（TENSOR_GROUP） |
| `act_max` | `input_global_scale` | 激活全局缩放 |

## 使用示例

### 示例 1：W4A16 AutoRound

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")

dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

recipe = AutoRoundModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
    iters=200,            # 优化迭代数
)

oneshot(
    model=model,
    recipe=recipe,
    dataset=dataset,
    num_calibration_samples=512,
    max_seq_length=2048,
)

model.save_pretrained("Llama-3-8B-W4A16-AutoRound")
```

### 示例 2：FP8 Block 量化

```python
recipe = AutoRoundModifier(
    targets="Linear",
    scheme="FP8_BLOCK",
    ignore=["lm_head"],
    iters=200,
    batch_size=8,
    enable_torch_compile=True,  # 加速优化
)
```

### 示例 3：NVFP4 量化

```python
recipe = AutoRoundModifier(
    targets="Linear",
    scheme="W4A4",            # NVFP4 权重 + 激活
    ignore=["lm_head", "re:.*mlp.gate$"],
    iters=200,
)
```

### 示例 4：多 GPU 加速

```python
recipe = AutoRoundModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
    iters=200,
    device_ids="0,1",  # 将不同层分配到不同 GPU
)
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `iters` | 200 | 每层的优化迭代次数 |
| `batch_size` | 8 | 优化时使用的批次大小 |
| `lr` | None（自动） | 学习率，None 时由 auto_round 自动确定 |
| `enable_torch_compile` | True | 启用 torch.compile 编译加速 |
| `device_ids` | None | 多 GPU 设备映射（如 "0,1" 或 "auto"） |

## 参数调优指南

| 需求 | 调整 |
|------|------|
| 提高精度 | 增大 `iters`（300-500）、增大 `batch_size`（16） |
| 加速量化 | 减小 `iters`（100）、启用 `torch_compile`、使用多 GPU |
| 减少内存 | 减小 `batch_size`（4）、使用 `device_ids` 分配 |

### 详细参数调优建议

#### iters 的选择策略

```
低比特（2-3 bit）→ 增大 iters（300-500），因为搜索空间中更多权重需要翻转
标准 4-bit      → 默认 200 通常足够
高比特（8-bit）  → 可减少到 100，因为 RTN 本身已经接近最优
FP8/NVFP4       → 200 是安全选择
```

#### lr 的手动设置

通常不需要手动设置 lr（auto_round 会自动选择）。但如果需要：

```python
# 较大 lr：更快收敛，但可能振荡
recipe = AutoRoundModifier(iters=100, lr=0.02, ...)

# 较小 lr：更稳定，但需要更多迭代
recipe = AutoRoundModifier(iters=500, lr=0.005, ...)
```

#### enable_torch_compile 的注意事项

```python
# torch.compile 在以下情况可能不适用：
# 1. 模型含动态 shape（如可变序列长度）→ 首次编译后 shape 变化会触发重编译
# 2. 使用了不支持的自定义算子
# 3. Python 3.12+ 某些版本的兼容性问题
# 遇到问题时关闭：
recipe = AutoRoundModifier(enable_torch_compile=False, ...)
```

## AutoRound 内存开销分析

对于一个 Linear 层（权重形状 $[C_{out}, C_{in}]$）：

| 组件 | 内存 | 说明 |
|---|---|---|
| 原始权重 $W$ | $C_{out} \times C_{in} \times 2$ bytes | FP16 |
| 舍入参数 $\theta_V$ | $C_{out} \times C_{in} \times 4$ bytes | FP32（需要梯度） |
| 裁剪参数 $\alpha, \beta$ | $2 \times C_{out} \times 4$ bytes | FP32 per-row |
| 梯度（$\theta_V$ 的） | $C_{out} \times C_{in} \times 4$ bytes | FP32 |
| 缓存激活 | $B \times T \times C_{in} \times 2$ bytes | FP16 |
| 参考输出 | $B \times T \times C_{out} \times 2$ bytes | FP16 |

**示例**：对于 Llama-3-8B 的一个 FFN 层（$C_{in}=4096, C_{out}=14336$）：
- $\theta_V$ + 梯度：$4096 \times 14336 \times 8 = 448$ MB
- 权重：$4096 \times 14336 \times 2 = 112$ MB
- 缓存输入（512 tokens）：$512 \times 4096 \times 2 = 4$ MB
- 总额外开销：约 450 MB/层

这就是为什么 AutoRound 内存消耗大于 GPTQ（后者只需要一个 $C_{in} \times C_{in}$ 的 Hessian 矩阵 = 64 MB）。

## AutoRound 的优缺点

**优点**：
- 4-bit 精度优异（通常略优于 GPTQ）
- 支持多种格式（W4A16、W4A4、FP8、MXFP4、NVFP4）
- 可学习的裁剪范围（适应异常值）
- 支持 torch.compile 加速
- 数值稳定（无 Cholesky 分解风险）

**缺点**：
- 量化速度最慢（每层 200 次迭代）
- 依赖外部 `auto_round` 库
- 内存开销较大（需要缓存输入和中间梯度）
- 超参数较多（iters、lr、batch_size）

## 与 GPTQ 的对比

| | AutoRound | GPTQ |
|--|-----------|------|
| 优化方式 | 迭代梯度优化 | 一次性 Hessian 补偿 |
| 优化目标 | 舍入方向 + 裁剪范围 | 误差传播到后续列 |
| 每层耗时 | 较长（200 次迭代） | 较短（单次遍历） |
| 内存开销 | 梯度 + 中间激活 | Hessian 矩阵 |
| 数值稳定性 | 好（梯度法） | 可能不稳定（Cholesky） |
| 精度 | 略优（尤其低比特） | 优秀 |

### 深度对比：AutoRound vs GPTQ 的本质区别

**GPTQ 的视角**：将量化看作一个有约束的线性代数问题。利用 Hessian（二阶信息）在闭式下计算最优补偿。一次性完成，无需迭代。但只优化"误差补偿到后续列"，不改变当前列的舍入方向。

**AutoRound 的视角**：将量化看作一个优化问题。通过梯度下降搜索最优的舍入方向组合。多次迭代逼近最优解。可以翻转任何权重的舍入方向，自由度更大。

**类比**：
- GPTQ 像是"下棋时走一步看一步，但每步都是局部最优"
- AutoRound 像是"全局考虑所有棋子的位置，通过多轮推演找到整体最优"

**精度差异的来源**：
1. AutoRound 可以翻转 RTN 已经选择的舍入方向（GPTQ 不行）
2. AutoRound 考虑层内所有 Linear 层的交互（GPTQ 逐模块独立）
3. AutoRound 的裁剪优化相当于在 scale 维度也做了优化

## 常见问题

### Q: 为什么 AutoRound 比 GPTQ 慢这么多？

**答**：GPTQ 对每个 Linear 层只需一次遍历（单次 Cholesky + 逐列量化），而 AutoRound 需要 200 次前向 + 反向传播。对于一个 Transformer 层含 7 个 Linear 层（q/k/v/o_proj + gate/up/down_proj），AutoRound 的计算量约为 GPTQ 的 50-100 倍。

但 AutoRound 的优势在于：
- 不需要 Cholesky 分解（可能失败）
- torch.compile 可以显著加速
- 可以利用多 GPU 并行

### Q: 什么时候选 AutoRound 而非 GPTQ？

选择 AutoRound 的场景：
1. 极低比特（2-3 bit）：AutoRound 的优势更明显
2. GPTQ Cholesky 失败时：AutoRound 无此问题
3. 需要同时量化权重和激活（W4A4）：AutoRound 的裁剪优化对此有帮助
4. 时间不是瓶颈：追求最高精度

选择 GPTQ 的场景：
1. 标准 4-bit 量化：GPTQ 精度已经很好
2. 需要快速量化：GPTQ 快 5-10 倍
3. 资源受限：GPTQ 内存开销更小
