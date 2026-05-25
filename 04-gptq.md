# GPTQ 量化详解

## 算法概述

GPTQ（Accurate Post-Training Quantization for Generative Pre-trained Transformers）是目前最主流的 4-bit 权重量化算法。它基于 OBQ（Optimal Brain Quantization）框架，通过 Hessian 矩阵指导的逐列量化与误差补偿，在低比特量化时显著优于 RTN。

**核心思想**：量化每一列权重时，利用 Hessian 逆矩阵将该列的量化误差最优地分配到尚未量化的列上，从而最小化整体输出误差。

## 理论基础

### 问题形式化

对于线性层 $Y = XW^T$，量化的目标是找到量化权重 $\hat{W}$ 使得输出误差最小：

$$\min_{\hat{W}} ||XW^T - X\hat{W}^T||_2^2 = \min_{\hat{W}} ||X(W - \hat{W})^T||_2^2$$

定义 Hessian 矩阵 $H = X^TX$（输入激活的协方差矩阵），上述目标等价于：

$$\min_{\hat{W}} (W - \hat{W})^T H (W - \hat{W})$$

### 从 OBQ 到 GPTQ

**OBQ 核心公式**：当量化第 $i$ 列权重 $w_i$ 到 $\hat{w}_i$ 时，最优的误差补偿为：

$$\delta_j = -\frac{w_i - \hat{w}_i}{[H^{-1}]_{ii}} \cdot [H^{-1}]_{ij}, \quad \forall j > i$$

其中：
- $[H^{-1}]_{ii}$：Hessian 逆矩阵的第 $i$ 个对角元素
- $[H^{-1}]_{ij}$：Hessian 逆矩阵第 $i$ 行第 $j$ 列的元素
- $\delta_j$：施加到第 $j$ 列权重的补偿值

**直觉理解**：
- $[H^{-1}]_{ii}$ 反映第 $i$ 列的"量化代价"——值越大，该列量化损失越小
- $[H^{-1}]_{ij}$ 反映列之间的"耦合程度"——用于在列间传播误差

**为什么 $[H^{-1}]_{ii}$ 越大表示量化代价越小？** 这是因为量化第 $i$ 列的损失公式为 $L_i = \frac{(w_i - \hat{w}_i)^2}{[H^{-1}]_{ii}}$，分母越大则损失越小。从物理意义上说，$[H^{-1}]_{ii}$ 大意味着该通道的输入激活能量较低，因此该列权重的微小变化不会对输出造成太大影响。

### Hessian 矩阵的构造

在 LLM Compressor 中，Hessian 通过校准数据的输入激活累积计算：

$$H = \frac{2}{N} \sum_{n=1}^{N} x_n x_n^T$$

其中 $x_n$ 是第 $n$ 个样本的输入向量（已展平）。代码中乘以 $\sqrt{2}$ 来简化：

```python
# src/llmcompressor/modifiers/gptq/gptq_quantize.py
inp = math.sqrt(2) * inp        # 预乘 √2
H += inp.matmul(inp.t())        # H = Σ (√2·x)(√2·x)^T = 2·Σ xx^T
```

**Hessian 构造的细节**：每个校准样本的输入激活 $x_n$ 的维度为 $[C_{in}]$（对于一个 token），对于序列中的多个 token，每个 token 都贡献一个 rank-1 更新到 Hessian 矩阵。最终 Hessian 的尺寸为 $[C_{in} \times C_{in}]$，这决定了 GPTQ 的内存开销。例如对于 hidden_size=4096 的模型，一个 Hessian 矩阵占用 $4096^2 \times 4 = 64$ MB（FP32）。

### 逆 Hessian 的计算

GPTQ 需要 $H^{-1}$。为确保数值稳定性：

1. **阻尼（Dampening）**：$H \leftarrow H + \lambda I$，其中 $\lambda = \text{percdamp} \cdot \text{mean}(\text{diag}(H))$
2. **Cholesky 分解**：$H = LL^T$
3. **Cholesky 求逆**：$H^{-1} = (LL^T)^{-1}$
4. **上三角 Cholesky**：$H^{-1} = U^T U$（方便后续逐列访问）

```python
damp = percdamp * torch.mean(torch.diag(H))
H[diag, diag] += damp
H = torch.linalg.cholesky(H)
H = torch.cholesky_inverse(H)
H = torch.linalg.cholesky(H, upper=True)  # 上三角形式
Hinv = H
```

**为什么使用上三角 Cholesky 分解 $H^{-1}$？** 在逐列量化过程中，需要频繁访问 $[H^{-1}]_{i, i:}$（从第 $i$ 个对角元素到行尾的切片）。将 $H^{-1}$ 分解为上三角形式 $U^TU$ 后，$U$ 的第 $i$ 行恰好对应需要的向量，使得内存访问模式连续，提高 GPU 利用率。

### dampening_frac（阻尼系数）的作用详解

阻尼系数 `dampening_frac`（即 `percdamp`）默认为 0.01，它的作用是防止 Hessian 矩阵奇异或病态导致 Cholesky 分解失败。

**阻尼过程的数值含义**：

假设 Hessian 对角线的均值为 $\bar{h} = 100$，则 $\lambda = 0.01 \times 100 = 1.0$。这意味着对角线上每个元素增加 1.0，确保最小特征值不小于 1.0。

**增大 dampening_frac 时的效果**：

| `dampening_frac` | 效果 |
|---|---|
| 0.01（默认） | 最小正则化，保留 Hessian 的原始信息，量化精度最高 |
| 0.05 | 中等正则化，数值更稳定，但会轻微降低补偿精度 |
| 0.10 | 强正则化，几乎不会出现 Cholesky 失败，但误差补偿效果衰减 |
| 0.50+ | 过强正则化，$H^{-1}$ 接近单位阵的标量倍，退化为无补偿的朴素量化 |

**何时需要增大 dampening_frac？**
- 校准样本过少（如 < 64），Hessian 秩不足
- 某些通道的激活完全为零，导致 Hessian 对角线有零元素
- 模型含有未训练的层（如新增加的 adapter 层）
- 出现 `RuntimeError: linalg.cholesky: the leading minor of order N is not positive-definite` 错误

### Block-wise 量化

为提高计算效率，GPTQ 将列分为大小为 `blocksize`（默认 128）的块：

```
W = [Block_0 | Block_1 | ... | Block_K]
     (128列)   (128列)         (128列)
```

**块内**：逐列量化 + 误差传播（仅在块内）
**块间**：将整个块的累积误差一次性传播到后续列

```python
for i1 in range(0, num_columns, blocksize):
    i2 = min(i1 + blocksize, num_columns)
    
    # 块内逐列量化
    for i in range(i2 - i1):
        w = W1[:, i]               # 当前列
        d = Hinv1[i, i]            # 对角元素
        q = fake_quantize(w, ...)  # 量化
        
        err1 = (w - q) / d         # 标准化误差
        # 块内误差传播
        W1[:, i:] -= err1.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
    
    # 块间误差传播
    W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
```

#### Block-wise 为什么能节省计算？

考虑一个 $4096 \times 4096$ 的权重矩阵：

**不分块（纯逐列）**：每量化一列，需要更新剩余所有列。第 $i$ 列量化后需要更新 $4096 - i - 1$ 列，总计约 $4096^2 / 2 \approx 8.4M$ 次列更新操作。由于每次更新只涉及单列，无法利用矩阵乘法的并行性。

**分块（blocksize=128）**：
- **块内**：每块 128 列内部逐列更新，每次只更新块内剩余列（最多 127 列），操作量小
- **块间**：每处理完一个块，将整个块的误差矩阵 $Err \in \mathbb{R}^{4096 \times 128}$ 通过一次矩阵乘法传播到后续所有列，充分利用 GPU 的矩阵乘法加速

计算量对比：
- 块内小循环：$32 \times 128^2/2 \times 4096 \approx 1.07B$ FLOPS（但内循环小，开销低）
- 块间矩阵乘法：$32 \times 4096 \times 128 \times (4096 - 128k)$，GPU 高效执行

实际加速：blocksize=128 通常比纯逐列方式快 **3-5 倍**。

### 量化损失度量

每列的量化损失为：

$$L_i = \frac{(w_i - \hat{w}_i)^2}{[H^{-1}]_{ii}^2}$$

总损失为所有列的累加（除以 2）。此值可用于评估量化质量。

## Activation Ordering（激活排序）

GPTQ 支持三种列处理顺序，影响精度：

### Static Ordering（默认）

按 Hessian 对角线 $[H^{-1}]_{ii}$ 降序排列——先量化"代价最小"的列：

```python
perm = torch.argsort(torch.diag(H), descending=True)
W = W[:, perm]
H = H[perm][:, perm]
```

**原理**：对角值大的列量化误差对输出影响小，先处理它们可以为后续的"敏感列"留出更多补偿空间。

#### actorder="static" 的详细步骤

以一个 4 列权重矩阵为例，假设 Hessian 逆矩阵对角线为 $[H^{-1}]_{00}=0.5, [H^{-1}]_{11}=2.0, [H^{-1}]_{22}=0.8, [H^{-1}]_{33}=1.5$：

1. **排序**：按降序排列得到索引 $[1, 3, 2, 0]$（对应值 $2.0 > 1.5 > 0.8 > 0.5$）
2. **重排权重**：$W_{reordered} = W[:, [1, 3, 2, 0]]$
3. **重排 Hessian**：行列同步置换 $H_{reordered} = H[[1,3,2,0]][:, [1,3,2,0]]$
4. **按新顺序执行 GPTQ**：先量化原始第 1 列（代价最小），最后量化原始第 0 列（代价最大）
5. **量化完成后**：将权重恢复原始顺序 $W_{final}[:, perm] = W_{quantized}$

**为什么先量化代价小的列更好？**

- 早期量化的列误差会传播到后续列
- 代价小的列本身量化误差就小（$[H^{-1}]_{ii}$ 大，损失公式的分母大）
- 后续的"敏感列"（$[H^{-1}]_{ii}$ 小）能获得前面所有列的误差补偿，有更多"修正余地"
- 实验表明，`actorder="static"` 通常比顺序量化提升 0.1-0.3 perplexity

### Group Ordering

仅在分组量化中使用。在每个 group 内按重要性排序：

- 先计算排列后的 scale/zero_point
- 使用 `g_idx` 追踪每列属于哪个 group

### Weight Ordering

基于权重重要性（而非 Hessian）的排列，量化后将权重恢复原始顺序并存储 `g_idx` 映射。

## 完整数值示例：4x4 权重矩阵的 GPTQ 量化

下面通过一个完整的数值示例展示 GPTQ 的内部工作过程。

### 设定

假设有一个 $4 \times 4$ 的权重矩阵 $W$ 和 4 个校准样本的输入激活：

$$W = \begin{pmatrix} 0.8 & -0.3 & 0.5 & 0.1 \\ -0.2 & 0.6 & -0.4 & 0.7 \\ 0.3 & 0.1 & 0.9 & -0.5 \\ -0.1 & 0.4 & 0.2 & 0.8 \end{pmatrix}$$

校准输入（4 个样本，每个 4 维）：

$$X = \begin{pmatrix} 1.0 & 0.5 & -0.3 & 0.2 \\ 0.8 & -0.2 & 0.6 & 0.1 \\ -0.5 & 1.0 & 0.4 & -0.3 \\ 0.3 & 0.7 & -0.1 & 0.9 \end{pmatrix}$$

量化为 4-bit 对称（$q_{min}=-8, q_{max}=7$），per-channel scale。

### 步骤 1：构造 Hessian 矩阵

$$H = \frac{2}{4} \sum_{n=1}^{4} x_n x_n^T = \frac{1}{2} X^T X$$

计算 $X^T X$：

$$X^T X = \begin{pmatrix} 1.98 & -0.15 & 0.13 & 0.34 \\ -0.15 & 1.78 & -0.25 & 0.26 \\ 0.13 & -0.25 & 0.62 & -0.33 \\ 0.34 & 0.26 & -0.33 & 0.95 \end{pmatrix}$$

$$H = \frac{1}{2} X^T X = \begin{pmatrix} 0.99 & -0.075 & 0.065 & 0.17 \\ -0.075 & 0.89 & -0.125 & 0.13 \\ 0.065 & -0.125 & 0.31 & -0.165 \\ 0.17 & 0.13 & -0.165 & 0.475 \end{pmatrix}$$

### 步骤 2：添加阻尼

对角线均值 $\bar{h} = (0.99 + 0.89 + 0.31 + 0.475)/4 = 0.666$

阻尼值 $\lambda = 0.01 \times 0.666 = 0.00666$

$$H_{damped} = H + 0.00666 \cdot I$$

对角线变为：$[0.997, 0.897, 0.317, 0.482]$

### 步骤 3：Cholesky 分解与求逆

对 $H_{damped}$ 执行 Cholesky 分解得到 $L$，然后求逆得到：

$$H^{-1} \approx \begin{pmatrix} 1.08 & 0.11 & -0.13 & -0.35 \\ 0.11 & 1.22 & 0.43 & -0.15 \\ -0.13 & 0.43 & 3.82 & 1.08 \\ -0.35 & -0.15 & 1.08 & 2.61 \end{pmatrix}$$

对角线：$[H^{-1}]_{00}=1.08,\ [H^{-1}]_{11}=1.22,\ [H^{-1}]_{22}=3.82,\ [H^{-1}]_{33}=2.61$

### 步骤 4：逐列量化（不启用 actorder，按顺序处理）

#### 量化第 0 列

当前 $W$ 的第 0 列：$w_0 = [0.8, -0.2, 0.3, -0.1]^T$

确定 scale：$s_0 = \max(|w_0|) / 7 = 0.8 / 7 = 0.114$

量化：
- $0.8 / 0.114 = 7.0 \to$ clip 到 7 $\to q = 7 \times 0.114 = 0.8$
- $-0.2 / 0.114 = -1.75 \to$ round 到 -2 $\to q = -2 \times 0.114 = -0.228$
- $0.3 / 0.114 = 2.63 \to$ round 到 3 $\to q = 3 \times 0.114 = 0.342$
- $-0.1 / 0.114 = -0.877 \to$ round 到 -1 $\to q = -1 \times 0.114 = -0.114$

量化误差 $e_0 = w_0 - \hat{w}_0 = [0.0, 0.028, -0.042, 0.014]^T$

标准化误差 $err_0 = e_0 / [H^{-1}]_{00} = e_0 / 1.08 = [0.0, 0.026, -0.039, 0.013]^T$

#### 误差补偿到第 1, 2, 3 列

对第 1 列：$\delta_1 = -err_0 \times [H^{-1}]_{01} = -[0.0, 0.026, -0.039, 0.013] \times 0.11$
$$W[:, 1] \leftarrow W[:, 1] - err_0 \times 0.11$$

对第 2 列：$\delta_2 = -err_0 \times [H^{-1}]_{02} = -err_0 \times (-0.13)$
$$W[:, 2] \leftarrow W[:, 2] + err_0 \times 0.13$$

对第 3 列：$\delta_3 = -err_0 \times [H^{-1}]_{03} = -err_0 \times (-0.35)$
$$W[:, 3] \leftarrow W[:, 3] + err_0 \times 0.35$$

更新后：
- $W[:, 1] = [-0.3, 0.597, 0.104, 0.399]^T$（变化微小）
- $W[:, 2] = [0.5, -0.397, 0.905, 0.198]^T$
- $W[:, 3] = [0.1, 0.709, -0.514, 0.805]^T$

#### 量化第 1 列（使用更新后的值）

重复上述过程：确定 scale、量化、计算误差、向第 2、3 列传播补偿...

#### 关键观察

- 第 2 列的 $[H^{-1}]_{22} = 3.82$ 最大，说明该列量化"代价最小"——如果启用 `actorder="static"`，它会被第一个处理
- 第 0 列的 $[H^{-1}]_{00} = 1.08$ 最小，说明它最"敏感"——启用 actorder 后最后处理，能享受前面所有列的误差补偿

### 步骤 5（actorder="static" 变体）

若启用 actorder，处理顺序变为 $[2, 3, 1, 0]$（按 $[H^{-1}]_{ii}$ 降序）：
1. 先量化第 2 列（$[H^{-1}]_{22}=3.82$，最不敏感）
2. 再量化第 3 列（$[H^{-1}]_{33}=2.61$）
3. 再量化第 1 列（$[H^{-1}]_{11}=1.22$）
4. 最后量化第 0 列（$[H^{-1}]_{00}=1.08$，最敏感，此时已接收了前 3 列的补偿）

## 在 LLM Compressor 中的实现

### GPTQModifier 生命周期

```python
# src/llmcompressor/modifiers/gptq/base.py

class GPTQModifier(Modifier, QuantizationMixin):
    # 参数
    block_size: int = 128           # 块大小
    dampening_frac: float = 0.01    # 阻尼系数
    actorder: str = "static"        # 激活排序方式
    offload_hessians: bool = False  # 将 Hessian 卸载到 CPU
    
    def on_initialize(self, state):
        # 1. 应用量化配置到模型（从 QuantizationMixin）
        self.initialize_quantization(state.model)
        # 2. 为每个目标模块创建空 Hessian 矩阵
        for module in target_modules:
            H = make_empty_hessian(module)
    
    def on_start(self, state):
        # 注册前向钩子：捕获输入以累积 Hessian
        for module in target_modules:
            self.register_hook(module, self._accumulate_hessian_hook, "forward_pre")
    
    def on_event(self, state, event):
        if event.type_ == SEQUENTIAL_EPOCH_END:
            # 对每个模块执行 GPTQ 量化
            for module in target_modules:
                loss, W_q, scale, zp, g_idx = quantize_weight(
                    module, quant_args, hessian,
                    blocksize=self.block_size,
                    percdamp=self.dampening_frac,
                )
                # 更新模块权重和量化参数
                module.weight.data = W_q
                module.weight_scale = scale
                module.weight_zero_point = zp
```

### Hessian 累积钩子

```python
def _accumulate_hessian_hook(self, module, inp):
    """每次前向传播时累积 Hessian"""
    H, num_samples = accumulate_hessian(
        inp[0], module, module._hessian, module._num_samples
    )
    module._hessian = H
    module._num_samples = num_samples
```

**Hessian 累积的精确过程**：

```python
def accumulate_hessian(inp, module, H, num_samples):
    """
    inp: [batch_size, seq_len, hidden_dim] 或 [batch_size, hidden_dim]
    """
    # 1. 展平为 2D：[num_tokens, hidden_dim]
    if inp.ndim == 3:
        inp = inp.reshape(-1, inp.shape[-1])  # [B*T, C_in]
    
    num_new = inp.shape[0]
    
    # 2. 增量更新 Hessian（维持滑动均值）
    # H_new = (num_samples * H_old + inp^T @ inp) / (num_samples + num_new)
    H *= num_samples / (num_samples + num_new)
    inp = math.sqrt(2.0 / (num_samples + num_new)) * inp.float()
    H += inp.t() @ inp
    
    return H, num_samples + num_new
```

### DDP 分布式 Hessian 汇总

多 GPU 时，各 rank 分别在自己的数据上计算部分 Hessian，然后汇总：

```python
# 各 rank 的 Hessian 求和
dist.reduce(hessian, dst=owner_rank, op=dist.ReduceOp.SUM)
# owner_rank 执行 quantize_weight，然后广播结果
```

### 完整的量化流程内部细节

```python
def quantize_weight(module, quant_args, H, blocksize=128, percdamp=0.01):
    """GPTQ 权重量化的完整流程"""
    
    W = module.weight.data.clone().float()  # [C_out, C_in]
    num_rows, num_columns = W.shape
    
    # === 步骤 1: Hessian 预处理 ===
    dead_mask = torch.diag(H) == 0  # 找出"死"通道（从未被激活）
    H[dead_mask, dead_mask] = 1     # 避免除以零
    W[:, dead_mask] = 0             # 死通道权重归零
    
    # === 步骤 2: 激活排序（如果启用）===
    if actorder == "static":
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
    
    # === 步骤 3: Cholesky 求逆 ===
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(num_columns)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    Hinv = torch.linalg.cholesky(H, upper=True)
    
    # === 步骤 4: Block-wise 量化 ===
    Losses = torch.zeros(num_rows)
    
    for i1 in range(0, num_columns, blocksize):
        i2 = min(i1 + blocksize, num_columns)
        block_cols = i2 - i1
        
        W1 = W[:, i1:i2].clone()           # 当前块权重
        Hinv1 = Hinv[i1:i2, i1:i2]         # 当前块的 Hinv 子矩阵
        Err1 = torch.zeros_like(W1)         # 累积误差
        
        for i in range(block_cols):
            w = W1[:, i]                     # 当前列
            d = Hinv1[i, i]                  # 对角元素
            
            # 量化
            q = fake_quantize(w, scale, zero_point, qmin, qmax)
            
            # 记录损失
            Losses += ((w - q) ** 2 / d ** 2)
            
            # 计算标准化误差
            err = (w - q) / d
            Err1[:, i] = err
            
            # 块内误差传播
            W1[:, i+1:] -= err.unsqueeze(1) * Hinv1[i, i+1:].unsqueeze(0)
        
        # 将量化后的块写回
        W[:, i1:i2] = Q1  # Q1 包含量化值
        
        # 块间误差传播（一次矩阵乘法）
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    
    # === 步骤 5: 恢复原始列顺序 ===
    if actorder == "static":
        invperm = torch.argsort(perm)
        W = W[:, invperm]
    
    return Losses.sum() / 2, W, scale, zero_point, g_idx
```

## 使用示例

### 示例 1：标准 W4A16 GPTQ 量化

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 准备校准数据
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

# GPTQ W4A16 量化
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
    block_size=128,
    dampening_frac=0.01,
)

oneshot(
    model=model,
    recipe=recipe,
    dataset=dataset,
    num_calibration_samples=512,
    max_seq_length=2048,
)

model.save_pretrained("Llama-3-8B-W4A16-GPTQ")
```

### 示例 2：GPTQ + Activation Ordering

```python
from compressed_tensors.quantization import QuantizationArgs

recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
    actorder="static",    # 按 Hessian 对角线排序（推荐）
    block_size=128,
)
```

### 示例 3：DDP 多卡加速

```python
# 脚本 gptq_ddp.py
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

oneshot(
    model=model,
    recipe=recipe,
    dataset=dataset,
    num_calibration_samples=512,
)
```

启动命令：
```bash
torchrun --nproc_per_node=4 gptq_ddp.py
```

### 示例 4：GPTQ + FP4 格式

```python
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A4",           # NVFP4 权重 + FP4 激活
    ignore=["lm_head", "re:.*mlp.gate$"],
    block_size=128,
)
```

## 关键参数调优

| 参数 | 默认值 | 调优建议 |
|------|--------|---------|
| `block_size` | 128 | 增大（256）可能略提精度但更慢；减小（64）加速但精度下降 |
| `dampening_frac` | 0.01 | Cholesky 失败时增大到 0.05-0.1 |
| `actorder` | "static" | 启用 static 通常提升 0.1-0.3 perplexity |
| `offload_hessians` | False | 内存不足时启用，将 Hessian 放到 CPU |
| `num_calibration_samples` | 512 | 128-1024 范围内调整 |

### 参数交互效应详解

**block_size 与精度的关系**：

- `block_size=1`：退化为纯逐列处理，每列误差立即传播到所有后续列，理论精度最高但极慢
- `block_size=128`：块内误差只在 128 列内传播，块间通过矩阵乘法传播，精度与速度的最佳平衡
- `block_size=num_columns`：所有列在一个块内处理，块间传播为空操作，精度同 block_size=1 但内存更大

**校准样本数的影响**：

| 样本数 | Hessian 质量 | 建议场景 |
|---|---|---|
| 32-64 | 低秩估计，可能需增大 dampening | 快速测试 |
| 128-256 | 适中，大多数情况足够 | 标准量化 |
| 512 | 稳定的 Hessian 估计 | 推荐默认 |
| 1024+ | 边际收益递减 | 追求极致精度 |

## GPTQ 的优缺点

**优点**：
- 4-bit 量化精度极高（接近 FP16 原始模型）
- 理论基础扎实（最优误差补偿）
- 支持多种量化策略（per-channel、per-group、per-block）
- 广泛的硬件支持

**缺点**：
- 需要校准数据（通常 512 样本）
- 量化速度较慢（需要计算 Hessian + Cholesky 分解）
- Hessian 占用额外显存（$C_{in} \times C_{in}$ 矩阵）
- 数值不稳定时 Cholesky 可能失败

## 与其他算法的对比

| | GPTQ | AWQ | AutoRound | RTN |
|--|------|-----|-----------|-----|
| 精度（W4A16） | ★★★★★ | ★★★★★ | ★★★★★ | ★★★ |
| 量化速度 | ★★★ | ★★★★ | ★★ | ★★★★★ |
| 内存开销 | ★★★ | ★★★★ | ★★★ | ★★★★★ |
| 理论基础 | 最优补偿 | 通道缩放 | 优化舍入 | 无 |
| 校准数据 | 必需 | 必需 | 必需 | 可选 |

## 常见问题与故障排除

### Cholesky 分解失败

**错误信息**：`RuntimeError: linalg.cholesky: the leading minor of order N is not positive-definite`

**原因及解决方案**：

1. **Hessian 接近奇异**：增大 `dampening_frac`（0.01 → 0.05 → 0.1）
2. **死通道**：某些输入通道在校准数据中从未被激活，导致 Hessian 对应行列全为零。代码会自动检测并处理（`dead_mask`），但极端情况可能仍有问题
3. **校准数据过少**：增加 `num_calibration_samples`
4. **数值精度**：确保 Hessian 累积在 FP32 下进行（代码默认如此）

### 显存不足（OOM）

**Hessian 内存估算**：
- hidden_size=4096: $4096^2 \times 4 = 64$ MB per layer
- hidden_size=8192: $8192^2 \times 4 = 256$ MB per layer
- hidden_size=12288: $12288^2 \times 4 = 576$ MB per layer

**解决方案**：
- 设置 `offload_hessians=True` 将 Hessian 卸载到 CPU
- 使用 DDP 多卡分摊显存
- 减少 `num_calibration_samples`
