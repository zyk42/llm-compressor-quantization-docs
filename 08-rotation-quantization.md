# SpinQuant & QuIP 旋转量化详解

## 概述

旋转量化的核心思想：通过**正交变换（旋转矩阵）**将权重变换到更适合量化的空间。正交变换保持向量的 L2 范数不变，因此不改变模型的数学行为，但可以降低权重分布的"不相干性"（incoherence），使得量化更加均匀。

## 理论基础

### 相干性与量化误差

**相干性（Coherence）**衡量权重矩阵中元素大小的不均匀程度：

$$\mu(W) = \frac{n \cdot \max_{i,j} |W_{ij}|^2}{||W||_F^2}$$

- 相干性高：存在少数极大值，量化时产生大误差
- 相干性低：元素大小均匀，量化误差分布均匀

**关键定理**：对于相干性为 $\mu$ 的矩阵，均匀量化的 MSE 上界为：

$$E[||W - \hat{W}||_F^2] \leq O(\mu \cdot s^2 \cdot n)$$

因此，降低相干性可以直接降低量化误差的理论上界。

### 直觉理解：为什么降低相干性有帮助

考虑一个极端例子：2 维向量 $v = [10, 0]^T$。

- 范数：$||v|| = 10$
- 相干性：$\mu = 2 \times 100 / 100 = 2$（最大值）

用 INT4（范围 [-8, 7]，scale = 10/7 ≈ 1.43）量化：
- $v_1 = 10$ → 量化为 $7 \times 1.43 = 10.0$（误差 0）
- $v_2 = 0$ → 量化为 $0$（误差 0）

看似完美，但如果向量是 $v = [10, 0.1]^T$：
- $v_2 = 0.1$ → 量化为 $0$（误差 0.1，**相对误差 100%！**）

旋转后 $Rv = [7.07, 7.00]^T$（近似）：
- 两个分量的量化精度都很均匀
- 总体量化误差更小

### 正交变换的保范性

对于正交矩阵 $R$（$R^TR = I$）：

$$||RW||_F = ||W||_F$$

即旋转不改变权重的 Frobenius 范数——模型的"信息量"不变，但分布更均匀。

**更严格的等价性证明**：

对于线性层 $Y = XW^T$，施加右侧旋转 $W' = WR$（输入侧旋转）：

$$Y = XW^T = X(R^T R)W^T = (XR^T)(RW)^T = X' \cdot W'^T$$

其中 $X' = XR^T$（输入也做对应旋转）。由于 $R$ 是正交的，这个变换是可逆的，且不改变输出 $Y$。

### Hadamard 矩阵

Hadamard 矩阵是最常用的旋转基：

$$H_1 = [1], \quad H_{2n} = \frac{1}{\sqrt{2}}\begin{bmatrix} H_n & H_n \\ H_n & -H_n \end{bmatrix}$$

**性质**：
- 正交：$H^TH = I$
- 仅含 $\pm 1/\sqrt{n}$：计算高效
- 快速变换：$O(n \log n)$ 复杂度（类似 FFT）
- 降低相干性效果好

---

## 具体数值示例：Hadamard 旋转降低相干性

### 4×4 Hadamard 矩阵

$$H_4 = \frac{1}{2}\begin{bmatrix} 1 & 1 & 1 & 1 \\ 1 & -1 & 1 & -1 \\ 1 & 1 & -1 & -1 \\ 1 & -1 & -1 & 1 \end{bmatrix}$$

验证正交性：$H_4^T H_4 = I$（每行是单位向量，两两正交）。

### 示例：旋转一个有离群值的权重矩阵

假设权重矩阵（4×4）有一个离群值元素：

```
W = [[ 0.1,  0.2, -0.1,  8.0],    ← 第 0 行第 3 列有离群值
     [ 0.3, -0.1,  0.2,  0.1],
     [-0.2,  0.4,  0.1, -0.3],
     [ 0.1,  0.2, -0.2,  0.1]]
```

**变换前的相干性分析**：
- $\max|W_{ij}| = 8.0$
- $||W||_F^2 = 0.01+0.04+0.01+64+0.09+0.01+0.04+0.01+0.04+0.16+0.01+0.09+0.01+0.04+0.04+0.01 = 64.61$
- $\mu(W) = 4 \times 64 / 64.61 = 3.96$（接近最大值 4，很不均匀）

**施加右侧 Hadamard 旋转** $W' = W \cdot H_4^T$：

```
W'[0,:] = W[0,:] × H_4^T
        = [0.1, 0.2, -0.1, 8.0] × H_4^T
        = 1/2 × [0.1+0.2-0.1+8.0, 0.1-0.2-0.1-8.0, 0.1+0.2+0.1-8.0, 0.1-0.2+0.1+8.0]
        = 1/2 × [8.2, -8.2, -7.6, 8.0]
        = [4.1, -4.1, -3.8, 4.0]
```

```
W'[1,:] = 1/2 × [0.3-0.1+0.2+0.1, 0.3+0.1+0.2-0.1, 0.3-0.1-0.2-0.1, 0.3+0.1-0.2+0.1]
        = 1/2 × [0.5, 0.5, -0.1, 0.3]
        = [0.25, 0.25, -0.05, 0.15]
```

```
W'[2,:] = 1/2 × [-0.2+0.4+0.1-0.3, -0.2-0.4+0.1+0.3, -0.2+0.4-0.1+0.3, -0.2-0.4-0.1-0.3]
        = 1/2 × [0.0, -0.2, 0.4, -1.0]
        = [0.0, -0.1, 0.2, -0.5]
```

```
W'[3,:] = 1/2 × [0.1+0.2-0.2+0.1, 0.1-0.2-0.2-0.1, 0.1+0.2+0.2-0.1, 0.1-0.2+0.2+0.1]
        = 1/2 × [0.2, -0.4, 0.4, 0.2]
        = [0.1, -0.2, 0.2, 0.1]
```

变换后：

```
W' = [[ 4.10, -4.10, -3.80,  4.00],
      [ 0.25,  0.25, -0.05,  0.15],
      [ 0.00, -0.10,  0.20, -0.50],
      [ 0.10, -0.20,  0.20,  0.10]]
```

**变换后的相干性分析**：
- $\max|W'_{ij}| = 4.10$
- $||W'||_F^2 = ||W||_F^2 = 64.61$（保范性！）
- $\mu(W') = 4 \times 4.10^2 / 64.61 = 4 \times 16.81 / 64.61 = 1.04$

**相干性从 3.96 降低到 1.04**！原来集中在一个元素的能量被分散到了整行。

### 量化误差对比

用 INT4 对称量化（范围 [-8, 7]）：

**变换前**：scale = 8.0/7 = 1.143
- W[0,0] = 0.1 → round(0.1/1.143) = 0 → 量化值 0（误差 0.1）
- W[0,3] = 8.0 → round(8.0/1.143) = 7 → 量化值 8.0（误差 0）
- 第 0 行 MSE = (0.1² + 0.03² + 0.04² + 0²) / 4 ≈ 0.0031 ... 但关键是小值完全丢失

**变换后**：scale = 4.1/7 = 0.586
- W'[0,0] = 4.10 → round(4.10/0.586) = 7 → 量化值 4.10（误差 0）
- W'[0,2] = -3.80 → round(-3.80/0.586) = -6 → 量化值 -3.52（误差 0.28）
- 所有值都在有效量化范围内，精度分布均匀

---

## SpinQuant

### 算法原理

SpinQuant 针对 Transformer 架构定义了四类旋转：

**R1：残差流旋转（可融合）**
- 应用于 Attention 的 Q/K/V 输入和 FFN 的输入
- 可以融合到 LayerNorm 和 Linear 层的权重中
- **无运行时开销**

**R2：头内旋转（可融合）**
- 在 Attention 的每个头内部应用旋转
- 旋转 Q 和 K 的输出（head_dim 维度）
- 可融合到 Q/K 投影权重中
- **无运行时开销**

**R3：FFN 中间旋转（在线）**
- 在 FFN 的 gate/up 和 down 之间应用
- 不能融合（中间有非线性激活函数）
- **需要运行时计算**

**R4：Attention 输出旋转（在线）**
- 在 V 投影输出和 O 投影之间应用
- 不能融合（中间有 softmax 注意力）
- **需要运行时计算**

### Transformer Block 中旋转位置的详细图示

```
Input x ──────────────────────────────────────────────────────────┐
    │                                                              │ (残差连接)
    ▼                                                              │
┌─────────────────┐                                                │
│  RMSNorm        │ ← R1 融合到 norm.weight                       │
└────────┬────────┘                                                │
         │ (输出已被 R1 旋转)                                       │
         ▼                                                         │
    ┌────┼────┐                                                    │
    ▼    ▼    ▼                                                    │
┌──────┐┌──────┐┌──────┐                                          │
│Q_proj││K_proj││V_proj│ ← R1 融合到这些权重的列方向                │
│      ││      ││      │ ← R2 融合到输出方向(head_dim 维度)         │
└──┬───┘└──┬───┘└──┬───┘                                          │
   │       │       │                                               │
   ▼       ▼       ▼                                               │
┌──────────────────────┐                                           │
│  Multi-Head Attn     │                                           │
│  Q×K^T → softmax     │                                           │
│  → ×V                │                                           │
└──────────┬───────────┘                                           │
           │ (V 加权输出，维度 [heads, seq, head_dim])              │
           ▼                                                       │
    ┌─────────────┐                                                │
    │  R4 旋转     │ ← 在线计算！不能融合（softmax 是非线性的）      │
    └──────┬──────┘                                                │
           ▼                                                       │
    ┌─────────────┐                                                │
    │   O_proj    │ ← R4 融合到 O_proj 的输入方向                   │
    └──────┬──────┘   R1 融合到 O_proj 的输出方向                   │
           │                                                       │
           ▼                                                       │
         (+) ← ────────────────────────────────────────────────────┘
           │
           ▼                                                       ┐
┌─────────────────┐                                                │ (残差连接)
│  RMSNorm        │ ← R1 融合到 norm.weight                        │
└────────┬────────┘                                                │
         │                                                         │
    ┌────┴────┐                                                    │
    ▼         ▼                                                    │
┌────────┐┌────────┐                                               │
│Gate_proj││Up_proj │ ← R1 融合到列方向                             │
└────┬───┘└───┬────┘                                               │
     │        │                                                    │
     ▼        ▼                                                    │
┌──────────────────┐                                               │
│ SiLU(gate) × up  │ ← 非线性激活！                                │
└────────┬─────────┘                                               │
         ▼                                                         │
  ┌─────────────┐                                                  │
  │  R3 旋转     │ ← 在线计算！不能融合（SiLU 是非线性的）           │
  └──────┬──────┘                                                  │
         ▼                                                         │
  ┌─────────────┐                                                  │
  │  Down_proj  │ ← R3 融合到 down_proj 的输入方向                  │
  └──────┬──────┘   R1 融合到 down_proj 的输出方向                  │
         │                                                         │
         ▼                                                         │
       (+) ← ──────────────────────────────────────────────────────┘
         │
         ▼
   (下一层输入)
```

### 为什么 R1/R2 可以融合而 R3/R4 不行

**R1 可融合的关键**：R1 旋转发生在残差流上，而 RMSNorm 和 Linear 层都是线性操作。

设 R1 旋转矩阵为 $R$，原始计算流：

```
x → RMSNorm(x) → Linear(x) = W · RMSNorm(x)
```

插入 R1 后：

```
x → R·x → RMSNorm(R·x) → Linear(RMSNorm(R·x))
```

由于 RMSNorm 是 scale-invariant 的（只依赖于向量的方向），有：

$$\text{RMSNorm}(Rx) = \gamma' \cdot \frac{Rx}{||Rx||} = \gamma' \cdot \frac{Rx}{||x||}$$

（因为 $||Rx|| = ||x||$）

因此可以将 $R$ 吸收到 $\gamma$ 参数和后续 $W$ 的列方向中：
- $\gamma_{\text{new}} = \gamma$（RMSNorm 的归一化对旋转透明）
- $W_{\text{new}} = W \cdot R^T$（R 被吸收到权重的列方向）

**R2 可融合的原因**：R2 在 head_dim 维度上旋转 Q 和 K，等价于：

$$Q_{\text{new}} = Q \cdot R_2^T \Leftrightarrow W_Q^{\text{new}} = R_2 \cdot W_Q$$

直接修改 $W_Q$ 和 $W_K$ 的权重即可，无运行时开销。

**R3 不能融合的原因**：R3 位于 SiLU 激活函数之后：

```
hidden = SiLU(gate_proj(x)) * up_proj(x)
rotated = R3 · hidden
output = down_proj(rotated)
```

由于 `SiLU` 是非线性的，$R3$ 不能"穿过"它融入 gate_proj/up_proj 的权重。$R3$ 只能融入 down_proj 的输入方向（即 $W_{\text{down}}^{\text{new}} = W_{\text{down}} \cdot R_3^T$），但旋转本身仍需在激活上执行。

**R4 不能融合的原因**类似：softmax 注意力是非线性操作，R4 不能穿过它。

### 可融合 vs 在线旋转

```
可融合旋转（R1, R2）：
  W_new = R · W    # 旋转融入权重，推理无额外计算
  
在线旋转（R3, R4）：
  x_rotated = R · x  # 推理时需要额外矩阵乘法
```

**在线旋转的推理开销**：

对于维度为 $d$ 的向量，完整旋转的计算量为 $O(d^2)$。使用 Hadamard 快速变换可降至 $O(d \log d)$。使用块对角旋转可进一步降至 $O(d \cdot b)$（其中 $b$ 是块大小）。

### Block-Diagonal 旋转

当 hidden_size 很大时，完整旋转矩阵太大。使用块对角结构：

$$R = \begin{bmatrix} R_1 & & \\ & R_2 & \\ & & \ddots \end{bmatrix}$$

每个 $R_i$ 大小为 `transform_block_size × transform_block_size`。

### transform_block_size 对旋转结构的影响

`transform_block_size` 参数控制块对角旋转中每个子块的尺寸，直接影响旋转效果和计算开销：

**示例：hidden_size = 4096**

| transform_block_size | 块数量 | 每块大小 | 降相干性效果 | 在线旋转开销 |
|---------------------|--------|---------|-------------|-------------|
| None (全维度) | 1 | 4096×4096 | 最好 | $O(4096^2)$ = 16.7M FLOPs |
| 1024 | 4 | 1024×1024 | 好 | $O(4 \times 1024^2)$ = 4.2M FLOPs |
| 256 | 16 | 256×256 | 中等 | $O(16 \times 256^2)$ = 1.0M FLOPs |
| 64 | 64 | 64×64 | 较弱 | $O(64 \times 64^2)$ = 0.26M FLOPs |

**关键权衡**：
- 块越大 → 降相干性越好（跨通道的"能量分散"越充分）
- 块越小 → 在线旋转计算越快
- 当使用 Hadamard 快速变换时，块大小必须是 2 的幂

**块对角旋转的降相干性局限**：

块对角旋转只能在每个块内部重新分配元素的能量。如果两个离群值恰好位于不同的块中，它们之间无法互相"稀释"。因此：

```
全维度旋转：W 的任何离群值都会被分散到所有位置
块对角旋转：离群值只在其所在的块内被分散
```

**实践建议**：
- 仅 R1+R2（可融合）：建议用 `None`（全维度），因为无运行时开销
- R3/R4（在线）：建议用 `128` 或 `256`，平衡精度和速度
- GPU 批量推理：较大块（512+）可以更好利用 tensor core

### Transform 类型

| 类型 | 要求 | 计算效率 | 降相干性 |
|------|------|---------|---------|
| `hadamard` | 维度须为 2 的幂 | 最快 ($O(n\log n)$) | 好 |
| `random-hadamard` | 任意维度 | 较快 | 更好 |
| `random-matrix` | 任意维度 | 最慢 ($O(n^2)$) | 最好 |

**`random-hadamard` 的构造**：

```
R = D · H
```

其中 $D$ 是随机对角矩阵（对角元素为 $\pm 1$，随机选取），$H$ 是 Hadamard 矩阵。$D$ 引入随机性，打破 Hadamard 矩阵的固定结构（防止特定权重模式恰好与 Hadamard 结构对齐）。

**`random-matrix` 的构造**：

通过 QR 分解一个随机高斯矩阵得到正交矩阵：

```python
random_matrix = torch.randn(n, n)
Q, R = torch.linalg.qr(random_matrix)
# Q 即为均匀分布在正交群 O(n) 上的随机正交矩阵
```

### Norm 融合与 Embedding 居中

SpinQuant 还包含两个预处理步骤：

1. **Norm 融合**：将 RMSNorm/LayerNorm 的缩放系数融入后续 Linear 权重
2. **Embedding 居中**：将 Embedding 的均值移除（减小激活范围）

```python
# 伪代码
W_linear = diag(norm_weight) @ W_linear  # norm 融合
embedding -= embedding.mean(dim=-1, keepdim=True)  # 居中
```

**Norm 融合的详细说明**：

RMSNorm 的计算为 $\text{RMSNorm}(x) = \gamma \cdot x / \text{RMS}(x)$。后续的 Linear 层为 $y = W \cdot \text{RMSNorm}(x)$。

融合后：$y = W \cdot \gamma \cdot x / \text{RMS}(x) = (W \cdot \text{diag}(\gamma)) \cdot x / \text{RMS}(x)$

即将 $\gamma$ 吸收到 $W$ 中，然后将 RMSNorm 变为不含 $\gamma$ 的纯归一化操作。这为后续的 R1 旋转提供了更干净的计算图。

**Embedding 居中的作用**：

```
原始: embedding[token_id] = [0.5, 0.8, -0.3, 0.6, ...]
均值: mean = [0.2, 0.1, -0.1, 0.3, ...]
居中后: embedding[token_id] = [0.3, 0.7, -0.2, 0.3, ...]
```

移除均值后，激活的动态范围减小，有利于后续的量化。注意居中操作需要在首个 RMSNorm 之前补偿（或融入其参数）。

## QuIP / QuIP#

### 算法原理

QuIP（Quantization with Incoherence Processing）对每个 Linear 层独立应用两个旋转：

- **V 旋转（输入侧）**：旋转权重的列方向 → $W' = W \cdot V^T$
- **U 旋转（输出侧）**：旋转权重的行方向 → $W'' = U \cdot W'$

完整变换：

$$\hat{W} = U \cdot W \cdot V^T$$

推理时：

$$Y = X \cdot W^T = X \cdot V \cdot \hat{W}^T \cdot U^T$$

### QuIP vs SpinQuant 的区别

| | QuIP | SpinQuant |
|--|------|-----------|
| 旋转粒度 | 每个 Linear 独立旋转 | 考虑残差连接的全局一致性 |
| 可融合性 | 部分可融合 | R1/R2 可融合，R3/R4 在线 |
| 残差连接 | 不特殊处理 | 设计考虑残差流 |
| 运行时开销 | 可能较大 | R1/R2 无开销 |

**关键区别展开**：

SpinQuant 的 R1 之所以能做到"免费"，是因为它利用了 Transformer 的残差流结构：残差连接要求加法两侧的表示空间一致，因此 R1 旋转一旦施加，就必须在整个残差流中保持一致。这意味着：

- 所有从残差流读取的层（Q/K/V/Gate/Up）都自动获得 R1 旋转
- 所有写回残差流的层（O/Down）也必须补偿 R1 旋转

整个变换可以通过修改权重一次性完成，无需运行时计算。

而 QuIP 不考虑残差连接的约束，对每个层独立优化旋转，导致层与层之间可能需要"解旋转-再旋转"的额外步骤。

### QuIP# 改进

QuIP# 在 QuIP 基础上增加了格编码簿（lattice codebook），用非均匀量化进一步提升精度。但 LLM Compressor 中的实现主要使用 Hadamard 旋转部分。

## 在 LLM Compressor 中的实现

### SpinQuantModifier

```python
# src/llmcompressor/modifiers/transform/spinquant/base.py

class SpinQuantModifier(Modifier):
    rotations: list[str] = ["R1", "R2", "R4"]  # 应用哪些旋转
    transform_type: str = "hadamard"            # 旋转矩阵类型
    transform_block_size: int | None = None     # 块大小（None=full）
    fuse_norm_linears: bool = True              # 是否融合 Norm
    center_embeddings: bool = True              # 是否居中 Embedding
```

### QuIPModifier

```python
# src/llmcompressor/modifiers/transform/quip/base.py

class QuIPModifier(Modifier):
    rotations: list[str] = ["v", "u"]           # V 旋转和/或 U 旋转
    transform_type: str = "hadamard"            # 旋转矩阵类型
    transform_block_size: int | None = None     # 块大小
```

### DataFree 特性

旋转量化的一个重要优势是**不需要校准数据**：

- 旋转矩阵可以随机生成（或使用 Hadamard）
- 不依赖数据分布
- 使用 `DataFreePipeline`

**为什么不需要校准数据**：

与 SmoothQuant/AWQ 等方法不同，旋转量化的目标是降低权重的相干性，这完全是权重矩阵自身的属性，与输入数据无关。Hadamard 旋转的效果是确定的（给定维度就唯一确定），random-hadamard 的随机性来自对角矩阵 D，也不依赖数据。

### 旋转矩阵的生成与缓存

```python
def _get_rotation_matrix(self, size, block_size):
    """获取或生成旋转矩阵"""
    if self.transform_type == "hadamard":
        # Hadamard 矩阵是确定的，可以缓存
        R = hadamard_matrix(size)  # 递归构造
    elif self.transform_type == "random-hadamard":
        # D · H，其中 D 是随机 ±1 对角矩阵
        H = hadamard_matrix(size)
        D = torch.randint(0, 2, (size,)) * 2 - 1  # {-1, +1}
        R = torch.diag(D.float()) @ H
    elif self.transform_type == "random-matrix":
        # QR 分解随机矩阵
        G = torch.randn(size, size)
        R, _ = torch.linalg.qr(G)
    
    if block_size and block_size < size:
        # 构造块对角矩阵
        R_block = torch.block_diag(*[
            self._get_rotation_matrix(block_size, None)
            for _ in range(size // block_size)
        ])
        return R_block
    return R
```

## 使用示例

### 示例 1：SpinQuant + W4A16 量化

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.transform.spinquant import SpinQuantModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")

recipe = [
    # SpinQuant 旋转预处理（DataFree，无需校准）
    SpinQuantModifier(
        rotations=["R1", "R2", "R4"],
        transform_type="hadamard",
    ),
    # 量化
    QuantizationModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
    ),
]

# 无需 dataset（SpinQuant 不需要校准数据）
# 但 QuantizationModifier 的 W4A16 建议配合 GPTQ
oneshot(model=model, recipe=recipe)
```

### 示例 2：SpinQuant + FP8 DataFree

```python
recipe = [
    SpinQuantModifier(
        rotations=["R1", "R2"],       # 仅用可融合旋转
        transform_type="hadamard",
        fuse_norm_linears=True,
        center_embeddings=True,
    ),
    QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    ),
]

# 完全无需校准数据
oneshot(model=model, recipe=recipe)
```

### 示例 3：QuIP + W4A16

```python
from llmcompressor.modifiers.transform.quip import QuIPModifier

recipe = [
    QuIPModifier(
        rotations=["v", "u"],           # V + U 双侧旋转
        transform_type="random-hadamard",
        transform_block_size=128,       # 块对角旋转
    ),
    QuantizationModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
    ),
]

oneshot(model=model, recipe=recipe)
```

### 示例 4：SpinQuant + GPTQ 组合（最高精度）

```python
from llmcompressor.modifiers.gptq import GPTQModifier

recipe = [
    SpinQuantModifier(rotations=["R1", "R2", "R4"]),
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
        block_size=128,
    ),
]

oneshot(
    model=model,
    recipe=recipe,
    dataset=dataset,
    num_calibration_samples=512,
)
```

### 示例 5：仅可融合旋转 + 块对角在线旋转

```python
recipe = [
    SpinQuantModifier(
        rotations=["R1", "R2", "R3", "R4"],  # 全部旋转
        transform_type="hadamard",
        transform_block_size=128,  # R3/R4 使用 128×128 块对角
        # 注意：R1/R2 始终用全维度（因为它们可融合，无运行时开销）
        fuse_norm_linears=True,
        center_embeddings=True,
    ),
    QuantizationModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
    ),
]
```

## 参数说明

### SpinQuant 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rotations` | ["R1","R2","R4"] | 应用的旋转类型 |
| `transform_type` | "hadamard" | 旋转矩阵类型 |
| `transform_block_size` | None（全维度） | 块对角大小 |
| `fuse_norm_linears` | True | 融合 Norm 到 Linear |
| `center_embeddings` | True | Embedding 居中 |

### QuIP 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rotations` | ["v", "u"] | V 侧和/或 U 侧旋转 |
| `transform_type` | "hadamard" | 旋转矩阵类型 |
| `transform_block_size` | None（全维度） | 块对角大小 |

## 旋转量化的优缺点

**优点**：
- 无需校准数据（DataFree）
- 理论保证：降低量化误差上界
- R1/R2 可融合：无运行时开销
- 可与任何量化器组合
- 对所有权重分布都有效（不像 SmoothQuant 只针对离群通道）

**缺点**：
- R3/R4 在线旋转有推理开销（可通过块对角缓解）
- Hadamard 要求维度为 2 的幂（可用 random-hadamard 绕过）
- 降低相干性的效果依赖权重分布（已经很均匀的权重收益小）
- 不能修复量化本身的误差（只是预处理）
- 在线旋转增加了推理 kernel 的复杂度

## 何时使用旋转量化

| 场景 | 推荐 |
|------|------|
| DataFree 量化（无校准数据） | SpinQuant R1+R2 + FP8 |
| 追求零推理开销 | 仅使用 R1+R2（可融合） |
| 与 GPTQ/AWQ 组合提升精度 | SpinQuant + GPTQ |
| 每层独立优化 | QuIP |
| Blackwell GPU 部署 | SpinQuant + NVFP4 |
| 激活也需要量化 (W8A8) | SpinQuant R4 帮助均匀化注意力输出激活 |

## 与其他 Transform 方法的组合

旋转量化可以与 SmoothQuant、IMatrix 等方法组合使用：

```python
# 最强组合：SpinQuant + IMatrix + GPTQ
recipe = [
    SpinQuantModifier(rotations=["R1", "R2"]),    # 先旋转降相干
    IMatrixGatherer(targets="Linear"),             # 收集重要性
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        observer={"weights": "imatrix_mse"},       # 重要性加权
    ),
]
```

组合逻辑：旋转让权重分布更均匀 → IMatrix 识别残余的重要通道 → GPTQ 在最优初始化下迭代优化。
