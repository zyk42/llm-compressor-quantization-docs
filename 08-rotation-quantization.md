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

### 正交变换的保范性

对于正交矩阵 $R$（$R^TR = I$）：

$$||RW||_F = ||W||_F$$

即旋转不改变权重的 Frobenius 范数——模型的"信息量"不变，但分布更均匀。

### Hadamard 矩阵

Hadamard 矩阵是最常用的旋转基：

$$H_1 = [1], \quad H_{2n} = \frac{1}{\sqrt{2}}\begin{bmatrix} H_n & H_n \\ H_n & -H_n \end{bmatrix}$$

**性质**：
- 正交：$H^TH = I$
- 仅含 $\pm 1/\sqrt{n}$：计算高效
- 快速变换：$O(n \log n)$ 复杂度（类似 FFT）
- 降低相干性效果好

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

### 可融合 vs 在线旋转

```
可融合旋转（R1, R2）：
  W_new = R · W    # 旋转融入权重，推理无额外计算
  
在线旋转（R3, R4）：
  x_rotated = R · x  # 推理时需要额外矩阵乘法
```

### Block-Diagonal 旋转

当 hidden_size 很大时，完整旋转矩阵太大。使用块对角结构：

$$R = \begin{bmatrix} R_1 & & \\ & R_2 & \\ & & \ddots \end{bmatrix}$$

每个 $R_i$ 大小为 `transform_block_size × transform_block_size`。

### Transform 类型

| 类型 | 要求 | 计算效率 | 降相干性 |
|------|------|---------|---------|
| `hadamard` | 维度须为 2 的幂 | 最快 ($O(n\log n)$) | 好 |
| `random-hadamard` | 任意维度 | 较快 | 更好 |
| `random-matrix` | 任意维度 | 最慢 ($O(n^2)$) | 最好 |

### Norm 融合与 Embedding 居中

SpinQuant 还包含两个预处理步骤：

1. **Norm 融合**：将 RMSNorm/LayerNorm 的缩放系数融入后续 Linear 权重
2. **Embedding 居中**：将 Embedding 的均值移除（减小激活范围）

```python
# 伪代码
W_linear = diag(norm_weight) @ W_linear  # norm 融合
embedding -= embedding.mean(dim=-1, keepdim=True)  # 居中
```

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

**缺点**：
- R3/R4 在线旋转有推理开销
- Hadamard 要求维度为 2 的幂
- 降低相干性的效果依赖权重分布
- 不能修复量化本身的误差（只是预处理）

## 何时使用旋转量化

| 场景 | 推荐 |
|------|------|
| DataFree 量化（无校准数据） | SpinQuant R1+R2 + FP8 |
| 追求零推理开销 | 仅使用 R1+R2（可融合） |
| 与 GPTQ/AWQ 组合提升精度 | SpinQuant + GPTQ |
| 每层独立优化 | QuIP |
| Blackwell GPU 部署 | SpinQuant + NVFP4 |
