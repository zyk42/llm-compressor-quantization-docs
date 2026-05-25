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

### Hessian 矩阵的构造

在 LLM Compressor 中，Hessian 通过校准数据的输入激活累积计算：

$$H = \frac{2}{N} \sum_{n=1}^{N} x_n x_n^T$$

其中 $x_n$ 是第 $n$ 个样本的输入向量（已展平）。代码中乘以 $\sqrt{2}$ 来简化：

```python
# src/llmcompressor/modifiers/gptq/gptq_quantize.py
inp = math.sqrt(2) * inp        # 预乘 √2
H += inp.matmul(inp.t())        # H = Σ (√2·x)(√2·x)^T = 2·Σ xx^T
```

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

### Group Ordering

仅在分组量化中使用。在每个 group 内按重要性排序：

- 先计算排列后的 scale/zero_point
- 使用 `g_idx` 追踪每列属于哪个 group

### Weight Ordering

基于权重重要性（而非 Hessian）的排列，量化后将权重恢复原始顺序并存储 `g_idx` 映射。

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

### DDP 分布式 Hessian 汇总

多 GPU 时，各 rank 分别在自己的数据上计算部分 Hessian，然后汇总：

```python
# 各 rank 的 Hessian 求和
dist.reduce(hessian, dst=owner_rank, op=dist.ReduceOp.SUM)
# owner_rank 执行 quantize_weight，然后广播结果
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
