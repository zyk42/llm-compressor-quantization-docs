# IMatrix 重要性加权校准详解

## 算法概述

IMatrix（Importance Matrix）是一种校准增强技术，通过收集输入激活的**二阶矩统计**来为每个权重通道分配重要性权重。传统的 MSE Observer 对所有通道等权处理，而 IMatrix 让频繁被大激活值"激活"的通道获得更大的量化保护。

**核心思想**：不是所有通道同等重要——激活值经常很大的通道对输出影响更大，量化时应该更精确地保护它们。

## 理论基础

### 通道重要性定义

IMatrix 定义第 $j$ 个输入通道的重要性为激活的二阶矩：

$$\text{importance}_j = E[x_j^2] = \frac{1}{N} \sum_{n=1}^{N} x_{n,j}^2$$

其中 $x_{n,j}$ 是第 $n$ 个样本在第 $j$ 个通道的激活值。

### 为什么用二阶矩？

对于线性层 $y = Wx$，权重 $w_j$ 的量化误差 $\Delta w_j$ 对输出的影响为：

$$\Delta y = \sum_j \Delta w_j \cdot x_j$$

输出误差的期望：

$$E[\Delta y^2] = \sum_j (\Delta w_j)^2 \cdot E[x_j^2] = \sum_j (\Delta w_j)^2 \cdot \text{importance}_j$$

因此，$E[x_j^2]$ 大的通道，其量化误差会被更大地放大。

### 加权 MSE 目标

传统 MSE Observer 的目标：

$$\min_{s, z} \sum_j (w_j - \hat{w}_j)^2$$

IMatrix 加权 MSE Observer 的目标：

$$\min_{s, z} \sum_j \text{importance}_j \cdot (w_j - \hat{w}_j)^2$$

这使得高重要性通道获得更小的量化误差。

### Grid Search 中的重要性加权

IMatrix MSE Observer 的 grid search 过程：

```
for each candidate (min_val, max_val):
    scale = (max_val - min_val) / (qmax - qmin)
    w_q = quantize(w, scale, zero_point)
    # 加权 MSE
    loss = sum(importance * (w - w_q)^2)
    
choose (min_val, max_val) with minimum weighted loss
```

## 两阶段工作流

IMatrix 的使用分为两个阶段：

### 阶段一：IMatrixGatherer 收集统计

```
校准数据 → 前向传播 → 收集每层的 E[x²] → 存储在模块上
```

### 阶段二：量化器使用 imatrix_mse Observer

```
量化器读取 E[x²] → 加权 MSE 优化 → 得到更优的 scale/zp
```

## 在 LLM Compressor 中的实现

### IMatrixGatherer

```python
# src/llmcompressor/modifiers/transform/imatrix/base.py

class IMatrixGatherer(Modifier):
    """
    收集 E[x²] 统计量，存储在目标模块上供后续 Observer 使用。
    本身不执行量化。
    """
    targets: str | list = "Linear"
    ignore: list = []
    
    def on_start(self, state):
        # 为每个目标模块注册输入捕获钩子
        for module in target_modules:
            module._imatrix_sum = torch.zeros(in_features)
            module._imatrix_count = 0
            self.register_hook(module, self._accumulate_hook, "forward_pre")
    
    def _accumulate_hook(self, module, inp):
        """累积 x² 统计"""
        x = inp[0]
        # x shape: [batch, seq_len, hidden_dim]
        x_flat = x.reshape(-1, x.shape[-1])
        module._imatrix_sum += (x_flat ** 2).sum(dim=0)
        module._imatrix_count += x_flat.shape[0]
    
    def on_event(self, state, event):
        if event.type_ == SEQUENTIAL_EPOCH_END:
            # DDP 同步：汇总各 rank 的统计
            for module in target_modules:
                dist.all_reduce(module._imatrix_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(module._imatrix_count, op=dist.ReduceOp.SUM)
```

### IMatrix MSE Observer

```python
# src/llmcompressor/observers/imatrix.py

class IMatrixMSEObserver(Observer):
    """使用 imatrix 重要性加权的 MSE Observer"""
    
    def get_qparams(self):
        # 获取 imatrix 统计
        imatrix = self.module._imatrix_sum / self.module._imatrix_count
        
        # 加权 MSE grid search
        best_loss = float('inf')
        for shrink in linspace(1-maxshrink, 1, grid):
            # 候选范围
            min_val = self.min_vals * shrink
            max_val = self.max_vals * shrink
            scale = (max_val - min_val) / (qmax - qmin)
            
            # 量化
            w_q = fake_quantize(weight, scale, zero_point)
            
            # 加权 MSE
            loss = (imatrix * (weight - w_q) ** 2).sum()
            
            if loss < best_loss:
                best_loss = loss
                best_scale = scale
        
        return {"scale": best_scale, "zero_point": best_zp, ...}
```

### 统计量传递机制

IMatrixGatherer 将统计量直接存储在模块的属性上：

```python
# IMatrixGatherer 存储
module._imatrix_sum = accumulated_sum      # Σ x²
module._imatrix_count = sample_count       # N

# IMatrixMSEObserver 读取
importance = module._imatrix_sum / module._imatrix_count  # E[x²]
```

这种设计使得两个 Modifier 可以独立配置但通过模块属性通信。

## 使用示例

### 示例 1：IMatrix + RTN W4A16

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer
from llmcompressor.modifiers.quantization import QuantizationModifier

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

recipe = [
    # 第一步：收集 IMatrix 统计
    IMatrixGatherer(
        targets="Linear",
        ignore=["lm_head"],
    ),
    # 第二步：使用 imatrix_mse observer 进行量化
    QuantizationModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
        observer={"weights": "imatrix_mse"},  # 指定加权 MSE observer
    ),
]

oneshot(
    model=model,
    recipe=recipe,
    dataset=dataset,
    num_calibration_samples=512,
    max_seq_length=2048,
)
```

### 示例 2：IMatrix + GPTQ

```python
from llmcompressor.modifiers.gptq import GPTQModifier

recipe = [
    IMatrixGatherer(targets="Linear", ignore=["lm_head"]),
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
        observer={"weights": "imatrix_mse"},
    ),
]

oneshot(
    model=model,
    recipe=recipe,
    dataset=dataset,
    num_calibration_samples=512,
)
```

### 示例 3：IMatrix + FP8

```python
recipe = [
    IMatrixGatherer(targets="Linear", ignore=["lm_head"]),
    QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
        observer={"weights": "imatrix_mse"},
    ),
]
```

## 参数说明

### IMatrixGatherer 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `targets` | "Linear" | 收集统计的目标层 |
| `ignore` | [] | 跳过的层 |

### IMatrix MSE Observer 参数

Observer 的参数通过量化配置传递：

| 参数 | 说明 |
|------|------|
| `maxshrink` | Grid search 的最大收缩比例 |
| `grid` | Grid search 的网格点数 |
| `patience` | 连续无改善后停止搜索 |
| `norm` | 误差范数类型（L2 等） |

## IMatrix 的优缺点

**优点**：
- 显著提升低比特量化精度（尤其 4-bit）
- 计算简单（仅需一次前向传播收集统计）
- 与任何量化器兼容
- 理论基础清晰（加权最优化）

**缺点**：
- 需要额外的校准步骤（两阶段流程）
- 增加少量内存开销（存储 E[x²]）
- 对 FP8 等高精度格式收益有限
- 需要有代表性的校准数据

## 适用场景

| 场景 | 推荐度 | 原因 |
|------|:---:|------|
| W4A16 INT4 量化 | ★★★★★ | 4-bit 下通道重要性差异大 |
| W4A16 + GPTQ | ★★★★ | 为 GPTQ 提供更好的初始 scale |
| W8A8 量化 | ★★★ | 8-bit 精度足够，收益有限 |
| FP8 量化 | ★★ | FP8 精度很高，几乎无需额外优化 |
| 模型有明显离群通道 | ★★★★★ | 正是 IMatrix 设计解决的问题 |
