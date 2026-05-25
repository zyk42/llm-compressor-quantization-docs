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

### Block-wise 损失函数

对每个 Transformer 层（block），优化目标为：

$$\mathcal{L} = ||f(X; W + V \cdot \Delta) - f(X; W)||_2^2$$

其中：
- $f(X; W)$：层使用原始权重的输出
- $f(X; W + V \cdot \Delta)$：层使用量化权重的输出
- $\Delta$：舍入产生的增量（$\Delta_i = s$ 或 $-s$）

### SignSGD 优化

AutoRound 使用 SignSGD 而非 Adam/SGD：

$$\theta \leftarrow \theta - \eta \cdot \text{sign}(\nabla_\theta \mathcal{L})$$

**优势**：
- 梯度只取方向（sign），对梯度尺度不敏感
- 适合离散化搜索空间（舍入本质上是二值决策）
- 收敛速度快，通常 200 次迭代即可

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

## AutoRound 的优缺点

**优点**：
- 4-bit 精度优异（通常略优于 GPTQ）
- 支持多种格式（W4A16、W4A4、FP8、MXFP4、NVFP4）
- 可学习的裁剪范围（适应异常值）
- 支持 torch.compile 加速

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
