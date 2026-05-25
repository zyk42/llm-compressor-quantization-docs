# LLM Compressor 项目架构深度解析

## 整体架构

LLM Compressor 采用模块化的分层架构，核心由五大系统组成：

```
┌─────────────────────────────────────────────────────────┐
│                    用户 API 层                            │
│              oneshot() / model_free_ptq()                │
├─────────────────────────────────────────────────────────┤
│                   Recipe 配方系统                         │
│            定义量化方案、目标层、参数配置                    │
├─────────────────────────────────────────────────────────┤
│                  Modifier 修改器系统                      │
│    QuantizationModifier / GPTQModifier / AWQModifier ... │
├─────────────────────────────────────────────────────────┤
│                  Pipeline 管线系统                        │
│     Sequential / Independent / DataFree / Basic         │
├─────────────────────────────────────────────────────────┤
│                  Observer 观测器系统                      │
│            MinMax / MSE / IMatrix 统计收集               │
├─────────────────────────────────────────────────────────┤
│               compressed-tensors 格式层                  │
│              量化参数序列化 & vLLM 兼容输出               │
└─────────────────────────────────────────────────────────┘
```

## 1. Modifier（修改器）系统

Modifier 是 LLM Compressor 的核心抽象，每个量化算法对应一个 Modifier 实现。所有 Modifier 共享统一的生命周期：

### 生命周期

```
on_initialize()          # 初始化：加载配置，准备量化参数
       ↓
on_start()               # 启动：注册校准钩子（hooks）
       ↓
[校准循环]                # 前向传播收集统计量
       ↓
on_event(SEQUENTIAL_EPOCH_END)  # 事件：执行量化/变换
       ↓
on_end()                 # 结束：冻结量化参数，移除钩子
       ↓
on_finalize()            # 清理：释放临时数据
```

### Modifier 继承体系

```
Modifier (基类)
├── QuantizationModifier     # RTN/标准 PTQ
├── GPTQModifier             # GPTQ 量化
├── AutoRoundModifier        # AutoRound 量化
├── AWQModifier              # AWQ 变换（Transform 类）
├── SmoothQuantModifier      # SmoothQuant 变换
├── SpinQuantModifier        # SpinQuant 旋转变换
├── QuIPModifier             # QuIP 旋转变换
├── IMatrixGatherer          # IMatrix 统计收集
├── SparseGPTModifier        # SparseGPT 剪枝
└── WandaPruningModifier     # Wanda 剪枝
```

### 关键机制：Hook 系统

Modifier 通过 PyTorch 的 forward hook 机制介入模型推理：

```python
# 典型的校准钩子注册
def on_start(self, state, ...):
    for module in target_modules:
        # 注册前向钩子：在 forward 时捕获输入/输出激活
        self.register_hook(module, calibrate_input_hook, "forward_pre")
        self.register_hook(module, calibrate_output_hook, "forward")
```

## 2. Recipe（配方）系统

Recipe 定义了"对模型做什么压缩操作"。支持三种配置方式：

### Python 对象（最常用）

```python
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.gptq import GPTQModifier

# 单一修改器
recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC")

# 多修改器组合
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", block_size=128),
]
```

### YAML 配方

```yaml
quantization_stage:
  quant_modifiers:
    QuantizationModifier:
      targets: "Linear"
      scheme: "FP8_DYNAMIC"
      ignore: ["lm_head"]
```

### 从 Hugging Face Hub 加载

```python
from llmcompressor.recipe import Recipe
recipe = Recipe.create_from("neuralmagic/Llama-3-8B-FP8-recipe")
```

## 3. Pipeline（管线）系统

Pipeline 控制校准数据如何流经模型。根据 Modifier 类型和模型大小自动选择：

### Sequential Pipeline（默认）

逐层处理模型，每次只在 GPU 上加载一层：

```
Layer 0 → 校准 → 量化 → 卸载到 CPU
Layer 1 → 校准 → 量化 → 卸载到 CPU
...
Layer N → 校准 → 量化 → 卸载到 CPU
```

**优势**：单卡即可处理 70B+ 模型
**配置**：`sequential_targets=["LlamaDecoderLayer"]`

### Independent Pipeline

每个 Modifier 独立运行自己的前向传播：

```
Modifier 1: 完整前向传播 → 收集统计 → 应用变换
Modifier 2: 完整前向传播 → 收集统计 → 应用变换
```

**适用**：多个 Modifier 需要独立校准数据时

### DataFree Pipeline

无需校准数据，直接初始化量化参数：

```
初始化 → 应用量化配置 → 完成
```

**适用**：FP8 RTN 等不需要校准的方案

### Basic Pipeline

所有 Modifier 共享同一次前向传播：

```
单次前向传播 → 所有 Modifier 同时收集统计
```

### 自动选择逻辑

```python
# 简化的管线选择逻辑
if all modifiers are data-free:
    pipeline = DataFreePipeline
elif modifiers need independent calibration:
    pipeline = IndependentPipeline
else:
    pipeline = SequentialPipeline  # 默认
```

## 4. Observer（观测器）系统

Observer 负责在校准过程中收集激活统计量，用于计算量化参数（scale、zero_point）：

### 可用 Observer

| Observer | 策略 | 适用场景 |
|----------|------|---------|
| `memoryless_minmax` | 仅使用当前 batch 的 min/max | 内存敏感，权重量化默认 |
| `static_minmax` | 维护全局 min/max | 稳定，大数据集 |
| `minmax` | 指数移动平均 min/max | 平衡稳定与适应性 |
| `mse` | Grid search 最小化量化 MSE | 更高精度，较慢 |
| `imatrix_mse` | 重要性加权 MSE | 最高精度，需 IMatrix |

### Observer 工作流

```python
# 1. 初始化 Observer（on_initialize 阶段）
observer = MinMaxObserver(quantization_args)

# 2. 收集统计（校准阶段，每个 batch）
observer.forward(input_tensor)  # 更新内部 min_val/max_val

# 3. 计算量化参数（on_event 阶段）
scale, zero_point = observer.get_qparams()
```

### 配置方式

```python
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W4A16",
    # 为权重和激活分别指定 Observer
    observer={
        "weights": "mse",        # 权重用 MSE observer
        "input": "minmax",       # 输入激活用 MinMax
    }
)
```

## 5. oneshot() 完整调用链

`oneshot()` 是最常用的入口函数，其完整执行流程如下：

```
oneshot(model, recipe, dataset, ...)
│
├── 1. 预处理（pre_process）
│   ├── 加载模型（如传入字符串 model_id）
│   ├── 解除共享 Embedding（untie_shared_embeddings）
│   └── Patch save_pretrained 方法（支持 compressed-tensors 格式）
│
├── 2. 构建校准 DataLoader
│   ├── 加载数据集（HF datasets 或自定义）
│   ├── Tokenize & 截断到 max_seq_length
│   └── 取 num_calibration_samples 个样本
│
├── 3. 压缩主流程（CompressionSession）
│   ├── 初始化所有 Modifier（on_initialize）
│   ├── 选择 Pipeline（Sequential/Independent/DataFree/Basic）
│   ├── 执行校准循环：
│   │   ├── Modifier.on_start() — 注册钩子
│   │   ├── 前向传播校准数据 — Observer 收集统计
│   │   ├── Modifier.on_event(SEQUENTIAL_EPOCH_END) — 量化/变换
│   │   └── Modifier.on_end() — 冻结参数
│   └── 完成所有 Modifier（on_finalize）
│
├── 4. 后处理（post_process）
│   ├── model.save_pretrained(output_dir)  # 保存压缩权重
│   ├── 保存 tokenizer/processor
│   └── 保存 recipe 配置
│
└── 返回量化后的模型
```

## 6. 量化状态管理

模型中的每个量化模块维护三种状态：

```
INITIALIZED  →  CALIBRATION  →  FROZEN
(配置已应用)    (正在校准)      (参数冻结)
```

- **INITIALIZED**：量化配置已注入模块，但未开始收集统计
- **CALIBRATION**：Observer 处于活跃状态，每次前向传播更新统计
- **FROZEN**：Observer 已移除，scale/zero_point 已固定

## 7. 分布式支持（DDP）

LLM Compressor 支持通过 `torchrun` 启动多 GPU 分布式量化：

```bash
torchrun --nproc_per_node=4 quantize_script.py
```

DDP 同步机制：
- **Hessian 矩阵**（GPTQ）：各 rank 分别计算部分 Hessian，通过 `dist.reduce(SUM)` 汇总
- **激活统计**（Observer）：通过 `dist.all_reduce` 同步 min/max 或移动平均值
- **权重更新**：源 rank 计算量化参数后广播到其他 rank

## 8. 输出格式

量化模型以 `compressed-tensors` 格式保存，兼容 vLLM 直接加载：

```
output_dir/
├── config.json                 # 模型配置 + 量化元数据
├── model-00001-of-00004.safetensors  # 压缩权重
├── model-00002-of-00004.safetensors
├── ...
├── tokenizer.json
└── recipe.yaml                 # 量化配方（可复现）
```

每个量化层的参数存储：
- `weight_scale`：缩放因子
- `weight_zero_point`：零点（非对称量化）
- `weight_global_scale`：全局缩放（TENSOR_GROUP 策略）
- `weight_g_idx`：分组索引（GPTQ grouped）
