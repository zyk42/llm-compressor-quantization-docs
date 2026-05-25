# 高级实践：大模型、多模态、MoE 与组合配方

## 一、大模型量化

### Sequential Onloading（逐层加载）

对于 70B+ 的大模型，单卡 GPU 无法同时加载整个模型。Sequential Onloading 的策略是逐层处理：

```
Disk/CPU → 加载 Layer 0 到 GPU → 校准 & 量化 → 保存 → 卸载
         → 加载 Layer 1 到 GPU → 校准 & 量化 → 保存 → 卸载
         → ...
         → 加载 Layer N 到 GPU → 校准 & 量化 → 保存 → 卸载
```

**关键配置**：`device_map=None`（不让 transformers 自动分配设备）

```python
from transformers import AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# 关键：device_map=None，由 LLM Compressor 控制加载
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.3-70B-Instruct",
    dtype="auto",
    device_map=None,  # 不自动分配设备
)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

oneshot(
    model=model,
    recipe=recipe,
    sequential_targets=["LlamaDecoderLayer"],  # 指定层级单位
)
```

**Sequential Onloading 内部执行流程详解**：

```
┌────────────────────────────────────────────────────────────────────────┐
│              Sequential Onloading Pipeline 完整流程                      │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  初始状态:                                                               │
│    CPU RAM: model.embed_tokens + model.norm + lm_head (常驻)            │
│    GPU: 空                                                              │
│    Disk: 全部层权重 (SafeTensors shards)                                 │
│                                                                         │
│  Phase 1: 校准数据前向传播 (收集激活统计)                                  │
│  ─────────────────────────────────────────                              │
│  for layer_idx in range(num_layers):                                    │
│    ┌─────────────────────────────────────────────┐                      │
│    │ Step 1: Load layer to GPU                   │                      │
│    │   CPU→GPU: layer_idx 的所有权重              │                      │
│    │   (q_proj, k_proj, v_proj, o_proj,          │                      │
│    │    gate_proj, up_proj, down_proj,            │                      │
│    │    input_layernorm, post_attention_layernorm)│                      │
│    │   GPU 内存: ~1.7GB (70B 模型单层)            │                      │
│    ├─────────────────────────────────────────────┤                      │
│    │ Step 2: Forward calibration data            │                      │
│    │   输入: 所有校准样本的 hidden_states          │                      │
│    │   (这些是前一层的输出, 存在 CPU 上)          │                      │
│    │   处理: 逐 batch 送入 GPU 做前向             │                      │
│    │   收集: 每个 Linear 层的输入/输出激活统计     │                      │
│    │   输出: 所有校准样本经过本层后的 hidden_states│                      │
│    ├─────────────────────────────────────────────┤                      │
│    │ Step 3: Quantize layer                      │                      │
│    │   根据收集的激活统计, 计算量化参数:           │                      │
│    │   - FP8: 直接从 max(|W|) 计算 scale         │                      │
│    │   - GPTQ: 使用 Hessian 进行逐列量化         │                      │
│    │   将权重就地量化                             │                      │
│    ├─────────────────────────────────────────────┤                      │
│    │ Step 4: Offload quantized layer to CPU      │                      │
│    │   GPU→CPU: 量化后的权重 + scale/zero_point   │                      │
│    │   释放 GPU 内存                              │                      │
│    └─────────────────────────────────────────────┘                      │
│                                                                         │
│  Phase 2: 保存模型                                                       │
│  ─────────────────                                                      │
│  将 CPU 上的所有量化层保存为 SafeTensors                                   │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

**CPU↔GPU 数据移动时序图**：

```
时间 →  ════════════════════════════════════════════════════════►

GPU:    [Layer0权重加载] [Layer0前向+量化] [卸载]  [Layer1加载] [Layer1前向+量化] [卸载] ...
         ▲                              │         ▲                              │
         │ CPU→GPU                      │GPU→CPU  │ CPU→GPU                      │GPU→CPU
         │ ~1.7GB                       │~0.9GB   │ ~1.7GB                       │~0.9GB
         │                              ▼         │                              ▼
CPU:    [全模型FP16权重]──────────────────[Layer0量化后]────────────[Layer1量化后]────────→

GPU 峰值内存: ~1.7GB(层权重) + ~2GB(激活+Hessian) ≈ 4GB
CPU 峰值内存: 全模型FP16 (~140GB for 70B) + 校准数据激活 (~8GB)

注: 量化后的层更小(FP8=原始一半, INT4=约1/4), 所以 GPU→CPU 传输量小于 CPU→GPU
```

**为什么需要 `device_map=None`？**：

```
如果使用 device_map="auto":
  - transformers 会自动将层分配到各 GPU/CPU
  - 模型加载后层的设备位置就固定了
  - LLM Compressor 无法控制逐层加载/卸载
  
使用 device_map=None:
  - 模型加载到 "meta" device (不分配实际内存)
  - 或全部加载到 CPU
  - LLM Compressor 完全控制 GPU 内存使用
  - 可以在单张 24GB GPU 上量化 70B 模型!
```

**内存预算分析（以 Llama-3.3-70B + GPTQ 为例）**：

```
单层内存需求:
  权重 (FP16): 80层, 每层 ~1.7GB → 单层 1.7GB
  校准激活缓存: batch_size × seq_len × hidden_dim × 2 bytes
    = 512 × 2048 × 8192 × 2 ≈ 17GB  (太大!)
    
  解决: sequential pipeline 只缓存每层输入/输出
    每层输入: num_samples × seq_len × hidden_dim × 2
    = 512 × 2048 × 8192 × 2 = 17GB → 存在 CPU 上, 分 batch 送 GPU
    
  GPU 上实际需要:
    - 当前层权重: 1.7 GB
    - 小 batch 激活: 8 × 2048 × 8192 × 2 = 268 MB
    - Hessian 矩阵 (GPTQ): hidden_dim² × 4 = 8192² × 4 = 268 MB
    - 临时计算: ~1 GB
    
  GPU 总计: ~3.2 GB → 轻松在 24GB GPU 上运行
```

### Disk Offloading（磁盘卸载）

当模型甚至无法完全加载到 CPU 内存时，使用磁盘卸载：

```python
from compressed_tensors.offload import dispatch_model

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V3",
    dtype="auto",
    device_map=None,
)

# 模型会自动利用磁盘卸载中间层
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_BLOCK",
    ignore=["lm_head", "re:.*mlp.gate$"],
)

oneshot(model=model, recipe=recipe)
```

### DDP 分布式量化

使用多 GPU 并行加速校准过程：

```python
# quantize_ddp.py
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
)

oneshot(
    model=model,
    recipe=recipe,
    dataset=dataset,
    num_calibration_samples=512,
)
```

启动：

```bash
torchrun --nproc_per_node=4 quantize_ddp.py
```

**DDP 如何加速**：
- 校准数据被均匀分配到各 rank
- 每个 rank 计算部分 Hessian/激活统计
- 通过 all-reduce 汇总结果
- 量化参数在所有 rank 上同步

**DDP 分布式量化详细机制**：

```
┌────────────────────────────────────────────────────────────────────────┐
│                    DDP GPTQ 量化流程                                     │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  假设: 4 GPU, 512 个校准样本, 模型完整复制到每张 GPU                       │
│                                                                         │
│  Step 1: 数据分片                                                       │
│  ────────────────                                                       │
│  GPU 0 (Rank 0): samples[0:128]     (128 个样本)                        │
│  GPU 1 (Rank 1): samples[128:256]   (128 个样本)                        │
│  GPU 2 (Rank 2): samples[256:384]   (128 个样本)                        │
│  GPU 3 (Rank 3): samples[384:512]   (128 个样本)                        │
│                                                                         │
│  Step 2: 并行计算 Hessian (对每个 Linear 层)                             │
│  ────────────────────────────────────────────                           │
│  GPTQ 需要的 Hessian: H = X^T × X (其中 X 是该层的输入激活)              │
│                                                                         │
│  Rank 0: H_0 = X_0^T × X_0  (使用 128 个样本的激活)                     │
│  Rank 1: H_1 = X_1^T × X_1  (使用 128 个样本的激活)                     │
│  Rank 2: H_2 = X_2^T × X_2                                             │
│  Rank 3: H_3 = X_3^T × X_3                                             │
│                                                                         │
│  Step 3: All-Reduce 汇总 Hessian                                        │
│  ────────────────────────────────                                       │
│  H_total = H_0 + H_1 + H_2 + H_3   (通过 NCCL all-reduce SUM)         │
│                                                                         │
│  数学等价性:                                                             │
│  H_total = X_all^T × X_all                                             │
│          = [X_0; X_1; X_2; X_3]^T × [X_0; X_1; X_2; X_3]              │
│          = X_0^T×X_0 + X_1^T×X_1 + X_2^T×X_2 + X_3^T×X_3            │
│          = H_0 + H_1 + H_2 + H_3  ✓                                   │
│                                                                         │
│  Step 4: 各 Rank 独立执行 GPTQ 量化 (结果相同)                            │
│  ──────────────────────────────────────────────                         │
│  每个 Rank 使用相同的 H_total + 相同的权重                                │
│  → 量化结果完全一致 (无需额外同步)                                        │
│                                                                         │
│  Step 5: Rank 0 保存模型                                                 │
│  ────────────────────────                                               │
│  只有 Rank 0 执行 model.save_pretrained()                                │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

**Hessian 矩阵通信的开销分析**：

```
每个 Linear 层的 Hessian 大小:
  H 形状: [input_dim, input_dim]
  
  Llama-3-70B 各层 Hessian 大小:
    q/k/v/o_proj: [8192, 8192] × 4 bytes = 256 MB
    gate/up_proj:  [8192, 8192] × 4 bytes = 256 MB  
    down_proj:     [28672, 28672] × 4 bytes = 3.2 GB  ← 最大!
    
  All-reduce 通信量 (per layer):
    ≈ 256MB × 6 + 3.2GB × 1 ≈ 4.7 GB
    
  80 层总通信: ~376 GB
  NVLink 带宽 (600 GB/s): 理论时间 ≈ 0.6 秒
  PCIe 4.0 x16 (32 GB/s): 理论时间 ≈ 12 秒
  
  结论: 通信开销相对于计算可忽略, DDP 近线性加速
```

**DDP vs 数据并行 vs 模型并行的区别**：

```
DDP 量化 (数据并行):
  - 每张 GPU 持有完整模型副本
  - 数据分片: 不同 GPU 处理不同校准样本
  - 适合: GPU 内存能放下整个模型的场景

Sequential + 模型并行 (手动):
  - 模型不完整加载, 逐层处理
  - 适合: 单卡放不下模型的场景

两者可以结合:
  torchrun --nproc_per_node=4 quantize.py
  + sequential_targets=["LlamaDecoderLayer"]
  = 每张 GPU 逐层处理, 但校准数据并行
```

### Model-Free PTQ

对于没有 Hugging Face 模型定义的超大模型（如 Mistral Large 675B）：

```python
from llmcompressor import model_free_ptq

model_free_ptq(
    model_stub="deepseek-ai/DeepSeek-V3",
    save_directory="DeepSeek-V3-FP8-Block",
    scheme="FP8_BLOCK",
    ignore=["re:.*gate$", "lm_head"],
    max_workers=15,     # 并行处理线程数
    device="cuda:0",
)
```

**限制**：目前仅支持 Data-Free 方案（FP8、MXFP4、MXFP8）。

---

## 二、多模态模型量化

### 核心原则

多模态模型（Vision-Language、Audio-Language）量化的关键：**视觉/音频编码器通常不量化**。

原因：
- 视觉编码器参数量小（相对于语言模型），量化收益有限
- 视觉特征对量化更敏感
- 语言模型部分占主要参数量

### 使用 ignore 排除非文本模块

```python
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",
        "re:.*vision_tower.*",          # 视觉编码器
        "re:.*multi_modal_projector.*",  # 多模态投射层
        "re:.*visual.*",                 # 视觉相关模块
    ],
)
```

### 示例：Qwen-VL 量化

```python
from transformers import AutoModelForCausalLM, AutoProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    dtype="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head", "re:.*visual.*"],
)

oneshot(
    model=model,
    recipe=recipe,
    dataset=calibration_dataset,  # 多模态校准数据
    num_calibration_samples=512,
)
```

### 示例：LLaVA 量化

```python
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=[
        "lm_head",
        "re:.*vision_tower.*",
        "re:.*multi_modal_projector.*",
    ],
)
```

### 多模态校准数据

多模态模型的校准数据需要包含图像/音频：

```python
# 使用 processor 处理多模态输入
def prepare_multimodal_calibration(dataset, processor):
    samples = []
    for item in dataset:
        inputs = processor(
            text=item["text"],
            images=item["image"],
            return_tensors="pt",
        )
        samples.append(inputs)
    return samples
```

---

## 三、MoE 模型量化

### MoE 量化的挑战

1. **Expert 数量多**：如 Mixtral 有 8 个 Expert，DeepSeek-V3 有 256 个
2. **门控层特殊处理**：Router/Gate 层不应量化
3. **内存压力大**：AWQ 缓存激活时内存爆炸
4. **不同 Expert 的激活分布差异**：一刀切可能不够精确

### 门控层排除

MoE 模型的门控（routing）层必须用 `ignore` 排除：

```python
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=[
        "lm_head",
        "re:.*mlp.gate$",      # MoE 门控层
        "re:.*router.*",       # 路由器
    ],
)
```

**为什么门控层量化是危险的？详细分析**：

```
┌────────────────────────────────────────────────────────────────────────┐
│                 MoE 门控层 (Router/Gate) 工作机制                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  MoE 路由过程:                                                           │
│                                                                         │
│  hidden_states [batch, seq, hidden_dim]                                 │
│       │                                                                  │
│       ↓                                                                  │
│  gate(x) = x @ W_gate   →  logits [batch, seq, num_experts]             │
│       │                                                                  │
│       ↓                                                                  │
│  routing_weights = softmax(topk(logits, k=2))                           │
│       │                                                                  │
│       ↓                                                                  │
│  selected_experts = argmax(logits, dim=-1)[:k]                          │
│       │                                                                  │
│       ↓                                                                  │
│  output = sum(routing_weights[i] * expert_i(x) for i in selected)       │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘

为什么量化 gate 是灾难性的:

1. 路由决策是离散的（argmax/topk）
   - gate 输出的微小变化可能改变 expert 选择
   - 例如: expert_0 score=0.31, expert_1 score=0.30
     量化后: expert_0 score=0.29, expert_1 score=0.32
     → 完全不同的 expert 被选中! Token 被路由到错误的 expert

2. gate 层通常很小（hidden_dim × num_experts）
   - Mixtral: 4096 × 8 = 32K 参数
   - DeepSeek-V3: 7168 × 256 = 1.8M 参数
   - 占总模型参数 < 0.01%，量化收益微乎其微

3. 路由错误的级联效应
   - Token 被路由到错误 expert → 该 token 的特征完全错误
   - 错误的特征传递到下一层 → 下一层的路由也可能出错
   - 在 60+ 层的深度网络中，路由错误会快速级联放大

4. 实验验证:
   - 量化 gate 后的典型退化:
     Mixtral-8x7B: perplexity +2.5 (正常量化仅 +0.3)
     DeepSeek-V3: 部分任务准确率下降 10-15%
   - 不量化 gate 时:
     几乎无额外损失 (gate 参数占比太小)
```

**Expert 路由分布与量化的交互**：

```
观察: 不同 Expert 的激活频率差异很大

DeepSeek-V3 的典型 Expert 激活模式:
  Expert 0-10:  高频 (每层 >50% token 激活)  → "通用 expert"
  Expert 11-50: 中频 (10-50% token 激活)     → "领域 expert"
  Expert 51-255: 低频 (<10% token 激活)       → "稀疏 expert"

量化影响:
  - 高频 Expert: 校准数据充分, 量化精度高
  - 低频 Expert: 校准数据稀少, 量化可能欠准确
    → 这些 expert 处理的往往是"困难"或"罕见"输入
    → 精度损失可能在特定任务上更明显

建议: 对于需要精确服务长尾任务的场景，考虑:
  - 增加校准样本数量 (>1024)
  - 使用 per-expert 的量化参数
  - 或对低频 expert 保持更高精度
```

### AWQ 对 MoE 的优化

AWQ 检测到 MoE 模型时自动启用 CPU offload：

```python
# AWQ 自动处理
recipe = [
    AWQModifier(
        # offload_device 自动设为 CPU（检测到 MoE）
        duo_scaling=True,
    ),
    QuantizationModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head", "re:.*mlp.gate$"],
    ),
]
```

### 示例：Qwen3 MoE 量化

```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B",
    dtype="auto",
)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_BLOCK",
    ignore=["lm_head", "re:.*mlp.gate$"],
)

oneshot(model=model, recipe=recipe)
```

### 示例：DeepSeek-V3/R1 量化

```python
from llmcompressor import model_free_ptq

# DeepSeek-V3 参数量极大，使用 model_free_ptq
model_free_ptq(
    model_stub="deepseek-ai/DeepSeek-V3",
    save_directory="DeepSeek-V3-FP8-Block",
    scheme="FP8_BLOCK",
    ignore=["re:.*gate$", "lm_head"],
    max_workers=15,
    device="cuda:0",
)
```

---

## 四、组合配方（Combined Recipes）

### 组合原则

```
Transform Modifier（变换）→ Quantization Modifier（量化）
     ↑ 第一步                    ↑ 第二步
  修改权重分布                  执行实际量化
```

**规则**：
1. Transform 修改器（SmoothQuant、AWQ、SpinQuant、QuIP）在前
2. 量化修改器（QuantizationModifier、GPTQModifier）在后
3. IMatrixGatherer 必须在使用 imatrix_mse 的量化器之前

### 经典组合 1：SmoothQuant + GPTQ（W8A8 INT8）

最适合需要 INT8 加速的场景：

```python
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]
```

**SmoothQuant + GPTQ 逐步执行追踪**：

```
┌────────────────────────────────────────────────────────────────────────┐
│         SmoothQuant + GPTQ 组合配方执行详情                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Pipeline: sequential (逐层处理)                                         │
│  Layer: LlamaDecoderLayer[0]                                            │
│                                                                         │
│  ═══════════ Phase 1: SmoothQuant (Transform) ═══════════               │
│                                                                         │
│  Step 1.1: 收集激活统计                                                  │
│    对该层的所有 Linear 输入运行校准数据:                                    │
│    X_stats[q_proj] = max(|X|, dim=token_dim)  → per-channel 激活最大值   │
│    形状: [hidden_dim]                                                    │
│                                                                         │
│  Step 1.2: 计算平滑因子 (per-channel)                                    │
│    α = smoothing_strength = 0.8                                         │
│    对每个 channel j:                                                     │
│      act_scale[j] = max(|X[:, j]|)  # 激活第 j 维的最大值               │
│      w_scale[j] = max(|W[j, :]|)    # 权重第 j 行(输入维度)的最大值      │
│      s[j] = (act_scale[j]^α) / (w_scale[j]^(1-α))                     │
│                                                                         │
│    直觉: s[j] 大 → 该 channel 激活大/权重小 → 需要更多平滑                 │
│                                                                         │
│  Step 1.3: 应用平滑变换                                                  │
│    修改前一层的 LayerNorm (或等效操作):                                    │
│      LayerNorm.weight /= s    (缩小输出的大 channel)                     │
│    修改当前层的 Linear 权重:                                              │
│      W_new = W × diag(s)      (放大权重的对应 channel 补偿)              │
│                                                                         │
│    效果: X_new = X / s (激活变小), W_new = W × s (权重变大)              │
│          X_new @ W_new^T = (X/s) @ (W×s)^T = X @ W^T  (数学等价!)       │
│          但 X_new 和 W_new 各自的分布更均匀了 → 更适合量化                  │
│                                                                         │
│  ═══════════ Phase 2: GPTQ (Quantize) ═══════════                       │
│                                                                         │
│  Step 2.1: 使用变换后的权重收集 Hessian                                   │
│    注意: 此时权重已经被 SmoothQuant 修改过!                                │
│    H = X_smoothed^T × X_smoothed                                        │
│    (使用经过平滑后的激活来构建 Hessian)                                    │
│                                                                         │
│  Step 2.2: GPTQ 逐列量化                                                │
│    对变换后的 W_new 执行 GPTQ:                                            │
│    for col in range(out_features):                                       │
│      w = W_new[:, col]                                                   │
│      q = quantize(w)           # 量化为 INT8                             │
│      error = w - dequantize(q) # 量化误差                                │
│      # 将误差分配到未量化的列 (Hessian 加权):                             │
│      W_new[:, col+1:] -= error × H_inv[col, col+1:] / H_inv[col, col]  │
│                                                                         │
│  Step 2.3: 存储结果                                                      │
│    保存: INT8 量化权重 + per-channel scale + zero_point                   │
│                                                                         │
│  ═══════════ 然后处理 Layer[1], Layer[2], ... ═══════════                │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

**为什么 SmoothQuant 在 GPTQ 前效果好？**：

```
不使用 SmoothQuant:
  激活分布: 某些 channel 有极端离群值 (如 channel 127 最大值达到 100)
  权重分布: 相对均匀 (最大值 ~0.5)
  
  量化时:
    激活: scale = 100/127 = 0.79 → 大部分值 (范围 -2~2) 只用到 2/0.79 ≈ 3 个级别
    权重: scale = 0.5/127 = 0.004 → 精度OK
    
使用 SmoothQuant (α=0.8):
  平滑后:
    激活: 离群 channel 被缩小, 所有 channel 范围相近 (~5)
    权重: 对应 channel 被放大补偿
    
  量化时:
    激活: scale = 5/127 = 0.04 → 大部分值充分利用量化区间
    权重: 即使被放大, GPTQ 可以很好地处理 (权重量化本来就容易)
    
结论: SmoothQuant 解决了激活量化的核心难题 (离群值),
     GPTQ 解决了权重量化的精度优化, 两者互补.
```

### 经典组合 2：AWQ + RTN（W4A16 快速量化）

精度优于纯 RTN，速度优于 GPTQ：

```python
recipe = [
    AWQModifier(duo_scaling=True, n_grid=20),
    QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
]
```

### 经典组合 3：SpinQuant + GPTQ（W4A16 最高精度）

旋转降低不相干性 + GPTQ 精细量化：

```python
recipe = [
    SpinQuantModifier(rotations=["R1", "R2", "R4"]),
    GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
]
```

### 经典组合 4：IMatrix + GPTQ（W4A16 重要性加权）

```python
recipe = [
    IMatrixGatherer(targets="Linear", ignore=["lm_head"]),
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
        observer={"weights": "imatrix_mse"},
    ),
]
```

### 经典组合 5：AWQ + FP8（W8A8 高精度）

```python
recipe = [
    AWQModifier(duo_scaling=True),
    QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]),
]
```

### 经典组合 6：非均匀量化（不同层不同精度）

```python
recipe = [
    # Attention 层用 W8A8
    GPTQModifier(
        targets=r"re:.*self_attn\.(q|k|v|o)_proj$",
        scheme="W8A8",
    ),
    # MLP 层用 W4A16（MLP 对低精度更鲁棒）
    GPTQModifier(
        targets=r"re:.*mlp\.(gate|up|down)_proj$",
        scheme="W4A16",
    ),
]
```

### 组合配方的执行顺序可视化

```
以 recipe = [SmoothQuantModifier, GPTQModifier] 为例:

Sequential Pipeline 执行时间线:
═══════════════════════════════════════════════════════════════

Layer 0:
  ├─ SmoothQuant.calibrate(layer_0_inputs)    ← 收集激活统计
  ├─ SmoothQuant.apply(layer_0)               ← 修改权重
  ├─ GPTQ.calibrate(layer_0_smoothed_inputs)  ← 用平滑后的输入算 Hessian
  ├─ GPTQ.quantize(layer_0)                   ← 量化
  └─ Forward all samples through quantized layer_0 → 得到 layer_1 的输入

Layer 1:
  ├─ SmoothQuant.calibrate(layer_1_inputs)
  ├─ SmoothQuant.apply(layer_1)
  ├─ GPTQ.calibrate(layer_1_smoothed_inputs)
  ├─ GPTQ.quantize(layer_1)
  └─ Forward → layer_2 inputs

...

关键: 每层的校准数据都是前面量化层的实际输出!
     这确保了误差不会在层间累积 (每层都在"真实"输入上校准)
```

---

## 五、Pipeline 选择策略

| 场景 | 推荐 Pipeline | 原因 |
|------|--------------|------|
| DataFree 量化（FP8 RTN） | `datafree` | 无需前向传播 |
| 标准校准量化 | `sequential`（默认） | 逐层处理，内存友好 |
| 多个独立 Modifier | `independent` | 每个 Modifier 独立校准 |
| Transform + 量化组合 | `sequential` | Transform 需要看到量化效果 |
| 大模型（70B+） | `sequential` | 逐层加载到 GPU |

### 手动指定 Pipeline

```python
oneshot(
    model=model,
    recipe=recipe,
    dataset=dataset,
    pipeline="sequential",          # 或 "independent", "basic"
    sequential_targets=["LlamaDecoderLayer"],
)
```

---

## 六、实战案例总结

### 场景 → 推荐方案

| 场景 | 方案 | 配方 |
|------|------|------|
| 快速部署 8B 模型 | FP8 Dynamic | `QuantizationModifier(scheme="FP8_DYNAMIC")` |
| 单卡量化 70B | Sequential + FP8 | `device_map=None` + sequential |
| 多卡加速 GPTQ | DDP | `torchrun --nproc_per_node=N` |
| Blackwell 部署 | NVFP4 | `scheme="NVFP4"` |
| VLM 量化 | GPTQ + ignore vision | `ignore=["re:.*vision.*"]` |
| MoE 量化 | FP8 + ignore gate | `ignore=["re:.*gate$"]` |
| 最高精度 W4 | SpinQuant + GPTQ | 两步组合配方 |
| 无 HF 定义 675B | model_free_ptq | `model_free_ptq(...)` |
