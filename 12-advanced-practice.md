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
