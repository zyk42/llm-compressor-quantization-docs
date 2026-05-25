# 问题排查与最佳实践

## 常见错误与解决方案

### 1. OOM（内存溢出）

**症状**：`RuntimeError: CUDA out of memory` 或 `torch.cuda.OutOfMemoryError`

**解决方案**：

| 原因 | 解决 |
|------|------|
| 模型太大无法加载 | 使用 `device_map=None` + Sequential Onloading |
| GPTQ Hessian 占满显存 | 设置 `offload_hessians=True` |
| AWQ 激活缓存太大 | 设置 `offload_device=torch.device("cpu")` |
| 校准 batch_size 太大 | 减小 `batch_size`（设为 1） |
| 序列太长 | 减小 `max_seq_length`（如 1024） |
| 模型加 CPU 内存也不够 | 使用磁盘卸载或 `model_free_ptq` |

```python
# OOM 时的保守配置
oneshot(
    model=model,
    recipe=recipe,
    dataset=dataset,
    num_calibration_samples=256,  # 减少样本
    max_seq_length=1024,          # 减短序列
)
```

### 2. Cholesky 分解失败（GPTQ）

**症状**：`torch._C._LinAlgError: linalg_cholesky_ex: ...` 或警告 `Failed to invert hessian due to numerical instability`

**原因**：Hessian 矩阵条件数过大或存在零行/列

**解决方案**：

```python
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
    dampening_frac=0.05,  # 增大阻尼（默认 0.01）
    # 严重时可以增大到 0.1
)
```

其他措施：
- 增加校准样本数（从 512 增到 1024）
- 打乱校准数据顺序（避免样本过于相似）
- 检查是否有全零输入导致 Hessian 退化

### 3. 量化后精度严重下降

**排查检查清单**：

1. **检查 ignore 列表**：`lm_head` 是否被排除？MoE 的 gate 层？
2. **检查校准数据质量**：是否覆盖实际使用场景？
3. **检查量化方案是否匹配**：4-bit RTN 精度本就有限，应使用 GPTQ/AWQ
4. **检查 group_size**：INT4 建议 group_size=128

**逐步提升精度的策略**：

```
RTN W4A16           （基线，精度最低）
→ + actorder        （启用激活排序）
→ GPTQ W4A16       （Hessian 补偿）
→ + AWQ 预处理      （通道缩放保护）
→ + IMatrix         （重要性加权）
→ + SpinQuant       （旋转降相干性）
```

### 4. 校准数据格式错误

**症状**：`KeyError`、`TypeError`、或输出全为 padding

**常见问题**：
- 数据集没有 `text` 字段
- Tokenizer 没有正确处理
- 多模态模型需要 processor 而非 tokenizer

**解决**：

```python
# 确保数据集格式正确
from datasets import load_dataset

dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

# 检查字段
print(dataset.column_names)  # 应包含 'messages' 或 'text'

# 自定义数据处理
def preprocess(example):
    return {"text": example["messages"][0]["content"]}

dataset = dataset.map(preprocess)
```

### 5. vLLM 加载量化模型失败

**常见原因**：

| 问题 | 解决 |
|------|------|
| vLLM 版本太旧 | 升级到最新版本 |
| 格式不兼容 | 确认使用 `compressed-tensors` 格式保存 |
| NVFP4 在非 Blackwell GPU 上 | 检查硬件兼容性 |
| config.json 缺少量化元数据 | 使用 `model.save_pretrained()` 保存 |

```python
# 正确保存方式
model.save_pretrained("output_dir")
tokenizer.save_pretrained("output_dir")

# vLLM 加载
from vllm import LLM
llm = LLM(model="output_dir")
```

## 精度调优检查清单

### 必做项

- [ ] `lm_head` 层已加入 `ignore` 列表
- [ ] MoE 模型的 gate/router 层已排除
- [ ] 校准数据量 ≥ 512 样本
- [ ] 校准序列长度 ≥ 2048
- [ ] 校准数据覆盖目标使用场景

### 可选优化

- [ ] 启用 `actorder="static"`（GPTQ）
- [ ] 增大 `dampening_frac` 到 0.02-0.05（GPTQ 数值更稳定）
- [ ] 添加 SmoothQuant/AWQ 前置处理
- [ ] 使用 IMatrix 加权（4-bit 场景）
- [ ] 添加 SpinQuant 旋转（DataFree 场景）
- [ ] 使用 `mse` observer 替代 `minmax`

## 性能优化

### 减少量化时间

| 方法 | 加速倍数 | 代价 |
|------|---------|------|
| DDP 多卡 | ~线性加速 | 需要多卡 |
| 减少校准样本（512→256） | ~2× | 略降精度 |
| 减小 max_seq_length | ~线性 | 长序列精度下降 |
| AutoRound 减少 iters | 与减少比例成正比 | 略降精度 |
| 启用 torch.compile | ~1.3-1.5× | 首次编译较慢 |

### 内存优化

```python
# 大模型量化的内存优化配置
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
    offload_hessians=True,  # Hessian 放 CPU
)

oneshot(
    model=model,  # device_map=None
    recipe=recipe,
    dataset=dataset,
    num_calibration_samples=256,
    max_seq_length=1024,
    sequential_targets=["LlamaDecoderLayer"],
)
```

## vLLM 推理部署

### 基本部署

```python
from vllm import LLM, SamplingParams

# 加载量化模型（自动识别 compressed-tensors 格式）
llm = LLM(model="path/to/quantized_model")

# 推理
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Hello, world!"], sampling_params)
```

### FP8 KV Cache 推理

```python
llm = LLM(
    model="path/to/model_with_kv_quant",
    kv_cache_dtype="fp8",  # 启用 FP8 KV Cache
)
```

### 兼容性矩阵

| 量化格式 | vLLM 支持 | 最低 vLLM 版本 |
|----------|:---:|------|
| FP8_DYNAMIC | ✅ | v0.5.0+ |
| FP8_BLOCK | ✅ | v0.7.0+ |
| W4A16 (GPTQ) | ✅ | v0.4.0+ |
| W8A8 INT8 | ✅ | v0.4.0+ |
| NVFP4 | ✅ | v0.8.0+ |
| MXFP4 W4A16 | ✅ | v0.14.0+ |
| MXFP4 W4A4 | ❌ | 计划中 |
| MXFP8 | ✅ | v0.14.0+ |
| KV Cache FP8 | ✅ | v0.5.0+ |
| Per-head KV | ✅ | v0.8.0+ |

## 评估方法

### Perplexity 评估

```python
from lm_eval import evaluator

# 使用 lm-eval-harness 评估
results = evaluator.simple_evaluate(
    model="hf",
    model_args=f"pretrained=path/to/quantized_model",
    tasks=["wikitext"],
    batch_size=1,
)
print(f"Perplexity: {results['results']['wikitext']['word_perplexity']}")
```

### 下游任务评估

```bash
# 使用 lm-eval 评估多个基准
lm_eval --model hf \
    --model_args pretrained=path/to/quantized_model \
    --tasks mmlu,hellaswag,arc_challenge,winogrande \
    --batch_size 1
```

### 快速验证

```python
# 简单的生成质量验证
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/quantized_model")
tokenizer = AutoTokenizer.from_pretrained("path/to/quantized_model")

inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## 最佳实践总结

1. **从 FP8 开始**：如果精度够用，FP8_DYNAMIC 是最简单最快的方案
2. **逐步压缩**：从 8-bit 到 4-bit，从 RTN 到 GPTQ，逐步增加复杂度
3. **始终排除 lm_head**：语言模型头部对精度极其敏感
4. **MoE 排除 gate**：路由器的错误会级联放大
5. **多模态排除 vision**：视觉编码器量化收益极低
6. **使用代表性校准数据**：数据质量 > 数据数量
7. **验证再部署**：量化后必须评估精度，不要盲目信任
8. **保留原始模型**：量化是不可逆操作，保留原始权重备份
