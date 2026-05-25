# RTN（Round-to-Nearest）最近舍入量化

## 算法概述

RTN 是最简单也是最基础的量化方法——将每个权重值舍入到最近的量化级别。虽然简单，但对于高精度格式（如 FP8），RTN 已经能获得接近无损的效果。

LLM Compressor 中 RTN 通过 `QuantizationModifier` 实现，支持多种 scheme 配置。

---

## 详细原理

### 核心数学过程

RTN 量化由三步组成：

**Step 1: 确定量化范围与缩放因子**

对称量化模式：
$$s = \frac{\max(|W|)}{q_{max}}$$

其中 $q_{max}$ 取决于数据类型：
- INT8 对称：$q_{max} = 127$
- INT4 对称：$q_{max} = 7$
- FP8 E4M3：$q_{max} = 448$

非对称量化模式：
$$s = \frac{W_{max} - W_{min}}{q_{max} - q_{min}}, \quad z = q_{min} - \text{round}\left(\frac{W_{min}}{s}\right)$$

**Step 2: 量化（float → int/fp8）**
$$W_q = \text{clamp}\left(\text{round}\left(\frac{W}{s}\right) + z,\ q_{min},\ q_{max}\right)$$

**Step 3: 反量化（推理时恢复近似浮点值）**
$$\hat{W} = s \cdot (W_q - z)$$

### 为什么 FP8_DYNAMIC + RTN 就足够了？

**FP8 E4M3 的精度分析**：

FP8 E4M3 格式有 4 位指数和 3 位尾数：
- 可表示的不同非零值约 **240 个**（正负各 120）
- 最大值 448，最小正规数 $2^{-6}$
- 在 0 附近的精度比 INT8 高得多（对数间距）

对于典型的 LLM 权重分布（均值约 0，标准差约 0.01-0.1）：
- 大部分权重落在 [-0.5, 0.5] 范围内
- FP8 在这个范围内有约 100 个不同的表示
- 相对量化误差 < 1%

**量化误差的数学分析**：

对于 FP8，由于对数间距，在值 $x$ 处的量化步长约为 $x \cdot 2^{-3}$（3 位尾数）。因此相对误差：

$$\frac{|\Delta W|}{|W|} \approx \frac{1}{2} \cdot 2^{-3} = 6.25\%$$

但由于 per-channel 量化，每个通道的 scale 适配其实际范围，实际误差远小于此理论上界。

### FP8_DYNAMIC 的工作机制详解

FP8_DYNAMIC scheme 实际包含两部分：

**权重量化（静态，per-channel）**：
```
对于每一行 W[i, :]:
    scale[i] = max(|W[i, :]|) / 448.0
    W_q[i, :] = cast_to_fp8_e4m3(W[i, :] / scale[i])
```

- 在量化时一次性完成，scale 保存在模型中
- 推理时不需要重新计算

**激活量化（动态，per-token）**：
```
对于每个输入 token X[t, :]:
    scale[t] = max(|X[t, :]|) / 448.0   # 推理时实时计算
    X_q[t, :] = cast_to_fp8_e4m3(X[t, :] / scale[t])
```

- 每次推理时动态计算 per-token scale
- 不需要提前校准——适应任何输入分布
- 这就是为什么 FP8_DYNAMIC 不需要校准数据

**矩阵乘法执行**：
$$Y[t, i] = \text{scale\_x}[t] \cdot \text{scale\_w}[i] \cdot \sum_j X_q[t,j] \cdot W_q[i,j]$$

GPU 执行 FP8×FP8 矩阵乘法（Hopper/Ada FP8 Tensor Core），然后乘以两个 scale 恢复正确数值范围。

### FP8_BLOCK 的工作机制详解

FP8_BLOCK 是 DeepSeek-V3 采用的量化风格，更细粒度：

**权重量化（block-wise，128×128 块）**：
```
将权重矩阵切分为 128×128 的块：
W = [B_00  B_01  ...]
    [B_10  B_11  ...]
    [...]

对于每个块 B[r,c]:
    scale[r,c] = max(|B[r,c]|) / 448.0
    B_q[r,c] = cast_to_fp8_e4m3(B[r,c] / scale[r,c])
```

- 比 per-channel 更细粒度（每 128 列一个独立 scale）
- 权重中不同区域的量化更精确

**激活量化（block-wise 动态）**：
```
将激活切分为 1×128 的块（沿隐藏维度每 128 个元素一个 scale）：
对于 X[t, j*128 : (j+1)*128]:
    scale[t,j] = max(|X[t, j*128:(j+1)*128]|) / 448.0
    X_q[t, j*128:(j+1)*128] = cast_to_fp8(X / scale)
```

**与 FP8_DYNAMIC 的关键区别**：

| 特性 | FP8_DYNAMIC | FP8_BLOCK |
|------|------------|-----------|
| 权重 scale 粒度 | per-channel (每行 1 个 scale) | per-block (每 128×128 一个) |
| 激活 scale 粒度 | per-token (每行 1 个) | per-block (每 128 元素一个) |
| 精度 | 高 | 更高（scale 粒度更细） |
| 兼容内核 | 标准 FP8 GEMM | DeepGEMM / Block FP8 GEMM |
| 存储开销 | scale 很少 | scale 稍多（但仍为 FP32） |

### W4A16 RTN 的精度问题详解

当用 RTN 做 4-bit 量化时，仅有 **16 个量化级别**（INT4 对称：-8 到 7）：

```
原始权重: [-0.23, 0.15, -0.08, 0.31, -0.45, 0.02, ...]

scale = 0.45 / 7 = 0.064 (per-group, group_size=128)

量化:
  -0.23 / 0.064 = -3.59 → round → -4 → 反量化 = -0.256 (误差 0.026)
   0.15 / 0.064 =  2.34 → round →  2 → 反量化 =  0.128 (误差 0.022)
  -0.45 / 0.064 = -7.03 → round → -7 → 反量化 = -0.448 (误差 0.002)
   0.02 / 0.064 =  0.31 → round →  0 → 反量化 =  0.000 (误差 0.020)
```

**问题**：
1. 量化步长 = 0.064，对于小权重（如 0.02），相对误差可达 100%
2. 所有值被迫映射到 16 个级别，大量信息丢失
3. 误差在各层之间**无补偿**，逐层累积

这就是为什么 4-bit 部署需要 GPTQ/AWQ——它们通过误差补偿和通道保护来克服这些问题。

---

## 在 LLM Compressor 中的完整实现流程

### QuantizationModifier 的内部机制

```python
# 完整的 RTN 量化执行流程

class QuantizationModifier:
    
    def on_initialize(self, state):
        """Phase 1: 将量化配置注入模型"""
        model = state.model
        for name, module in model.named_modules():
            if should_quantize(module, self.targets, self.ignore):
                # 为模块附加量化 scheme
                module.quantization_scheme = resolve_scheme(self.scheme)
                # 创建 Observer 实例
                if module.quantization_scheme.weights:
                    module.weight_observer = create_observer(
                        "memoryless_minmax",  # 权重默认用 memoryless
                        module.quantization_scheme.weights
                    )
                if module.quantization_scheme.input_activations:
                    module.input_observer = create_observer(
                        "minmax",
                        module.quantization_scheme.input_activations
                    )
    
    def on_start(self, state):
        """Phase 2: 注册校准钩子"""
        for module in quantized_modules:
            if has_input_observer(module):
                # 前向钩子：每次推理时 Observer 观测输入激活
                register_hook(module, calibrate_input_hook, "forward_pre")
            if has_output_observer(module):
                register_hook(module, calibrate_output_hook, "forward")
    
    def on_event(self, state, event):
        """Phase 3: 在校准完成后计算量化参数"""
        if event.type_ == SEQUENTIAL_EPOCH_END:
            for module in quantized_modules:
                # --- 权重量化 ---
                module.weight_observer(module.weight)
                qparams = module.weight_observer.get_qparams()
                module.weight_scale = qparams["scale"]
                module.weight_zero_point = qparams["zero_point"]
                
                # --- 激活量化（如果有校准数据）---
                if has_input_observer(module):
                    qparams = module.input_observer.get_qparams()
                    module.input_scale = qparams["scale"]
    
    def on_end(self, state):
        """Phase 4: 冻结——移除 Observer，固定参数"""
        for module in quantized_modules:
            del module.weight_observer
            if hasattr(module, 'input_observer'):
                del module.input_observer
            remove_all_hooks()
```

### DataFree Pipeline 的工作方式

当不提供校准数据时（FP8_DYNAMIC、FP8_BLOCK）：

```python
class DataFreePipeline:
    def run(self, model, modifiers):
        """无前向传播，直接初始化和完成"""
        for mod in modifiers:
            mod.on_initialize(state)
        
        for mod in modifiers:
            mod.on_start(state)
            mod.on_event(state, SEQUENTIAL_EPOCH_END)
            mod.on_end(state)
        
        for mod in modifiers:
            mod.on_finalize(state)
```

---

## 各场景详细使用指南

### 场景 1：快速部署，精度要求高 → FP8_DYNAMIC

**适用条件**：
- 有 Hopper (H100) 或 Ada (RTX 4090/L40S) GPU
- 希望快速量化，不想准备校准数据
- 可以接受约 2× 压缩
- 精度损失需要极小（perplexity 增加 < 0.1）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from compressed_tensors.offload import dispatch_model
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

oneshot(model=model, recipe=recipe)

# 验证
dispatch_model(model)
input_ids = tokenizer("Hello", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))

model.save_pretrained("Llama-3-8B-FP8-Dynamic")
tokenizer.save_pretrained("Llama-3-8B-FP8-Dynamic")
```

### 场景 2：DeepSeek 风格部署 → FP8_BLOCK

```python
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_BLOCK",
    ignore=["lm_head", "re:.*mlp.gate$"],
)
oneshot(model=model, recipe=recipe)
```

### 场景 3：快速验证 → W4A16 RTN（仅测试用）

```python
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
)
oneshot(model=model, recipe=recipe)
# 注意：生产环境请用 GPTQ/AWQ！
```

### 场景 4：生产 4-bit → GPTQ

```python
from llmcompressor.modifiers.gptq import GPTQModifier

recipe = GPTQModifier(
    targets="Linear", scheme="W4A16", ignore=["lm_head"],
    block_size=128, dampening_frac=0.01, actorder="static",
)
oneshot(model=model, recipe=recipe, dataset=dataset,
        num_calibration_samples=512, max_seq_length=2048)
```

---

## 性能对比总结

| 方法 | 压缩比 | 量化时间(8B) | Perplexity 增加 | 需要数据 | 推荐场景 |
|------|:---:|:---:|:---:|:---:|------|
| FP8_DYNAMIC RTN | 2× | ~1 分钟 | < 0.05 | 否 | **快速部署，精度高** |
| FP8_BLOCK RTN | 2× | ~1 分钟 | < 0.03 | 否 | **DeepSeek 风格** |
| W8A8 RTN | 2× | ~5 分钟 | 0.1-0.3 | 是 | INT8 加速 |
| W4A16 RTN | 3.9× | ~1 分钟 | 0.5-1.0 | 否 | **仅快速验证** |
| W4A16 GPTQ | 3.9× | ~30 分钟 | 0.1-0.2 | 是 | **生产 4-bit** |
| W4A16 AWQ | 3.9× | ~20 分钟 | 0.1-0.2 | 是 | **生产 4-bit** |
| W4A16 AutoRound | 3.9× | ~60 分钟 | 0.08-0.15 | 是 | 追求极致精度 |
