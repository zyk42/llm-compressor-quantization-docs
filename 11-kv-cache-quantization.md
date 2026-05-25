# KV Cache 量化

## 背景：KV Cache 的内存问题

在 LLM 推理中，KV Cache 存储每个 token 的 Key 和 Value 状态，随着序列长度增长线性增加。对于长序列场景，KV Cache 可能成为主要内存瓶颈。

### 内存占用分析

单层 KV Cache 的内存：

$$\text{Memory} = 2 \times \text{batch} \times \text{seq\_len} \times \text{num\_heads} \times \text{head\_dim} \times \text{bytes\_per\_element}$$

以 Llama-3-70B 为例（80 层，64 头，head_dim=128）：

| 序列长度 | FP16 KV Cache | FP8 KV Cache | 节省 |
|----------|:---:|:---:|:---:|
| 2K | 2.5 GB | 1.25 GB | 50% |
| 8K | 10 GB | 5 GB | 50% |
| 32K | 40 GB | 20 GB | 50% |
| 128K | 160 GB | 80 GB | 50% |

KV Cache 量化到 FP8 可以将缓存占用减半，等效于将可服务的序列长度翻倍。

### 内存节省计算公式

**通用 KV Cache 内存公式**：

```
KV_Memory (bytes) = 2 × num_layers × batch_size × seq_len × num_kv_heads × head_dim × bytes_per_elem
                    ↑                                        ↑
                    K和V各一份                                GQA 时 kv_heads < q_heads
```

**实际计算示例（Llama-3-70B）**：

```
模型参数:
  num_layers = 80
  num_kv_heads = 8  (GQA, 8 KV heads shared by 64 Q heads)
  head_dim = 128
  
FP16 (2 bytes/elem):
  KV_mem = 2 × 80 × 1 × 8192 × 8 × 128 × 2
         = 2 × 80 × 8192 × 8 × 128 × 2
         = 2,684,354,560 bytes ≈ 2.5 GB (per batch, seq=8K)

FP8 (1 byte/elem):
  KV_mem = 2 × 80 × 1 × 8192 × 8 × 128 × 1
         = 1,342,177,280 bytes ≈ 1.25 GB (per batch, seq=8K)

节省 = 2.5 - 1.25 = 1.25 GB per batch per 8K tokens
```

**多 batch 场景下的收益放大**：

```
场景: Llama-3-70B, 128K 上下文, 32 并发请求 (continuous batching)
FP16: 2 × 80 × 32 × 131072 × 8 × 128 × 2 = 2.62 TB  (不可能!)
FP8:  2 × 80 × 32 × 131072 × 8 × 128 × 1 = 1.31 TB  (仍然太大)

实际部署使用 PagedAttention:
  - 物理内存按 page 分配 (page_size=16 tokens)
  - KV Cache 量化从 FP16→FP8 使得同等 GPU 内存可服务:
    → 2× 的并发 batch 数
    → 或 2× 的序列长度
    → 或 sqrt(2)× 的两者组合提升
```

**vLLM 中 KV Cache 占用估算**：

```
vLLM 启动时显示:
  # GPU blocks: 2048, # CPU blocks: 512
  
每个 GPU block = page_size × num_layers × 2(K+V) × num_kv_heads × head_dim × dtype_size
例如 Llama-3-8B (FP8 KV):
  block = 16 × 32 × 2 × 8 × 128 × 1 = 1,048,576 bytes = 1 MB
  
总 KV Cache 容量 = 2048 blocks × 1 MB = 2 GB
可缓存的总 token 数 = 2048 × 16 = 32,768 tokens
```

## 量化原理

### 目标模块

KV Cache 量化作用于 Attention 层的 Key 和 Value 输出：

```
q_proj → Q states（Query，每次推理重新计算）
k_proj → K states → 存入 KV Cache ← 量化目标
v_proj → V states → 存入 KV Cache ← 量化目标
```

在 LLM Compressor 中，KV Cache 量化通过对 `k_proj` 和 `v_proj` 的**输出**施加量化实现（注意：量化的是这些投影层的输出激活，即将要存入缓存的 K 和 V 向量）。

### 完整数据流详解

以下是 KV Cache 量化在推理时的完整数据流：

```
┌──────────────────────────────────────────────────────────────────┐
│                    Attention Layer 数据流                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  输入 hidden_states [batch, seq_len, hidden_dim]                  │
│       │                                                           │
│       ├──→ q_proj(x) → Q [batch, seq, num_heads, head_dim]       │
│       │                     │                                     │
│       │                     │ (Q 不缓存，每次重新计算)             │
│       │                     ↓                                     │
│       ├──→ k_proj(x) → K [batch, seq, num_kv_heads, head_dim]    │
│       │                     │                                     │
│       │                     ↓  ★ 量化步骤 ★                       │
│       │               K_scale = max(|K|) / 448.0                  │
│       │               K_fp8 = cast_to_fp8(K / K_scale)            │
│       │                     │                                     │
│       │                     ↓  存入 KV Cache                      │
│       │              ┌──────────────────┐                         │
│       │              │  KV Cache (FP8)  │                         │
│       │              │  K_cache[layer]  │ ← 追加新 token 的 K     │
│       │              │  V_cache[layer]  │ ← 追加新 token 的 V     │
│       │              └──────────────────┘                         │
│       │                     │                                     │
│       │                     ↓  ★ 反量化步骤 ★                     │
│       │               K_dequant = K_fp8 * K_scale                 │
│       │               (将历史所有 K 反量化回 FP16/BF16)            │
│       │                     │                                     │
│       │                     ↓                                     │
│       │              Attention = softmax(Q @ K^T / sqrt(d)) @ V   │
│       │                                                           │
│       └──→ v_proj(x) → V [batch, seq, num_kv_heads, head_dim]    │
│                             │                                     │
│                             ↓  (同 K 的量化/反量化流程)            │
│                             ↓                                     │
│                      V_dequant 参与 attention 计算                 │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**逐步执行追踪（单 token 生成，decode 阶段）**：

```
时刻 t: 生成第 t 个 token

1. 输入: x_t = embedding[token_{t-1}] 经过前面层的输出
   形状: [1, 1, hidden_dim]  (batch=1, seq=1 for decode)

2. 计算投影:
   q_t = x_t @ W_q  → [1, 1, num_heads × head_dim]
   k_t = x_t @ W_k  → [1, 1, num_kv_heads × head_dim]
   v_t = x_t @ W_v  → [1, 1, num_kv_heads × head_dim]

3. 对 k_t 量化并存入 cache:
   k_t_reshaped = k_t.view(1, 1, num_kv_heads, head_dim)
   // Per-head 量化 (以 head 0 为例):
   k_head0 = k_t_reshaped[0, 0, 0, :]  // shape: [head_dim=128]
   scale_k_head0 = max(|k_head0|) / 448.0
   k_head0_fp8 = round_to_fp8(k_head0 / scale_k_head0)
   
   // 存入 cache (追加到位置 t):
   kv_cache.k[layer][0, t, 0, :] = k_head0_fp8
   kv_cache.k_scale[layer][0, 0] = scale_k_head0  // per-head scale

4. 对 v_t 同样处理并存入 cache

5. Attention 计算 (需要反量化历史 cache):
   // 读取完整 K cache [0:t+1] 并反量化
   K_all_fp8 = kv_cache.k[layer][0, 0:t+1, :, :]  // [t+1, num_kv_heads, head_dim]
   K_all = K_all_fp8.to(float16) * kv_cache.k_scale[layer]  // 广播乘 scale
   
   // Q @ K^T
   attn_weights = q_t @ K_all.transpose(-1, -2) / sqrt(head_dim)
   attn_weights = softmax(attn_weights)
   
   // 读取完整 V cache 并反量化
   V_all = kv_cache.v[layer][0, 0:t+1, :, :].to(float16) * kv_cache.v_scale[layer]
   
   // Attention output
   output = attn_weights @ V_all
```

**vLLM 中的实际实现路径**：

```python
# vLLM PagedAttention with FP8 KV Cache (简化伪代码)
class PagedAttentionWithFP8KVCache:
    def forward(self, query, key, value, kv_cache, attn_metadata):
        # 1. 量化 key/value 到 FP8
        key_cache, value_cache = kv_cache[0], kv_cache[1]
        
        # 2. 使用 Flashinfer/FlashAttention 内核
        #    内核内部处理: FP8→FP16 反量化 + attention 计算融合
        #    避免了显式的反量化 + 存储中间 FP16 张量的开销
        output = flash_attn_with_kvcache(
            q=query,
            k_cache=key_cache,      # FP8 格式存储
            v_cache=value_cache,    # FP8 格式存储
            cache_seqlens=attn_metadata.seq_lens,
            k_scale=kv_scale,       # 反量化 scale
            v_scale=kv_scale,
        )
        return output
```

### 量化策略

**Per-Tensor（默认）**：
- 整个 KV Cache 共享一个 scale
- 最简单，推理开销最小
- 精度可能不足（不同头的分布差异大）

**Per-Head（推荐用于高精度需求）**：
- 每个注意力头独立量化
- 精度更高（适应各头不同的值域）
- vLLM 支持 per-head FP8 KV Cache

**Per-Head 策略详解：为什么不同头需要不同 scale？**

```
观察: 在多头注意力中，不同头学习了不同的"角色":
  - 某些头关注局部位置信息 → K 值范围较小 (如 [-0.5, 0.5])
  - 某些头关注全局语义 → K 值范围较大 (如 [-3.0, 3.0])
  - 某些头呈现离群值模式 → 偶尔出现极端值

如果使用 per-tensor scale (所有头共享一个 scale):
  scale = max(|all_heads|) / 448 = 3.0 / 448 = 0.0067
  对于范围小的头 ([-0.5, 0.5]):
    有效利用的量化级别 = 0.5 / 0.0067 ≈ 75 (out of 448)
    → 精度损失大! 大量量化级别被浪费

使用 per-head scale:
  Head_0 (范围 [-0.5, 0.5]): scale_0 = 0.5/448 = 0.00112
    → 充分利用所有 448 个量化级别
  Head_1 (范围 [-3.0, 3.0]): scale_1 = 3.0/448 = 0.0067
    → 同样充分利用

结论: Per-head 使每个头都能最大限度利用 FP8 的精度
     代价: 存储 num_kv_heads 个 scale 而非 1 个 (极小开销)
```

**实际精度差异度量**：

```
实验数据 (Llama-3-8B, 8K context):
  Head 0: K 值 std=0.15, max=0.62
  Head 1: K 值 std=0.43, max=2.87
  Head 2: K 值 std=0.08, max=0.31
  Head 3: K 值 std=0.55, max=3.41
  ...
  
  Per-tensor MSE: 1.2e-4
  Per-head MSE:   3.8e-5  (3.2× 更精确)
```

### 与权重量化的区别

| | 权重量化 | KV Cache 量化 |
|--|---------|--------------|
| 量化对象 | 模型权重（静态） | 中间激活（动态） |
| 量化时机 | 离线一次性 | 推理时每次存入缓存 |
| scale 计算 | 校准时确定 | 动态 或 校准时确定 |
| 存储位置 | SafeTensors 文件 | GPU 显存中的缓存 |
| 影响范围 | GEMM 计算精度 | Attention 计算精度 |
| 计算开销 | 无（推理时已是低精度） | 有（量化/反量化每次都执行） |

### kv_cache_scheme 与权重量化的交互

KV Cache 量化与权重量化是**正交**的两个维度，可以独立组合：

```
┌─────────────────────────────────────────────────────────┐
│                   组合矩阵                               │
├──────────────────┬──────────┬───────────┬───────────────┤
│                  │ 无KV量化  │ FP8 KV    │ INT8 KV       │
├──────────────────┼──────────┼───────────┼───────────────┤
│ FP8 权重 (W8A8)  │ 标准      │ 常见组合   │ 可行但少见    │
│ INT4 权重 (W4A16)│ 标准      │ 最佳平衡   │ 可行          │
│ NVFP4 权重       │ 标准      │ 最大压缩   │ 不兼容        │
│ 无权重量化       │ 基准      │ 长序列优化  │ 长序列优化    │
└──────────────────┴──────────┴───────────┴───────────────┘
```

**组合时的数据流**：

```
以 W4A16 + FP8 KV Cache 为例:

hidden_states (FP16)
    │
    ├─→ k_proj: INT4 权重反量化为 FP16 → FP16 GEMM → K (FP16)
    │                                                    │
    │                                            量化到 FP8 存入 cache
    │                                                    │
    │                                            反量化回 FP16 做 attention
    │
    ├─→ v_proj: 同上流程
    │
    └─→ q_proj: INT4 权重反量化为 FP16 → FP16 GEMM → Q (FP16, 不缓存)

整体压缩效果:
  权重: 4× 压缩 (INT4)
  KV Cache: 2× 压缩 (FP8)
  总内存节省 = 权重节省 + KV Cache 节省（两者独立计算）
```

**关键注意事项**：

1. **kv_cache_scheme 不影响权重量化**：它只控制 k_proj/v_proj 输出的缓存精度
2. **scale 的来源不同**：
   - 权重 scale：来自校准时的权重统计（静态）
   - KV cache scale：可以是校准时对激活统计得到的静态 scale，也可以是推理时动态计算
3. **校准数据的作用**：当使用静态 KV cache scale 时，校准数据用于观察 k_proj/v_proj 输出的分布范围，确定合适的 scale

## 在 LLM Compressor 中的配置

### kv_cache_scheme 参数

KV Cache 量化通过 `QuantizationModifier` 的 `kv_cache_scheme` 参数配置：

```python
from compressed_tensors.quantization import QuantizationArgs

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
    kv_cache_scheme=QuantizationArgs(
        num_bits=8,
        type="float",           # FP8
        strategy="tensor",      # per-tensor
        symmetric=True,
    ),
)
```

### 配置选项

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `num_bits` | 8 | 量化位数 |
| `type` | "float" / "int" | 数据类型 |
| `strategy` | "tensor" / "head" | 量化粒度 |
| `symmetric` | True / False | 对称/非对称 |
| `dynamic` | True / False | 是否动态计算 scale |

### 配置如何映射到模型配置文件

LLM Compressor 在保存模型时，将 KV Cache 量化信息写入 `config.json` 的 `quantization_config` 字段：

```json
{
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
    "kv_cache_scheme": {
      "num_bits": 8,
      "type": "float",
      "strategy": "tensor",
      "symmetric": true
    },
    "config_groups": {
      "group_0": {
        "targets": ["Linear"],
        "weights": { "num_bits": 8, "type": "float", ... },
        "input_activations": { "num_bits": 8, "type": "float", ... }
      }
    }
  }
}
```

vLLM 在加载模型时读取此配置，自动设置 KV Cache 的 dtype 和 scale。

## 使用示例

### 示例 1：FP8 KV Cache（Per-Tensor）

```python
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationArgs

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
    kv_cache_scheme=QuantizationArgs(
        num_bits=8,
        type="float",
        strategy="tensor",
        symmetric=True,
    ),
)

oneshot(
    model=model,
    recipe=recipe,
    dataset=dataset,
    num_calibration_samples=512,
    max_seq_length=2048,
)

model.save_pretrained("Llama-3-8B-FP8-KV")
```

### 示例 2：Per-Head KV Cache 量化

```python
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
    kv_cache_scheme=QuantizationArgs(
        num_bits=8,
        type="float",
        strategy="head",        # per-head 量化
        symmetric=True,
    ),
)
```

### 示例 3：权重量化 + KV Cache 量化组合

```python
from llmcompressor.modifiers.gptq import GPTQModifier

# GPTQ W4A16 + FP8 KV Cache
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
    kv_cache_scheme=QuantizationArgs(
        num_bits=8,
        type="float",
        strategy="tensor",
        symmetric=True,
    ),
)

oneshot(model=model, recipe=recipe, dataset=dataset, num_calibration_samples=512)
```

### 示例 4：YAML 配方格式

```yaml
quantization_stage:
  quant_modifiers:
    QuantizationModifier:
      targets: "Linear"
      scheme: "FP8_DYNAMIC"
      ignore: ["lm_head"]
      kv_cache_scheme:
        num_bits: 8
        type: float
        strategy: tensor
        symmetric: true
```

## vLLM 推理配置

量化后的模型在 vLLM 中使用 KV Cache 量化：

```python
from vllm import LLM

# 加载带 KV Cache 量化的模型
llm = LLM(
    model="Llama-3-8B-FP8-KV",
    kv_cache_dtype="fp8",      # 启用 FP8 KV Cache
    # 或 kv_cache_dtype="fp8_e4m3"
)
```

**注意**：vLLM 会自动从模型的 `config.json` 中读取 KV Cache 量化参数（scale 等）。

**vLLM 内部处理流程**：

```
模型加载时:
  1. 读取 config.json 中的 kv_cache_scheme
  2. 分配 FP8 格式的 KV Cache block pool
  3. 加载预计算的 k_scale / v_scale (如果有)

推理时 (每个 attention 层):
  1. 计算 k_proj(x), v_proj(x) 得到 FP16 的 K, V
  2. 使用预存的 scale 将 K, V 量化到 FP8
  3. 写入对应 page 的 KV Cache slot (FP8 格式)
  4. Attention 计算时:
     - FlashAttention 内核直接读取 FP8 KV Cache
     - 内核内部进行 FP8→FP16 转换 + attention 计算（融合操作）
     - 无需显式反量化到临时 FP16 buffer
```

## 精度影响

KV Cache 量化对模型精度的影响通常很小：

| 方案 | Perplexity 增加 | 下游任务影响 |
|------|:---:|------|
| FP8 per-tensor | +0.01 ~ +0.05 | 几乎无影响 |
| FP8 per-head | +0.005 ~ +0.02 | 几乎无影响 |
| INT8 per-tensor | +0.05 ~ +0.1 | 极小影响 |

**原因**：
1. KV Cache 中的值通常范围有限（softmax 后的注意力权重限制了 V 的使用方式）
2. FP8 的精度对于 KV 值已经足够
3. 量化误差在多头注意力中被平均

**精度分析深入**：

```
为什么 KV Cache 量化精度损失比权重量化小?

1. 权重量化: 每次 GEMM 都使用量化权重，误差在层间累积
   误差传播: layer_0 误差 → layer_1 输入偏差 → layer_1 误差 → ...
   
2. KV Cache 量化: 仅影响 attention 计算
   - K 量化误差: 影响 attention weights (QK^T)
     但 softmax 的归一化效应会抑制小误差的影响
   - V 量化误差: 影响 attention output (weighted sum of V)
     多头平均进一步抑制误差
   
3. 定量分析:
   FP8 相对误差 ≈ 2^(-3) / value ≈ 0.1% (对于典型 KV 值)
   经过 softmax 归一化后的 attention weight 误差 ≈ 0.01%
   多头 (32-128 头) 平均后误差 ≈ 0.01% / sqrt(num_heads)
```

## 最佳实践

1. **首选 FP8 per-tensor**：精度损失极小，推理开销最低
2. **长序列场景必备**：序列长度 > 8K 时，KV Cache 量化收益明显
3. **与权重量化正交**：可以同时使用 W4A16 + FP8 KV Cache
4. **Per-head 用于高要求场景**：如果 per-tensor 精度不够，升级到 per-head
5. **校准数据使用长序列**：确保校准数据包含长序列样本
6. **GQA 模型收益计算**：注意 GQA 模型的 kv_heads 远少于 q_heads，因此 KV Cache 本身就较小，量化节省的绝对内存量也相应减少
7. **监控长序列精度**：KV Cache 量化的精度影响随序列长度增加而略微增大（更多历史 token 累积量化误差），超长序列（>64K）时建议使用 per-head 策略
