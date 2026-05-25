# 量化基础理论

## 什么是量化

量化（Quantization）是将高精度浮点数（如 FP16/BF16）映射到低精度表示（如 INT4/INT8/FP8）的过程。对于 LLM，量化的核心目标是：

1. **减少模型体积**：4-bit 量化可将模型大小压缩至原来的 1/4
2. **降低显存占用**：更小的数据类型意味着更少的 GPU 内存需求
3. **加速推理**：低精度算术运算（如 INT8 矩阵乘法）通常更快

## 均匀量化（Uniform Quantization）

### 基本公式

均匀量化将连续浮点值映射到等间距的离散级别：

**量化（Quantize）**：
$$q = \text{clamp}\left(\text{round}\left(\frac{x}{s}\right) + z,\ q_{min},\ q_{max}\right)$$

**反量化（Dequantize）**：
$$\hat{x} = s \cdot (q - z)$$

其中：
- $x$：原始浮点权重/激活值
- $q$：量化后的整数值
- $s$：缩放因子（scale）
- $z$：零点（zero point）
- $q_{min}, q_{max}$：量化范围（如 INT8 为 [-128, 127]）

### 对称量化 vs 非对称量化

**对称量化（Symmetric）**：

$$s = \frac{\max(|x|)}{q_{max}}$$
$$z = 0$$

- 零点固定为 0，简化计算
- 适合权重（通常近似对称分布）
- 范围：$[-q_{max} \cdot s,\ q_{max} \cdot s]$

**非对称量化（Asymmetric）**：

$$s = \frac{x_{max} - x_{min}}{q_{max} - q_{min}}$$
$$z = q_{min} - \text{round}\left(\frac{x_{min}}{s}\right)$$

- 更精确地覆盖实际值域
- 适合激活（可能存在偏移，如 ReLU 后全为正值）
- 范围：$[x_{min},\ x_{max}]$

## 量化粒度（Granularity）

量化粒度决定了"多少个元素共享一组量化参数（scale, zero_point）"：

### Per-Tensor（张量级）

整个权重矩阵共享一个 scale：

$$s = \frac{\max(|W|)}{q_{max}}$$

- 最简单，参数开销最小
- 精度最低（一个异常值会影响整个矩阵）

### Per-Channel（通道级）

每个输出通道（矩阵的每一行）独立计算 scale：

$$s_i = \frac{\max(|W[i,:]|)}{q_{max}}$$

- 精度显著优于 per-tensor
- 对 LLM 权重矩阵是标准选择
- 额外存储：$C_{out}$ 个 scale 值

### Per-Group（分组级）

将通道内连续的 $g$ 个元素分为一组，每组独立量化：

$$s_{i,j} = \frac{\max(|W[i, j \cdot g : (j+1) \cdot g]|)}{q_{max}}$$

- 典型 group_size：32、64、128
- 精度高于 per-channel，是 4-bit 量化的标准配置
- 额外存储：$C_{out} \times \lceil C_{in} / g \rceil$ 个 scale 值

### Per-Token（令牌级）

对激活按每个 token 独立量化：

$$s_t = \frac{\max(|X[t,:]|)}{q_{max}}$$

- 用于动态激活量化
- 每个 token 的激活分布差异大，per-token 显著优于 per-tensor
- 在推理时动态计算，不需要提前校准

### TENSOR_GROUP（张量分组级）

Per-group 基础上增加 global_scale：

$$\text{local\_scale}_{i,j} = \frac{\max(|W[i, j \cdot g : (j+1) \cdot g]|)}{\text{global\_scale} \cdot q_{max}}$$

- 用于 NVFP4、FP8_BLOCK 等格式
- global_scale 为所有 local_scale 的缩放因子
- 两级缩放结构可以用低精度存储 local_scale

## 量化误差分析

### 舍入误差

量化的核心误差来源于 `round()` 操作：

$$\epsilon_{round} = x - \hat{x} = x - s \cdot \text{round}(x/s)$$

对于均匀分布的权重，期望均方误差为：

$$E[\epsilon^2] = \frac{s^2}{12}$$

即 scale 越大（量化越粗糙），误差越大。

### 截断误差

超出量化范围的值被截断（clamp），造成额外误差：

$$\epsilon_{clip} = x - s \cdot q_{max}, \quad \text{当}\ x > s \cdot q_{max}$$

### 对模型输出的影响

单层量化误差通过模型前向传播累积。对 Transformer 的线性层：

$$Y = X \cdot W^T$$

权重量化误差传播为：

$$\Delta Y = X \cdot \Delta W^T$$

其中 $\Delta W = W - \hat{W}$ 是权重量化误差。关键洞察：**输入激活 $X$ 的大小会放大权重误差** — 这就是为什么 AWQ 和 SmoothQuant 关注激活分布。

## Weight-Only vs Weight-Activation 量化

### Weight-Only 量化（W4A16、W8A16）

- 仅量化权重，激活保持 FP16/BF16
- 主要收益：减少模型体积和显存占用
- 推理时需要反量化权重再做矩阵乘法
- 精度损失较小，适合高压缩比（4-bit）
- 典型方案：GPTQ W4A16、AWQ W4A16

### Weight-Activation 量化（W8A8、W4A4）

- 同时量化权重和激活
- 主要收益：可以使用低精度矩阵乘法核（如 INT8 GEMM、FP8 GEMM）
- 推理速度提升显著（利用硬件加速）
- 需要更精心的校准（激活分布更复杂）
- 典型方案：SmoothQuant W8A8、FP8_DYNAMIC

## 浮点量化 vs 整数量化

### 整数量化（INT4/INT8）

$$q = \text{round}(x / s) + z$$

- 等间距分布，每个量化级别间隔相同
- 硬件支持广泛（所有 NVIDIA GPU）
- 需要更精细的校准来处理异常值
- 典型精度：INT8 损失 ~0.1-0.5%，INT4 损失 ~1-3%

### 浮点量化（FP8/FP4）

FP8 有两种格式：
- **E4M3**：4 位指数 + 3 位尾数，范围 ±448，精度较高
- **E5M2**：5 位指数 + 2 位尾数，范围 ±57344，精度较低

$$q = \text{cast\_to\_fp8}(x / s)$$

- 非等间距分布（对数间距），小值附近更密集
- 天然适合神经网络的权重/激活分布（集中在 0 附近）
- FP8 量化精度通常优于 INT8（同等比特数下）
- 需要 Hopper (H100) 或更新 GPU 的硬件加速

### 微缩放格式（Microscaling：MXFP4/MXFP8）

OCP（Open Compute Project）定义的标准：

- 每 32 个元素共享一个 8-bit 缩放因子（shared exponent）
- 组内每个元素用 FP4 或 FP8 存储
- 兼顾压缩率和精度
- NVIDIA Blackwell GPU 原生支持

### NVFP4（NVIDIA 专有）

- 每 16 个元素共享一个缩放因子
- 全局缩放因子（global_scale）+ 局部缩放因子（per-group scale）
- 专为 Blackwell SM100+ 架构设计
- 需要校准数据确定 global_scale

## 校准（Calibration）

### 为什么需要校准

量化参数（scale, zero_point）需要知道数据的实际分布范围。对于：
- **权重**：可以直接从权重值计算（不需要数据）
- **激活**：依赖输入数据，必须用代表性样本进行校准

### 校准过程

```python
# 伪代码：校准流程
for batch in calibration_data:
    output = model(batch)        # 前向传播
    # Observer 在 hook 中自动收集统计量：
    #   min_val = min(min_val, x.min())
    #   max_val = max(max_val, x.max())

# 校准结束后计算量化参数
scale = (max_val - min_val) / (qmax - qmin)
zero_point = qmin - round(min_val / scale)
```

### 校准样本数量的影响

| 样本数 | 精度 | 速度 | 建议 |
|--------|------|------|------|
| 128 | 较低 | 最快 | 快速验证 |
| 512 | 良好 | 适中 | **推荐默认值** |
| 1024 | 最佳 | 较慢 | 追求最高精度 |
| 2048+ | 边际递减 | 最慢 | 通常无必要 |

### 校准数据选择

- 应覆盖模型实际使用场景的分布
- 常用数据集：`HuggingFaceH4/ultrachat_200k`、`wikitext`、C4
- 序列长度建议：2048（覆盖长上下文场景）
- 避免使用分布极端或单一主题的数据

## 量化对推理的影响

### 内存节省

| 原始格式 | 量化格式 | 压缩比 | 70B 模型显存 |
|----------|----------|--------|-------------|
| FP16 | FP8 | 2× | ~70GB → ~35GB |
| FP16 | INT8 | 2× | ~70GB → ~35GB |
| FP16 | INT4 (group=128) | ~3.5× | ~70GB → ~20GB |
| FP16 | FP4 | 4× | ~70GB → ~17.5GB |

### 推理加速

- **Weight-Only (W4A16)**：主要节省带宽，对 batch=1 推理加速明显
- **W8A8 (FP8/INT8)**：利用 FP8/INT8 GEMM 核心，大 batch 下加速 1.5-2×
- **W4A4 (NVFP4)**：Blackwell GPU 原生 FP4 核心，极致性能
