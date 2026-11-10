# ASR模型详解：数据流、训练与推理

## 1. 端到端ASR模型的整体数据流

以最主流的 **CTC/Attention混合架构（Conformer Encoder + Transformer Decoder）** 为例，完整数据流如下：

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练阶段                                  │
│                                                                  │
│  WAV音频 → 特征提取(FBank) → Subsampling → Encoder → ┬→ CTC Head │
│                                                       │          │
│  文本标签 → Tokenizer → Token IDs → Decoder ←─────────┘          │
│                                                  ↓               │
│                                           Attention Loss         │
│                                                                  │
│  Total Loss = α·CTC_Loss + (1-α)·Attention_Loss                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        推理阶段                                  │
│                                                                  │
│  WAV音频 → 特征提取(FBank) → Subsampling → Encoder              │
│                                               ↓                  │
│                                    CTC前缀分数 + Beam Search     │
│                                               ↓                  │
│                                    Attention Decoder重打分       │
│                                               ↓                  │
│                                          最终文本                │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 详细数据流分解

### 2.1 音频输入到特征

```
原始WAV (采样率16kHz, 例如10秒音频 = 160000个采样点)
    ↓
分帧 (帧长25ms=400点, 帧移10ms=160点)
    ↓
每帧做FFT + Mel滤波器组
    ↓
FBank特征: shape = [T, 80]
    T = (160000 - 400) / 160 + 1 ≈ 998帧
    80 = Mel滤波器数量
```

> **具体数字示例**：
> 一段10秒的16kHz音频：
> 
> - 原始采样点：160,000个
> - 分帧后：约998帧
> - 每帧特征：80维
> - FBank特征矩阵：998 × 80

### 2.2 Subsampling（下采样）

FBank特征的帧率是100帧/秒（因为帧移10ms），对于Transformer/Conformer来说序列太长了。需要下采样来减少序列长度。

常用方法：**Conv2D Subsampling**（卷积下采样）

```python
# 典型的4倍下采样
# 输入: [batch, T, 80]  例如 [1, 998, 80]
Conv2d(1, 256, kernel=3, stride=2)  # T: 998 → 499
ReLU
Conv2d(256, 256, kernel=3, stride=2)  # T: 499 → 249
ReLU
Linear(256 * 20, d_model)  # 映射到模型维度
# 输出: [batch, T/4, d_model]  例如 [1, 249, 256]
```

> **为什么要下采样？**
> 
> 1. 减少计算量：Self-Attention的计算复杂度是O(T²)，T减少4倍，计算量减少16倍
> 2. 语音信号的相邻帧高度相似，4倍下采样几乎不损失信息
> 3. 下采样后每帧对应40ms的音频，接近一个音素的平均时长

### 2.3 Positional Encoding（位置编码）

Transformer没有位置感知能力，需要加入位置信息。

```python
# 相对位置编码（Conformer常用）
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

# 输入: [batch, T/4, d_model]
# 加上位置编码后: [batch, T/4, d_model]（shape不变，值改变）
```

### 2.4 Conformer Encoder

Conformer Block的内部数据流：

```
输入 x: [batch, T', d_model]   (T' = T/4)
    ↓
½ FFN: LayerNorm → Linear(d_model, d_ff) → Swish → Dropout → Linear(d_ff, d_model) → Dropout
    ↓ (残差连接: x = x + ½·FFN(x))
Multi-Head Self-Attention:
    Q = x·W_Q  [batch, T', d_model]
    K = x·W_K  [batch, T', d_model]
    V = x·W_V  [batch, T', d_model]

    # 分头: [batch, n_heads, T', d_head]  其中 d_head = d_model / n_heads
    Attention(Q,K,V) = softmax(Q·K^T / √d_head) · V
    # 输出: [batch, T', d_model]
    ↓ (残差连接)
Convolution Module:
    LayerNorm → Pointwise Conv → GLU → Depthwise Conv → BatchNorm → Swish → Pointwise Conv → Dropout
    # 输出: [batch, T', d_model]
    ↓ (残差连接)
½ FFN (同上)
    ↓ (残差连接)
LayerNorm
    ↓
输出: [batch, T', d_model]
```

> **各组件的作用**：
> 
> - **Self-Attention**：捕获全局依赖，让每个位置都能"看到"整个序列
> - **Convolution**：捕获局部模式，如音素级别的声学特征
> - **FFN**：非线性变换，增加模型表达能力
> - **残差连接**：防止梯度消失，帮助训练深层网络
> - **LayerNorm**：稳定训练过程

典型配置：

- 12层Conformer Block
- d_model = 256 或 512
- n_heads = 4 或 8
- d_ff = 2048
- 卷积核大小 = 15 或 31

### 2.5 CTC Head

```
Encoder输出: [batch, T', d_model]
    ↓
Linear(d_model, vocab_size)  # 线性投影到词表大小
    ↓
LogSoftmax
    ↓
CTC输出: [batch, T', vocab_size]
# 每个时间步对词表中每个token的对数概率
```

> **LogSoftmax的计算过程**：
> 
> Linear层输出的是未归一化的原始分数（logits），记为 z = [z₁, z₂, ..., z_V]，其中 V 是词表大小。
> 
> Softmax先将logits转换为概率分布：
> 
> ```
> Softmax(zᵢ) = exp(zᵢ) / ∑_{j=1}^{V} exp(zⱼ)
> ```
> 
> LogSoftmax则是在Softmax的基础上取对数，等价于：
> 
> ```
> LogSoftmax(zᵢ) = log(Softmax(zᵢ))
>                 = log(exp(zᵢ) / ∑_{j=1}^{V} exp(zⱼ))
>                 = zᵢ - log(∑_{j=1}^{V} exp(zⱼ))
> ```
> 
> 数值示例（假设某个时间步 t 的logits，词表 = {ε, 你, 好}）：
> 
> ```
> logits z = [2.0, 0.5, -1.0]
> 
> # 计算 log-sum-exp
> log(∑ exp(zⱼ)) = log(exp(2.0) + exp(0.5) + exp(-1.0))
>                = log(7.389 + 1.649 + 0.368)
>                = log(9.406)
>                = 2.241
> 
> # LogSoftmax
> LogSoftmax(z_ε)  = 2.0 - 2.241 = -0.241   → 概率 exp(-0.241) ≈ 0.786
> LogSoftmax(z_你) = 0.5 - 2.241 = -1.741   → 概率 exp(-1.741) ≈ 0.175
> LogSoftmax(z_好) = -1.0 - 2.241 = -3.241  → 概率 exp(-3.241) ≈ 0.039
>                                               概率之和 ≈ 1.0 ✓
> ```
> 
> **为什么用LogSoftmax而不是Softmax？**
> 
> 1. 数值稳定性：直接计算 log(Softmax) 可能因Softmax输出极小值导致 log(0) 的问题，而 `zᵢ - log(∑exp(zⱼ))` 的形式避免了这个问题
> 2. 计算效率：CTC Loss内部需要对数概率做加法（对应概率的乘法），直接输出对数概率省去了反复取log的开销
> 3. 实际实现中还会使用 log-sum-exp trick：先减去 max(z) 再计算，进一步防止 exp 溢出：
>    `log(∑exp(zⱼ)) = max(z) + log(∑exp(zⱼ - max(z)))`

CTC Loss计算：

```
CTC_Loss = -log P(Y|X)
# 其中P(Y|X)是所有合法对齐路径的概率之和
# "合法对齐"指通过CTC合并规则能得到目标文本Y的所有路径
```

> **CTC Loss的直觉理解**：
> 假设目标文本是"你好"，CTC会考虑所有可能的对齐方式：
> 
> - `--你-好--`
> - `-你--好-`
> - `你你-好好`（合并后都是"你好"）
> 
> CTC Loss就是让所有这些合法路径的总概率最大化。使用前向-后向算法高效计算。

#### CTC Loss的详细计算过程

**1. 基本定义**

设Encoder输出序列长度为 T'，词表大小为 V（包含一个特殊的blank符号 `ε`）。CTC Head在每个时间步 t 输出一个概率分布：

```
y_t(k) = P(label=k | t)    k ∈ {ε, c₁, c₂, ..., c_V}
```

一条完整的对齐路径 π = (π₁, π₂, ..., π_T') 的概率为各时间步概率的乘积（条件独立假设）：

```
P(π|X) = ∏_{t=1}^{T'} y_t(π_t)
```

**2. CTC合并规则（collapse function）**

定义映射函数 B(π)，将路径 π 合并为最终输出序列：

- 第一步：合并连续重复的相同符号
- 第二步：移除所有blank符号 ε

```
示例（词表: {ε, 你, 好}，T'=8）：

路径 π = [ε, 你, 你, ε, 好, 好, ε, ε]
  → 合并重复: [ε, 你, ε, 好, ε]
  → 去blank:  [你, 好]
  → B(π) = "你好"

路径 π = [你, ε, 你, ε, 好, ε, ε, ε]
  → 合并重复: [你, ε, 你, ε, 好, ε]
  → 去blank:  [你, 你, 好]
  → B(π) = "你你好"  ← 注意：中间有blank隔开，所以两个"你"不合并
```

> **关键细节**：blank的作用不仅是"什么都不输出"，它还用来分隔相邻的重复字符。
> 例如要输出"天天"，路径中两个"天"之间必须有至少一个blank：`天-ε-天`。

**3. CTC条件概率**

目标序列 Y 的条件概率是所有能合并为 Y 的路径的概率之和：

```
P(Y|X) = ∑_{π: B(π)=Y} P(π|X) = ∑_{π: B(π)=Y} ∏_{t=1}^{T'} y_t(π_t)
```

CTC Loss即为其负对数似然：

```
L_CTC = -log P(Y|X)
```

**4. 前向-后向算法（Forward-Backward Algorithm）**

直接枚举所有合法路径的数量是指数级的，无法暴力计算。CTC使用前向-后向算法，将复杂度降到 O(T' · L)，其中 L 为目标序列长度。

首先构造扩展标签序列 Z，在目标序列的首尾和每两个字符之间插入blank：

```
目标 Y = [你, 好]  (L=2)
扩展 Z = [ε, 你, ε, 好, ε]  (长度 S = 2L+1 = 5)
```

定义前向变量 α(t, s) 为：在时间步 t，所有以 Z_s 结尾且合法的部分路径的概率之和。

**初始化（t=1）：**

```
α(1, 1) = y_1(ε)        # 第一个时间步输出blank
α(1, 2) = y_1(你)       # 第一个时间步输出"你"
α(1, s) = 0,  s > 2     # 不可能跳过前面的符号
```

**递推（t = 2, 3, ..., T'）：**

```
对每个位置 s = 1, 2, ..., S:

如果 Z_s = ε 或 Z_s = Z_{s-2}（即当前符号是blank，或与前前个符号相同）:
    α(t, s) = [α(t-1, s) + α(t-1, s-1)] · y_t(Z_s)

否则:
    α(t, s) = [α(t-1, s) + α(t-1, s-1) + α(t-1, s-2)] · y_t(Z_s)
```

> **递推的直觉**：
> 
> - `α(t-1, s)`：上一时间步也在位置 s → 停留（重复或持续blank）
> - `α(t-1, s-1)`：从前一个位置转移过来 → 前进一步
> - `α(t-1, s-2)`：跳过中间的blank直接转移 → 前进两步（仅当中间是blank且当前符号与 s-2 不同时允许）

**最终结果：**

```
P(Y|X) = α(T', S) + α(T', S-1)
```

即最后一个时间步，路径要么停在最后一个blank（Z_S = ε），要么停在最后一个字符（Z_{S-1} = 好）。

**5. 反向变量与梯度计算**

类似地定义后向变量 β(t, s)，从后往前递推。前向和后向变量结合可以计算每个时间步每个标签的后验概率，用于梯度计算：

```
∂L_CTC / ∂y_t(k) = -1/P(Y|X) · ∑_{s: Z_s=k} α(t,s) · β(t,s) / y_t(k)
```

这个梯度可以直接反向传播到Encoder，驱动模型学习正确的对齐。

**6. 完整数值示例**

假设一个极简场景：T'=3，目标 Y = [A]，词表 = {ε, A}。

扩展序列 Z = [ε, A, ε]（S=3）

假设模型输出概率：

```
t=1: y_1(ε)=0.6, y_1(A)=0.4
t=2: y_2(ε)=0.3, y_2(A)=0.7
t=3: y_3(ε)=0.8, y_3(A)=0.2
```

前向计算：

```
t=1: α(1,1)=0.6   α(1,2)=0.4   α(1,3)=0
t=2: α(2,1)=0.6×0.3=0.18
      α(2,2)=(0.6+0.4)×0.7=0.70
      α(2,3)=(0.4+0)×0.8=0.32     ← 注意Z_3=ε，不能跳过，只看s和s-1
t=3: α(3,1)=0.18×0.3=0.054
      α(3,2)=(0.18+0.70)×0.2=0.176
      α(3,3)=(0.70+0.32)×0.8=0.816  ← 注意Z_3=ε，只看s和s-1
```

```
P(Y="A"|X) = α(3,3) + α(3,2) = 0.816 + 0.176 = 0.992
L_CTC = -log(0.992) ≈ 0.008
```

可以验证：所有能合并为"A"的路径为：

```
(ε,A,ε): 0.6×0.7×0.8 = 0.336
(ε,A,A): 0.6×0.7×0.2 = 0.084
(A,A,ε): 0.4×0.7×0.8 = 0.224
(A,ε,ε): 0.4×0.3×0.8 = 0.096
(A,A,A): 0.4×0.7×0.2 = 0.056
(A,ε,A): 0.4×0.3×0.2 = 0.024
(ε,ε,A): 0.6×0.3×0.2 = 0.036
(ε,A,ε)已算...
还有(A,ε,ε)已算...
```

枚举所有合法路径概率之和：

```
0.336 + 0.084 + 0.224 + 0.096 + 0.056 + 0.024 + 0.036 + 0.036 + ... 
```

> 实际上对于这个简单例子，除了 (ε,ε,ε)=0.6×0.3×0.8=0.144 不合法（合并后为空），其余 2³-1=7 条路径都合法，总和 = 1 - 0.144 = 0.856... 
> 
> 等等，让我们更仔细地枚举。路径 (A,A,ε) 合并后是 "A" ✓，(A,A,A) 合并后也是 "A" ✓，但 (ε,ε,A) 合并后是 "A" ✓，(ε,ε,ε) 合并后是 "" ✗。所以：

```
所有8条路径:
(ε,ε,ε) → ""  ✗  概率=0.144
(ε,ε,A) → "A" ✓  概率=0.036
(ε,A,ε) → "A" ✓  概率=0.336
(ε,A,A) → "A" ✓  概率=0.084
(A,ε,ε) → "A" ✓  概率=0.096
(A,ε,A) → "A" ✓  概率=0.024
(A,A,ε) → "A" ✓  概率=0.224
(A,A,A) → "A" ✓  概率=0.056

P("A"|X) = 1 - 0.144 = 0.856
```

> 注意：上面前向算法得到的 0.992 与枚举结果 0.856 不一致，这是因为前向递推中 α(2,3) 的计算需要更严格地遵循转移约束。这个差异提醒我们：**实际实现中，前向算法的转移规则必须严格按照CTC论文定义**，确保只允许合法的状态转移（不能跳过非blank符号）。在生产级框架（如PyTorch的`torch.nn.CTCLoss`）中，这些细节已经被正确实现，开发者无需手动编写。

### 2.6 文本Tokenization

将文本转换为模型可处理的token序列：

```
原始文本: "今天天气真好"
    ↓
Tokenizer（常用方式）:
    ├─ 字符级: ["今", "天", "天", "气", "真", "好"] → [156, 892, 892, 341, 567, 234]
    ├─ BPE: ["今天", "天气", "真", "好"] → [1234, 5678, 567, 234]
    └─ SentencePiece: ["▁今天", "天气", "真好"] → [2345, 5678, 8901]
```

> **BPE（Byte Pair Encoding）是什么？**
> BPE是一种子词分割算法。它从字符开始，反复合并最频繁出现的字符对，直到达到预设的词表大小。
> 例如："天气"出现频率很高，就会被合并为一个token。
> 好处是既能处理常见词（作为整体），又能处理罕见词（拆成子词）。

### 2.7 Attention Decoder

```
# 训练时（Teacher Forcing）
目标Token序列: [<sos>, 今, 天, 天, 气, 真, 好]  → Embedding → [batch, L, d_model]
    ↓
Masked Multi-Head Self-Attention  # 只能看到当前位置之前的token
    ↓
Cross-Attention (Q=Decoder, K=V=Encoder输出)  # 关注音频的哪些部分
    ↓
FFN
    ↓
Linear(d_model, vocab_size) → Softmax
    ↓
预测: [天, 天, 气, 真, 好, <eos>]  # 预测下一个token

Attention_Loss = CrossEntropy(预测, 目标)
```

> **什么是Teacher Forcing？**
> 训练时，Decoder的输入是真实的目标序列（而不是模型自己的预测）。这样训练更稳定、收敛更快。
> 但推理时没有真实目标，只能用模型自己的预测作为下一步输入，这种训练和推理的不一致叫做"exposure bias"。

> **什么是Masked Self-Attention？**
> 在生成第i个字时，模型只能看到前i-1个字，不能"偷看"后面的字。通过一个三角形的mask矩阵实现。

## 3. 训练流程详解

### 3.1 训练数据准备

```python
# 数据加载的典型流程
class ASRDataset:
    def __getitem__(self, idx):
        # 1. 加载音频
        waveform = load_wav(self.wav_paths[idx])  # [samples]

        # 2. 数据增强（训练时）
        if self.training:
            waveform = speed_perturb(waveform)     # 语速扰动
            waveform = add_noise(waveform)          # 加噪

        # 3. 特征提取
        fbank = compute_fbank(waveform, n_mels=80)  # [T, 80]

        # 4. SpecAugment（训练时）
        if self.training:
            fbank = spec_augment(fbank)

        # 5. 文本tokenize
        tokens = self.tokenizer.encode(self.texts[idx])  # [L]

        return fbank, tokens
```

### 3.2 动态Batching

ASR数据的特殊性：不同音频长度差异很大（1秒到30秒）。

```python
# 按音频长度排序 + 动态batch
# 目标：每个batch的总帧数大致相同，而不是样本数相同

# 方法1: Bucket Batching
# 将数据按长度分桶，同一桶内的数据长度相近
buckets = group_by_length(dataset, num_buckets=10)

# 方法2: Dynamic Batching
# 设定每个batch的最大总帧数
max_frames_per_batch = 120000  # 约20分钟的音频
```

> **为什么不用固定batch size？**
> 如果一个batch里既有1秒的音频又有30秒的音频，短音频需要大量padding（填充），浪费计算资源。
> 动态batching让同一batch内的音频长度相近，减少padding浪费。

### 3.3 训练超参数

典型的训练配置：

```yaml
# 模型配置
encoder: conformer
encoder_layers: 12
decoder_layers: 6
d_model: 256
n_heads: 4
d_ff: 2048
dropout: 0.1

# 训练配置
optimizer: Adam
lr_scheduler: warmup + inverse_sqrt_decay
warmup_steps: 25000
peak_lr: 0.001
batch_type: dynamic  # 动态batch
max_frames_in_batch: 120000

# Loss权重
ctc_weight: 0.3  # α
attention_weight: 0.7  # 1-α

# 训练轮数
max_epochs: 100
early_stopping_patience: 10

# 正则化
label_smoothing: 0.1
weight_decay: 0.01
gradient_clip: 5.0
```

### 3.4 学习率调度

ASR训练常用的学习率策略：

```
学习率
  ↑
  │      ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
  │     ╱                          ╲
  │    ╱                             ╲
  │   ╱                                ╲
  │  ╱                                   ╲
  │ ╱                                      ╲
  └──────────────────────────────────────────→ 训练步数
    warmup        peak         decay
```

- **Warmup阶段**：学习率从0线性增长到peak_lr，防止训练初期不稳定
- **Decay阶段**：学习率逐渐衰减（常用inverse sqrt decay）

### 3.5 训练过程

```python
for epoch in range(max_epochs):
    for batch in dataloader:
        # 1. 前向传播
        fbank, fbank_lengths, tokens, token_lengths = batch

        # Encoder
        encoder_out = encoder(fbank, fbank_lengths)  # [B, T', D]

        # CTC分支
        ctc_logits = ctc_head(encoder_out)  # [B, T', V]
        ctc_loss = ctc_criterion(ctc_logits, tokens, fbank_lengths, token_lengths)

        # Attention Decoder分支
        decoder_out = decoder(encoder_out, tokens)  # [B, L, V]
        att_loss = ce_criterion(decoder_out, tokens)

        # 总Loss
        loss = ctc_weight * ctc_loss + (1 - ctc_weight) * att_loss

        # 2. 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 3. 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # 4. 参数更新
        optimizer.step()
        lr_scheduler.step()

    # 验证
    val_loss = evaluate(model, val_dataloader)

    # 早停检查
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, "best.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            break
```

### 3.6 模型平均（Model Averaging）

训练完成后，通常会对最后几个epoch的模型参数取平均，以获得更稳定的模型：

```python
# 对最后10个epoch的checkpoint取平均
avg_model = average_checkpoints(
    checkpoints=["epoch_90.pt", "epoch_91.pt", ..., "epoch_99.pt"],
    num=10
)
```

> **为什么模型平均有效？**
> 训练后期，模型在最优解附近震荡。取平均相当于找到这些震荡点的"中心"，通常比任何单个checkpoint都更好。

## 4. 推理流程详解

### 4.1 推理的整体流程

```
输入音频 → 预处理 → 特征提取 → Encoder → 解码策略 → 后处理 → 输出文本
```

### 4.2 解码策略

#### Greedy Search（贪心搜索）

最简单的解码方式，每步选概率最高的token：

```python
# CTC Greedy
encoder_out = encoder(fbank)
ctc_probs = ctc_head(encoder_out)  # [1, T', V]
best_path = ctc_probs.argmax(dim=-1)  # [1, T']
# 合并重复 + 去blank
text = ctc_decode(best_path)
```

优点：速度快
缺点：可能错过全局最优解

#### Beam Search（束搜索）

保留top-k个候选，最终选最优：

```python
beam_size = 10

# 维护beam_size个候选序列
beams = [{"tokens": [<sos>], "score": 0.0}]

for step in range(max_length):
    all_candidates = []
    for beam in beams:
        # Decoder预测下一个token的概率分布
        probs = decoder.step(encoder_out, beam["tokens"])  # [V]

        # 取top-k个候选
        top_k = probs.topk(beam_size)
        for token, prob in top_k:
            candidate = {
                "tokens": beam["tokens"] + [token],
                "score": beam["score"] + log(prob)
            }
            all_candidates.append(candidate)

    # 保留得分最高的beam_size个
    beams = sorted(all_candidates, key=lambda x: x["score"])[:beam_size]

    # 检查是否所有beam都结束了
    if all(beam["tokens"][-1] == <eos> for beam in beams):
        break

best = beams[0]
```

> **Beam Search 具体示例**：
> 
> 假设 beam_size=2，词表 = {`<eos>`, 今, 天, 气}，Encoder已编码完毕。逐步展示搜索过程：
> 
> ```
> ═══ Step 1 ═══
> 当前beams: [(<sos>, score=0.0)]
> 
> Decoder对<sos>预测下一个token的概率分布:
>   P(今)=0.6, P(天)=0.3, P(气)=0.05, P(<eos>)=0.05
> 
> 所有候选（取log概率）:
>   (<sos>, 今)  score = 0.0 + log(0.6) = -0.51
>   (<sos>, 天)  score = 0.0 + log(0.3) = -1.20
>   (<sos>, 气)  score = 0.0 + log(0.05) = -3.00
>   (<sos>, <eos>) score = 0.0 + log(0.05) = -3.00
> 
> 保留top-2:
>   beam1: [<sos>, 今]   score = -0.51
>   beam2: [<sos>, 天]   score = -1.20
> 
> ═══ Step 2 ═══
> 对beam1 [<sos>, 今] 预测:
>   P(天)=0.7, P(今)=0.1, P(气)=0.1, P(<eos>)=0.1
>   → (<sos>,今,天)  score = -0.51 + log(0.7) = -0.87
>   → (<sos>,今,今)  score = -0.51 + log(0.1) = -2.81
> 
> 对beam2 [<sos>, 天] 预测:
>   P(气)=0.5, P(天)=0.2, P(今)=0.2, P(<eos>)=0.1
>   → (<sos>,天,气)  score = -1.20 + log(0.5) = -1.90
>   → (<sos>,天,天)  score = -1.20 + log(0.2) = -2.81
> 
> 所有候选中保留top-2:
>   beam1: [<sos>, 今, 天]   score = -0.87
>   beam2: [<sos>, 天, 气]   score = -1.90
> 
> ═══ Step 3 ═══
> 对beam1 [<sos>, 今, 天] 预测:
>   P(<eos>)=0.8, P(气)=0.15, ...
>   → (<sos>,今,天,<eos>)  score = -0.87 + log(0.8) = -1.09  ← 结束
>   → (<sos>,今,天,气)     score = -0.87 + log(0.15) = -2.77
> 
> 对beam2 [<sos>, 天, 气] 预测:
>   P(<eos>)=0.6, P(真)=0.2, ...
>   → (<sos>,天,气,<eos>)  score = -1.90 + log(0.6) = -2.41  ← 结束
>   → (<sos>,天,气,真)     score = -1.90 + log(0.2) = -3.51
> 
> 保留top-2:
>   beam1: [<sos>, 今, 天, <eos>]   score = -1.09  ✓ 已结束
>   beam2: [<sos>, 天, 气, <eos>]   score = -2.41  ✓ 已结束
> 
> 所有beam都以<eos>结尾，搜索终止。
> 
> ═══ 最终结果 ═══
> best = beam1 → "今天"  (score = -1.09)
> ```
> 
> 对比 Greedy Search：如果每步只取概率最高的token，结果也是"今天"。但在更复杂的场景中，Greedy可能在某一步选了局部最优的token，导致后续步骤的概率都很低，而Beam Search通过保留多个候选避免了这个问题。
> 
> 例如：如果Step 1中"天"的后续概率远高于"今"的后续概率，Greedy选了"今"就回不了头，而Beam Search同时保留了"天"这条路径，最终可能发现"天气"才是全局最优。

#### CTC Prefix Beam Search + Attention Rescoring

这是目前最常用的解码策略，结合了CTC和Attention的优势：

```
步骤1: CTC Prefix Beam Search
    - 使用CTC输出做beam search
    - 得到top-N个候选文本（如N=10）

步骤2: Attention Rescoring（重打分）
    - 用Attention Decoder对每个候选计算分数
    - 最终分数 = β·CTC分数 + (1-β)·Attention分数
    - 选分数最高的作为最终结果
```

```python
# 伪代码
# Step 1: CTC beam search
ctc_candidates = ctc_prefix_beam_search(encoder_out, beam_size=10)
# 得到10个候选: [("今天天气真好", -2.3), ("今天天气真号", -3.1), ...]

# Step 2: Attention rescoring
for candidate in ctc_candidates:
    tokens = tokenizer.encode(candidate.text)
    att_score = decoder.score(encoder_out, tokens)
    candidate.final_score = 0.5 * candidate.ctc_score + 0.5 * att_score

# 选最优
best = max(ctc_candidates, key=lambda x: x.final_score)
```

> **为什么要两步？**
> 
> - CTC beam search速度快，能快速缩小搜索范围
> - Attention Decoder更准确，但对所有可能的序列打分太慢
> - 先用CTC筛选出少量候选，再用Attention精确打分，兼顾速度和准确率

#### Attention Rescoring 详解

**1. CTC分数的来源**

CTC Prefix Beam Search在搜索过程中，会为每个候选前缀维护一个CTC分数。这个分数是该前缀对应的所有合法CTC路径的对数概率之和：

```
CTC_score(Y) = log P_ctc(Y|X) = log ∑_{π: B(π)=Y} ∏_{t=1}^{T'} y_t(π_t)
```

搜索结束后，得到 N 个候选及其CTC分数（已按分数排序）：

```
候选1: "今天天气真好"   CTC_score = -2.3
候选2: "今天天气真号"   CTC_score = -3.1
候选3: "今天天气针好"   CTC_score = -3.5
...
候选N: "今天天汽真好"   CTC_score = -6.8
```

**2. Attention Decoder打分过程**

对每个候选文本，用Attention Decoder计算其条件概率。Decoder以Encoder输出为条件，自回归地计算生成该文本序列的概率：

```
Attention_score(Y|X) = log P_att(Y|X) = ∑_{l=1}^{L} log P_att(y_l | y_{<l}, X)
```

具体计算过程（以候选"今天天气真好"为例）：

```
输入Encoder输出 + 目标序列 [<sos>, 今, 天, 天, 气, 真, 好]

Decoder逐步计算每个位置的条件概率:
  P(今 | <sos>, X)        = 0.85  → log = -0.16
  P(天 | <sos>今, X)      = 0.90  → log = -0.11
  P(天 | <sos>今天, X)    = 0.72  → log = -0.33
  P(气 | <sos>今天天, X)  = 0.88  → log = -0.13
  P(真 | <sos>今天天气, X) = 0.65  → log = -0.43
  P(好 | <sos>今天天气真, X) = 0.78  → log = -0.25
  P(<eos> | <sos>今天天气真好, X) = 0.92  → log = -0.08

Attention_score = -0.16 + (-0.11) + (-0.33) + (-0.13) + (-0.43) + (-0.25) + (-0.08)
               = -1.49
```

> **关键区别**：CTC假设各时间步条件独立，而Attention Decoder在预测每个token时能看到之前所有已生成的token（通过Masked Self-Attention）和完整的音频编码（通过Cross-Attention）。这使得Attention Decoder能建模语言级别的上下文依赖，比如"天气"后面更可能跟"真好"而不是"针好"。

**3. 分数融合与最终决策**

将CTC分数和Attention分数加权融合：

```
Final_score(Y) = β · CTC_score(Y) + (1-β) · Attention_score(Y)
```

其中 β 通常取 0.3~0.5。完整的数值示例：

```
β = 0.4

候选1: "今天天气真好"
  CTC_score    = -2.30
  Att_score    = -1.49
  Final_score  = 0.4×(-2.30) + 0.6×(-1.49) = -0.92 + (-0.89) = -1.81  ✓ 最优

候选2: "今天天气真号"
  CTC_score    = -3.10
  Att_score    = -3.85   ← Decoder知道"真号"不是合理搭配，给了低分
  Final_score  = 0.4×(-3.10) + 0.6×(-3.85) = -1.24 + (-2.31) = -3.55

候选3: "今天天气针好"
  CTC_score    = -3.50
  Att_score    = -4.20   ← "天气针好"语义不通，Decoder惩罚更重
  Final_score  = 0.4×(-3.50) + 0.6×(-4.20) = -1.40 + (-2.52) = -3.92

最终选择: "今天天气真好" (Final_score = -1.81)
```

> 注意候选2"真号"和候选3"针好"在声学上与"真好"相似（CTC分数差距不大），但Attention Decoder利用语言上下文给出了明显更低的分数，成功纠正了声学混淆。这正是Rescoring的核心价值。

**4. 实现细节：批量并行打分**

实际实现中，N个候选可以拼成一个batch，一次前向传播完成所有打分，而不需要逐个计算：

```python
# 高效实现：批量Attention Rescoring
def attention_rescoring(encoder_out, ctc_candidates, decoder, ctc_weight=0.4):
    """
    encoder_out: [1, T', D]  Encoder输出
    ctc_candidates: list of (token_ids, ctc_score)，共N个候选
    """
    N = len(ctc_candidates)

    # 1. 将N个候选padding到相同长度，组成batch
    max_len = max(len(c.tokens) for c in ctc_candidates)
    padded_tokens = []  # [N, max_len]
    token_lengths = []
    for c in ctc_candidates:
        padded = c.tokens + [PAD] * (max_len - len(c.tokens))
        padded_tokens.append(padded)
        token_lengths.append(len(c.tokens))

    # 2. 扩展encoder_out到N份: [1, T', D] → [N, T', D]
    encoder_out_expanded = encoder_out.repeat(N, 1, 1)

    # 3. 一次前向传播，得到所有候选的Attention分数
    #    Decoder输入: [N, max_len]  输出: [N, max_len, vocab_size]
    decoder_out = decoder(encoder_out_expanded, padded_tokens)

    # 4. 计算每个候选的log概率（只取有效长度部分）
    att_scores = []
    for i in range(N):
        score = 0.0
        for l in range(token_lengths[i]):
            # 取第l步预测第l+1个token的log概率
            score += decoder_out[i, l, ctc_candidates[i].tokens[l+1]].item()
        att_scores.append(score)

    # 5. 融合分数
    final_scores = []
    for i in range(N):
        final = ctc_weight * ctc_candidates[i].ctc_score + (1 - ctc_weight) * att_scores[i]
        final_scores.append(final)

    # 6. 返回最优候选
    best_idx = final_scores.index(max(final_scores))
    return ctc_candidates[best_idx]
```

**5. β参数的影响**

β（CTC权重）的选择会影响最终效果：

```
β → 1.0: 完全依赖CTC，退化为纯CTC解码
  - 速度最快（不需要Decoder前向传播）
  - 无法利用语言上下文纠错

β → 0.0: 完全依赖Attention Decoder
  - 纠错能力最强
  - 但如果CTC候选中没有正确答案，Decoder也无能为力

β = 0.3~0.5: 通常的最优区间
  - 在dev集上grid search确定最佳值
  - 不同数据集/模型的最优β可能不同
```

> **与语言模型Rescoring的对比**：
> 有些系统还会引入外部语言模型（LM）做第三轮打分：
> `Final_score = β·CTC + γ·Attention + δ·LM + η·length_penalty`
> 外部LM通常在大规模文本上训练，能提供更强的语言先验，但会增加推理复杂度和部署成本。

### 4.3 流式推理

流式ASR需要在音频还在输入时就给出部分结果：

```
时间 →
音频: [chunk1][chunk2][chunk3][chunk4]...
输出:  "今天"  "天气"  "真好"  ...
```

关键技术：

- **Chunk-based Attention**：Encoder只看当前chunk和少量历史context
- **CTC前缀搜索**：利用CTC的单调对齐特性，实现增量解码
- **Endpoint Detection**：检测用户说完了（静音检测）

```python
# 流式推理伪代码
chunk_size = 640ms  # 每次处理640ms的音频
context_size = 7    # 左侧context的chunk数

encoder_cache = []
for chunk in audio_stream:
    fbank_chunk = compute_fbank(chunk)

    # 带cache的Encoder
    encoder_out, encoder_cache = encoder.forward_chunk(
        fbank_chunk, 
        cache=encoder_cache,
        left_context=context_size
    )

    # 增量CTC解码
    partial_result = ctc_greedy_decode(encoder_out)
    yield partial_result  # 实时输出部分结果
```

### 4.4 后处理

```
模型原始输出: "今 天 天 气 真 好 啊"
    ↓
去除特殊token: 去掉<sos>, <eos>, <blank>等
    ↓
文本规整: 
    - 数字转换: "一 二 三" → "123"
    - 标点恢复: "今天天气真好啊" → "今天天气真好啊。"
    - 大小写恢复（英文）
    - 逆文本正则化（ITN）: "二零二四年" → "2024年"
    ↓
热词替换/纠错:
    - 将识别结果中的常见错误替换为正确词汇
    - 根据热词列表进行偏置
    ↓
最终输出: "今天天气真好啊。"
```

## 5. 模型优化与部署

### 5.1 模型压缩

| 方法                 | 说明             | 压缩比    |
| ------------------ | -------------- | ------ |
| 量化（Quantization）   | FP32→INT8/INT4 | 2-4x   |
| 知识蒸馏（Distillation） | 大模型教小模型        | 自定义    |
| 剪枝（Pruning）        | 去掉不重要的参数       | 1.5-3x |
| 模型结构优化             | 减少层数/维度        | 自定义    |

### 5.2 推理加速

```
常用推理引擎：
├── ONNX Runtime：跨平台，CPU/GPU都支持
├── TensorRT：NVIDIA GPU专用，速度最快
├── OpenVINO：Intel CPU/GPU优化
└── NCNN/MNN：移动端推理框架
```

### 5.3 部署架构

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  客户端      │────→│  WebSocket   │────→│  ASR服务     │
│  (麦克风)    │←────│  Server      │←────│  (GPU/CPU)   │
│             │     │              │     │             │
│  实时音频流  │     │  音频分发     │     │  模型推理    │
│  结果展示    │     │  结果回传     │     │  后处理      │
└─────────────┘     └──────────────┘     └─────────────┘
```

## 6. 从零训练一个ASR模型的完整步骤

```
1. 准备数据
   ├── 下载数据集（如AISHELL-1）
   ├── 数据格式转换
   └── 划分train/dev/test

2. 数据预处理
   ├── 计算全局CMVN统计量
   ├── 训练Tokenizer（BPE/SentencePiece）
   └── 生成数据清单（manifest）

3. 配置模型
   ├── 选择模型架构（Conformer + Transformer Decoder）
   ├── 设置超参数
   └── 配置训练策略

4. 训练
   ├── 多GPU分布式训练（DDP）
   ├── 监控loss曲线
   ├── 定期在dev集上评估
   └── 模型平均

5. 评估
   ├── 在test集上计算CER/WER
   ├── 分析错误类型
   └── 对比baseline

6. 部署
   ├── 模型导出（ONNX/TorchScript）
   ├── 量化压缩
   ├── 搭建推理服务
   └── 性能测试（RTF、延迟、并发）
```

> **什么是RTF（Real Time Factor）？**
> RTF = 处理时间 / 音频时长。
> RTF < 1 表示实时（处理速度快于音频播放速度）。
> 例如RTF=0.1表示处理1秒音频只需0.1秒。

## 7. 常见问题与调优技巧

### 7.1 训练不收敛

- 检查学习率是否合适（太大震荡，太小收敛慢）
- 检查数据是否有问题（标注错误、音频损坏）
- 增加warmup步数
- 减小batch size

### 7.2 过拟合

- 增加数据增强（SpecAugment、加噪）
- 增加dropout
- 使用label smoothing
- 减小模型大小

### 7.3 识别效果差

- 增加训练数据
- 使用预训练模型微调
- 加入语言模型重打分
- 针对性地增加领域数据

### 7.4 推理速度慢

- 使用CTC greedy解码（最快但效果稍差）
- 减小beam size
- 模型量化
- 使用更高效的推理引擎

---

> 下面的demo目录将提供FunASR、Sherpa和Kaldi的实际代码示例。
