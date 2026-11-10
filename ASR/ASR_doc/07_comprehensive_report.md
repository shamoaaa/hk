# ASR 综合报告：从数据到部署的完整流程

## 1. 数据准备

### 1.1 数据格式标准

中文 ASR 领域的音频数据遵循统一标准：

| 属性 | 标准值 |
|------|--------|
| 格式 | WAV (PCM) |
| 采样率 | 16,000 Hz |
| 位深 | 16 bit |
| 声道 | 单声道 (Mono) |
| 编码 | 线性 PCM (Little-Endian) |

### 1.2 主要开源数据集

| 数据集 | 规模 | 说话人 | 许可证 | 适用场景 |
|--------|------|--------|--------|----------|
| THCHS-30 | ~30h | ~40 | Apache 2.0 | 入门学习 |
| AISHELL-1 | ~178h | 400 | Apache 2.0 | 标准基准 |
| aidatatang_200zh | ~200h | 600 | CC BY-NC-ND 4.0 | 中等规模训练 |
| ST-CMDS | ~110h | 855 | CC BY-NC-ND 4.0 | 手机录音场景 |
| MAGICDATA | ~755h | 1080 | CC BY-NC-ND 4.0 | 大规模训练 |
| TAL_ASR | ~587h | - | 开源 | 中英混合 |

### 1.3 数据组织格式

FunASR 使用 jsonl 格式：

```jsonl
{"key": "BAC009S0002W0122", "source": "/data/wav/BAC009S0002W0122.wav", "target": "甚至出现交易"}
```

Kaldi 风格（被广泛沿用）：

```
# wav.scp
BAC009S0002W0122 /data/wav/BAC009S0002W0122.wav

# text
BAC009S0002W0122 甚 至 出 现 交 易
```

### 1.4 数据预处理流程

```
原始音频 → 格式统一(16kHz/16bit/Mono) → 音量归一化 → VAD切分 → 质量筛选 → 数据增强
```

关键数据增强方法：

| 方法 | 作用 | 实现 |
|------|------|------|
| Speed Perturbation | 模拟不同语速 | 0.9x/1.0x/1.1x，等效3倍数据 |
| SpecAugment | 防止过拟合 | 随机遮挡频谱图的时间/频率区域 |
| 加噪 | 提升噪声鲁棒性 | 混入白噪声、环境噪声 |
| 混响 | 模拟真实环境 | 卷积房间冲激响应 |

### 1.5 文本归一化

训练前必须统一文本格式：

```
TN (训练标签统一): "100元" → "一百元"
ITN (推理后处理): "一百元" → "100元"
```

去标点、统一大小写、全角转半角、繁简统一等操作确保标签一致性。


## 2. 特征提取

### 2.1 FBank 特征（现代端到端模型标准输入）

```
原始音频 → 预加重(0.97) → 分帧(25ms帧长,10ms帧移) → 汉明窗 → FFT(512点)
         → 功率谱 → Mel滤波器组(80个三角滤波器) → 取对数 → 80维FBank
```

关键参数：

| 参数 | 值 | 说明 |
|------|------|------|
| num_mel_bins | 80 | 特征维度 |
| frame_length | 25ms | 每帧400个采样点 |
| frame_shift | 10ms | 每帧160个采样点，帧率100fps |
| sample_rate | 16000 | 采样率 |
| nfft | 512 | FFT点数 |

### 2.2 CMVN 归一化

```
x_norm = (x - global_mean) / sqrt(global_var)
```

消除不同录音条件（麦克风、环境）带来的特征分布差异。

### 2.3 LFR（Low Frame Rate，可选）

Paraformer 使用 LFR 拼帧替代 Conv2D 下采样：

```
每7帧拼接为1帧，步长6帧
维度变化: [T, 80] → [T/6, 560]  (7×80=560)
```

## 3. 模型架构

### 3.1 端到端 ASR 架构演进

```
CTC (2006)           ← 最早的端到端方法，条件独立假设
    ↓
AED/Seq2Seq (2015)   ← 注意力机制，自回归解码
    ↓
CTC/Attention混合     ← 取长补短，目前主流
    ↓
Transducer (RNN-T)   ← 天然支持流式
    ↓
NAR (Paraformer)     ← 非自回归，并行解码，速度最快
```

### 3.2 主流 Encoder 架构对比

| 架构 | 核心组件 | 特点 | 代表框架 |
|------|----------|------|----------|
| Transformer | Self-Attention + FFN | 全局建模 | ESPnet |
| Conformer | Self-Attention + Conv + FFN | 全局+局部 | WeNet |
| SAN-M | Self-Attention + FSMN + FFN | 全局+局部记忆 | FunASR (Paraformer) |

SAN-M 与 Conformer 的区别：
- Conformer 用 Depthwise Conv 捕获局部模式
- SAN-M 用 FSMN Memory Block 捕获局部模式
- FSMN 天然支持因果模式，更适合流式；且在达摩院大规模实验中效果更优

### 3.3 Paraformer 架构（FunASR 旗舰模型）

```
┌─────────────────────────────────────────────────────────────┐
│  输入: FBank [T, 80] → LFR [T', 560]                        │
│      ↓                                                       │
│  Encoder: SAN-M 50层 → [T', 512]                            │
│      ↓                    ↓                                  │
│  CIF Predictor         CTC辅助分支                           │
│  (预测长度L', 生成声学嵌入)                                    │
│      ↓                                                       │
│  Sampler (GLM策略，训练时混合真实token和声学嵌入)              │
│      ↓                                                       │
│  Parallel Decoder: SAN-M 16层 (无causal mask, NAR)           │
│      ↓                                                       │
│  Output Layer → [L', 8404] → argmax → 文本                  │
└─────────────────────────────────────────────────────────────┘
```

核心创新：
- **CIF**: 连续积分触发机制，预测输出长度并生成声学嵌入
- **GLM**: 渐进式训练策略，解决 NAR 训练-推理不一致问题
- **NAR 解码**: 一次前向传播输出所有 token，速度比 AR 快 10 倍以上

### 3.4 SeACo-Paraformer（热词定制扩展）

在 Paraformer 基础上增加三个模块：

```
Bias Encoder (LSTM 2层): 将热词编码为固定向量
Bias Decoder (Transformer 4层): 检测热词位置，生成偏置概率
概率融合: P_merged = λ·P_bias + (1-λ)·P_ASR (仅在热词位置激活)
```

训练解耦：ASR backbone 冻结，仅训练热词模块，互不影响。

### 3.5 完整流水线模型

```
FSMN-VAD (语音检测) → Paraformer (语音识别) → CT-Punc (标点恢复)
```

| 模块 | 作用 | 架构 |
|------|------|------|
| FSMN-VAD | 检测语音段 | 4层FSMN，输出每帧语音概率 |
| Paraformer | 语音转文字 | SAN-M Encoder + NAR Decoder |
| CT-Punc | 添加标点 | Transformer Encoder (类BERT) |

## 4. 训练流程

### 4.1 损失函数

**Paraformer 训练损失：**

```
L_total = α·L_CTC + (1-α)·L_CE + β·L_MAE

L_CTC: CTC Loss (辅助对齐，α通常=0或0.3)
L_CE:  Cross-Entropy Loss (主损失，带label smoothing=0.1)
L_MAE: MAE Loss (CIF长度预测，β=1.0)
```

**SeACo 模块训练损失：**

```
L_bias = CrossEntropy(P_bias, y_bias)

y_bias: 热词位置标记为对应token，非热词位置标记为<no-bias>
```

### 4.2 训练策略

| 策略 | 说明 |
|------|------|
| Warmup + Decay | 学习率先线性增长再衰减 |
| Dynamic Batching | 按总帧数组batch，减少padding浪费 |
| Label Smoothing | 防止模型过度自信 (0.1) |
| Gradient Clipping | 防止梯度爆炸 (max_norm=5.0) |
| Model Averaging | 最后N个epoch参数平均 |
| SpecAugment | 频谱图随机遮挡 |

### 4.3 两阶段训练（SeACo-Paraformer）

```
阶段一: 训练 ASR Backbone
  数据: 50000~60000h 标注语音
  模块: Encoder + CIF + Decoder + CTC (全部可训练)
  Epoch: ~100

阶段二: 训练热词模块
  数据: 同样数据 (或专门热词数据)
  模块: Bias Encoder + Bias Decoder (仅这些可训练)
  冻结: Encoder + CIF + Decoder (全部冻结)
  Epoch: 6
```

### 4.4 关键超参数

```yaml
# Paraformer-large 配置
encoder: SANMEncoder (50层, d=512, heads=4, ffn=2048, kernel=11)
decoder: ParaformerSANMDecoder (16层, d=512, heads=4, ffn=2048)
predictor: CifPredictorV2 (threshold=1.0)
vocab_size: 8404
optimizer: Adam (peak_lr=0.001, warmup=25000 steps)
```

## 5. 评估指标

### 5.1 核心指标

| 指标 | 公式 | 适用 |
|------|------|------|
| CER | (S+D+I)/N × 100% | 中文（以字为单位） |
| WER | (S+D+I)/N × 100% | 英文（以词为单位） |
| SER | 错误句数/总句数 × 100% | 句级评估 |
| RTF | 处理时间/音频时长 | 速度评估 |

其中 S=替换错误, D=删除错误, I=插入错误, N=参考文本总字/词数。

### 5.2 基准性能（AISHELL-1 测试集）

| 模型 | CER(%) | RTF(CPU) | 解码方式 |
|------|--------|----------|----------|
| Paraformer-large (离线) | ~1.95 | 0.05 | NAR 并行 |
| Paraformer-zh-streaming | ~5.0 | 0.08 | NAR 流式 |
| Conformer (WeNet) | ~4.5 | 0.15 | AR Beam Search |
| Whisper-large-v3 | ~3.0 | 0.80 | AR 自回归 |

### 5.3 评估注意事项

1. **文本归一化一致性**: ref 和 hyp 必须用相同规则归一化
2. **分词标准统一**: 中文推荐用 CER 避免分词差异
3. **测试集泄露**: 确保训练数据不包含测试集内容
4. **多维度评估**: CER 低不代表用户体验好，需结合 SER 和主观评估

## 6. 调优策略

### 6.1 训练不收敛

| 问题 | 解决方案 |
|------|----------|
| Loss 震荡 | 降低学习率，增加 warmup 步数 |
| Loss 不下降 | 检查数据标注质量，检查特征提取是否正确 |
| NaN/Inf | 增加 gradient clipping，检查数据中是否有异常值 |

### 6.2 过拟合

| 问题 | 解决方案 |
|------|----------|
| 训练 loss 低但验证 loss 高 | 增加 SpecAugment 强度 |
| 小数据集过拟合 | 增加 dropout (0.1→0.2)，使用预训练模型微调 |
| 特定词汇过拟合 | 增加 label smoothing |

### 6.3 识别效果差

| 场景 | 解决方案 |
|------|----------|
| 通用效果差 | 增加训练数据量，使用更大模型 |
| 领域词汇错误 | 使用 SeACo 热词模块，或领域数据微调 |
| 噪声环境差 | 加噪数据增强，使用更强的 VAD |
| 长音频效果差 | 确保 VAD 切分合理，避免过长段 |

### 6.4 推理速度优化

| 方法 | 加速比 | 代价 |
|------|--------|------|
| NAR 解码 (Paraformer) | 10x vs AR | 需要专门模型 |
| ONNX 导出 | 1.5-3x | 需要导出步骤 |
| INT8 量化 | 2-4x | 精度略降 |
| 减小 beam size | 线性 | 效果略降 |
| GPU batch 推理 | 线性 | 需要 GPU |

## 7. 部署方案

### 7.1 部署框架选择

| 框架 | 适用场景 | 特点 |
|------|----------|------|
| FunASR 服务端 | 生产环境 | Docker 一键部署，WebSocket |
| Sherpa-onnx | 跨平台/端侧 | ONNX Runtime，支持移动端 |
| ONNX Runtime | 通用 | 跨平台，多硬件支持 |
| TensorRT | NVIDIA GPU | 速度最快 |

### 7.2 生产环境架构

```
客户端 → 负载均衡(Nginx) → FunASR服务集群(GPU) → 模型存储(共享文件系统)
  ↑                                    ↓
  └──────── WebSocket 实时返回结果 ←────┘
```


## 8. 完整示例：Paraformer 从输入到输出的每个阶段

以识别一段 3 秒中文语音 **"今天去达摩院开会"**（8个字）为例，使用 SeACo-Paraformer-large 模型，热词列表为 `["达摩院"]`，逐阶段追踪数据的精确输入和输出。

---

### 阶段 1：音频加载

```
输入: "meeting.wav" 文件 (3秒, 16kHz, 16bit, 单声道)
处理: 读取 WAV 文件，转为浮点数组
输出: waveform = [48000] (float32, 范围 [-1.0, 1.0])

具体数值示例 (前10个采样点):
  [0.0012, -0.0034, 0.0056, 0.0023, -0.0089, 0.0145, -0.0067, 0.0098, 0.0034, -0.0012]
```

---

### 阶段 2：FBank 特征提取

```
输入: waveform [48000]
处理:
  (1) 预加重: y[n] = x[n] - 0.97·x[n-1]
  (2) 分帧: 帧长25ms=400点, 帧移10ms=160点
      帧数 = (48000 - 400) / 160 + 1 = 298
  (3) 加窗: 每帧 × 汉明窗
  (4) FFT: 512点FFT → 取前257点(对称性)
  (5) 功率谱: |X(k)|² / 512 → [298, 257]
  (6) Mel滤波: 80个三角滤波器 → [298, 80]
  (7) 取对数: log(energy + 1e-10)
输出: fbank = [298, 80] (float32)

每帧含义: 第i帧 = 音频第 i×10ms ~ i×10ms+25ms 区间的80维频谱能量
  fbank[0] = 0~25ms 的频谱
  fbank[1] = 10~35ms 的频谱
  ...
  fbank[297] = 2970~2995ms 的频谱
```

---

### 阶段 3：LFR 拼帧 + CMVN

```
输入: fbank [298, 80]
处理:
  (1) LFR拼帧: 每7帧拼接为1帧, 步长6帧
      新帧数 = (298 - 7) / 6 + 1 ≈ 49
      新维度 = 7 × 80 = 560
  (2) CMVN: x_norm = (x - mean) / sqrt(var)
      使用预计算的全局统计量 (mean: [560], var: [560])
输出: features = [49, 560] (float32, 均值≈0, 方差≈1)

加batch维度后: [1, 49, 560]
```

---

### 阶段 4：Encoder (SAN-M, 50层)

```
输入: [1, 49, 560]
处理:
  (1) 输入投影: Linear(560, 512) → [1, 49, 512]
  (2) 正弦位置编码: x = x + PE → [1, 49, 512]
  (3) SAN-M Block × 50:
      每层:
        - Multi-Head Self-Attention (4头, 每头128维)
          Q=K=V=x → Attention(QK^T/√128)·V → [1, 49, 512]
        - FSMN Memory (kernel=11, 捕获±5帧局部模式)
        - Residual + LayerNorm
        - FFN: Linear(512,2048) → ReLU → Linear(2048,512)
        - Residual + LayerNorm
输出: encoder_out = [1, 49, 512]

含义: 49个帧位置, 每个位置是512维的上下文感知声学表示
  encoder_out[0, 0, :] = 第1帧的编码 (融合了全局信息)
  encoder_out[0, 48, :] = 第49帧的编码
```

---

### 阶段 5：CIF Predictor

```
输入: encoder_out [1, 49, 512]
处理:
  (1) Conv1D(512, 512, kernel=3, padding=1) → ReLU → Dropout
      → [1, 49, 512]
  (2) Linear(512, 1) → Sigmoid
      → α权重 [1, 49, 1], 每个值∈(0,1)
  (3) 累积触发:
      逐帧累积α值, 当累积≥1.0时触发, 生成一个声学嵌入
      触发次数 = 预测输出长度 L'
  (4) 加权求和:
      每次触发时, 对触发窗口内的encoder帧做加权求和

输出:
  acoustic_embedding = [1, 8, 512]  (8个声学嵌入, 对应8个字)
  predicted_length = 8

α权重示例 (49个值):
  [0.15, 0.12, 0.18, 0.20, 0.35,  ← 累积=1.00, 触发→"今"
   0.14, 0.16, 0.22, 0.28, 0.25,  ← 累积=1.05, 触发→"天"
   0.30, 0.25, 0.20, 0.30,        ← 累积=1.05, 触发→"去"
   0.18, 0.22, 0.28, 0.35,        ← 累积=1.03, 触发→"达"
   0.20, 0.25, 0.30, 0.28,        ← 累积=1.03, 触发→"摩"
   0.22, 0.30, 0.25, 0.25,        ← 累积=1.02, 触发→"院"
   0.18, 0.20, 0.28, 0.35,        ← 累积=1.01, 触发→"开"
   0.20, 0.25, 0.30, 0.28, ...]   ← 累积=1.03, 触发→"会"
```

---

### 阶段 6：Sampler (推理时跳过)

```
输入: acoustic_embedding [1, 8, 512]
处理:
  训练时: 按GLM策略混合声学嵌入和真实token embedding
  推理时: 直接使用声学嵌入 (无真实标签可用)
输出: semantic_embedding = [1, 8, 512] (推理时等于acoustic_embedding)
```

---

### 阶段 7：Parallel Decoder (SAN-M, 16层)

```
输入:
  query: semantic_embedding [1, 8, 512]
  memory: encoder_out [1, 49, 512]
处理:
  SAN-M Decoder Block × 16:
    每层:
      - Self-Attention (无causal mask, 8个位置互相可见)
        Q=K=V=decoder_state → [1, 8, 512]
        + FSMN Memory
      - Cross-Attention
        Q=decoder_state [1, 8, 512]
        K=V=encoder_out [1, 49, 512]
        → 每个输出位置从49帧编码中提取相关信息
        → [1, 8, 512]
      - FFN + Residual + LayerNorm
输出: decoder_hidden = [1, 8, 512]

含义: 8个位置的解码表示, 融合了声学信息和位置间的相互关系
```

---

### 阶段 8：ASR Output Layer

```
输入: decoder_hidden [1, 8, 512]
处理: Linear(512, 8404) → Softmax
输出: P_ASR = [1, 8, 8404]

含义: 每个位置对8404个token的概率分布
  P_ASR[0, 0, :] = 位置1的概率分布 → argmax = "今" (id=156, prob=0.95)
  P_ASR[0, 1, :] = 位置2的概率分布 → argmax = "天" (id=892, prob=0.93)
  P_ASR[0, 2, :] = 位置3的概率分布 → argmax = "去" (id=341, prob=0.88)
  P_ASR[0, 3, :] = 位置4的概率分布 → argmax = "达" (id=1024, prob=0.40)  ← 不确定!
  P_ASR[0, 4, :] = 位置5的概率分布 → argmax = "摩" (id=2048, prob=0.85)
  P_ASR[0, 5, :] = 位置6的概率分布 → argmax = "院" (id=3072, prob=0.82)
  P_ASR[0, 6, :] = 位置7的概率分布 → argmax = "开" (id=567, prob=0.91)
  P_ASR[0, 7, :] = 位置8的概率分布 → argmax = "会" (id=234, prob=0.89)

注意位置4: "达"的概率只有0.40, "大"的概率是0.45 → 基础ASR可能输出"大"(错误)
```

---

### 阶段 9：Bias Encoder (热词编码)

```
输入: 热词列表 ["达摩院"]
处理:
  (1) Tokenize: "达摩院" → [达_id=1024, 摩_id=2048, 院_id=3072]
  (2) Embedding (共享ASR的embedding层):
      [1024, 2048, 3072] → lookup → [3, 512]
  (3) LSTM (2层, hidden=512):
      输入序列 [3, 512] → 取最后时刻隐状态 → [1, 512]
  (4) 加上默认<blank>热词: → [2, 512]
  (5) 扩展batch维度: → [1, 2, 512]
输出: hotword_embedding = [1, 2, 512]  (2个热词: "达摩院" + <blank>)
```

---

### 阶段 10：Bias Decoder (热词解码, 4层)

```
输入:
  D = decoder_hidden [1, 8, 512]  (来自阶段7)
  E = acoustic_embedding [1, 8, 512]  (来自阶段5)
  Z = hotword_embedding [1, 2, 512]  (来自阶段9)
处理:
  Transformer Layer × 4:
    (1) Cross-Attention 1: D attend to Z
        Q=D [1,8,512], K=V=Z [1,2,512]
        → D⁺ [1, 8, 512]
        位置4,5,6 对热词"达摩院"产生高注意力
    (2) Cross-Attention 2: E attend to Z
        Q=E [1,8,512], K=V=Z [1,2,512]
        → E⁺ [1, 8, 512]
    (3) FFN + LayerNorm
  Bias Output: Linear(512, 8405) → Softmax
    (8405 = 8404 + 1个<no-bias>标记)
输出: P_bias = [1, 8, 8405]

P_bias 各位置的 argmax:
  位置1: <no-bias> (id=8404, prob=0.95)  → 不是热词
  位置2: <no-bias> (id=8404, prob=0.93)  → 不是热词
  位置3: <no-bias> (id=8404, prob=0.90)  → 不是热词
  位置4: "达" (id=1024, prob=0.92)       → 检测到热词开始!
  位置5: "摩" (id=2048, prob=0.94)       → 热词中间
  位置6: "院" (id=3072, prob=0.91)       → 热词结束
  位置7: <no-bias> (id=8404, prob=0.88)  → 不是热词
  位置8: <no-bias> (id=8404, prob=0.92)  → 不是热词
```

---

### 阶段 11：概率融合

```
输入:
  P_ASR [1, 8, 8404]  (基础ASR概率)
  P_bias [1, 8, 8405] (热词偏置概率)
  λ = 1.0 (偏置强度)
处理:
  对每个位置 i:
    if argmax(P_bias[i]) == <no-bias>:
        P_merged[i] = P_ASR[i]
    else:
        P_merged[i] = λ·P_bias[i][:8404] + (1-λ)·P_ASR[i]

逐位置融合结果:
  位置1: <no-bias> → P_merged = P_ASR → argmax = "今" ✓
  位置2: <no-bias> → P_merged = P_ASR → argmax = "天" ✓
  位置3: <no-bias> → P_merged = P_ASR → argmax = "去" ✓
  位置4: "达"激活  → P_merged = P_bias → argmax = "达" ✓ (纠正了"大"→"达")
  位置5: "摩"激活  → P_merged = P_bias → argmax = "摩" ✓
  位置6: "院"激活  → P_merged = P_bias → argmax = "院" ✓
  位置7: <no-bias> → P_merged = P_ASR → argmax = "开" ✓
  位置8: <no-bias> → P_merged = P_ASR → argmax = "会" ✓

输出: P_merged = [1, 8, 8404]
```

---

### 阶段 12：Token 解码 → 最终文本

```
输入: P_merged [1, 8, 8404]
处理:
  (1) argmax: [1024, 892, 341, 1024, 2048, 3072, 567, 234]
  (2) Token ID → 文字查表:
      1024→"达", 892→"天", 341→"去", 1024→"达",
      2048→"摩", 3072→"院", 567→"开", 234→"会"
  (3) 拼接: "今天去达摩院开会"
输出: "今天去达摩院开会"
```

---

### 完整数据流维度变化总结

```
阶段                          输入形状              输出形状              耗时(估)
─────────────────────────────────────────────────────────────────────────────────
1. 音频加载                   WAV文件              [48000]              <1ms
2. FBank特征提取              [48000]              [298, 80]            ~5ms
3. LFR + CMVN                [298, 80]            [1, 49, 560]         <1ms
4. Encoder (50层SAN-M)       [1, 49, 560]         [1, 49, 512]         ~80ms
5. CIF Predictor             [1, 49, 512]         [1, 8, 512] + L'=8   ~2ms
6. Sampler (推理跳过)         [1, 8, 512]          [1, 8, 512]          0ms
7. Decoder (16层SAN-M)       [1,8,512]+[1,49,512] [1, 8, 512]          ~30ms
8. ASR Output Layer          [1, 8, 512]          [1, 8, 8404]         <1ms
9. Bias Encoder              ["达摩院"]            [1, 2, 512]          ~2ms
10. Bias Decoder (4层)       D+E+Z                [1, 8, 8405]         ~10ms
11. 概率融合                  P_ASR + P_bias       [1, 8, 8404]         <1ms
12. Token解码                 [1, 8, 8404]         "今天去达摩院开会"    <1ms
─────────────────────────────────────────────────────────────────────────────────
总计                          3秒音频               8个字                ~130ms
RTF ≈ 0.13/3.0 ≈ 0.043 (远小于1, 满足实时要求)
```

---

### 关键观察

1. **Encoder 是计算瓶颈**: 50层 SAN-M 占总推理时间的 ~60%
2. **NAR 的速度优势**: Decoder 只做 1 次前向传播输出 8 个字；AR 模型需要 8 次
3. **热词模块开销小**: Bias Encoder + Decoder 仅增加 ~12ms (~9% 额外开销)
4. **CIF 的精确性**: 准确预测了输出长度 L'=8，与真实文本长度一致
5. **热词纠错**: 位置4 的 "达" 在基础 ASR 中概率不是最高的，热词模块成功纠正

## 9. 总结

```
完整 ASR 系统构建路径:

数据准备 → 特征提取 → 模型训练 → 评估调优 → 部署上线
  │           │          │          │          │
  ├─格式统一   ├─FBank    ├─Encoder  ├─CER/WER  ├─ONNX导出
  ├─标注质量   ├─CMVN     ├─CIF      ├─错误分析  ├─量化压缩
  ├─数据增强   ├─LFR      ├─Decoder  ├─热词调优  ├─服务部署
  └─文本归一化  └─batch化   ├─SeACo    └─A/B测试  └─监控运维
                           └─Loss优化
```

核心选型建议:
- **中文离线识别**: Paraformer-large (CER最低, 速度最快)
- **需要热词定制**: SeACo-Paraformer (显式热词, 不伤基础性能)
- **流式实时识别**: Paraformer-zh-streaming (低延迟)
- **多语言场景**: SenseVoice 或 Whisper
- **端侧部署**: Sherpa-onnx + 量化模型