# SeACo-Paraformer 模型架构详解与 FunASR 训练数据流转

## 1. 模型概述

### 1.1 模型全称解析

`speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`

这个模型名称包含了丰富的信息：

| 字段 | 含义 |
|------|------|
| speech | 语音任务 |
| seaco | Semantic-Augmented Contextual（语义增强上下文） |
| paraformer | Parallel Transformer（并行Transformer） |
| large | 大模型版本 |
| asr | 自动语音识别 |
| nat | Non-Autoregressive Transformer（非自回归） |
| zh-cn | 中文普通话 |
| 16k | 16kHz 采样率 |
| common | 通用场景 |
| vocab8404 | 词表大小 8404 |
| pytorch | PyTorch 框架实现 |

### 1.2 模型定位

SeACo-Paraformer 是阿里达摩院在 Paraformer-large 基础上扩展的热词定制化模型。它在保持 Paraformer 高精度、高效率的非自回归解码能力的同时，增加了灵活且高效的热词（Hotword）定制能力。

核心优势：
- AED 模型的高精度
- NAR 模型的高效率（比自回归快 10 倍以上）
- 显式的热词定制能力（可控、可解释）
- 训练解耦：热词模块独立训练，不影响基础 ASR 性能


## 2. 完整架构详解

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SeACo-Paraformer 完整架构                                  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        输入处理层                                       │ │
│  │  WAV 音频 (16kHz, 单声道)                                              │ │
│  │      ↓                                                                 │ │
│  │  FBank 特征提取 (80维, 帧长25ms, 帧移10ms)                              │ │
│  │      ↓                                                                 │ │
│  │  CMVN 全局归一化                                                        │ │
│  │      ↓                                                                 │ │
│  │  特征: [batch, T, 560]  (经过 LFR 拼帧后，7帧×80维=560)                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│      ↓                                                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                   Encoder: SAN-M (50层)                                 │ │
│  │                                                                         │ │
│  │  Linear(560, 512) + Positional Encoding                                │ │
│  │      ↓                                                                  │ │
│  │  ┌──────────────────────────────────────┐                              │ │
│  │  │   SAN-M Block × 50                   │                              │ │
│  │  │   ┌────────────────────────────────┐ │                              │ │
│  │  │   │ Self-Attention with Memory     │ │  ← SAN-M: 在标准Self-Attention│ │
│  │  │   │ (Multi-Head Attention + FSMN)  │ │    基础上加入FSMN记忆模块      │ │
│  │  │   │ Feed-Forward Network           │ │                              │ │
│  │  │   │ LayerNorm                      │ │                              │ │
│  │  │   └────────────────────────────────┘ │                              │ │
│  │  └──────────────────────────────────────┘                              │ │
│  │      ↓                                                                  │ │
│  │  Encoder 输出 e₁:T: [batch, T', 512]                                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│      ↓                              ↓                                        │
│  ┌──────────────────┐   ┌──────────────────────────────────────────────┐   │
│  │   CTC 辅助分支    │   │         CIF Predictor (CifPredictorV2)       │   │
│  │                    │   │                                              │   │
│  │  Linear(512, V)   │   │  Conv1D(512,512,k=3) → ReLU → Dropout       │   │
│  │  LogSoftmax       │   │  Linear(512, 1) → Sigmoid → α权重            │   │
│  │      ↓            │   │  累积α直到超过阈值β(=1.0)时触发               │   │
│  │  CTC Loss         │   │      ↓                                        │   │
│  │  (训练辅助)        │   │  预测输出长度 L'                               │   │
│  │                    │   │  声学嵌入 E₁:L': [batch, L', 512]            │   │
│  └──────────────────┘   └──────────────────────────────────────────────┘   │
│                                 ↓                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │           Sampler (GLM 采样器，仅训练时使用)                              │ │
│  │                                                                         │ │
│  │  训练时: S₁:L = Sampler(E₁:L', EMB(y₁:L))                             │ │
│  │    根据当前准确率，混合声学嵌入和真实token embedding                      │ │
│  │  推理时: S₁:L' = E₁:L' (直接使用声学嵌入)                               │ │
│  │      ↓                                                                  │ │
│  │  语义嵌入: [batch, L', 512]                                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│      ↓                                                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │           Parallel Decoder: SAN-M Decoder (16层)                        │ │
│  │                                                                         │ │
│  │  ┌──────────────────────────────────────┐                              │ │
│  │  │  SAN-M Decoder Block × 16           │                              │ │
│  │  │  ┌────────────────────────────────┐  │                              │ │
│  │  │  │ Self-Attention (无 causal mask)│  │  ← NAR: 所有位置互相可见     │ │
│  │  │  │ Cross-Attention (Q=dec, KV=enc)│  │                              │ │
│  │  │  │ Feed-Forward Network           │  │                              │ │
│  │  │  │ LayerNorm                      │  │                              │ │
│  │  │  └────────────────────────────────┘  │                              │ │
│  │  └──────────────────────────────────────┘                              │ │
│  │      ↓                                                                  │ │
│  │  Decoder 隐状态 D₁:L': [batch, L', 512]                                │ │
│  │      ↓                                                                  │ │
│  │  ASR Output Layer: Linear(512, 8404) → Softmax                         │ │
│  │      ↓                                                                  │ │
│  │  P_ASR: [batch, L', 8404]  (ASR 概率分布)                              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ════════════════════════════════════════════════════════════════════════════ │
│  ↓↓↓  以下为 SeACo 热词定制模块（独立训练，ASR backbone 冻结）  ↓↓↓         │
│  ════════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │           Bias Encoder (热词编码器)                                      │ │
│  │                                                                         │ │
│  │  热词列表 H₁:n (n个热词，如 ["达摩院", "Paraformer", "阿里云"])          │ │
│  │      ↓                                                                  │ │
│  │  Embedding Layer (与ASR共享参数)                                         │ │
│  │      ↓                                                                  │ │
│  │  LSTM (2层, hidden_size=512)                                            │ │
│  │      ↓                                                                  │ │
│  │  热词嵌入 Z₁:n: [batch, n, 512]                                        │ │
│  │  (取LSTM最后时刻输出作为每个热词的表示)                                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│      ↓                                                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │           Bias Decoder (热词解码器, 4层 Transformer)                     │ │
│  │                                                                         │ │
│  │  输入1: CIF声学嵌入 E₁:L' (来自CIF模块)                                 │ │
│  │  输入2: Decoder隐状态 D₁:L' (来自Parallel Decoder)                      │ │
│  │  输入3: 热词嵌入 Z₁:n (来自Bias Encoder)                                │ │
│  │                                                                         │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Cross-Attention 1:                                              │  │ │
│  │  │    D⁺₁:L' = MHA(Q=D₁:L', K=Z₁:n, V=Z₁:n)                      │  │ │
│  │  │    (Decoder输出 attend to 热词嵌入)                               │  │ │
│  │  │                                                                  │  │ │
│  │  │  Cross-Attention 2:                                              │  │ │
│  │  │    E⁺₁:L' = MHA(Q=E₁:L', K=Z₁:n, V=Z₁:n)                      │  │ │
│  │  │    (CIF输出 attend to 热词嵌入)                                   │  │ │
│  │  │                                                                  │  │ │
│  │  │  Feed-Forward + LayerNorm                                        │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  │      ↓                                                                  │ │
│  │  Bias Output Layer: Linear(512, 8404+1) → Softmax                      │ │
│  │  (多一个 <no-bias> token，表示该位置不是热词)                             │ │
│  │      ↓                                                                  │ │
│  │  P_bias: [batch, L', 8405]  (热词偏置概率分布)                          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│      ↓                                                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │           概率融合 (Inference Only)                                      │ │
│  │                                                                         │ │
│  │  对于第 i 个位置:                                                        │ │
│  │    if argmax(P_bias_i) == <no-bias>:                                    │ │
│  │        P_merged_i = P_ASR_i          (不激活热词，使用原始ASR结果)        │ │
│  │    else:                                                                 │ │
│  │        P_merged_i = λ·P_bias_i + (1-λ)·P_ASR_i  (融合热词偏置)         │ │
│  │                                                                         │ │
│  │  λ 为可调参数，控制热词偏置强度（默认 λ=1.0）                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│      ↓                                                                       │
│  最终输出: argmax(P_merged) → Token IDs → 文本                               │
└─────────────────────────────────────────────────────────────────────────────┘
```


### 2.2 各模块详细参数

#### Encoder: SAN-M (Self-Attention with Memory)

SAN-M 是达摩院提出的改进型 Self-Attention 结构，在标准 Multi-Head Attention 基础上融合了 FSMN 的记忆机制，兼顾全局建模和局部特征捕获。

```
SAN-M Encoder 配置:
├── 类型: SANMEncoder
├── 层数: 50
├── d_model (output_size): 512
├── 注意力头数: 4 (每头 128 维)
├── FFN 维度: 2048
├── Dropout: 0.1
├── 位置编码: SinusoidalPositionEncoder
├── 卷积核大小 (kernel_size): 11
├── SANM shift: 0
├── Self-Attention 类型: sanm
├── 输入层: pe (positional encoding)
└── normalize_before: true

SAN-M Block 内部结构:
┌─────────────────────────────────────────┐
│  输入 x                                  │
│    ↓                                     │
│  LayerNorm                               │
│    ↓                                     │
│  Multi-Head Self-Attention               │
│    + FSMN Memory Block (kernel_size=11)  │
│    ↓                                     │
│  Residual Connection (+ x)               │
│    ↓                                     │
│  LayerNorm                               │
│    ↓                                     │
│  Feed-Forward Network                    │
│    Linear(512, 2048) → ReLU → Dropout    │
│    → Linear(2048, 512) → Dropout         │
│    ↓                                     │
│  Residual Connection                     │
│    ↓                                     │
│  输出                                    │
└─────────────────────────────────────────┘

FSMN Memory 的作用:
  标准 Self-Attention: 全局注意力，计算量 O(T²)
  FSMN Memory: 局部记忆，捕获 kernel_size=11 范围内的局部模式
  SAN-M = Self-Attention (全局) + FSMN (局部) → 兼顾全局和局部
```

#### CIF Predictor (CifPredictorV2)

```
CIF Predictor 配置:
├── 输入维度 (idim): 512
├── 阈值 (threshold): 1.0
├── 左序 (l_order): 1
├── 右序 (r_order): 1
├── 尾部阈值 (tail_threshold): 0.45
└── 输出: 声学嵌入 + 预测长度

CIF 内部计算:
  输入: Encoder输出 e₁:T [batch, T', 512]
    ↓
  Conv1D(512, 512, kernel_size=3, padding=1)
    ↓
  ReLU → Dropout
    ↓
  Linear(512, 1) → Sigmoid
    ↓
  α权重: [batch, T', 1]  (每帧的信息量权重)
    ↓
  累积触发机制:
    累积 α 值，当累积 ≥ 1.0 时触发
    触发时对窗口内的 encoder 输出做加权求和
    生成一个声学嵌入
    ↓
  输出: E₁:L' [batch, L', 512]  (L' = 预测的token数)
```

#### Parallel Decoder: ParaformerSANMDecoder

```
Decoder 配置:
├── 类型: ParaformerSANMDecoder
├── 层数: 16
├── 注意力头数: 4
├── FFN 维度: 2048
├── Dropout: 0.1
├── 卷积核大小: 11
├── SANM shift: 0
└── att_layer_num: 16

Decoder Block 内部结构:
┌─────────────────────────────────────────┐
│  输入: 语义嵌入 S₁:L'                    │
│    ↓                                     │
│  Self-Attention (无 causal mask)         │
│    + FSMN Memory                         │
│    所有 L' 个位置互相可见                 │
│    ↓                                     │
│  Cross-Attention                         │
│    Q = Decoder 状态                      │
│    K, V = Encoder 输出 e₁:T              │
│    ↓                                     │
│  Feed-Forward Network                    │
│    ↓                                     │
│  输出: D₁:L' [batch, L', 512]           │
└─────────────────────────────────────────┘
```

#### Bias Encoder (热词编码器)

```
Bias Encoder 配置:
├── Embedding Layer: 与 ASR 模型共享 (vocab_size=8404, dim=512)
├── LSTM: 2层, hidden_size=512, bidirectional=False
└── 输出: 每个热词的固定长度表示 [n, 512]

处理流程:
  热词 "达摩院" → Token IDs [达, 摩, 院]
    ↓
  Embedding: [3, 512]
    ↓
  LSTM 2层: 取最后时刻隐状态
    ↓
  热词表示: [1, 512]  (一个固定维度向量代表整个热词)
```

#### Bias Decoder (热词解码器)

```
Bias Decoder 配置:
├── 类型: 4层 Transformer Decoder
├── Hidden size: 1024
├── 注意力头数: 4
├── 输入: E₁:L' (CIF输出) + D₁:L' (Decoder隐状态) + Z₁:n (热词嵌入)
└── 输出: 热词偏置概率 [batch, L', 8405]

Bias Decoder 内部:
┌─────────────────────────────────────────────────┐
│  Layer × 4:                                      │
│                                                  │
│  MHA_1: Q=D₁:L', K=Z₁:n, V=Z₁:n → D⁺₁:L'     │
│    (Decoder输出查询热词库，找到相关热词)           │
│                                                  │
│  MHA_2: Q=E₁:L', K=Z₁:n, V=Z₁:n → E⁺₁:L'     │
│    (CIF声学嵌入查询热词库，从声学角度匹配热词)    │
│                                                  │
│  Feed-Forward + LayerNorm                        │
│                                                  │
│  最终: BiasOutLayer(D⁺₁:L' + E⁺₁:L')           │
│    → [batch, L', 8405]                           │
└─────────────────────────────────────────────────┘
```

### 2.3 模型参数统计

```
模块                    参数量(约)      说明
─────────────────────────────────────────────────────
Encoder (SAN-M 50层)    ~150M          主要参数所在
CIF Predictor           ~0.5M          轻量级
Parallel Decoder (16层) ~60M           第二大模块
ASR Output Layer        ~4.3M          512 × 8404
─────────────────────────────────────────────────────
ASR Backbone 总计       ~215M
─────────────────────────────────────────────────────
Bias Encoder (LSTM)     ~4M            2层LSTM
Bias Decoder (4层)      ~25M           4层Transformer
Bias Output Layer       ~4.3M          512 × 8405
─────────────────────────────────────────────────────
SeACo 模块总计          ~33M
─────────────────────────────────────────────────────
整体模型总计            ~248M
```


## 3. 训练流程详解

### 3.1 两阶段训练策略

SeACo-Paraformer 采用两阶段训练策略，这是其设计的核心优势之一：

```
阶段一：训练 Paraformer ASR Backbone（基础语音识别能力）
═══════════════════════════════════════════════════════════
  训练数据: 50000~60000 小时标注语音
  训练模块: Encoder + CIF + Decoder + CTC
  损失函数: L_total = α·L_CTC + (1-α)·L_CE + L_MAE
  
  其中:
    L_CTC: CTC Loss（辅助训练，α=0.0 在最终配置中关闭）
    L_CE: Cross-Entropy Loss（主损失，带 label smoothing=0.1）
    L_MAE: MAE Loss（CIF长度预测损失，权重=1.0）

阶段二：训练 SeACo 热词模块（热词定制能力）
═══════════════════════════════════════════════════════════
  训练数据: 同样的 50000 小时数据（或专门的热词数据）
  冻结模块: Encoder + CIF + Decoder（全部冻结，不更新梯度）
  训练模块: Bias Encoder + Bias Decoder + Bias Output Layer
  损失函数: L_bias = CrossEntropy(P_bias, y_bias)
  
  其中 y_bias 是热词位置感知标签:
    原始标签: "今 天 去 达 摩 院 开 会"
    热词 "达摩院" 被采样后:
    bias标签: "# # # 达 摩 院 # #"  (# = <no-bias>)
```

### 3.2 热词采样策略

训练 SeACo 模块时，热词通过随机采样获得：

```
采样超参数:
├── r_b = 0.75  (batch激活比例: 75%的batch会进行热词采样)
├── r_u = 0.75  (utterance激活比例: 激活batch中75%的utterance参与)
├── l_min = 2   (最小热词长度: 至少2个字符)
├── l_max = 8   (最大热词长度: 最多8个字符)
└── 默认热词: <blank> (未激活的batch使用空白热词)

采样流程:
  1. 对当前 batch 中的所有 utterance 的转写文本
  2. 随机从中抽取满足长度要求的子串作为热词
  3. 每个激活的 batch 平均有 r_u × batch_size + 1 个热词
  4. 构建热词列表 H₁:n 和对应的 bias 标签 y_bias

示例:
  Batch 中有4条utterance:
    utt1: "今天天气真好我们出去玩"
    utt2: "达摩院发布了新模型"
    utt3: "语音识别技术进步很快"
    utt4: "阿里云提供云计算服务"
  
  随机采样热词: ["达摩院", "语音识别", "阿里云"]
  
  对应 bias 标签:
    utt1: [#, #, #, #, #, #, #, #, #, #, #]
    utt2: [达, 摩, 院, #, #, #, #, #, #]
    utt3: [语, 音, 识, 别, #, #, #, #, #, #]
    utt4: [阿, 里, 云, #, #, #, #, #, #, #]
```

### 3.3 FunASR 训练配置示例

```yaml
# SeACo-Paraformer 训练配置 (config.yaml)

# ============ 基础模型配置 ============
model: SeACo-Paraformer
model_conf:
    ctc_weight: 0.0          # CTC权重（SeACo阶段不使用CTC）
    lsm_weight: 0.1          # Label Smoothing
    length_normalized_loss: true
    predictor_weight: 1.0     # CIF预测器损失权重
    predictor_bias: 1
    sampling_ratio: 0.75      # GLM采样比例

# ============ Encoder 配置 ============
encoder: SANMEncoder
encoder_conf:
    output_size: 512
    attention_heads: 4
    linear_units: 2048
    num_blocks: 50
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    attention_dropout_rate: 0.1
    input_layer: pe
    pos_enc_class: SinusoidalPositionEncoder
    normalize_before: true
    kernel_size: 11
    sanm_shfit: 0
    selfattention_layer_type: sanm

# ============ Decoder 配置 ============
decoder: ParaformerSANMDecoder
decoder_conf:
    attention_heads: 4
    linear_units: 2048
    num_blocks: 16
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    self_attention_dropout_rate: 0.1
    src_attention_dropout_rate: 0.1
    att_layer_num: 16
    kernel_size: 11
    sanm_shfit: 0

# ============ CIF Predictor 配置 ============
predictor: CifPredictorV2
predictor_conf:
    idim: 512
    threshold: 1.0
    l_order: 1
    r_order: 1
    tail_threshold: 0.45

# ============ SeACo 热词模块配置 ============
bias_encoder:
    type: LSTM
    num_layers: 2
    hidden_size: 512
    share_embedding: true     # 与ASR共享embedding

bias_decoder:
    type: TransformerDecoder
    num_layers: 4
    hidden_size: 1024
    attention_heads: 4

# ============ 热词采样配置 ============
seaco_conf:
    r_b: 0.75                 # batch激活比例
    r_u: 0.75                 # utterance激活比例
    l_min: 2                  # 最小热词长度
    l_max: 8                  # 最大热词长度

# ============ 训练配置 ============
train_conf:
    max_epoch: 6              # SeACo阶段训练6个epoch
    batch_size: 6000          # 每batch 6000帧 × 16 GPU
    freeze_modules:           # 冻结ASR backbone
        - encoder
        - decoder
        - predictor
        - ctc

# ============ 优化器配置 ============
optim: adam
optim_conf:
    lr: 0.0005
scheduler: warmuplr
scheduler_conf:
    warmup_steps: 5000

# ============ 数据配置 ============
dataset_conf:
    data_dir: /path/to/data
    data_type: sound          # 音频文件类型
    frontend: wav_frontend
    frontend_conf:
        fs: 16000             # 采样率
        window: hamming
        n_mels: 80            # FBank维度
        frame_length: 25      # 帧长 ms
        frame_shift: 10       # 帧移 ms
        lfr_m: 7              # LFR拼帧数
        lfr_n: 6              # LFR跳帧数
```

### 3.4 FunASR 训练数据格式

FunASR 使用 jsonl 格式组织训练数据：

```jsonl
{"key": "BAC009S0002W0122", "source": "/data/wav/train/S0002/BAC009S0002W0122.wav", "target": "甚至出现交易"}
{"key": "BAC009S0002W0123", "source": "/data/wav/train/S0002/BAC009S0002W0123.wav", "target": "在这期间她始终没有放弃寻找自己的亲生父母"}
{"key": "BAC009S0002W0124", "source": "/data/wav/train/S0002/BAC009S0002W0124.wav", "target": "这一次我们是受到了公司的委托"}
```

或者使用 scp 格式：

```
# wav.scp
BAC009S0002W0122 /data/wav/train/S0002/BAC009S0002W0122.wav
BAC009S0002W0123 /data/wav/train/S0002/BAC009S0002W0123.wav

# text
BAC009S0002W0122 甚至出现交易
BAC009S0002W0123 在这期间她始终没有放弃寻找自己的亲生父母
```


## 4. 数据在模型中的完整流转（训练阶段）

### 4.1 具体示例

以一条训练样本为例，追踪数据在整个模型中的流转过程：

```
训练样本:
  音频: "今天去达摩院开会.wav" (3秒, 16kHz)
  标签: "今天去达摩院开会"
  采样热词: ["达摩院"]
```

### 4.2 Step-by-Step 数据流转

```
═══════════════════════════════════════════════════════════════════════════
Step 1: 音频加载与预处理
═══════════════════════════════════════════════════════════════════════════

输入: WAV文件 → 48000个采样点 (3s × 16000Hz)
数据形状: [48000] (float32, 范围 [-1, 1])

  ↓ FBank特征提取

分帧: 帧长25ms=400点, 帧移10ms=160点
帧数: (48000 - 400) / 160 + 1 ≈ 298帧
每帧做FFT → Mel滤波器组 → 取对数 → 80维FBank

数据形状: [298, 80]

  ↓ LFR (Low Frame Rate) 拼帧

每7帧拼接为1帧，步长6帧
帧数: (298 - 7) / 6 + 1 ≈ 49帧
每帧维度: 7 × 80 = 560

数据形状: [49, 560]

  ↓ CMVN归一化

使用预计算的全局均值和方差
x_norm = (x - global_mean) / sqrt(global_var)

数据形状: [49, 560] (值被标准化到均值0、方差1附近)

  ↓ 加batch维度

数据形状: [1, 49, 560]  (batch_size=1 简化说明)


═══════════════════════════════════════════════════════════════════════════
Step 2: Encoder 编码 (SAN-M 50层)
═══════════════════════════════════════════════════════════════════════════

输入: [1, 49, 560]

  ↓ 输入投影 Linear(560, 512)

数据形状: [1, 49, 512]

  ↓ 正弦位置编码 (Sinusoidal Positional Encoding)

PE(pos, 2i) = sin(pos / 10000^(2i/512))
PE(pos, 2i+1) = cos(pos / 10000^(2i/512))
x = x + PE

数据形状: [1, 49, 512]

  ↓ SAN-M Block × 50

每层内部:
  (1) LayerNorm → Multi-Head Self-Attention (4头, 每头128维)
      Q, K, V = Linear(x) → [1, 4, 49, 128]
      Attention = softmax(QK^T / sqrt(128)) × V
      + FSMN Memory (kernel=11, 捕获局部11帧范围的模式)
      → [1, 49, 512]
  (2) Residual + LayerNorm → FFN(512→2048→512)
      → [1, 49, 512]

经过50层后:
数据形状: [1, 49, 512]  (shape不变，但包含了丰富的声学语义信息)

输出: e₁:₄₉ = Encoder输出


═══════════════════════════════════════════════════════════════════════════
Step 3: CIF Predictor (预测输出长度 + 生成声学嵌入)
═══════════════════════════════════════════════════════════════════════════

输入: Encoder输出 [1, 49, 512]

  ↓ Conv1D(512, 512, kernel=3, padding=1) → ReLU → Dropout

数据形状: [1, 49, 512]

  ↓ Linear(512, 1) → Sigmoid

α权重: [1, 49, 1]
示例α值: [0.15, 0.12, 0.18, 0.20, 0.35, 0.14, 0.16, 0.22, ...]
         (每帧的"信息量"权重，范围0~1)

  ↓ 累积触发 (Integrate-and-Fire)

α累积过程 (简化示意):
  帧1:  0.15 → 累积 0.15
  帧2:  0.12 → 累积 0.27
  帧3:  0.18 → 累积 0.45
  帧4:  0.20 → 累积 0.65
  帧5:  0.35 → 累积 1.00 → 触发! 生成 E₁ (对应"今")
  帧6:  0.14 → 累积 0.14 (溢出0.00)
  帧7:  0.16 → 累积 0.30
  帧8:  0.22 → 累积 0.52
  帧9:  0.28 → 累积 0.80
  帧10: 0.25 → 累积 1.05 → 触发! 生成 E₂ (对应"天"), 溢出0.05
  ...
  最终触发 8 次 → 预测长度 L'=8 (对应"今天去达摩院开会" 8个字)

  ↓ 加权求和生成声学嵌入

E₁ = 0.15×e₁ + 0.12×e₂ + 0.18×e₃ + 0.20×e₄ + 0.35×e₅
E₂ = 0.14×e₆ + 0.16×e₇ + 0.22×e₈ + 0.28×e₉ + 0.20×e₁₀
...
E₈ = (对应"会"的加权声学表示)

输出: 声学嵌入 E₁:₈ = [1, 8, 512]
      预测长度 L' = 8

  ↓ MAE Loss 计算

L_MAE = |L' - L_true| = |8 - 8| = 0  (预测正确)
(如果预测错误，通过padding/截断对齐到真实长度L=8)


═══════════════════════════════════════════════════════════════════════════
Step 4: Sampler (GLM 采样，仅训练时)
═══════════════════════════════════════════════════════════════════════════

输入:
  声学嵌入 E₁:₈: [1, 8, 512]
  真实标签 "今天去达摩院开会" → Token IDs → Embedding: [1, 8, 512]

  ↓ 计算当前准确率 (Pass1 结果)

假设当前模型准确率 accuracy = 0.6
采样比例 ratio = 1 - accuracy = 0.4
→ 40% 的位置使用真实 token embedding，60% 使用声学嵌入

  ↓ 随机采样 mask

mask = [0, 1, 0, 0, 1, 1, 0, 0]  (1=使用真实token, 0=使用声学嵌入)
       今 天 去 达 摩 院 开 会

  ↓ 混合生成语义嵌入

S₁ = E₁ (声学嵌入，对应"今")
S₂ = EMB("天") (真实token embedding)
S₃ = E₃ (声学嵌入，对应"去")
S₄ = E₄ (声学嵌入，对应"达")
S₅ = EMB("摩") (真实token embedding)
S₆ = EMB("院") (真实token embedding)
S₇ = E₇ (声学嵌入，对应"开")
S₈ = E₈ (声学嵌入，对应"会")

输出: 语义嵌入 S₁:₈ = [1, 8, 512]


═══════════════════════════════════════════════════════════════════════════
Step 5: Parallel Decoder (16层 SAN-M Decoder)
═══════════════════════════════════════════════════════════════════════════

输入:
  语义嵌入 S₁:₈: [1, 8, 512] (作为 Decoder 的输入)
  Encoder输出 e₁:₄₉: [1, 49, 512] (作为 Cross-Attention 的 KV)

  ↓ SAN-M Decoder Block × 16

每层:
  (1) Self-Attention (无 causal mask，8个位置互相可见)
      Q=K=V=S → Attention → [1, 8, 512]
      + FSMN Memory
  (2) Cross-Attention
      Q=Decoder状态, K=V=Encoder输出
      → 从49帧的编码中提取与当前位置相关的信息
      → [1, 8, 512]
  (3) FFN(512→2048→512)
      → [1, 8, 512]

经过16层后:
输出: Decoder隐状态 D₁:₈ = [1, 8, 512]

  ↓ ASR Output Layer

Linear(512, 8404) → LogSoftmax
P_ASR = [1, 8, 8404]  (每个位置对8404个token的概率分布)

  ↓ ASR CE Loss

L_CE = CrossEntropy(P_ASR, y_true)
y_true = TokenIDs("今天去达摩院开会") = [今_id, 天_id, 去_id, 达_id, 摩_id, 院_id, 开_id, 会_id]


═══════════════════════════════════════════════════════════════════════════
Step 6: Bias Encoder (热词编码) — SeACo 模块
═══════════════════════════════════════════════════════════════════════════

输入: 采样的热词列表 ["达摩院", <blank>]
      (实际训练中会有多个热词 + 一个默认<blank>)

  ↓ Tokenize

"达摩院" → [达_id, 摩_id, 院_id]
"<blank>" → [blank_id]

  ↓ Embedding (共享ASR的embedding层)

"达摩院": [3, 512]  (3个字符，每个512维)
"<blank>": [1, 512]

  ↓ LSTM (2层, hidden=512)

对每个热词序列过LSTM，取最后时刻的hidden state:
"达摩院" → LSTM → 最终隐状态: [1, 512]
"<blank>" → LSTM → 最终隐状态: [1, 512]

输出: 热词嵌入 Z₁:₂ = [1, 2, 512]  (2个热词，每个512维)
      (batch维度上会repeat以匹配batch_size)


═══════════════════════════════════════════════════════════════════════════
Step 7: Bias Decoder (热词解码) — SeACo 模块
═══════════════════════════════════════════════════════════════════════════

输入:
  CIF声学嵌入 E₁:₈: [1, 8, 512]
  Decoder隐状态 D₁:₈: [1, 8, 512]
  热词嵌入 Z₁:₂: [1, 2, 512]

  ↓ Bias Decoder Layer × 4

每层:
  (1) Cross-Attention 1: D attend to Z
      Q=D₁:₈ [1,8,512], K=V=Z₁:₂ [1,2,512]
      Attention权重: [1, 4heads, 8, 2]
      → D⁺₁:₈ [1, 8, 512]
      
      直觉: Decoder的每个位置去"查询"热词库
      位置4(达)、5(摩)、6(院) 会对热词"达摩院"产生高注意力
      其他位置对<blank>产生高注意力

  (2) Cross-Attention 2: E attend to Z
      Q=E₁:₈ [1,8,512], K=V=Z₁:₂ [1,2,512]
      → E⁺₁:₈ [1, 8, 512]
      
      直觉: CIF声学嵌入从声学角度匹配热词
      "达"的声学特征与热词"达摩院"的开头匹配

  (3) Feed-Forward + LayerNorm

  ↓ Bias Output Layer

BiasOutLayer(D⁺₁:₈ + E⁺₁:₈)
= Linear(512, 8405) → Softmax
→ P_bias = [1, 8, 8405]

其中 8405 = 8404(原始词表) + 1(<no-bias>标记)

  ↓ Bias CE Loss

构建 bias 标签:
  原始标签: "今 天 去 达 摩 院 开 会"
  热词 "达摩院" 对应位置4,5,6
  bias标签: [#, #, #, 达_id, 摩_id, 院_id, #, #]
  (# = no_bias_id = 8404)

L_bias = CrossEntropy(P_bias, y_bias)


═══════════════════════════════════════════════════════════════════════════
Step 8: 总损失计算与反向传播
═══════════════════════════════════════════════════════════════════════════

阶段一 (训练ASR Backbone):
  L_total = L_CE + predictor_weight × L_MAE
  反向传播更新: Encoder, Decoder, CIF, Output Layer

阶段二 (训练SeACo模块):
  L_total = L_bias
  反向传播更新: Bias Encoder, Bias Decoder, Bias Output Layer
  冻结不更新: Encoder, Decoder, CIF, ASR Output Layer
```


## 5. 推理阶段数据流转

### 5.1 推理示例

```
推理输入:
  音频: "今天去达摩院开会.wav" (3秒, 16kHz)
  热词列表: ["达摩院", "FunASR", "阿里云"]  (用户提供)
```

### 5.2 推理数据流

```
═══════════════════════════════════════════════════════════════════════════
Step 1-2: 音频预处理 + Encoder (与训练相同)
═══════════════════════════════════════════════════════════════════════════

WAV [48000] → FBank [298,80] → LFR [49,560] → CMVN → Encoder
→ e₁:₄₉ = [1, 49, 512]


═══════════════════════════════════════════════════════════════════════════
Step 3: CIF Predictor
═══════════════════════════════════════════════════════════════════════════

e₁:₄₉ → CIF → 声学嵌入 E₁:₈ = [1, 8, 512], 预测长度 L'=8


═══════════════════════════════════════════════════════════════════════════
Step 4: Sampler (推理时不使用GLM)
═══════════════════════════════════════════════════════════════════════════

推理时没有真实标签，直接使用声学嵌入:
S₁:₈ = E₁:₈ = [1, 8, 512]


═══════════════════════════════════════════════════════════════════════════
Step 5: Parallel Decoder
═══════════════════════════════════════════════════════════════════════════

S₁:₈ + e₁:₄₉ → Decoder 16层 → D₁:₈ = [1, 8, 512]
→ ASR Output Layer → P_ASR = [1, 8, 8404]

初步ASR结果 (argmax): "今天去达摩院开会"
(假设基础模型已经能正确识别)


═══════════════════════════════════════════════════════════════════════════
Step 6: Bias Encoder (编码用户热词)
═══════════════════════════════════════════════════════════════════════════

热词列表: ["达摩院", "FunASR", "阿里云"]
  ↓ Tokenize + Embedding + LSTM
热词嵌入 Z₁:₃ = [1, 3, 512]


═══════════════════════════════════════════════════════════════════════════
Step 7: Bias Decoder + ASF (注意力分数过滤)
═══════════════════════════════════════════════════════════════════════════

第一轮: 全量热词推理 (获取注意力分数)
  D₁:₈ + E₁:₈ + Z₁:₃ → Bias Decoder
  → 注意力分数矩阵 A ∈ R^{8×3}
  
  A = [[0.1, 0.0, 0.0],   # 位置1 "今" → 对所有热词注意力低
       [0.1, 0.0, 0.0],   # 位置2 "天"
       [0.1, 0.0, 0.0],   # 位置3 "去"
       [0.8, 0.0, 0.1],   # 位置4 "达" → 对"达摩院"注意力高!
       [0.9, 0.0, 0.0],   # 位置5 "摩" → 对"达摩院"注意力高!
       [0.8, 0.0, 0.1],   # 位置6 "院" → 对"达摩院"注意力高!
       [0.1, 0.0, 0.0],   # 位置7 "开"
       [0.1, 0.0, 0.0]]   # 位置8 "会"

  按列求和 (每个热词的总注意力):
    "达摩院": 0.1+0.1+0.1+0.8+0.9+0.8+0.1+0.1 = 3.0 ← 最高!
    "FunASR": 0.0+0.0+0.0+0.0+0.0+0.0+0.0+0.0 = 0.0
    "阿里云": 0.0+0.0+0.0+0.1+0.0+0.1+0.0+0.0 = 0.2

  ASF 过滤: 选择 top-k (k=50) 个最活跃的热词
  → 保留 ["达摩院"] (其他热词注意力太低，被过滤)

第二轮: 过滤后热词推理 (精确偏置)
  D₁:₈ + E₁:₈ + Z_filtered → Bias Decoder
  → P_bias = [1, 8, 8405]


═══════════════════════════════════════════════════════════════════════════
Step 8: 概率融合
═══════════════════════════════════════════════════════════════════════════

对每个位置 i:
  if argmax(P_bias[i]) == <no-bias> (id=8404):
      P_merged[i] = P_ASR[i]  (使用原始ASR结果)
  else:
      P_merged[i] = λ × P_bias[i] + (1-λ) × P_ASR[i]

λ = 1.0 时:
  位置1 "今": argmax(P_bias)=<no-bias> → 使用 P_ASR → "今"
  位置2 "天": argmax(P_bias)=<no-bias> → 使用 P_ASR → "天"
  位置3 "去": argmax(P_bias)=<no-bias> → 使用 P_ASR → "去"
  位置4 "达": argmax(P_bias)="达"      → 使用 P_bias → "达" ✓
  位置5 "摩": argmax(P_bias)="摩"      → 使用 P_bias → "摩" ✓
  位置6 "院": argmax(P_bias)="院"      → 使用 P_bias → "院" ✓
  位置7 "开": argmax(P_bias)=<no-bias> → 使用 P_ASR → "开"
  位置8 "会": argmax(P_bias)=<no-bias> → 使用 P_ASR → "会"


═══════════════════════════════════════════════════════════════════════════
Step 9: 最终输出
═══════════════════════════════════════════════════════════════════════════

Token IDs → 文本解码 → "今天去达摩院开会"

完整输出结构:
{
    "key": "test_001",
    "text": "今天去达摩院开会",
    "timestamp": [[0,380], [380,620], [620,900], [900,1180],
                  [1180,1460], [1460,1740], [1740,2100], [2100,2500]],
    "hotwords_activated": ["达摩院"]
}
```

### 5.3 热词纠错场景示例

```
场景: 基础ASR模型将"达摩院"错误识别为"大魔院"（同音字错误）

无热词时:
  P_ASR 位置4: {"大":0.45, "达":0.40, "打":0.05, ...}
  argmax → "大" (错误!)

有热词"达摩院"时:
  P_bias 位置4: {"达":0.92, "#":0.05, "大":0.01, ...}
  argmax(P_bias) = "达" ≠ <no-bias>
  → P_merged = λ×P_bias + (1-λ)×P_ASR
  → 当 λ=1.0: P_merged = P_bias
  → argmax → "达" (正确!)

这就是热词定制的核心价值:
  通过 Bias Decoder 的显式热词检测，
  在模型"犹豫"的位置强制偏向热词，纠正同音字错误。
```


## 6. FunASR 训练实战

### 6.1 环境准备

```bash
# 安装 FunASR
pip install funasr
pip install modelscope

# 克隆 FunASR 源码（用于训练脚本）
git clone https://github.com/modelscope/FunASR.git
cd FunASR
pip install -e .
```

### 6.2 数据准备

```python
"""
数据准备脚本: 将原始数据转换为 FunASR 训练格式
"""
import json
import os

def prepare_data(wav_scp_path, text_path, output_jsonl):
    """
    将 wav.scp 和 text 文件转换为 FunASR 的 jsonl 格式
    
    wav.scp 格式:
        utterance_id /path/to/audio.wav
    text 格式:
        utterance_id 转写文本
    """
    # 读取 wav.scp
    wav_dict = {} 
    with open(wav_scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            wav_dict[parts[0]] = parts[1]
    
    # 读取 text
    text_dict = {}
    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            text_dict[parts[0]] = parts[1]
    
    # 生成 jsonl
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for utt_id in wav_dict:
            if utt_id in text_dict:
                record = {
                    "key": utt_id,
                    "source": wav_dict[utt_id],
                    "target": text_dict[utt_id]
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

# 准备训练集、验证集
prepare_data("data/train/wav.scp", "data/train/text", "data/train.jsonl")
prepare_data("data/dev/wav.scp", "data/dev/text", "data/dev.jsonl")
```

### 6.3 使用 FunASR 微调 SeACo-Paraformer

```python
"""
使用 FunASR 微调 SeACo-Paraformer 的完整示例

场景: 在特定领域数据上微调热词模块，提升领域专有名词的识别率
"""
import os
from funasr import AutoModel

# ============================================================
# 方式1: 使用 AutoModel 进行推理（验证基线效果）
# ============================================================

# 加载预训练的 SeACo-Paraformer 模型
model = AutoModel(
    model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    device="cuda:0",
)

# 不使用热词的基线识别
result_baseline = model.generate(input="test_audio.wav")
print(f"基线结果: {result_baseline[0]['text']}")

# 使用热词的识别
result_hotword = model.generate(
    input="test_audio.wav",
    hotword="达摩院 Paraformer FunASR 阿里云"
)
print(f"热词结果: {result_hotword[0]['text']}")


# ============================================================
# 方式2: 使用 FunASR 训练框架进行微调
# ============================================================

"""
训练命令 (命令行执行):

# 阶段一: 微调 ASR Backbone (如果需要)
python -m funasr.bin.train \
    --config conf/seaco_paraformer_large.yaml \
    --train_data_path data/train.jsonl \
    --valid_data_path data/dev.jsonl \
    --output_dir exp/seaco_paraformer_finetune \
    --init_param iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
    --max_epoch 10 \
    --batch_size 4000 \
    --learning_rate 0.0001 \
    --gpu_id 0,1,2,3

# 阶段二: 仅微调 SeACo 热词模块
python -m funasr.bin.train \
    --config conf/seaco_paraformer_large_bias.yaml \
    --train_data_path data/train.jsonl \
    --valid_data_path data/dev.jsonl \
    --output_dir exp/seaco_bias_finetune \
    --init_param exp/seaco_paraformer_finetune/best.pt \
    --freeze_modules encoder,decoder,predictor,ctc \
    --max_epoch 6 \
    --batch_size 6000 \
    --learning_rate 0.0005 \
    --gpu_id 0,1,2,3
"""
```

### 6.4 自定义热词数据增强

```python
"""
为 SeACo-Paraformer 训练准备热词增强数据

核心思路:
  1. 从训练文本中提取命名实体/专有名词作为热词候选
  2. 构建热词列表文件
  3. 训练时框架会自动从batch中随机采样热词
"""
import re
import json
from collections import Counter

def extract_hotword_candidates(text_file, min_len=2, max_len=8):
    """
    从训练文本中提取热词候选
    实际生产中建议使用 NER 工具或领域词典
    """
    word_counter = Counter()
    
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 2:
                continue
            text = parts[1]
            
            # 简单的中文词语提取（实际应使用分词工具）
            # 这里用连续中文字符作为候选
            words = re.findall(r'[\u4e00-\u9fff]{2,8}', text)
            for w in words:
                if min_len <= len(w) <= max_len:
                    word_counter[w] += 1
    
    # 选择出现频率适中的词作为热词候选
    # 太高频的词模型已经能识别，太低频的词样本不够
    candidates = [
        word for word, count in word_counter.items()
        if 3 <= count <= 100  # 出现3-100次的词
    ]
    
    return candidates

def create_hotword_list(candidates, output_file):
    """生成热词列表文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in candidates:
            f.write(word + '\n')

# 使用示例
candidates = extract_hotword_candidates("data/train/text")
create_hotword_list(candidates, "data/hotword_list.txt")
print(f"提取了 {len(candidates)} 个热词候选")
```

### 6.5 推理部署示例

```python
"""
SeACo-Paraformer 完整推理流水线
包含 VAD + ASR(热词) + 标点恢复
"""
from funasr import AutoModel

# 加载完整流水线
model = AutoModel(
    model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    device="cuda:0",
)

# 定义领域热词
hotwords = "达摩院 Paraformer SeACo FunASR 阿里云 ModelScope 语音识别"

# 识别长音频
result = model.generate(
    input="meeting_recording.wav",
    hotword=hotwords,
    batch_size_s=300,  # 每次处理300秒
)

# 输出结果
for item in result:
    print(f"文本: {item['text']}")
    if 'timestamp' in item:
        print(f"时间戳: {item['timestamp']}")
    print("---")


# ============================================================
# 批量处理多个文件
# ============================================================

import os
import glob

audio_files = glob.glob("audio_dir/*.wav")

results = model.generate(
    input=audio_files,
    hotword=hotwords,
    batch_size_s=300,
)

for audio_path, result in zip(audio_files, results):
    filename = os.path.basename(audio_path)
    print(f"[{filename}] {result['text']}")
```


## 7. 维度变化总结表

以 3 秒音频 "今天去达摩院开会"（8个字）为例，热词列表 3 个热词：

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 阶段                    │ 数据形状              │ 说明                    │
├──────────────────────────────────────────────────────────────────────────┤
│ 原始音频                │ [48000]              │ 3s × 16kHz              │
│ FBank特征               │ [298, 80]            │ 298帧 × 80维            │
│ LFR拼帧后              │ [49, 560]            │ 7帧拼接, 步长6           │
│ CMVN归一化             │ [49, 560]            │ 值标准化                 │
│ +batch维度             │ [1, 49, 560]         │ batch_size=1            │
│ Encoder输入投影         │ [1, 49, 512]         │ Linear(560,512)         │
│ Encoder输出 (50层后)    │ [1, 49, 512]         │ 丰富语义信息             │
│ CIF α权重              │ [1, 49, 1]           │ 每帧信息量               │
│ CIF声学嵌入            │ [1, 8, 512]          │ 8个token的声学表示       │
│ Sampler输出            │ [1, 8, 512]          │ 混合/纯声学嵌入          │
│ Decoder隐状态          │ [1, 8, 512]          │ 16层Decoder后           │
│ P_ASR                  │ [1, 8, 8404]         │ ASR概率分布              │
│ 热词Embedding          │ [3, max_len, 512]    │ 3个热词的字符嵌入        │
│ 热词LSTM输出           │ [1, 3, 512]          │ 3个热词的固定表示        │
│ Bias Decoder D⁺        │ [1, 8, 512]          │ 偏置后的Decoder状态      │
│ Bias Decoder E⁺        │ [1, 8, 512]          │ 偏置后的CIF嵌入         │
│ P_bias                 │ [1, 8, 8405]         │ 热词偏置概率(+1 no-bias) │
│ P_merged               │ [1, 8, 8404]         │ 融合后最终概率           │
│ 输出Token IDs          │ [1, 8]               │ argmax结果              │
│ 最终文本               │ "今天去达摩院开会"     │ Token解码               │
└──────────────────────────────────────────────────────────────────────────┘
```

## 8. 与标准 Paraformer 的关键区别

```
┌─────────────────────────────────────────────────────────────────────────┐
│                Paraformer vs SeACo-Paraformer 对比                       │
├─────────────────────────────────────────────────────────────────────────┤
│ 特性              │ Paraformer-large        │ SeACo-Paraformer-large   │
├─────────────────────────────────────────────────────────────────────────┤
│ Encoder           │ SAN-M 50层             │ SAN-M 50层 (相同)        │
│ CIF               │ CifPredictorV2         │ CifPredictorV2 (相同)    │
│ Decoder           │ SAN-M 16层             │ SAN-M 16层 (相同)        │
│ 热词支持          │ 无 / 简单CLAS          │ SeACo (Bias Enc+Dec)     │
│ 额外参数          │ 无                     │ ~33M (Bias模块)          │
│ 训练方式          │ 端到端联合训练          │ 两阶段解耦训练           │
│ 推理速度          │ 最快                   │ 略慢(多一次Bias Decoder)  │
│ 热词效果          │ 无/一般                │ 优秀(R1召回率65%+)       │
│ 通用ASR效果       │ 优秀                   │ 优秀(不受热词模块影响)    │
│ 无热词时          │ 正常识别               │ 退化为标准Paraformer      │
│ 模型ID(ModelScope)│ paraformer-zh          │ speech_seaco_paraformer.. │
├─────────────────────────────────────────────────────────────────────────┤
│ 核心区别: SeACo 在 Paraformer 基础上增加了独立训练的热词检测模块，       │
│ 通过 Bias Encoder + Bias Decoder 实现显式的热词激活与概率融合。          │
│ 当不提供热词时，SeACo-Paraformer 的行为与标准 Paraformer 完全一致。      │
└─────────────────────────────────────────────────────────────────────────┘
```

## 9. 关键技术总结

### 9.1 为什么用 CIF 输出和 Decoder 输出双路输入 Bias Decoder？

```
消融实验结果 (论文 Table 3):

方案                          CER    R1-Hotword Recall
─────────────────────────────────────────────────────
SeACo (D + E 分别attend)     5.40   60%    ← 最优
A1: D+E 合并后再attend       5.56   60%
A2: 仅用 E (CIF输出)         5.94   61%    ← 召回高但CER差
A3: 仅用 D (Decoder输出)     5.49   55%    ← CER好但召回低

分析:
- 仅用 CIF 输出 (E): 声学信息丰富，容易匹配热词发音
  → 召回率高，但容易误激活（CER上升）
- 仅用 Decoder 输出 (D): 语义信息丰富，判断更准确
  → CER好，但可能漏掉一些热词（召回低）
- 双路分别 attend: 声学+语义互补，既准确又全面
```

### 9.2 ASF (Attention Score Filtering) 的作用

```
问题: 当热词列表很大时（如4000个），Bias Decoder的Cross-Attention
      难以在大量热词中准确定位，性能下降。

解决: ASF 两轮推理策略
  第一轮: 用全量热词做一次轻量推理，获取注意力分数
  第二轮: 只保留 top-k 个最相关的热词，做精确推理

效果 (热词数=4000时):
  无ASF: CER上升，召回率下降
  有ASF (k=50): CER保持稳定，召回率维持优势
```

### 9.3 训练解耦的优势

```
传统 CLAS 方法:
  热词模块与ASR模型联合训练
  → 热词训练可能影响通用ASR性能
  → 难以单独优化热词能力
  → 更换热词策略需要重新训练整个模型

SeACo 解耦训练:
  ASR Backbone 独立训练 → 保证通用识别性能
  SeACo 模块独立训练 → 可以:
    ├── 使用专门的热词数据（低频词、专有名词）
    ├── 使用不同的训练策略
    ├── 快速迭代热词模块而不影响基础ASR
    └── 在不同领域快速适配（只需微调SeACo模块）
```

## 10. 参考资料

- 论文: [SeACo-Paraformer: A Non-Autoregressive ASR System with Flexible and Effective Hotword Customization Ability](https://arxiv.org/abs/2308.03266) (ICASSP 2024)
- 论文: [Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition](https://arxiv.org/abs/2206.08317) (INTERSPEECH 2022)
- 论文: [SAN-M: Memory Equipped Self-Attention for End-to-End Speech Recognition](https://arxiv.org/abs/2006.01713)
- 论文: [CIF: Continuous Integrate-and-Fire for End-to-End Speech Recognition](https://arxiv.org/abs/2005.11764) (ICASSP 2020)
- GitHub: [FunASR](https://github.com/modelscope/FunASR)
- GitHub: [SeACo-Paraformer 源码](https://github.com/R1ckShi/SeACo-Paraformer)
- ModelScope: [speech_seaco_paraformer_large](https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)
- HuggingFace: [funasr/paraformer-zh](https://huggingface.co/funasr/paraformer-zh)