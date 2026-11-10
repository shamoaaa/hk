# FunASR 详细介绍：从入门到掌握

## 1. FunASR 是什么

FunASR（Fundamental ASR）是阿里巴巴达摩院语音实验室于 2023 年开源的语音识别工具包。它提供了从学术研究到工业落地的完整解决方案，尤其在中文语音识别领域表现突出，是目前开源中文 ASR 中效果最好、生态最完善的框架之一。

FunASR 的名字中 "Fun" 代表 "Fundamental"，寓意为语音识别提供基础性的、根本性的能力支撑。

### 1.1 FunASR 的核心定位

```
工业级端到端语音识别工具包
├── 丰富的预训练模型：开箱即用，覆盖 ASR、VAD、标点恢复、说话人识别等
├── 非自回归旗舰模型 Paraformer：速度快、效果好
├── 完整的语音处理流水线：VAD + ASR + 标点 + 时间戳 + 热词
├── 支持离线和流式识别
├── 基于 ModelScope 生态：模型下载和管理方便
└── 已在阿里内部大规模生产验证
```

### 1.2 FunASR 在 ASR 生态中的位置

```
ASR 框架演进与定位：

Kaldi (2011)          ← 传统 GMM-HMM / DNN-HMM，模块化架构
    │
    ├──→ ESPnet (2018)     ← 学术界主流，功能全面，研究导向
    ├──→ k2/icefall (2020) ← WFST + 端到端，Kaldi 原作者新项目
    ├──→ WeNet (2021)      ← 出门问问，工业级，流式支持好
    ├──→ FunASR (2023)     ← 阿里达摩院，工业级，模型丰富，中文最强
    └──→ Whisper (2022)    ← OpenAI，多语言，零样本能力强

FunASR 的差异化优势：
├── 非自回归解码（Paraformer）：推理速度比自回归模型快 10 倍以上
├── 超大规模训练数据：60000+ 小时中文语音
├── 最完整的中文语音处理流水线
└── 工业级质量保证（阿里内部大规模使用）
```

### 1.3 FunASR 与其他框架的对比

| 特性       | FunASR           | WeNet          | ESPnet         | Whisper      | Kaldi       |
| -------- | ---------------- | -------------- | -------------- | ------------ | ----------- |
| 开发者      | 阿里达摩院            | 出门问问           | CMU/JHU        | OpenAI       | JHU         |
| 架构       | 端到端              | 端到端            | 端到端            | 端到端          | 传统模块化       |
| 核心模型     | Paraformer (NAR) | Conformer (AR) | Conformer (AR) | Whisper (AR) | GMM/DNN-HMM |
| 解码方式     | 非自回归（并行）         | 自回归（逐字）        | 自回归（逐字）        | 自回归（逐字）      | WFST 搜索     |
| 推理速度     | 很快               | 中等             | 中等             | 慢            | 快（WFST）     |
| 中文效果     | 最优               | 优秀             | 优秀             | 良好           | 一般          |
| 流式支持     | 支持               | 支持             | 支持             | 不支持          | 天然支持        |
| 预训练模型    | 非常丰富             | 较多             | 较多             | 多语言          | 无           |
| VAD/标点集成 | 完整集成             | 部分             | 部分             | 无            | 无           |
| 多语言      | 中英为主             | 中英为主           | 多语言            | 99 种语言       | 多语言         |
| 训练数据规模   | 60000h+          | 10000h+        | 取决于 recipe     | 680000h      | 取决于 recipe  |
| 部署友好度    | 高（ONNX/服务端）      | 高              | 中等             | 中等           | 低（多文件）      |
| 学习曲线     | 低（API 简洁）        | 中等             | 较高             | 低            | 高           |

## 2. 安装与环境配置

### 2.1 基础安装（推荐）

FunASR 的安装非常简单，通过 pip 即可完成：

```bash
# 基础安装
pip install funasr

# 安装 ModelScope（用于模型下载和管理）
pip install modelscope

# 音频处理依赖
pip install soundfile numpy
```

### 2.2 完整安装（包含所有可选依赖）

```bash
# 完整安装，包含所有功能
pip install "funasr[all]"

# 或者分别安装各组件
pip install funasr modelscope soundfile numpy torch torchaudio
```

### 2.3 GPU 支持

```bash
# 如果需要 GPU 加速，确保安装了 CUDA 版本的 PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证 GPU 是否可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2.4 环境要求

```
Python >= 3.8
PyTorch >= 1.13
操作系统: Linux / macOS / Windows
内存: >= 4GB（推荐 8GB+）
磁盘: 模型下载需要约 1-3GB 空间（取决于模型大小）
GPU: 可选，CPU 也能运行（速度较慢）
```

### 2.5 验证安装

```python
# 验证 FunASR 安装
import funasr
print(f"FunASR version: {funasr.__version__}")

# 快速测试（会自动下载模型，首次较慢）
from funasr import AutoModel
model = AutoModel(model="paraformer-zh")
result = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav")
print(result[0]["text"])
```

### 2.6 模型存储位置

FunASR 通过 ModelScope 下载模型，默认存储在：

```
~/.cache/modelscope/hub/
├── iic/
│   ├── speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/
│   ├── speech_fsmn_vad_zh-cn-16k-common-pytorch/
│   ├── punc_ct-transformer_cn-en-common-vocab471067-large/
│   └── ...
```

> 如果需要修改模型缓存路径，可以设置环境变量 `MODELSCOPE_CACHE`：
> 
> ```bash
> export MODELSCOPE_CACHE=/your/custom/path
> ```

## 3. 核心概念与模型体系

### 3.1 FunASR 的模型体系

FunASR 提供了一整套语音处理模型，覆盖从语音检测到文本后处理的完整流水线：

```
FunASR 模型体系：

┌─────────────────────────────────────────────────────────────────┐
│                    语音处理完整流水线                              │
│                                                                  │
│  原始音频                                                        │
│      ↓                                                           │
│  ┌──────────────┐                                               │
│  │ FSMN-VAD     │  语音活动检测：找出音频中有人说话的片段          │
│  │ (语音检测)    │  输出：语音段的起止时间 [[0, 3200], [4500, 8900]]│
│  └──────┬───────┘                                               │
│         ↓                                                        │
│  ┌──────────────┐                                               │
│  │ Paraformer   │  语音识别：将语音转为文字                       │
│  │ (语音识别)    │  输出：原始文本 "今天天气真好我们出去玩吧"       │
│  └──────┬───────┘                                               │
│         ↓                                                        │
│  ┌──────────────┐                                               │
│  │ CT-Punc      │  标点恢复：为无标点文本添加标点                  │
│  │ (标点恢复)    │  输出："今天天气真好，我们出去玩吧。"            │
│  └──────┬───────┘                                               │
│         ↓                                                        │
│  ┌──────────────┐                                               │
│  │ 时间戳对齐    │  可选：输出每个字/词的时间戳                    │
│  └──────┬───────┘                                               │
│         ↓                                                        │
│  ┌──────────────┐                                               │
│  │ 热词偏置      │  可选：提升特定词汇的识别准确率                  │
│  └──────────────┘                                               │
│                                                                  │
│  其他独立模型：                                                   │
│  ├── Emotion2Vec：语音情感识别                                   │
│  ├── CAM++：说话人验证/识别                                      │
│  └── SenseVoice：多语言语音理解                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 FunASR 的核心 API：AutoModel

FunASR 的所有模型都通过统一的 `AutoModel` 接口加载和使用，设计理念类似 HuggingFace 的 Transformers：

```python
from funasr import AutoModel

# 加载单个模型
model = AutoModel(model="paraformer-zh")

# 加载完整流水线（VAD + ASR + 标点）
model = AutoModel(
    model="paraformer-zh",       # ASR 模型
    vad_model="fsmn-vad",        # VAD 模型
    punc_model="ct-punc",        # 标点恢复模型
)

# 推理
result = model.generate(input="audio.wav")
```

`AutoModel` 的关键参数：

| 参数         | 说明          | 示例              |
| ---------- | ----------- | --------------- |
| model      | ASR 模型名称或路径 | "paraformer-zh" |
| vad_model  | VAD 模型（可选）  | "fsmn-vad"      |
| punc_model | 标点恢复模型（可选）  | "ct-punc"       |
| device     | 推理设备        | "cpu" 或 "cuda"  |
| ncpu       | CPU 线程数     | 4               |
| batch_size | 批处理大小       | 1               |

`generate` 方法的关键参数：

| 参数         | 说明                        | 示例                |
| ---------- | ------------------------- | ----------------- |
| input      | 输入音频（文件路径/URL/numpy数组/列表） | "test.wav"        |
| hotword    | 热词列表                      | "阿里巴巴 达摩院 FunASR" |
| cache      | 流式识别的缓存                   | {}                |
| is_final   | 是否为最后一个 chunk             | True/False        |
| chunk_size | 流式 chunk 配置               | [5, 10, 5]        |

## 4. 旗舰模型：Paraformer 深度解析

### 4.1 为什么 Paraformer 是核心

Paraformer（Parallel Transformer）是 FunASR 最重要的模型，也是目前开源中文 ASR 中效果最好的模型之一。它的核心创新在于实现了非自回归（Non-Autoregressive, NAR）解码，一次性并行生成所有字符。

```
自回归（AR）解码 vs 非自回归（NAR）解码：

AR 解码（传统方式，如 Whisper、WeNet）：
  时间步1: 输入 <sos>        → 预测 "今"
  时间步2: 输入 <sos> 今     → 预测 "天"
  时间步3: 输入 <sos> 今 天  → 预测 "天"
  时间步4: 输入 <sos> 今天天 → 预测 "气"
  ...
  必须逐字生成，前一个字生成完才能生成下一个字
  6个字需要6次 Decoder 前向传播

NAR 解码（Paraformer）：
  一次前向传播 → 同时输出 ["今", "天", "天", "气", "真", "好"]
  6个字只需要1次 Decoder 前向传播
  速度提升约 10 倍
```

> 为什么 NAR 解码难？因为 AR 解码时，生成每个字都能参考前面已经生成的字（上下文信息），而 NAR 解码时所有字同时生成，互相看不到对方的结果。Paraformer 通过 CIF 和 GLM 两个关键技术解决了这个问题。

### 4.2 Paraformer 完整架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Paraformer 架构                               │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     输入处理                                   │   │
│  │  WAV音频 (16kHz)                                              │   │
│  │      ↓                                                        │   │
│  │  FBank特征提取 (80维, 帧长25ms, 帧移10ms)                      │   │
│  │      ↓                                                        │   │
│  │  CMVN归一化（使用预计算的全局均值和方差）                        │   │
│  │      ↓                                                        │   │
│  │  特征: [batch, T, 80]    (T ≈ 音频秒数 × 100)                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
│      ↓                                                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   Encoder (Conformer 编码器)                    │   │
│  │                                                                │   │
│  │  Conv2D Subsampling (4倍下采样)                                │   │
│  │      [batch, T, 80] → [batch, T/4, d_model]                  │   │
│  │      ↓                                                        │   │
│  │  Positional Encoding (正弦位置编码)                             │   │
│  │      ↓                                                        │   │
│  │  ┌────────────────────────────────┐                           │   │
│  │  │   Conformer Block × 12        │                           │   │
│  │  │   ┌──────────────────────┐    │                           │   │
│  │  │   │ ½ Feed-Forward       │    │                           │   │
│  │  │   │ Multi-Head Attention │    │                           │   │
│  │  │   │ Convolution Module   │    │                           │   │
│  │  │   │ ½ Feed-Forward       │    │                           │   │
│  │  │   │ LayerNorm            │    │                           │   │
│  │  │   └──────────────────────┘    │                           │   │
│  │  └────────────────────────────────┘                           │   │
│  │      ↓                                                        │   │
│  │  Encoder输出: [batch, T/4, 512]                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│      ↓                          ↓                                    │
│  ┌──────────────────┐   ┌──────────────────────────────────────┐   │
│  │   CTC 辅助分支    │   │         CIF 模块（核心创新）           │   │
│  │                    │   │                                      │   │
│  │  Linear(512, V)   │   │  Conv1D → Sigmoid → α权重            │   │
│  │  LogSoftmax       │   │  累积α直到超过阈值β(=1.0)             │   │
│  │      ↓            │   │  触发时生成一个声学嵌入                 │   │
│  │  CTC Loss         │   │      ↓                                │   │
│  │  (辅助训练)        │   │  预测输出长度 N                        │   │
│  │                    │   │  声学嵌入: [batch, N, 512]            │   │
│  └──────────────────┘   └──────────────────────────────────────┘   │
│                                 ↓                                    │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Decoder (非自回归 Transformer 解码器)               │   │
│  │                                                                │   │
│  │  声学嵌入 + Token Embedding                                    │   │
│  │  （训练时使用 GLM 采样策略）                                     │   │
│  │      ↓                                                        │   │
│  │  ┌────────────────────────────────┐                           │   │
│  │  │  Transformer Decoder × 6      │                           │   │
│  │  │  ┌──────────────────────┐     │                           │   │
│  │  │  │ Self-Attention       │     │  ← NAR模式：不需要         │   │
│  │  │  │ Cross-Attention      │     │    causal mask，所有       │   │
│  │  │  │ Feed-Forward         │     │    位置可以互相看到        │   │
│  │  │  └──────────────────────┘     │                           │   │
│  │  └────────────────────────────────┘                           │   │
│  │      ↓                                                        │   │
│  │  Linear(512, vocab_size) → Softmax                            │   │
│  │      ↓                                                        │   │
│  │  输出: [batch, N, vocab_size]                                 │   │
│  │  一次性并行输出所有 N 个 token                                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Loss = α·CTC_Loss + (1-α)·CE_Loss + MAE_Loss(CIF长度预测)         │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 CIF 模块详解

CIF（Continuous Integrate-and-Fire）是 Paraformer 最核心的创新组件。它解决了 NAR 解码的两个关键问题：

1. 如何预测输出序列的长度（输出多少个字）
2. 如何为每个输出位置生成对应的声学表示

```
CIF 的工作原理：

Encoder输出: [h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈, ...]
                ↓
Conv1D(1D卷积) + Sigmoid → 每个位置产生一个权重 α ∈ (0, 1):
α值:           [0.3, 0.4, 0.5, 0.8, 0.2, 0.6, 0.7, 0.5, ...]
                ↓
累积求和，当累积值 ≥ 阈值 β(=1.0) 时触发：

位置:  h₁    h₂    h₃    h₄    h₅    h₆    h₇    h₈
α值:   0.3   0.4   0.5   0.8   0.2   0.6   0.7   0.5
累积:  0.3 → 0.7 → 1.2   ← 触发！溢出 0.2 留给下一轮
                    0.2+0.8=1.0  ← 触发！溢出 0.0
                          0.2 → 0.8 → 1.5  ← 触发！溢出 0.5
                                      0.5+0.5=1.0  ← 触发！

触发 4 次 → 预测输出长度 N=4

每次触发时，对触发窗口内的 Encoder 输出做加权求和：
  声学嵌入₁ = 0.3·h₁ + 0.4·h₂ + 0.3·h₃  (0.3是h₃的部分贡献)
  声学嵌入₂ = 0.2·h₃ + 0.8·h₄            (0.2是h₃的剩余贡献)
  声学嵌入₃ = 0.2·h₅ + 0.6·h₆ + 0.2·h₇
  声学嵌入₄ = 0.5·h₇ + 0.5·h₈

最终：4 个声学嵌入 → 对应 4 个输出 token
```

CIF 的直觉理解：

- α 值可以理解为每个 Encoder 帧"包含多少信息量"
- 信息量大的帧（如元音核心部分）α 值高，信息量小的帧（如静音、过渡段）α 值低
- 当信息量累积到一个完整 token 的量（阈值 β=1.0）时，就触发输出
- 这样自然地实现了从变长音频帧到定长文本 token 的软对齐

> CIF 的训练信号来自哪里？CIF 的长度预测通过 MAE Loss（Mean Absolute Error）监督——预测的输出长度应该等于真实文本的长度。α 权重则通过反向传播从最终的 CE Loss 间接学习。

### 4.4 GLM 训练策略详解

GLM（Glancing Language Model）解决了 NAR 模型训练和推理不一致的问题。

```
问题：
  训练时：Decoder 输入 = 声学嵌入 + 真实 Token Embedding（Teacher Forcing）
  推理时：Decoder 输入 = 声学嵌入（没有真实 Token，因为文本未知）

  这种不一致会导致模型在推理时性能下降。

GLM 的解决方案——渐进式训练：

训练初期（模型弱，预测准确率低）：
  采样比例高 → 大部分位置使用真实 Token
  Decoder输入: [今, 天, 天, 气, 真, 好]  ← 几乎全是真实 token
  模型先学会"在有提示的情况下预测"

训练中期（模型逐渐变强）：
  采样比例降低 → 随机 mask 一部分位置
  Decoder输入: [今, ?, 天, ?, 真, ?]    ← 一半被 mask
  模型学会"在部分提示下预测"

训练后期（模型强）：
  采样比例很低 → 大部分位置被 mask
  Decoder输入: [?, ?, ?, ?, ?, ?]        ← 几乎全被 mask
  模型学会"在没有提示的情况下独立预测"

采样比例的计算：
  ratio = 1 - accuracy  （accuracy 是模型当前的预测准确率）
  当模型准确率为 0.3 时，70% 的位置使用真实 token
  当模型准确率为 0.9 时，只有 10% 的位置使用真实 token
```

> GLM 的巧妙之处在于它是自适应的——模型越弱，给的提示越多；模型越强，给的提示越少。这样既保证了训练初期的稳定性，又让模型在训练后期逐渐适应推理时的条件。

### 4.5 Paraformer 的模型变体

FunASR 提供了多个 Paraformer 变体，适用于不同场景：

| 模型                      | 参数量   | 训练数据    | AISHELL-1 CER | 特点          |
| ----------------------- | ----- | ------- | ------------- | ----------- |
| paraformer-zh (large)   | ~220M | 60000h  | ~1.95%        | 效果最好，推荐离线使用 |
| paraformer-zh (标准)      | ~46M  | 60000h  | ~3.5%         | 资源受限场景      |
| paraformer-zh-streaming | ~46M  | 60000h  | ~5.0%         | 流式识别，低延迟    |
| paraformer-en           | ~220M | 50000h  | -             | 英文识别        |
| SenseVoice-small        | ~234M | 400000h | ~2.0%         | 多语言（50+语种）  |

#### Paraformer-large 详细参数

```
Encoder:
  类型: Conformer
  层数: 12
  d_model: 512
  注意力头数: 8
  FFN维度: 2048
  卷积核大小: 15
  Dropout: 0.1

Decoder:
  类型: Transformer (非自回归)
  层数: 6
  d_model: 512
  注意力头数: 8
  FFN维度: 2048

CIF:
  Conv1D 核大小: 3
  阈值 β: 1.0

词表:
  大小: 8404
  类型: 中文字符 + 常用子词
  特殊 token: <blank>, <sos>, <eos>, <unk>

输入:
  采样率: 16kHz
  特征: 80维 FBank
  CMVN: 全局归一化

训练:
  数据量: 60000+ 小时中文语音
  优化器: Adam
  学习率: warmup + inverse sqrt decay
  CTC权重 α: 0.3
```

### 4.6 Paraformer 推理数据流（逐步追踪）

以识别一段 3 秒的中文语音"今天天气真好"为例，追踪完整的数据流：

```
Step 1: 音频输入
  WAV: 3秒, 16kHz, 单声道 → 48000 个采样点
  数据: [48000] (float32)

Step 2: FBank 特征提取
  分帧: 帧长25ms(400点), 帧移10ms(160点)
  → (48000 - 400) / 160 + 1 ≈ 298 帧
  每帧 80 维 FBank
  → 特征矩阵: [1, 298, 80]

Step 3: CMVN 归一化
  使用预计算的全局均值和方差
  x_norm = (x - mean) / sqrt(var)
  → [1, 298, 80]  (shape不变，值被归一化)

Step 4: Conv2D Subsampling (4倍下采样)
  Conv2D(1, 256, kernel=3, stride=2): 298 → 149
  ReLU
  Conv2D(256, 256, kernel=3, stride=2): 149 → 74
  ReLU
  Linear(256×20, 512): 映射到 d_model
  → [1, 74, 512]

Step 5: Positional Encoding
  加入正弦位置编码
  → [1, 74, 512]  (shape不变)

Step 6: Conformer Encoder (12层)
  每层: ½FFN → MultiHead Attention → Conv → ½FFN → LayerNorm
  12层堆叠，特征被充分编码
  → [1, 74, 512]  (shape不变，但语义信息丰富)

Step 7: CIF 模块
  Conv1D + Sigmoid → 74 个 α 权重
  累积触发 → 预测输出长度 N=6（"今天天气真好"6个字）
  加权求和 → 声学嵌入: [1, 6, 512]

Step 8: Decoder (6层 Transformer)
  输入: 声学嵌入 [1, 6, 512]
  6层 Self-Attention + Cross-Attention + FFN
  注意：NAR 模式下 Self-Attention 没有 causal mask
  → [1, 6, 512]

Step 9: 输出投影
  Linear(512, 8404) → Softmax
  → [1, 6, 8404]
  Argmax → Token IDs: [156, 892, 892, 341, 567, 234]

Step 10: Token 解码
  Token IDs → 文本: "今天天气真好"
```

> 整个推理过程中，Decoder 只做了 1 次前向传播就输出了所有 6 个字。如果是自回归模型，需要做 6 次 Decoder 前向传播。这就是 Paraformer 速度快的根本原因。

## 5. 其他重要模型详解

### 5.1 FSMN-VAD（语音活动检测）

VAD（Voice Activity Detection）是语音处理流水线的第一步，负责检测音频中哪些部分包含语音。

```
FSMN-VAD 架构：

输入: 音频波形
    ↓
FBank 特征提取 (400维，注意这里用400维而非80维)
    ↓
┌────────────────────────────┐
│  FSMN 层 × 4               │
│  (前馈序列记忆网络)          │
│                             │
│  FSMN 的核心：记忆块         │
│  h_t = f(x_t + Σ a_i·h_{t-i})│
│  当前输出 = 当前输入 +       │
│            历史记忆的加权和   │
│                             │
│  类似简化版 LSTM，但更快     │
└────────────┬───────────────┘
             ↓
Linear → Sigmoid
             ↓
每帧的语音概率: [T, 1]  (0=静音, 1=语音)
             ↓
后处理：
  - 平滑：消除短暂的跳变
  - 阈值判断：概率 > 0.5 → 语音
  - 最小时长过滤：语音段 < 200ms 的丢弃
  - 合并：间隔 < 300ms 的相邻语音段合并
             ↓
输出: 语音段列表 [[start_ms, end_ms], ...]
```

FSMN 与 LSTM 的对比：

| 特性    | FSMN      | LSTM      |
| ----- | --------- | --------- |
| 结构    | 前馈 + 记忆块  | 循环 + 门控   |
| 并行性   | 可并行       | 不可并行      |
| 速度    | 快         | 慢         |
| 长距离依赖 | 有限（记忆块大小） | 理论上无限     |
| 适用场景  | VAD、关键词检测 | 语言模型、序列建模 |

> FSMN 非常适合 VAD 这种轻量级任务——不需要建模很长的依赖关系，但需要实时处理，速度要快。

### 5.2 CT-Punc（标点恢复模型）

标点恢复是 ASR 后处理的重要环节，将无标点的识别结果转为可读的文本。

```
CT-Punc 架构：

输入文本: "今天天气真好我们出去玩吧"
    ↓
Tokenizer → Token IDs
    ↓
┌────────────────────────────┐
│  Transformer Encoder       │
│  (类似 BERT 的结构)         │
│  12层, d_model=768         │
│                             │
│  每个 token 位置输出一个     │
│  上下文感知的表示向量        │
└────────────┬───────────────┘
             ↓
Linear(768, num_punc_classes) → Softmax
             ↓
每个字后面的标点预测:
  今→无  天→无  天→无  气→无  真→无  好→逗号
  我→无  们→无  出→无  去→无  玩→无  吧→句号
             ↓
输出: "今天天气真好，我们出去玩吧。"

标点类别（中文）:
  0: 无标点
  1: 逗号 ，
  2: 句号 。
  3: 问号 ？
  4: 感叹号 ！
```

### 5.3 时间戳模型

FunASR 支持字级别的时间戳输出，告诉你每个字在音频中的精确位置：

```python
# 使用时间戳
model = AutoModel(model="paraformer-zh")
result = model.generate(input="audio.wav")

# 输出示例
# {
#   "text": "今天天气真好",
#   "timestamp": [
#     [0, 380],      # "今" 0ms-380ms
#     [380, 580],    # "天" 380ms-580ms
#     [580, 820],    # "天" 580ms-820ms
#     [820, 1060],   # "气" 820ms-1060ms
#     [1060, 1340],  # "真" 1060ms-1340ms
#     [1340, 1680],  # "好" 1340ms-1680ms
#   ]
# }
```

时间戳的实现原理：

- Paraformer 的 CIF 模块天然提供了音频帧到文本 token 的对齐信息
- 每个 token 对应的 CIF 触发位置就是该 token 在音频中的大致时间
- 再通过 CTC 对齐进行精细化调整

### 5.4 热词（Hotword）功能

热词功能可以提升特定词汇的识别准确率，适用于专有名词、人名、产品名等场景：

```python
# 使用热词
model = AutoModel(model="paraformer-zh")

# 方式1: 字符串形式（空格分隔）
result = model.generate(
    input="audio.wav",
    hotword="达摩院 Paraformer FunASR 阿里云"
)

# 方式2: 列表形式
result = model.generate(
    input="audio.wav",
    hotword=["达摩院", "Paraformer", "FunASR", "阿里云"]
)
```

热词的工作原理：

```
普通识别: "语音识别" 可能被识别为 "语音时别"（同音字错误）

加入热词 "语音识别" 后:
  模型在解码时会给包含热词的路径额外加分
  相当于在语言模型层面提升了热词出现的先验概率
  → 正确识别为 "语音识别"
```

## 6. 实战使用指南

### 6.1 离线语音识别（非流式）

离线识别适用于已经录制好的音频文件，可以看到完整音频后再输出结果，准确率最高。

```python
from funasr import AutoModel

# 加载模型（首次运行会自动下载，约1-2GB）
model = AutoModel(model="paraformer-zh", device="cpu")

# === 基本用法 ===

# 识别单个文件
result = model.generate(input="test.wav")
print(result[0]["text"])
# 输出: "今天天气真好"

# 识别多个文件（批处理）
result = model.generate(input=["audio1.wav", "audio2.wav", "audio3.wav"])
for item in result:
    print(item["text"])

# 识别 URL 音频
result = model.generate(
    input="https://example.com/audio.wav"
)

# === 使用 numpy 数组输入 ===
import soundfile as sf
import numpy as np

speech, sr = sf.read("test.wav")
result = model.generate(input=speech, input_len=len(speech))
print(result[0]["text"])
```

### 6.2 流式语音识别

流式识别适用于实时场景（如语音助手、实时字幕），在音频还在输入时就逐步给出结果。

```python
import soundfile as sf
import numpy as np
from funasr import AutoModel

# 加载流式模型
model = AutoModel(model="paraformer-zh-streaming", device="cpu")

# 读取音频
speech, sample_rate = sf.read("test.wav")
if len(speech.shape) > 1:
    speech = speech[:, 0]  # 取单声道

# 模拟流式输入：将音频切成 chunk
chunk_ms = 600  # 每个 chunk 600ms
chunk_size = int(sample_rate * chunk_ms / 1000)
chunks = [speech[i:i+chunk_size] for i in range(0, len(speech), chunk_size)]

# 流式识别
cache = {}  # 维护流式状态
final_text = ""

for i, chunk in enumerate(chunks):
    is_final = (i == len(chunks) - 1)  # 最后一个 chunk

    result = model.generate(
        input=chunk,
        cache=cache,
        is_final=is_final,
        chunk_size=[5, 10, 5],  # [左context, 当前chunk, 右context] 单位：帧
    )

    if result and result[0]["text"]:
        text = result[0]["text"]
        final_text += text
        print(f"Chunk {i+1}: {text}")

print(f"最终结果: {final_text}")
```

流式识别的关键参数：

```
chunk_size = [5, 10, 5]  含义：
  5  = 左侧 context（看前面5帧 = 200ms 的历史信息）
  10 = 当前 chunk（处理10帧 = 400ms 的当前音频）
  5  = 右侧 context（等待后面5帧 = 200ms 的未来信息）

延迟 = 右侧 context × 帧移 = 5 × 40ms = 200ms
（Subsampling 4倍后，每帧对应 40ms）

实际端到端延迟 ≈ chunk时长 + 右侧context + 计算时间
                ≈ 400ms + 200ms + ~50ms ≈ 650ms
```

### 6.3 完整流水线（VAD + ASR + 标点）

这是生产环境中最常用的方式，一次调用完成所有处理：

```python
from funasr import AutoModel

# 方式1: 集成 Pipeline（推荐）
model = AutoModel(
    model="paraformer-zh",       # ASR 模型
    vad_model="fsmn-vad",        # VAD 模型
    punc_model="ct-punc",        # 标点恢复模型
    device="cpu",
)

result = model.generate(input="long_audio.wav")
print(result[0]["text"])
# 输出: "今天天气真好，我们出去玩吧。下午三点在公园门口见。"

# 方式2: 分步调用（更灵活，可以对中间结果做处理）

# Step 1: VAD
vad_model = AutoModel(model="fsmn-vad")
vad_result = vad_model.generate(input="long_audio.wav")
segments = vad_result[0]["value"]  # [[0, 3200], [4500, 8900], ...]
print(f"检测到 {len(segments)} 个语音段")

# Step 2: ASR
asr_model = AutoModel(model="paraformer-zh")
asr_result = asr_model.generate(input="long_audio.wav")
raw_text = asr_result[0]["text"]
print(f"原始识别: {raw_text}")

# Step 3: 标点恢复
punc_model = AutoModel(model="ct-punc")
punc_result = punc_model.generate(input=raw_text)
final_text = punc_result[0]["text"]
print(f"加标点后: {final_text}")
```

### 6.4 长音频处理

对于超长音频（几十分钟到几小时），VAD 的作用尤为重要：

```python
# 长音频处理流水线
model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
)

# FunASR 会自动：
# 1. 用 VAD 将长音频切成短语音段
# 2. 对每个语音段分别做 ASR
# 3. 合并所有识别结果
# 4. 对合并后的文本做标点恢复
result = model.generate(input="meeting_recording_2hours.wav")

# 输出包含完整的带标点文本
print(result[0]["text"])
```

> 长音频处理的注意事项：
> 
> - VAD 会自动过滤静音段，只对有语音的部分做识别
> - 每个语音段独立识别，段间没有上下文信息
> - 如果音频质量差（噪声大），VAD 可能会误切或漏切
> - 建议音频采样率为 16kHz，单声道

### 6.5 语音情感识别

```python
# 使用 Emotion2Vec 进行语音情感识别
model = AutoModel(model="iic/emotion2vec_plus_large", device="cpu")

result = model.generate(input="emotional_speech.wav")
print(result[0])
# 输出示例:
# {
#   "labels": ["happy", "neutral", "angry", "sad", "surprised", ...],
#   "scores": [0.85, 0.08, 0.03, 0.02, 0.01, ...],
# }
```

### 6.6 说话人验证

```python
# 使用 CAM++ 进行说话人验证
model = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common")

# 提取说话人嵌入向量
embedding1 = model.generate(input="speaker1_audio1.wav")
embedding2 = model.generate(input="speaker1_audio2.wav")
embedding3 = model.generate(input="speaker2_audio1.wav")

# 计算相似度（余弦相似度）
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 同一说话人 → 相似度高
sim_same = cosine_similarity(embedding1, embedding2)
# 不同说话人 → 相似度低
sim_diff = cosine_similarity(embedding1, embedding3)

print(f"同一说话人相似度: {sim_same:.4f}")  # 例: 0.92
print(f"不同说话人相似度: {sim_diff:.4f}")  # 例: 0.35
```

## 7. 流式 vs 离线：深入对比

### 7.1 架构差异

```
离线 Paraformer:
┌─────────────────────────────────────────────┐
│  完整音频 → Encoder (全局 Attention)          │
│                    → CIF → Decoder → 文本    │
│                                              │
│  Encoder 的每一帧都能看到整段音频的所有帧     │
│  信息最完整，效果最好                         │
└─────────────────────────────────────────────┘

流式 Paraformer:
┌─────────────────────────────────────────────┐
│  音频 chunk₁ → Encoder (局部 Attention)       │
│                       → CIF → Decoder → 部分文本│
│  音频 chunk₂ → Encoder (局部 Attention + cache)│
│                       → CIF → Decoder → 部分文本│
│  ...                                          │
│                                              │
│  Encoder 的每一帧只能看到当前 chunk           │
│  和有限的左右 context                         │
│  信息受限，效果略差，但延迟低                  │
└─────────────────────────────────────────────┘
```

### 7.2 性能对比

```
以 AISHELL-1 测试集为例：

模型                          CER(%)    RTF(CPU)    延迟
─────────────────────────────────────────────────────────
Paraformer-large (离线)       1.95      0.05        N/A（需完整音频）
Paraformer-zh-streaming       5.0       0.08        ~650ms
Conformer (WeNet, 离线)       4.5       0.15        N/A
Conformer (WeNet, 流式)       5.5       0.12        ~800ms
Whisper-large-v3 (离线)       3.0       0.80        N/A

RTF = 处理时间 / 音频时长
RTF < 1 表示实时（处理速度快于音频播放速度）
```

### 7.3 如何选择

```
选择离线模型的场景：
├── 音频已经录制完成（会议录音、播客、视频字幕）
├── 对准确率要求高
├── 不需要实时反馈
└── 批量处理大量音频文件

选择流式模型的场景：
├── 需要实时反馈（语音助手、实时字幕）
├── 音频是实时输入的（麦克风、电话）
├── 可以接受略低的准确率
└── 对延迟敏感（< 1秒）
```

## 8. 模型部署

### 8.1 ONNX 导出与推理

FunASR 支持将模型导出为 ONNX 格式，用于生产环境部署：

```python
# 方式1: 直接使用 FunASR 的 ONNX 推理
from funasr import AutoModel

# 指定使用 ONNX 后端（如果模型提供了 ONNX 版本）
model = AutoModel(
    model="paraformer-zh",
    device="cpu",
    # FunASR 会自动检测并使用 ONNX 模型（如果可用）
)

result = model.generate(input="test.wav")
```

```python
# 方式2: 手动使用 ONNX Runtime
import onnxruntime as ort
import numpy as np

# 加载 ONNX 模型
session = ort.InferenceSession("paraformer.onnx")

# 准备输入（需要自行做特征提取和预处理）
fbank_features = extract_fbank("test.wav")  # [1, T, 80]
feature_lengths = np.array([fbank_features.shape[1]], dtype=np.int32)

# 推理
outputs = session.run(
    None,
    {
        "speech": fbank_features,
        "speech_lengths": feature_lengths,
    }
)

# 解码输出
token_ids = outputs[0]  # [1, N]
text = decode_tokens(token_ids)
```

### 8.2 FunASR 服务端部署

FunASR 提供了开箱即用的服务端部署方案：

```bash
# 使用 Docker 一键部署（推荐）
# 拉取 FunASR 服务端镜像
docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-latest

# 启动服务
docker run -p 10095:10095 -it \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-latest
```

```python
# 客户端连接（WebSocket）
import asyncio
import websockets
import json

async def recognize(audio_path):
    uri = "ws://localhost:10095"
    async with websockets.connect(uri) as ws:
        # 发送配置
        config = {
            "mode": "offline",  # 或 "online" 流式
            "wav_name": "test",
            "is_speaking": True,
        }
        await ws.send(json.dumps(config))

        # 发送音频数据
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        await ws.send(audio_data)

        # 发送结束标志
        await ws.send(json.dumps({"is_speaking": False}))

        # 接收结果
        result = await ws.recv()
        return json.loads(result)

# 运行
result = asyncio.run(recognize("test.wav"))
print(result["text"])
```

### 8.3 部署架构参考

```
生产环境部署架构：

┌──────────┐     ┌──────────────────┐     ┌─────────────────┐
│ 客户端    │     │  负载均衡器       │     │  FunASR 服务集群  │
│          │     │  (Nginx/HAProxy) │     │                  │
│ 浏览器    │────→│                  │────→│  实例1 (GPU)     │
│ App      │←────│  WebSocket 代理  │←────│  实例2 (GPU)     │
│ 小程序    │     │                  │     │  实例3 (CPU)     │
└──────────┘     └──────────────────┘     └─────────────────┘
                                                   │
                                          ┌────────┴────────┐
                                          │  模型存储        │
                                          │  (共享文件系统)   │
                                          │  Paraformer      │
                                          │  FSMN-VAD        │
                                          │  CT-Punc         │
                                          └─────────────────┘

关键指标参考（单实例）：
├── CPU (8核): RTF ≈ 0.05, 并发 ≈ 20路
├── GPU (T4):  RTF ≈ 0.01, 并发 ≈ 100路
└── GPU (A10): RTF ≈ 0.005, 并发 ≈ 200路
```

### 8.4 模型量化

```python
# INT8 量化可以将模型大小减半，推理速度提升 1.5-2 倍
# FunASR 的 ONNX 模型通常已经提供了量化版本

# 使用量化模型
model = AutoModel(
    model="paraformer-zh",
    quantize=True,  # 使用量化版本（如果可用）
)
```

量化效果参考：

| 模型                    | 精度      | 大小     | CER   | RTF(CPU) |
| --------------------- | ------- | ------ | ----- | -------- |
| Paraformer-large FP32 | float32 | ~880MB | 1.95% | 0.05     |
| Paraformer-large INT8 | int8    | ~220MB | 2.0%  | 0.03     |
| Paraformer-large INT4 | int4    | ~110MB | 2.2%  | 0.02     |

> 量化带来的精度损失通常很小（CER 增加 0.05-0.2%），但模型大小和推理速度的改善非常显著。

### 8.5 FunASR 预训练模型全景

FunASR 在 ModelScope 和 Hugging Face 上开源了大量预训练模型，覆盖语音识别、语音检测、标点恢复、说话人识别、情感识别等多个任务。以下是完整的模型清单及其参数量信息。

> 模型来源：[FunASR Model Zoo](https://github.com/modelscope/FunASR)，数据基于官方公开信息整理。

#### 8.5.1 语音识别（ASR）模型

语音识别是 FunASR 的核心能力，提供了从轻量级到大规模、从单语到多语言的完整模型矩阵：

```
ASR 模型体系：

┌─────────────────────────────────────────────────────────────────┐
│                    按规模分层                                     │
│                                                                  │
│  超大规模（LLM 级别）                                             │
│  ├── Fun-ASR (7.7B)          ← 基于 LLM 的 ASR，效果最强         │
│  └── Fun-ASR-Nano (800M)     ← 轻量 LLM ASR，31 语种            │
│                                                                  │
│  大规模（工业级）                                                  │
│  ├── SenseVoice-Large (1587M) ← 多语言语音理解，50+ 语种          │
│  ├── Whisper-large-v3 (1550M) ← OpenAI 多语言模型                │
│  └── Whisper-large-v3-turbo (809M) ← Whisper 加速版              │
│                                                                  │
│  中等规模（推荐使用）                                              │
│  ├── Paraformer-zh (220M)     ← 中文离线识别首选                  │
│  ├── Paraformer-zh-streaming (220M) ← 中文流式识别                │
│  ├── Paraformer-en (220M)     ← 英文离线识别                     │
│  ├── Conformer-en (220M)      ← 英文 AR 模型                    │
│  └── SenseVoice-Small (234M)  ← 多语言理解，5 语种，速度极快      │
│                                                                  │
│  多模态                                                           │
│  ├── Qwen-Audio (8B)          ← 音频-文本多模态预训练             │
│  └── Qwen-Audio-Chat (8B)     ← 音频-文本多模态对话              │
└─────────────────────────────────────────────────────────────────┘
```

| 模型名称                    | 参数量   | 架构类型               | 解码方式 | 训练数据      | 支持语言      | 流式支持 | 核心特点                                       |
| ----------------------- | ----- | ------------------ | ---- | --------- | --------- | ---- | ------------------------------------------ |
| Fun-ASR                 | 7.7B  | LLM-based E2E      | 自回归  | 千万小时级     | 31 语种     | 支持   | 基于大语言模型，效果最强，支持方言和口音                       |
| Fun-ASR-Nano            | 800M  | LLM-based E2E      | 自回归  | 千万小时级     | 31 语种     | 支持   | Fun-ASR 的轻量版，7 大方言 + 26 地方口音               |
| Fun-ASR-MLT-Nano        | 800M  | LLM-based E2E      | 自回归  | 数十万小时     | 31 语种     | 支持   | 多语言版本，覆盖东南亚及欧洲语言                           |
| SenseVoice-Large        | 1587M | Autoregressive     | 自回归  | 400000h+  | 50+ 语种    | 不支持  | 多语言语音理解，ASR + LID + SER + AED              |
| SenseVoice-Small        | 234M  | Non-autoregressive | 非自回归 | 300000h   | 中/粤/英/日/韩 | 不支持  | 推理极快（10s 音频仅需 70ms），比 Whisper-Large 快 15 倍 |
| Paraformer-zh           | 220M  | Paraformer (NAR)   | 非自回归 | 60000h 中文 | 中文        | 不支持  | 中文离线识别首选，支持时间戳和热词                          |
| Paraformer-zh-streaming | 220M  | Paraformer (NAR)   | 非自回归 | 60000h 中文 | 中文        | 支持   | 中文流式识别，延迟约 650ms                           |
| Paraformer-en           | 220M  | Paraformer (NAR)   | 非自回归 | 50000h 英文 | 英文        | 不支持  | 英文离线识别                                     |
| Conformer-en            | 220M  | Conformer (AR)     | 自回归  | 50000h 英文 | 英文        | 不支持  | 英文自回归模型                                    |
| Whisper-large-v3        | 1550M | Whisper (AR)       | 自回归  | 多语言       | 99 语种     | 不支持  | OpenAI 多语言模型，支持语音翻译                        |
| Whisper-large-v3-turbo  | 809M  | Whisper (AR)       | 自回归  | 多语言       | 99 语种     | 不支持  | Whisper 加速版，速度更快                           |
| Qwen-Audio              | 8B    | 多模态 LLM            | 自回归  | 多语言       | 多语言       | 不支持  | 音频-文本多模态预训练模型                              |
| Qwen-Audio-Chat         | 8B    | 多模态 LLM            | 自回归  | 多语言       | 多语言       | 不支持  | 音频-文本多模态对话模型                               |

#### 8.5.2 语音活动检测（VAD）模型

| 模型名称     | 参数量  | 架构   | 训练数据     | 流式支持 | 核心特点             |
| -------- | ---- | ---- | -------- | ---- | ---------------- |
| FSMN-VAD | 0.4M | FSMN | 5000h 中英 | 支持   | 极轻量，适合实时场景，工业级质量 |

> FSMN-VAD 仅有 0.4M（40 万）参数，是整个流水线中最轻量的模型。它的设计目标是在极低计算开销下实现高精度的语音段检测，非常适合部署在边缘设备上。

#### 8.5.3 标点恢复模型

| 模型名称    | 参数量  | 架构                  | 训练数据     | 支持语言  | 核心特点                 |
| ------- | ---- | ------------------- | -------- | ----- | -------------------- |
| CT-Punc | 290M | Transformer Encoder | 1 亿句中英文本 | 中文、英文 | 基于 BERT 类架构，逐字预测标点类别 |

#### 8.5.4 时间戳预测模型

| 模型名称  | 参数量 | 训练数据     | 核心特点                 |
| ----- | --- | -------- | -------------------- |
| FA-zh | 38M | 5000h 中文 | 字级别时间戳对齐，基于 CTC 强制对齐 |

#### 8.5.5 关键词检测（KWS）模型

| 模型名称               | 参数量  | 架构   | 流式支持 | 核心特点            |
| ------------------ | ---- | ---- | ---- | --------------- |
| FSMN-KWS           | 0.7M | FSMN | 支持   | 轻量关键词检测，适合唤醒词场景 |
| FSMN-KWS-MT        | -    | FSMN | 支持   | 多任务版本           |
| SANM-KWS           | -    | SANM | 不支持  | 基于 SANM 架构      |
| SANM-KWS-Streaming | -    | SANM | 支持   | SANM 流式版本       |

#### 8.5.6 说话人识别模型

| 模型名称  | 参数量  | 训练数据  | 核心特点                  |
| ----- | ---- | ----- | --------------------- |
| CAM++ | 7.2M | 5000h | 说话人验证与说话人日志，提取说话人嵌入向量 |

#### 8.5.7 语音情感识别模型

emotion2vec 系列是 FunASR 生态中的语音情感识别基础模型，通过自监督预训练和数据迭代微调，实现了跨语言、跨场景的通用情感识别能力：

| 模型名称               | 参数量   | 微调数据        | 情感类别 | 核心特点                     |
| ------------------ | ----- | ----------- | ---- | ------------------------ |
| emotion2vec        | -     | 无（自监督预训练）   | 通用表征 | 基础预训练模型，提取情感表征向量         |
| emotion2vec+ seed  | -     | 201h（学术数据）  | 9 类  | 学术数据微调的种子模型              |
| emotion2vec+ base  | ~90M  | 4788h（伪标签）  | 9 类  | 大规模伪标签数据微调，base 尺寸       |
| emotion2vec+ large | ~300M | 42526h（伪标签） | 9 类  | 大规模伪标签数据微调，large 尺寸，效果最好 |

> emotion2vec+ 的 9 个情感类别：angry（愤怒）、disgusted（厌恶）、fearful（恐惧）、happy（开心）、neutral（中性）、other（其他）、sad（悲伤）、surprised（惊讶）、unknown（未知）。

#### 8.5.8 模型选型指南

```
按场景选择模型：

中文离线识别（效果优先）：
  → Paraformer-zh (220M) + FSMN-VAD (0.4M) + CT-Punc (290M)
  总参数量 ≈ 510M，中文效果最优

中文流式识别（实时场景）：
  → Paraformer-zh-streaming (220M) + FSMN-VAD (0.4M)
  延迟约 650ms

多语言识别（5 语种，速度优先）：
  → SenseVoice-Small (234M)
  推理速度极快，支持中/粤/英/日/韩

多语言识别（50+ 语种，精度优先）：
  → SenseVoice-Large (1587M) 或 Whisper-large-v3 (1550M)

最强中文识别（不限资源）：
  → Fun-ASR-Nano (800M) + FSMN-VAD (0.4M)
  支持方言、口音、歌词识别

语音情感分析：
  → emotion2vec+ large (300M)

说话人识别/日志：
  → CAM++ (7.2M)

完整语音理解（ASR + 情感 + 事件检测）：
  → SenseVoice-Small (234M) 或 SenseVoice-Large (1587M)
```

## 9. 训练自定义模型

[Paraformer 本地微调通用手册（含 XTTS 语音数据自建）_paraformer微调-CSDN博客](https://blog.csdn.net/m0_62505792/article/details/151019249#:~:text=%E9%80%82%E7%94%A8%E8%8C%83%E5%9B%B4%EF%BC%9A%E4%B8%AD%E6%96%87%E4%B8%BA%E4%B8%BB%EF%BC%8C%E5%85%BC%E9%A1%BE%E5%A4%9A%E8%AF%AD%2F%E6%96%B9%E8%A8%80%EF%BC%9B%E9%9F%B3%E9%A2%91%2016%20kHz%2F16-bit%2F%E5%8D%95%E5%A3%B0%E9%81%93%EF%BC%9BGPU%20%E2%89%A5%2012GB%EF%BC%88%E4%BD%8E%E6%98%BE%E5%AD%98%E9%85%8D%E6%96%B9%E5%8F%AF%208GB%20%E8%B5%B7%E6%AD%A5%EF%BC%89%E3%80%82%201.,CUDA%20%E5%AF%B9%E9%BD%90%E3%80%82%20%E5%AD%98%E5%82%A8%E5%BB%BA%E8%AE%AE%20%E2%89%A5%20200GB%E3%80%82%20%E6%8F%90%E7%A4%BA%EF%BC%9A%E4%B8%8D%E5%90%8C%20CUDA%2FPyTorch%20%E7%89%88%E6%9C%AC%E9%9C%80%E4%B8%A5%E6%A0%BC%E5%8C%B9%E9%85%8D%E3%80%82)

### 9.1 使用 FunASR 微调

如果预训练模型在你的特定领域效果不够好，可以用自己的数据进行微调：

```python
# FunASR 微调示例（简化版）
from funasr import AutoModel

# 加载预训练模型
model = AutoModel(
    model="paraformer-zh",
    device="cuda",
)

# 准备训练数据
# FunASR 支持多种数据格式，最简单的是 jsonl 格式：
# {"audio": "path/to/audio1.wav", "text": "对应的文本标注"}
# {"audio": "path/to/audio2.wav", "text": "对应的文本标注"}

# 微调
model.finetune(
    train_data="train.jsonl",
    dev_data="dev.jsonl",
    output_dir="exp/finetune",
    max_epochs=10,
    batch_size=16,
    learning_rate=1e-4,
)
```

### 9.2 数据准备

FunASR 支持多种数据格式：

```
格式1: JSONL（推荐）
每行一个 JSON 对象：
{"key": "utt_001", "source": "/data/audio/001.wav", "target": "今天天气真好"}
{"key": "utt_002", "source": "/data/audio/002.wav", "target": "我想订一张机票"}

格式2: Kaldi 风格
wav.scp:
  utt_001 /data/audio/001.wav
  utt_002 /data/audio/002.wav

text:
  utt_001 今天天气真好
  utt_002 我想订一张机票

格式3: 音频文件 + 同名文本文件
/data/
├── audio/
│   ├── 001.wav
│   └── 002.wav
└── text/
    ├── 001.txt  (内容: 今天天气真好)
    └── 002.txt  (内容: 我想订一张机票)
```

### 9.3 数据质量要求

```
音频要求：
├── 采样率: 16kHz（必须）
├── 位深: 16bit
├── 声道: 单声道
├── 格式: WAV（推荐）/ MP3 / FLAC
├── 时长: 1-30秒/条（推荐 3-15秒）
└── 信噪比: > 15dB（越高越好）

文本要求：
├── 与音频内容完全一致
├── 统一使用简体中文
├── 去除标点符号（或统一标点格式）
├── 数字统一格式（"100" 或 "一百"，不要混用）
└── 去除说话人标记、噪声标记等非文本内容

数据量参考：
├── 微调: 10-100 小时（在预训练模型基础上）
├── 从头训练（小模型）: 500-1000 小时
└── 从头训练（大模型）: 5000+ 小时
```

### 9.4 训练配置参考

```yaml
# ============================================================
# FunASR Paraformer 完整训练配置 (config.yaml)
# ============================================================

# ────────────────────────────────────────────────────────────
# 1. 模型总体配置
# ────────────────────────────────────────────────────────────

# 模型类型：指定使用哪种 ASR 模型架构
# 可选值: paraformer / conformer / transformer / branchformer 等
model: paraformer

# 模型级别的全局配置
model_conf:
    # CTC 损失的权重，与 Attention(CE) 损失配合使用
    # 最终 Loss = ctc_weight × CTC_Loss + (1 - ctc_weight) × CE_Loss + MAE_Loss
    # 值越大，CTC 分支的影响越大；Paraformer 推荐 0.3
    ctc_weight: 0.3

    # CIF 长度预测损失（MAE Loss）的权重
    # 用于监督 CIF 模块预测的输出序列长度与真实文本长度一致
    predictor_weight: 1.0

    # GLM（Glancing Language Model）采样比例的衰减系数
    # 控制训练过程中 Decoder 输入从"真实 token"过渡到"无提示"的速度
    # 值越小，过渡越快；推荐 1.0
    sampling_ratio: 1.0

    # 标签平滑系数：防止模型对预测结果过度自信
    # 将 one-hot 标签中的概率从 1.0 分散一部分给其他类别
    # 0.0 = 不使用标签平滑，0.1 = 常用值
    lsm_weight: 0.1

    # 输出序列的最大长度（token 数），超过此长度的样本会被丢弃
    # 防止异常长文本导致显存溢出
    length_normalized_loss: true

# ────────────────────────────────────────────────────────────
# 2. 前端（Frontend）配置 —— 音频特征提取
# ────────────────────────────────────────────────────────────

frontend: wav_frontend
frontend_conf:
    # 音频采样率（Hz），FunASR 的预训练模型统一使用 16kHz
    fs: 16000

    # FBank 特征的维度（滤波器组数量）
    # 80 是 ASR 领域的标准配置，与预训练模型保持一致
    n_mels: 80

    # 分帧窗口长度（毫秒）
    # 25ms 是语音处理的经典参数，一帧包含 400 个采样点（16kHz × 0.025s）
    frame_length: 25

    # 帧移（毫秒），相邻两帧的起始位置间隔
    # 10ms 帧移意味着每秒产生 100 帧特征
    frame_shift: 10

    # 是否对提取的特征做 CMVN（倒谱均值方差归一化）
    # 消除不同录音条件（麦克风、环境）带来的特征偏移
    lfr_m: 1          # 低帧率（Low Frame Rate）合并帧数，1 表示不合并
    lfr_n: 1          # 低帧率跳帧数，1 表示不跳帧

    # CMVN 统计文件路径，包含预计算的全局特征均值和方差
    # 训练前需要在全部训练数据上统计一次
    cmvn_file: null    # null 表示不使用外部 CMVN 文件（模型内部处理）

# ────────────────────────────────────────────────────────────
# 3. Encoder 配置 —— Conformer 编码器
# ────────────────────────────────────────────────────────────

encoder: conformer
encoder_conf:
    # 编码器输出维度（d_model），也是整个模型的隐藏层维度
    # 512 是 Paraformer-large 的标准配置；小模型可用 256
    output_size: 512

    # Multi-Head Self-Attention 的注意力头数
    # 每个头的维度 = output_size / attention_heads = 512 / 8 = 64
    attention_heads: 8

    # Feed-Forward Network（FFN）的中间层维度
    # 通常为 output_size 的 4 倍，是 Transformer 的标准设计
    linear_units: 2048

    # Conformer Block 的堆叠层数
    # 层数越多，模型容量越大，但计算量也越大
    # Paraformer-large: 12 层，Paraformer-small: 6 层
    num_blocks: 12

    # Dropout 比率：训练时随机丢弃神经元的概率
    # 防止过拟合；数据量大时可适当降低（如 0.05）
    dropout_rate: 0.1

    # Positional Encoding 的 Dropout
    # 对位置编码单独设置 Dropout，增强模型对位置信息的鲁棒性
    positional_dropout_rate: 0.1

    # Self-Attention 权重的 Dropout
    # 防止注意力权重过度集中在少数位置上
    attention_dropout_rate: 0.1

    # 输入层类型：将 FBank 特征映射到 d_model 维度并做下采样
    # conv2d: 使用两层 Conv2D 做 4 倍下采样（最常用）
    # conv2d6: 6 倍下采样（更快但信息损失更多）
    # conv2d8: 8 倍下采样
    # linear: 线性映射，不做下采样
    input_layer: conv2d

    # 是否在每个子层之前做 LayerNorm（Pre-Norm）
    # true = Pre-Norm（训练更稳定，Conformer 标准配置）
    # false = Post-Norm（原始 Transformer 的方式）
    normalize_before: true

    # Conformer 卷积模块的卷积核大小
    # 控制局部上下文的感受野范围；15 是常用值
    # 感受野 = (kernel_size - 1) × 下采样倍数 × 帧移 = 14 × 4 × 10ms = 560ms
    cnn_module_kernel: 15

    # 位置编码类型
    # rel_pos: 相对位置编码（Conformer 标准配置，效果最好）
    # abs_pos: 绝对正弦位置编码（原始 Transformer）
    # rope: 旋转位置编码（RoPE）
    pos_enc_layer_type: rel_pos

    # Self-Attention 类型
    # rel_selfattn: 相对位置自注意力（配合 rel_pos 使用）
    # selfattn: 标准自注意力
    selfattn_layer_type: rel_selfattn

    # 激活函数类型
    # swish: Conformer 论文推荐，效果优于 relu
    # relu / gelu: 备选
    activation_type: swish

    # Macaron 风格的 FFN：Conformer 的标志性设计
    # true = 使用两个半步 FFN 夹住 Attention 和 Conv（½FFN → Attn → Conv → ½FFN）
    # false = 只使用一个 FFN（标准 Transformer 风格）
    use_macaron_style_in_conformer: true

    # 是否使用卷积模块（Conformer 的核心组件之一）
    # true = 完整 Conformer（Attn + Conv）
    # false = 退化为标准 Transformer（只有 Attn）
    use_cnn_module: true

# ────────────────────────────────────────────────────────────
# 4. CIF Predictor 配置 —— 连续积分触发模块
# ────────────────────────────────────────────────────────────

predictor: cif_predictor
predictor_conf:
    # CIF 内部 1D 卷积的核大小
    # 用于从 Encoder 输出计算每帧的发射权重 α
    # 3 表示每次看当前帧及左右各 1 帧来预测 α
    cnn_kernel_size: 3

    # CIF 的触发阈值 β
    # 当累积的 α 权重达到此阈值时，触发一次输出
    # 1.0 是标准配置；调小会增加输出长度，调大会减少
    threshold: 1.0

    # 是否对 α 权重做 L2 正则化
    # 防止 α 值过大或过小，使触发行为更稳定
    l2_regularize: 0.0001

    # 尾部处理策略：处理音频末尾累积不足阈值的残余权重
    # true = 强制在末尾触发一次输出（避免丢失最后几个字）
    tail_handling: true

# ────────────────────────────────────────────────────────────
# 5. Decoder 配置 —— 非自回归 Transformer 解码器
# ────────────────────────────────────────────────────────────

decoder: paraformer_decoder
decoder_conf:
    # 注意力头数（与 Encoder 保持一致或独立设置）
    attention_heads: 8

    # FFN 中间层维度
    linear_units: 2048

    # Decoder 层数
    # 通常比 Encoder 少（Encoder 12 层，Decoder 6 层）
    # 因为 NAR 解码不需要像 AR 那样深的 Decoder
    num_blocks: 6

    # Dropout 比率
    dropout_rate: 0.1

    # Self-Attention Dropout
    self_attention_dropout_rate: 0.1

    # Cross-Attention Dropout（Decoder 对 Encoder 输出的注意力）
    src_attention_dropout_rate: 0.1

    # 是否在子层之前做 LayerNorm（与 Encoder 保持一致）
    normalize_before: true

    # 是否拼接 Encoder 输出的注意力上下文
    # Paraformer 特有：将 Cross-Attention 的输出与 Self-Attention 拼接
    concat_after: false

# ────────────────────────────────────────────────────────────
# 6. 训练流程配置
# ────────────────────────────────────────────────────────────

train_conf:
    # 最大训练轮数
    # 微调时通常 10-30 轮即可；从头训练可能需要 100+ 轮
    max_epoch: 100

    # 批处理类型
    # dynamic: 动态 batch，按帧数凑 batch（推荐，GPU 利用率高）
    # static: 固定 batch_size
    batch_type: dynamic

    # 动态 batch 时，每个 batch 的最大总帧数
    # 值越大，GPU 利用率越高，但显存占用也越大
    # 根据 GPU 显存调整：V100(32GB) ≈ 20000, A100(80GB) ≈ 40000
    max_frames_in_batch: 12000

    # 静态 batch 时的 batch size（batch_type=static 时生效）
    batch_size: 32

    # 梯度累积步数：每 N 个 batch 更新一次参数
    # 等效 batch_size = batch_size × accum_grad
    # 显存不够时可增大此值来模拟更大的 batch
    accum_grad: 1

    # 梯度裁剪阈值：梯度的 L2 范数超过此值时进行裁剪
    # 防止梯度爆炸导致训练不稳定；5.0 是常用值
    grad_clip: 5.0

    # 日志打印间隔（每 N 个 batch 打印一次训练信息）
    log_interval: 100

    # 是否使用 AMP（自动混合精度训练）
    # true = 使用 FP16 加速训练，显存占用减半，速度提升 1.5-2 倍
    # 需要 GPU 支持（Volta 架构及以上，如 V100/A100/RTX 30xx）
    use_amp: false

    # AMP 的数据类型
    # float16: 标准半精度（大多数 GPU）
    # bfloat16: Brain Float 16（A100/H100 推荐，数值更稳定）
    amp_dtype: float16

    # 多少轮保存一次 checkpoint
    save_checkpoint_interval: 1

    # 最多保留多少个 checkpoint（节省磁盘空间）
    keep_nbest_models: 10

    # 用于选择最佳模型的指标
    # acc: 准确率（越高越好）
    # loss: 损失值（越低越好）
    best_model_criterion:
        - - valid
          - acc
          - max

    # 早停（Early Stopping）：验证集指标连续 N 轮不提升时停止训练
    # 防止过拟合，节省训练时间
    patience: 10

    # 验证间隔：每 N 轮在验证集上评估一次
    val_interval: 1

    # 是否在训练开始前先做一次验证（检查数据和模型是否正常）
    validate_first: true

    # 随机种子：保证实验可复现
    seed: 42

    # 数据加载的 worker 数量（多进程预加载数据）
    # 推荐设为 CPU 核心数的一半
    num_workers: 4

    # 是否将数据固定在 GPU 内存中（加速数据传输）
    pin_memory: true

# ────────────────────────────────────────────────────────────
# 7. 优化器配置
# ────────────────────────────────────────────────────────────

# 优化器类型
# adam: 标准 Adam，最常用
# adamw: Adam + 权重衰减解耦（推荐，正则化效果更好）
optim: adam
optim_conf:
    # 学习率：训练中最关键的超参数
    # 从头训练: 0.001（配合 warmup）
    # 微调: 0.0001 ~ 0.00005（比从头训练小 10-20 倍）
    lr: 0.001

    # 权重衰减（L2 正则化）：防止模型参数过大导致过拟合
    # 0.01 是 AdamW 的推荐值；纯 Adam 可设为 0.0
    weight_decay: 0.01

    # Adam 的一阶矩估计衰减率
    # 控制梯度均值的指数移动平均速度；0.9 是标准值
    betas: [0.9, 0.98]

    # 数值稳定性常数，防止除零
    eps: 0.000000001  # 1e-9

# ────────────────────────────────────────────────────────────
# 8. 学习率调度器配置
# ────────────────────────────────────────────────────────────

# 调度器类型
# warmuplr: 线性 warmup + 逆平方根衰减（Transformer 经典策略）
# cosine: 余弦退火（适合微调）
# noam: Noam 调度（原始 Transformer 论文的策略）
# reduce_on_plateau: 验证集指标停滞时降低学习率
scheduler: warmuplr
scheduler_conf:
    # Warmup 步数：学习率从 0 线性增长到 lr 的步数
    # 训练初期模型参数随机，大学习率会导致不稳定
    # warmup 让学习率缓慢上升，等模型"热身"后再用完整学习率
    # 推荐值: 总训练步数的 5%-10%
    warmup_steps: 25000

    # Warmup 结束后的学习率衰减方式
    # inv_sqrt: 逆平方根衰减 lr = lr_peak / sqrt(step)（推荐）
    # linear: 线性衰减到 0
    # constant: warmup 后保持不变
    decay_type: inv_sqrt

    # 最终学习率（衰减的下限）
    # 防止学习率衰减到过小导致训练停滞
    final_lr: 0.00001  # 1e-5

# ────────────────────────────────────────────────────────────
# 9. 数据增强配置
# ────────────────────────────────────────────────────────────

dataset_conf:
    # ── 语速扰动（Speed Perturbation）──
    # 将音频以不同速度播放，模拟不同语速的说话人
    # [0.9, 1.0, 1.1] 表示分别以 0.9x、1.0x、1.1x 速度生成 3 份数据
    # 等效于将训练数据扩充 3 倍
    speed_perturb: [0.9, 1.0, 1.1]

    # ── SpecAugment（频谱增强）──
    # 在 FBank 特征上随机遮盖时间和频率区域
    # 迫使模型不依赖局部特征，提升泛化能力
    # 这是目前 ASR 训练中最重要的数据增强手段
    spec_aug: true
    spec_aug_conf:
        # 频率遮盖的宽度范围（FBank 维度方向）
        # [0, 30] 表示每次随机遮盖 0~30 个频率通道（共 80 个）
        freq_mask_width_range: [0, 30]

        # 时间遮盖的宽度范围（帧方向）
        # [0, 40] 表示每次随机遮盖 0~40 帧（每帧 10ms，即 0~400ms）
        time_mask_width_range: [0, 40]

        # 频率遮盖的次数：对同一条样本施加几次频率遮盖
        num_freq_mask: 2

        # 时间遮盖的次数：对同一条样本施加几次时间遮盖
        num_time_mask: 2

    # ── 音量扰动（Volume Perturbation）──
    # 随机调整音频音量，模拟不同录音距离和设备
    volume_perturb: true
    volume_perturb_conf:
        # 音量缩放范围（倍数）
        # [0.5, 2.0] 表示音量在 0.5x ~ 2.0x 之间随机缩放
        scale_range: [0.5, 2.0]

    # ── 噪声注入（Noise Augmentation）──
    # 将背景噪声叠加到干净语音上，提升模型的抗噪能力
    noise_aug: false           # 是否启用噪声注入
    noise_aug_conf:
        # 噪声文件列表（每行一个噪声音频路径）
        noise_scp: null        # 例: "data/noise/noise.scp"
        # 信噪比范围（dB）：叠加噪声后的信噪比在此范围内随机选取
        snr_range: [5, 30]     # 5dB（噪声较大）~ 30dB（几乎无噪声）

    # ── 混响增强（Reverberation Augmentation）──
    # 模拟不同房间的混响效果
    rir_aug: false             # 是否启用混响增强
    rir_aug_conf:
        # 房间冲激响应（RIR）文件列表
        rir_scp: null          # 例: "data/rir/rir.scp"

    # ── 数据过滤 ──
    # 过滤掉过短或过长的样本，避免影响训练稳定性
    min_duration: 0.5          # 最短音频时长（秒），短于此值的样本被丢弃
    max_duration: 30.0         # 最长音频时长（秒），长于此值的样本被丢弃
    min_token_length: 1        # 最短文本长度（token 数）
    max_token_length: 200      # 最长文本长度（token 数）

    # ── 数据排序与分桶 ──
    # 将相似长度的样本放在同一个 batch 中，减少 padding，提升 GPU 利用率
    sort_by_length: true       # 是否按音频长度排序
    shuffle: true              # 是否在每个 epoch 开始时打乱数据

# ────────────────────────────────────────────────────────────
# 10. 分布式训练配置（多 GPU）
# ────────────────────────────────────────────────────────────

dist_conf:
    # 分布式后端
    # nccl: NVIDIA GPU 推荐（最快）
    # gloo: CPU 或跨节点通信的备选
    dist_backend: nccl

    # 分布式初始化方法
    dist_init_method: env://

    # 是否使用 DistributedDataParallel
    distributed: true

# ────────────────────────────────────────────────────────────
# 11. 词表与 Tokenizer 配置
# ────────────────────────────────────────────────────────────

tokenizer: char_tokenizer
tokenizer_conf:
    # 词表文件路径
    # Paraformer-zh 使用 8404 大小的中文字符词表
    token_list: null           # null 表示使用模型自带词表

    # 特殊 token 定义
    # <blank>: CTC 的空白符（ID=0）
    # <sos>/<eos>: 序列起止符
    # <unk>: 未知字符（不在词表中的字映射到此 token）
    unk_symbol: "<unk>"
    sos_symbol: "<sos>"
    eos_symbol: "<eos>"

# ────────────────────────────────────────────────────────────
# 12. 推理/解码配置（训练配置文件中可一并指定）
# ────────────────────────────────────────────────────────────

decode_conf:
    # 解码时的 batch size
    batch_size: 1

    # 是否使用 CTC 前缀分数辅助解码（Paraformer 通常不需要）
    ctc_weight: 0.0

    # Beam size（NAR 模型通常不需要 beam search，设为 1 即可）
    beam_size: 1

    # 是否输出时间戳
    output_timestamp: true
```

> 配置调优建议：
> 
> - 微调时：降低 `lr`（0.0001）、减少 `max_epoch`（10-30）、关闭 `speed_perturb`（数据量已够）
> - 显存不足时：减小 `max_frames_in_batch`、增大 `accum_grad`、开启 `use_amp`
> - 数据量少时：增大 `dropout_rate`（0.2-0.3）、开启所有数据增强、增大 `weight_decay`
> - 数据量大时：减小 `dropout_rate`（0.05）、增大 `max_frames_in_batch`、增加 `num_blocks`

## 10. 常见问题与调优技巧

### 10.1 识别效果不好

```
问题排查清单：

1. 音频质量问题
   ├── 采样率不是 16kHz → 重采样: sox input.wav -r 16000 output.wav
   ├── 多声道 → 转单声道: sox input.wav -c 1 output.wav
   ├── 噪声大 → 考虑前端降噪处理
   └── 音量过低/过高 → 音量归一化

2. 模型选择问题
   ├── 中文识别 → 使用 paraformer-zh（不要用英文模型）
   ├── 英文识别 → 使用 paraformer-en 或 whisper
   ├── 长音频 → 必须加 VAD（fsmn-vad）
   └── 特定领域 → 考虑微调

3. 热词问题
   ├── 专有名词识别错误 → 添加热词
   ├── 热词太多 → 控制在 100 个以内
   └── 热词格式错误 → 检查是否正确传入

4. 后处理问题
   ├── 没有标点 → 加上 ct-punc 模型
   ├── 数字格式不对 → 使用 ITN（逆文本归一化）
   └── 繁简混用 → 统一转简体
```

### 10.2 推理速度慢

```
优化方案（按效果排序）：

1. 使用 GPU
   model = AutoModel(model="paraformer-zh", device="cuda")
   → 速度提升 5-10 倍

2. 使用量化模型
   → 速度提升 1.5-2 倍，精度损失极小

3. 使用 ONNX Runtime
   → CPU 上速度提升 2-3 倍

4. 批处理
   result = model.generate(input=["a1.wav", "a2.wav", ...])
   → 多条音频一起处理，GPU 利用率更高

5. 减小模型
   paraformer-zh-large (220M) → paraformer-zh (46M)
   → 速度提升 3-4 倍，精度略降

6. 调整 VAD 参数
   → 减少不必要的语音段，减少 ASR 处理量
```

### 10.3 内存不足

```
解决方案：

1. 使用 CPU 推理（GPU 显存不足时）
   model = AutoModel(model="paraformer-zh", device="cpu")

2. 减小 batch_size
   model = AutoModel(model="paraformer-zh", batch_size=1)

3. 使用更小的模型
   paraformer-zh (46M) 代替 paraformer-zh-large (220M)

4. 使用量化模型
   INT8 量化可以将显存占用减半

5. 长音频分段处理
   使用 VAD 切分后逐段处理，避免一次性加载整段音频
```

### 10.4 常见错误与解决

```python
# 错误1: 模型下载失败
# 原因: 网络问题或 ModelScope 服务不可用
# 解决: 设置镜像或手动下载
import os
os.environ["MODELSCOPE_CACHE"] = "/your/cache/path"
# 或使用代理

# 错误2: 音频格式不支持
# 原因: 缺少音频解码库
# 解决:
# pip install soundfile pydub ffmpeg-python

# 错误3: CUDA out of memory
# 原因: GPU 显存不足
# 解决: 减小 batch_size 或使用 CPU
model = AutoModel(model="paraformer-zh", device="cpu")

# 错误4: 识别结果为空
# 原因: 音频中没有语音（全是静音或噪声）
# 解决: 检查音频内容，确认有语音
import soundfile as sf
speech, sr = sf.read("test.wav")
print(f"采样率: {sr}, 时长: {len(speech)/sr:.2f}秒")
print(f"最大振幅: {abs(speech).max():.4f}")  # 如果接近0说明是静音
```

## 11. FunASR 与 Kaldi 的关系

### 11.1 技术传承

```
Kaldi 对 FunASR 的影响：

数据格式:
  Kaldi 的 wav.scp / text / utt2spk 格式 → FunASR 也支持
  Kaldi 的 ark/scp 特征格式 → FunASR 可以读取

特征提取:
  Kaldi 的 FBank 实现 → FunASR 使用兼容的 FBank 参数
  Kaldi 的 CMVN → FunASR 同样使用全局 CMVN

训练范式:
  Kaldi 的 GMM-HMM 对齐 → 不再需要（端到端模型自动对齐）
  Kaldi 的 WFST 解码 → 被 CTC/Attention 解码替代
  Kaldi 的 LF-MMI 训练 → 被 CTC + CE Loss 替代

概念延续:
  Kaldi 的声学模型 → Paraformer 的 Encoder
  Kaldi 的语言模型 → Paraformer 的 Decoder（隐式语言模型）
  Kaldi 的发音词典 → 不再需要（使用字符/BPE 词表）
  Kaldi 的 HCLG 解码图 → 不再需要（端到端解码）
```

### 11.2 核心差异

| 方面   | Kaldi                      | FunASR                    |
| ---- | -------------------------- | ------------------------- |
| 架构理念 | 模块化（AM + LM + 词典 + 解码器）    | 端到端（单一模型）                 |
| 声学模型 | GMM-HMM / DNN-HMM          | Conformer (Paraformer)    |
| 解码方式 | WFST Viterbi 搜索            | CIF + NAR 并行解码            |
| 训练流程 | 多阶段渐进式（mono→tri→sat→chain） | 一步到位（端到端训练）               |
| 对齐需求 | 需要强制对齐（每帧对应音素状态）           | CIF 自动学习软对齐               |
| 发音词典 | 必需（词→音素映射）                 | 不需要（字符/BPE 词表）            |
| 语言模型 | 外部 n-gram LM               | 内置于 Decoder               |
| 编程语言 | C++ + Shell                | Python + PyTorch          |
| 上手难度 | 高（需要理解大量概念）                | 低（几行代码即可使用）               |
| 可定制性 | 极高（每个模块可独立调整）              | 中等（主要通过微调）                |
| 效果   | CER ~8%（Chain TDNN）        | CER ~2%（Paraformer-large） |

### 11.3 从 Kaldi 迁移到 FunASR

如果你之前使用 Kaldi，迁移到 FunASR 的关键变化：

```
你不再需要：
├── 编译 C++ 源码
├── 准备发音词典（lexicon.txt）
├── 训练 n-gram 语言模型
├── 构建 HCLG 解码图
├── 多阶段渐进式训练（mono → tri → sat → chain）
├── 手动做强制对齐
└── 编写复杂的 Shell 脚本

你仍然需要：
├── 准备高质量的音频和文本标注数据
├── 理解 FBank 特征提取（参数基本一致）
├── 理解 CMVN 归一化
└── 数据格式转换（Kaldi 格式 → FunASR 格式）

新增需要了解的：
├── PyTorch 基础
├── Transformer / Conformer 架构
├── CTC / Attention 解码
├── BPE / SentencePiece 分词
└── 模型微调技巧
```

## 12. 进阶话题

### 12.1 Paraformer 的负样本采样

Paraformer 在训练时使用了负样本采样策略来增强模型的纠错能力：

```
正常训练：
  输入: 声学嵌入（对应"今天天气真好"）
  目标: [今, 天, 天, 气, 真, 好]

负样本训练：
  故意将部分 token 替换为错误的 token
  输入: 声学嵌入 + 部分错误 token
  目标: 仍然是 [今, 天, 天, 气, 真, 好]

  例如：
  输入: [金, 天, 填, 气, 真, 号]  ← 故意引入错误
  目标: [今, 天, 天, 气, 真, 好]  ← 模型需要纠正错误

效果：
  模型不仅学会了"从声学特征预测文本"
  还学会了"发现并纠正错误"
  这使得 NAR 解码的效果更接近 AR 解码
```

### 12.2 SenseVoice：多语言语音理解

SenseVoice 是 FunASR 最新推出的多语言语音理解模型，不仅能做语音识别，还能做语音情感识别、音频事件检测等：

```python
from funasr import AutoModel

model = AutoModel(model="iic/SenseVoiceSmall", device="cpu")

result = model.generate(input="test.wav")
# 输出可能包含：
# - 语音识别结果
# - 语言识别（中文/英文/日文/韩文/粤语等）
# - 情感标签（开心/悲伤/愤怒等）
# - 音频事件（笑声/掌声/音乐等）
```

SenseVoice 的特点：

- 支持 50+ 种语言
- 训练数据超过 40 万小时
- 推理速度极快（比 Whisper-large 快 15 倍）
- 同时输出语音识别、语言识别、情感识别结果

### 12.3 FunASR 的未来方向

```
FunASR 的发展趋势：

模型层面：
├── 更大规模的预训练（百万小时级别）
├── 多模态融合（语音 + 文本 + 视觉）
├── 更强的多语言/多方言支持
└── 与大语言模型（LLM）的结合

工程层面：
├── 更高效的推理引擎
├── 端侧部署（手机/嵌入式设备）
├── 更完善的服务端部署方案
└── 更丰富的后处理能力（ITN、纠错等）

应用层面：
├── 会议转写（多说话人 + 长音频）
├── 实时翻译（ASR + MT）
├── 语音交互（ASR + LLM + TTS）
└── 多模态理解（Audio + Video + Text）
```

## 13. 学习资源与推荐路径

### 13.1 推荐学习路径

```
入门阶段（1-3天）：
├── 安装 FunASR，运行离线识别 demo
├── 尝试不同的模型（paraformer-zh、fsmn-vad、ct-punc）
├── 理解完整流水线（VAD + ASR + 标点）
└── 用自己的音频测试效果

进阶阶段（1-2周）：
├── 理解 Paraformer 的架构（CIF、GLM、NAR 解码）
├── 尝试流式识别
├── 学习热词功能
├── 了解模型部署（ONNX、Docker）
└── 阅读 FunASR 的论文

深入阶段（1-2月）：
├── 用自己的数据微调模型
├── 理解训练配置和超参数调优
├── 搭建生产级的 ASR 服务
├── 对比不同模型和框架的效果
├── 阅读 FunASR 源码，理解实现细节
└── 关注 FunASR 的最新模型和功能更新
```

### 13.2 重要参考资料

```
官方资源：
├── FunASR GitHub: https://github.com/modelscope/FunASR
├── FunASR 文档: https://github.com/modelscope/FunASR/tree/main/docs
├── ModelScope 模型库: https://modelscope.cn/models?task=auto-speech-recognition
└── FunASR 论文集: https://github.com/modelscope/FunASR#citation

核心论文：
├── Paraformer: "Paraformer: Fast and Accurate Parallel Transformer
│   for Non-autoregressive End-to-End Speech Recognition" (2022)
├── CIF: "CIF: Continuous Integrate-and-Fire for End-to-End Speech
│   Recognition" (2019)
├── FunASR: "FunASR: A Fundamental End-to-End Speech Recognition
│   Toolkit" (2023)
└── SenseVoice: "SenseVoice: Versatile Speech Foundation Model" (2024)

社区资源：
├── FunASR GitHub Issues（问题反馈和讨论）
├── ModelScope 社区（模型分享和交流）
└── 阿里达摩院语音实验室技术博客
```

### 13.3 与其他文档的关联

```
本系列文档的阅读顺序建议：

1. doc/01_asr_basics.md
   → ASR 基础概念、特征提取、评估指标
   → 为理解 FunASR 打下基础

2. doc/02_asr_model_details.md
   → 端到端模型的数据流、训练和推理流程
   → 理解 Paraformer 的技术背景

3. doc/kaldi.md
   → 传统 ASR 方法（GMM-HMM、WFST）
   → 理解 FunASR 相对于传统方法的进步

4. doc/funasr.md（本文档）
   → FunASR 的完整介绍和实战指南
   → 从使用到部署的全流程

5. demo/funasr/
   → 实际代码示例
   → 动手实践
```

---

> FunASR 代表了当前中文语音识别的最高水平。它的 Paraformer 模型通过非自回归解码实现了速度和效果的最佳平衡，完整的流水线（VAD + ASR + 标点 + 热词）覆盖了生产环境的核心需求。无论是快速原型验证还是大规模生产部署，FunASR 都是中文 ASR 的首选方案。
