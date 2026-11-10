# Kaldi 详细介绍：从入门到掌握

## 1. Kaldi 是什么

Kaldi 是由 Daniel Povey（约翰霍普金斯大学）等人于 2011 年发起的开源语音识别（ASR）工具包。它是 ASR 领域最具影响力的开源项目之一，以 C++ 为核心实现，使用 Shell 脚本组织训练流程，以 WFST（加权有限状态转换器）为核心解码框架。

Kaldi 的名字来源于埃塞俄比亚传说中发现咖啡的牧羊人——寓意"发现新事物"。

### 1.1 Kaldi 的核心定位

```
传统ASR工具包（非端到端）
├── 模块化架构：声学模型 + 发音词典 + 语言模型 + 解码器
├── 基于 GMM-HMM / DNN-HMM 的声学建模
├── 基于 WFST 的高效解码
├── 丰富的训练 recipe（覆盖主流数据集）
└── 工业级性能与学术级灵活性兼备
```

### 1.2 Kaldi 的历史地位与影响

尽管端到端模型（如 Conformer、Whisper）已成为主流，Kaldi 的影响力依然深远：

| 贡献        | 说明                                    |
| --------- | ------------------------------------- |
| 数据格式标准    | wav.scp、text、utt2spk 等格式被广泛沿用         |
| WFST 解码框架 | 推广了 WFST 在 ASR 中的应用                   |
| 训练范式      | 建立了 GMM-HMM → DNN-HMM 的渐进式训练方法        |
| 特征提取标准    | Kaldi 的 FBank/MFCC 实现是事实标准            |
| 人才培养      | 培养了大量 ASR 研究者和工程师                     |
| 后续项目      | ESPnet、k2/icefall/sherpa、WeNet 等均受其影响 |

下面对每项贡献做更详细的展开：

#### 数据格式标准

Kaldi 定义了一套简洁而实用的纯文本数据组织格式，成为 ASR 领域的事实标准。`wav.scp` 将语音 ID 映射到音频路径（支持管道命令实现运行时格式转换），`text` 存储对应的转写文本，`utt2spk` / `spk2utt` 描述语音与说话人的归属关系，`segments` 处理长录音的分段信息。这套格式的设计哲学是"一切皆文本、按行组织、字典序排列"，极易被脚本处理和人工检查。以下是各核心文件的具体格式示例：

```
# wav.scp —— 语音ID到音频路径的映射
# 格式: <utterance_id> <wav_path_or_pipe_command>
spk01_utt001 /data/audio/spk01_001.wav
spk01_utt002 /data/audio/spk01_002.wav
spk02_utt001 sox /data/raw/spk02_001.flac -t wav -r 16000 -c 1 - |
# 注意：以 "|" 结尾表示管道命令，Kaldi 会在读取时执行该命令，
# 可用于运行时格式转换、重采样、声道选择等，无需预先转换文件

# text —— 语音ID到转写文本的映射
# 格式: <utterance_id> <transcription>
spk01_utt001 今天天气真好
spk01_utt002 我想订一张去北京的机票
spk02_utt001 语音识别技术发展很快

# utt2spk —— 语音ID到说话人ID的映射
# 格式: <utterance_id> <speaker_id>
spk01_utt001 spk01
spk01_utt002 spk01
spk02_utt001 spk02

# spk2utt —— 说话人ID到其所有语音ID的映射（由 utt2spk 自动生成）
# 格式: <speaker_id> <utt_id_1> <utt_id_2> ...
# 生成命令: utils/utt2spk_to_spk2utt.pl utt2spk > spk2utt
spk01 spk01_utt001 spk01_utt002
spk02 spk02_utt001

# segments —— 长录音的分段信息（可选，用于从一条长录音中切出多条语音）
# 格式: <utterance_id> <recording_id> <start_time_sec> <end_time_sec>
spk01_utt001 recording_001 0.50 3.20
spk01_utt002 recording_001 3.80 7.15
spk02_utt001 recording_002 1.00 4.60
# 当使用 segments 时，wav.scp 中的 key 是 recording_id 而非 utterance_id

# reco2dur —— 录音时长（可选）
# 格式: <recording_id> <duration_seconds>
recording_001 120.5
recording_002 85.3
```

> 所有文件中的 ID 必须按字典序（ASCII 排序）排列，这是 Kaldi 的硬性要求。如果排序不正确，可以用 `utils/fix_data_dir.sh data/train` 自动修复。`utils/validate_data_dir.sh data/train` 可以检查数据目录的完整性和格式正确性。

后来的 ESPnet、WeNet、FunASR 等框架在数据准备阶段几乎都兼容或直接复用了这套格式，连 Hugging Face 上很多语音数据集的元信息组织方式也能看到 Kaldi 格式的影子。Lhotse 等现代数据处理工具也提供了对 Kaldi 格式的原生读写支持。

#### WFST 解码框架

Kaldi 将加权有限状态转换器（WFST）从学术论文中的理论工具变成了工程可用的核心解码引擎。它通过 HCLG 组合（H=HMM 拓扑、C=上下文依赖、L=发音词典、G=语言模型），将声学模型、词典、语言模型统一编译为一张解码图，解码时只需一次图搜索即可完成从声学状态到文本的全部映射。Kaldi 集成了 OpenFst 库，并在其上实现了确定化（determinization）、最小化（minimization）、组合（composition）等关键操作，使得解码图的构建和优化变得标准化、可复现。这一框架不仅在传统 ASR 中被广泛使用，也深刻影响了后续的混合架构——Daniel Povey 的新项目 k2 正是将 WFST 与端到端模型结合的代表，WeNet 的 TLG 解码也借鉴了 WFST 的思想。

#### 训练范式

Kaldi 确立了"渐进式训练"的标准范式：先训练最简单的单音素（Monophone）GMM-HMM 模型获得初始对齐，再用对齐结果训练上下文相关的三音素（Triphone）模型，接着通过 LDA+MLLT 优化特征空间，再用 SAT/fMLLR 做说话人自适应，最后以 GMM-HMM 的对齐作为标签训练 DNN-HMM 或 Chain Model。每一阶段都建立在前一阶段的输出之上，逐步提升模型质量和对齐精度。这种"用简单模型引导复杂模型"的思路影响深远——即使在端到端时代，CTC 预训练后接 Attention 微调、用外部对齐初始化 Transducer 等做法，本质上都延续了 Kaldi 渐进式训练的哲学。

#### 特征提取标准

Kaldi 的 MFCC 和 FBank 特征提取实现（包括预加重系数 0.97、Povey 窗函数、帧长 25ms/帧移 10ms 等默认参数）已成为语音处理领域的参考实现。后续几乎所有框架在提取声学特征时都会注明"与 Kaldi 一致"或"兼容 Kaldi 配置"。独立的 Python 包如 `kaldi-native-fbank`（C++ 实现、Python 接口）和 `kaldifeat`（PyTorch 版本）让开发者无需安装完整 Kaldi 即可使用相同的特征提取逻辑。Torchaudio 中的 `kaldi_compliance` 模块也专门提供了与 Kaldi 对齐的特征提取函数。这种"特征提取以 Kaldi 为准"的共识极大地降低了不同框架之间结果对比和模型迁移的门槛。

#### 人才培养

Kaldi 自 2011 年发布以来，凭借其丰富的 recipe（覆盖 WSJ、LibriSpeech、AISHELL、TIMIT 等主流数据集）、详尽的官方文档和活跃的邮件列表社区，成为全球高校和研究机构培养 ASR 人才的首选工具。大量语音识别方向的硕博论文以 Kaldi 为实验平台，研究者通过阅读和修改 Kaldi 的 Shell 脚本和 C++ 源码，深入理解了声学建模、语言建模、解码搜索等核心技术。许多如今活跃在工业界（Google、Meta、Microsoft、字节跳动、阿里等）和学术界的 ASR 专家都是从 Kaldi 入门的。Kaldi 的 recipe 式组织方式（每个数据集一个完整的可复现训练流程）也成为后续框架的标准做法。

#### 后续项目

Kaldi 直接或间接催生了一系列重要的后续项目。ESPnet（2018）由 Kaldi 社区成员开发，将端到端模型与 Kaldi 风格的 recipe 组织方式结合，成为学术界最主流的端到端 ASR/TTS 框架。k2（2020）是 Daniel Povey 本人的新项目，用 PyTorch 重新实现了可微分的 WFST 操作，配套的 icefall 提供训练 recipe，sherpa/sherpa-onnx 提供部署方案，代表了"WFST + 端到端"的融合方向。WeNet（2021，出门问问）借鉴了 Kaldi 的工程实践，提供了工业级的流式/非流式一体化端到端方案。FunASR（2023，阿里达摩院）在模型丰富度和中文场景支持上进一步拓展。Lhotse（数据处理）、Pychain（PyTorch 版 LF-MMI）等工具也都是 Kaldi 生态的延伸。可以说，现代 ASR 开源生态的基本格局很大程度上是由 Kaldi 社区塑造的。

```
Kaldi (2011) 的技术传承：
    │
    ├──→ ESPnet（学术界主流端到端框架）
    ├──→ k2 / icefall / sherpa（Kaldi原作者的新项目，WFST + 端到端）
    ├──→ WeNet（出门问问，工业级端到端）
    ├──→ FunASR（阿里达摩院）
    └──→ 几乎所有现代ASR框架都在概念或工具层面受到Kaldi影响
```

## 2. 安装与环境配置

### 2.1 完整安装 Kaldi（从源码编译）

Kaldi 的完整安装需要编译 C++ 源码，依赖较多，适合需要使用全部功能的用户。

```bash
# 1. 克隆仓库
git clone https://github.com/kaldi-asr/kaldi.git
cd kaldi

# 2. 编译依赖工具（OpenFst等）
cd tools
# 检查依赖
extras/check_dependencies.sh
# 编译（-j4 表示4线程并行）
make -j4

# 3. 编译Kaldi核心库
cd ../src
./configure --shared
make depend -j4
make -j4
```

> 编译过程可能需要 30 分钟到数小时，取决于机器性能。常见依赖包括：gcc/g++（>=4.8）、automake、autoconf、libtool、subversion、wget、zlib、ATLAS/MKL/OpenBLAS（线性代数库）。

### 2.2 轻量级 Python 安装（推荐入门）

如果只需要使用 Kaldi 风格的特征提取，无需编译完整 Kaldi：

```bash
pip install kaldi-native-fbank   # Kaldi FBank 特征提取（C++实现，Python接口）
pip install kaldifeat             # PyTorch 版 Kaldi 特征提取
pip install soundfile numpy       # 音频读取和数值计算
```

### 2.3 目录结构概览

```
kaldi/
├── src/                    # C++ 源码
│   ├── base/               # 基础工具（日志、数学等）
│   ├── matrix/             # 矩阵运算库
│   ├── feat/               # 特征提取（MFCC、FBank等）
│   ├── gmm/                # GMM 相关
│   ├── hmm/                # HMM 相关
│   ├── tree/               # 决策树
│   ├── decoder/            # 解码器
│   ├── nnet3/              # 神经网络（第三代）
│   ├── chain/              # Chain 模型（LF-MMI）
│   ├── fstext/             # FST 工具
│   ├── lat/                # Lattice 处理
│   └── bin/                # 可执行文件
│
├── tools/                  # 第三方工具
│   ├── openfst/            # OpenFst 库
│   └── srilm/              # SRILM 语言模型工具（可选）
│
├── egs/                    # 训练示例（recipe）
│   ├── aishell/            # AISHELL-1 中文
│   ├── librispeech/        # LibriSpeech 英文
│   ├── wsj/                # Wall Street Journal
│   └── ...                 # 更多数据集
│
└── scripts/                # 通用脚本
    ├── steps/              # 训练步骤脚本
    └── utils/              # 工具脚本
```

## 3. 核心概念详解

### 3.1 Kaldi 的模块化架构

与端到端模型（一个神经网络从音频直接到文本）不同，Kaldi 采用经典的模块化架构，每个模块各司其职：

```
┌──────────────────────────────────────────────────────────────────┐
│                     Kaldi ASR 系统全景                            │
│                                                                   │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐  │
│  │ 声学模型  │   │ 发音词典  │   │ 语言模型  │   │   解码器     │  │
│  │(AM)      │   │(Lexicon) │   │(LM)      │   │  (Decoder)  │  │
│  │          │   │          │   │          │   │             │  │
│  │ GMM-HMM  │   │ 词→音素  │   │ n-gram   │   │  WFST搜索   │  │
│  │ 或DNN-HMM│   │ 映射     │   │          │   │             │  │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └──────┬──────┘  │
│       │               │              │                │          │
│       └───────────────┴──────────────┘                │          │
│                       │                               │          │
│                       ↓                               │          │
│               ┌───────────────┐                       │          │
│               │  HCLG.fst     │───────────────────────┘          │
│               │  (解码图)      │                                  │
│               └───────────────┘                                  │
│                                                                   │
│  音频 → 特征提取 → 声学模型打分 → WFST解码 → 输出文本             │
└──────────────────────────────────────────────────────────────────┘
```

> **什么是 n-gram？** n-gram 是一种基于统计的语言模型方法，核心思想是：一个词出现的概率取决于它前面的 n-1 个词。"n"代表考虑的词的个数——当 n=1 时称为 unigram（只看当前词本身的频率），n=2 时称为 bigram（看前一个词），n=3 时称为 trigram（看前两个词）。例如在 bigram 模型中，P("天气" | "今天") 表示在"今天"后面出现"天气"的概率，这个概率通过统计大量文本中"今天 天气"连续出现的次数除以"今天"出现的总次数来估算。n 越大，模型能捕获的上下文越长，但需要的训练数据也呈指数增长（因为可能的 n-gram 组合数量爆炸），实际中通常使用 3-gram 或 4-gram，并配合平滑技术（如 Kneser-Ney 平滑）来处理训练数据中未出现过的词组合。n-gram 语言模型虽然简单，但计算高效、易于与 WFST 框架集成，因此在 Kaldi 中被广泛使用。现代端到端 ASR 系统中，n-gram 仍常被用于解码阶段的浅层融合（shallow fusion）或首遍解码，再用神经网络语言模型做重打分。

> **什么是音素（Phone / Phoneme）？** 音素是语言中最小的能区分词义的语音单位。例如普通话中"八"（bā）和"趴"（pā）的区别仅在于声母 /b/ 和 /p/，所以 /b/ 和 /p/ 就是两个不同的音素。音素不等于文字，也不等于音节——一个汉字通常对应一个音节，而一个音节由一个或多个音素组成。例如"天"（tiān）可以拆分为音素序列 /t/ /ian/（或更细粒度的 /t/ /i/ /a/ /n/，取决于音素集的设计）。英文中"speech"可以拆分为 /S/ /P/ /IY/ /CH/ 四个音素。
> 
> 在 Kaldi 的传统 ASR 框架中，音素是连接声学模型和语言模型的核心中间表示：声学模型负责判断每帧音频对应哪个音素状态，发音词典负责定义每个词由哪些音素组成，两者通过音素这一桥梁将"声音"和"文字"关联起来。
> 
> 常见的音素集规模：普通话约 60~200 个音素（取决于是否将声调编码为独立音素），英文（CMU 音素集）约 39 个音素。音素集的设计是 ASR 系统的重要决策之一——粒度太粗会丢失区分信息，粒度太细则训练数据不足以覆盖所有音素组合。

各模块的职责：

| 模块   | 输入     | 输出              | 作用            |
| ---- | ------ | --------------- | ------------- |
| 特征提取 | 原始音频波形 | MFCC/FBank 特征矩阵 | 将音频转为紧凑的数值表示  |
| 声学模型 | 特征向量   | 每帧属于各音素状态的概率    | 判断"这帧声音像哪个音素" |
| 发音词典 | 词      | 音素序列            | 告诉系统"每个词怎么发音" |
| 语言模型 | 词序列    | 序列概率            | 判断"这句话通不通顺"   |
| 解码器  | 以上所有信息 | 最优文本            | 综合搜索最可能的文本    |

### 3.2 HMM（隐马尔可夫模型）

HMM 是 Kaldi 声学建模的骨架。在 ASR 中，每个音素用一个 3 状态的从左到右 HMM 表示：

```
每个音素的 HMM 结构：

    ┌───┐    ┌───┐    ┌───┐
    │ S1│───→│ S2│───→│ S3│───→ (下一个音素)
    └─┬─┘    └─┬─┘    └─┬─┘
      │↺       │↺       │↺
   自转移    自转移    自转移

S1 = 音素的起始部分（过渡段）
S2 = 音素的稳定部分（核心段）
S3 = 音素的结束部分（过渡段）
```

为什么用 3 个状态？因为一个音素的声学特征在时间上不是均匀的——开始、中间、结束阶段的声音特征不同。3 状态 HMM 能捕获这种时间结构。

自转移（self-loop）允许一个状态持续多帧，这样同一个音素可以对应不同长度的音频段（说话快的人和说话慢的人，同一个音素的持续时间不同）。

HMM 的两个核心概率：

- 转移概率：从一个状态跳到下一个状态（或停留在当前状态）的概率
- 发射概率：在某个状态下，产生某个特征向量的概率（由 GMM 建模）

### 3.3 GMM（高斯混合模型）

GMM 用于建模 HMM 每个状态的发射概率——即"在这个音素状态下，音频特征长什么样"。

```
P(x | state_j) = Σ(k=1 to K) w_k · N(x; μ_k, Σ_k)

其中：
  x = 当前帧的特征向量（如39维MFCC）
  K = 高斯分量数（通常8~64个）
  w_k = 第k个高斯的权重（所有权重之和为1）
  μ_k = 第k个高斯的均值向量
  Σ_k = 第k个高斯的协方差矩阵（通常用对角矩阵）
  N(x; μ, Σ) = 多维高斯分布的概率密度函数
```

> 为什么用混合模型而不是单个高斯？因为同一个音素在不同说话人、不同语境下的声学特征分布是多峰的（multimodal），单个高斯只能描述单峰分布，多个高斯的加权和才能拟合复杂的分布形状。

直觉理解：想象"a"这个元音，男性说的"a"和女性说的"a"在频谱上差异很大，一个高斯分布无法同时覆盖两者。用两个高斯分量，一个拟合男性的分布，一个拟合女性的分布，加权求和就能很好地描述"a"的整体分布。

### 3.4 上下文相关音素（Triphone）

单音素（monophone）模型假设每个音素的声学特征与上下文无关，但实际上同一个音素在不同上下文中发音差异很大。例如"an"中的"a"和"ai"中的"a"听起来不同。

[[语音识别] 单音素、三音素、决策树-腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/1604113)

[GMM-HMM在语音中的训练及和预测的简要流程 - wieneralan - 博客园](https://www.cnblogs.com/wieneralan/p/17439772.html)

三音素（triphone）考虑左右各一个音素的上下文：

```
三音素表示法：左音素-当前音素+右音素

例如 "今天" 的音素序列 j-in-t-ian：
  j/in+t    → "in"的左边是j，右边是t
  in/t+ian  → "t"的左边是in，右边是ian

问题：三音素状态数量爆炸
  假设有60个音素，每个3状态
  单音素状态数: 60 × 3 = 180
  三音素状态数: 60 × 60 × 60 × 3 = 648,000
  很多三音素组合在训练数据中从未出现过
```

解决方案：决策树聚类（Decision Tree Clustering）

```
通过提问将相似的三音素状态聚类到一起：

              [当前音素是元音?]
              /              \
           是                 否
          /                     \
    [左边是鼻音?]          [右边是元音?]
    /        \              /        \
  叶节点1   叶节点2     叶节点3    叶节点4
  (共享GMM) (共享GMM)  (共享GMM)  (共享GMM)

最终将 648,000 个三音素状态聚类为约 3,000~8,000 个共享状态（senone）
每个叶节点对应一个 senone，共享同一组 GMM 参数
```

决策树中的问题是语言学家设计的，例如：

- "当前音素是元音吗？"
- "左边的音素是鼻音吗？"
- "右边的音素是送气音吗？"
- "当前音素是静音吗？"

### 3.5 WFST（加权有限状态转换器）

WFST 是 Kaldi 最核心的技术之一，也是 Kaldi 区别于其他 ASR 工具包的关键特征。

#### 什么是 WFST

WFST 是一种图结构，每条边上有四个元素：输入符号、输出符号、权重、目标状态。

```
简单示例：一个将音素映射到词的 WFST

状态0 --[n:ε/0.0]--> 状态1 --[i:你/0.5]--> 状态2 --[h:ε/0.0]--> 状态3 --[ao:好/0.3]--> 状态4
                                                                                              (终态)

读法：
  从状态0出发
  输入"n"，不输出（ε），权重0.0，到状态1
  输入"i"，输出"你"，权重0.5，到状态2
  输入"h"，不输出，权重0.0，到状态3
  输入"ao"，输出"好"，权重0.3，到状态4（终态）

整体效果：输入音素序列 [n, i, h, ao]，输出词序列 [你, 好]，总权重 0.8
```

> WFST 边上的权重从哪里来？权重的来源取决于这条边属于 HCLG 中的哪一层：
> 
> - G（语言模型）层的权重来自 n-gram 语言模型训练，表示词序列的对数概率。例如 P("好" | "你") = 0.3，这个 0.3 是从大量文本语料中统计词共现频率后计算得到的。语料中"你好"出现得越频繁，对应边的权重（代价）就越小，解码时就越倾向于选择这条路径。
> - L（发音词典）层的权重通常为 0（即不带额外代价），因为词典只是确定性地描述"词→音素"的映射关系。但如果一个词有多种发音变体（如"谁" 可以读 shéi 或 shuí），不同变体的边可以赋予不同权重来表示发音偏好，这些权重可以从标注数据中统计得到，也可以人工设定。
> - H（HMM 拓扑）层的权重来自 HMM 的转移概率，即从一个状态跳到下一个状态（或自转移）的概率。这些转移概率在 GMM-HMM 训练的 EM 迭代过程中从训练数据中估计得到。
> - 在解码阶段，声学模型的打分（每帧特征对应各 HMM 状态的对数似然）会作为额外的权重叠加到 HCLG 图的搜索过程中。
> 
> 权重的作用是什么？WFST 中的权重本质上编码了"代价"（cost）——在热带半环（tropical semiring）下，权重越小表示路径越优。解码器在 HCLG 图上搜索时，会将路径上所有边的权重累加（声学分数 + 语言模型分数 + 转移概率），最终选择总权重最小（即总概率最大）的路径作为识别结果。权重机制使得 WFST 能够在一个统一的图搜索框架中同时考虑"这段声音像什么音素"（声学模型）、"这个词怎么发音"（词典）和"这句话通不通顺"（语言模型）三方面的信息，而不需要分别处理再合并。

#### HCLG 解码图

Kaldi 的核心创新是将所有知识源编译成一个统一的 WFST 解码图 HCLG：

```
HCLG = min(det(H ∘ C ∘ L ∘ G))

四个组成部分：

G (Grammar / 语言模型):
  输入: 词        输出: 词
  作用: 编码 n-gram 语言模型的概率
  例: P("天气" | "今天") = 0.15

L (Lexicon / 发音词典):
  输入: 音素      输出: 词
  作用: 将音素序列映射为词
  例: [t, ian, q, i] → "天气"

C (Context / 上下文依赖):
  输入: 上下文相关音素(triphone)    输出: 音素
  作用: 将三音素映射回单音素
  例: "t-ian+q" → "ian"

H (HMM / HMM拓扑):
  输入: HMM状态ID    输出: 上下文相关音素
  作用: 描述 HMM 的状态转移结构
  例: [s1, s2, s3] → "t-ian+q"

组合后的 HCLG:
  输入: HMM状态序列
  输出: 词序列
  一次 WFST 搜索就完成了从声学状态到文本的全部映射
```

```
HCLG 的组合过程：

G (语言模型)
  ↑ 组合 (∘)
L (词典)
  ↑ 组合 (∘)
C (上下文)
  ↑ 组合 (∘)
H (HMM拓扑)
  ↓
确定化 (det) + 最小化 (min)
  ↓
HCLG.fst（最终解码图，通常几百MB到几GB）
```

> WFST 的组合（composition）操作类似于函数复合：如果 L 把音素映射到词，G 给词序列打分，那么 L∘G 就能直接给音素序列打分。确定化和最小化是图优化操作，减少冗余路径，加速解码。

#### 为什么 WFST 如此重要

1. 统一框架：将声学模型、词典、语言模型统一到一个搜索图中，解码时只需一次图搜索
2. 高效解码：经过确定化和最小化后，解码图非常紧凑，搜索效率高
3. 灵活组合：可以方便地替换任何一个组件（换词典、换语言模型）
4. 理论优雅：基于有限状态自动机理论，有严格的数学基础

### 3.6 发音词典（Lexicon）

发音词典定义了每个词的发音方式，是连接声学模型和语言模型的桥梁：

```
# 中文发音词典示例 (lexicon.txt)
# 格式: 词  音素1 音素2 ...
你好    n i3 h ao3
天气    t ian1 q i4
今天    j in1 t ian1
语音    y v3 y in1
识别    sh ir2 b ie2

# 英文发音词典示例 (CMU Pronouncing Dictionary)
HELLO   HH AH0 L OW1
WORLD   W ER1 L D
SPEECH  S P IY1 CH
```

Kaldi 还需要几个特殊文件来完整描述词典：

```
data/local/dict/
├── lexicon.txt          # 发音词典（词→音素）
├── nonsilence_phones.txt # 非静音音素列表
├── silence_phones.txt    # 静音音素列表（SIL, SPN等）
├── optional_silence.txt  # 可选静音（通常是SIL）
└── extra_questions.txt   # 额外的决策树问题（如声调）
```

#### 如何构建发音词典

构建发音词典是 Kaldi 系统搭建中最关键的前期工作之一，词典质量直接影响最终识别效果。不同语言的构建方式差异较大。

**中文发音词典的构建**

中文的发音相对规则——每个汉字对应固定的拼音，因此中文词典通常可以自动生成：

1. 准备词表：从训练文本中提取所有出现过的词（如果做字级别建模，则提取所有汉字）。
2. 汉字转拼音：使用工具将每个汉字转为拼音，常用工具包括 `pypinyin`（Python 库）、`phonetisaurus`、或直接查 Unicode 拼音表。
3. 拼音拆分为音素：将拼音按声母和韵母拆分为音素序列。拆分方案取决于音素集的设计。

```python
# 示例：用 pypinyin 生成中文发音词典
from pypinyin import pinyin, Style

# 定义声母和韵母的拆分规则（简化版）
INITIALS = {'b','p','m','f','d','t','n','l','g','k','h',
            'j','q','x','zh','ch','sh','r','z','c','s','y','w'}

def word_to_phones(word):
    """将中文词转为音素序列"""
    phones = []
    # 获取带声调的拼音
    pys = pinyin(word, style=Style.TONE3, heteronym=False)
    for py_list in pys:
        py = py_list[0]  # 如 "jin1", "tian1"
        # 分离声调数字
        tone = py[-1] if py[-1].isdigit() else '5'  # 5表示轻声
        base = py.rstrip('12345')
        # 拆分声母和韵母
        initial = ''
        for init in sorted(INITIALS, key=len, reverse=True):
            if base.startswith(init):
                initial = init
                break
        final = base[len(initial):]
        # 生成音素（声调附加在韵母上）
        if initial:
            phones.append(initial)
        if final:
            phones.append(final + tone)
    return phones

# 生成词典
vocab = ["今天", "天气", "你好", "语音", "识别"]
with open("lexicon.txt", "w", encoding="utf-8") as f:
    for word in sorted(vocab):
        phones = word_to_phones(word)
        f.write(f"{word}\t{' '.join(phones)}\n")

# 输出示例：
# 今天    j in1 t ian1
# 你好    n i3 h ao3
# 天气    t ian1 q i4
# 识别    sh ir2 b ie2
# 语音    y v3 y in1
```

中文音素集的常见设计方案：

```
方案一：声母+韵母（最常用，约 60~80 个音素）
  声母: b p m f d t n l g k h j q x zh ch sh r z c s y w  (约23个)
  韵母: a o e i u v ai ei ao ou an en ang eng ong ia ie iu
        ian in iang ing iong ua uo uai ui uan un uang ueng
        van vn ve er ...  (约35~50个，带声调则 ×4~5)
  示例: "今天" → j in1 t ian1

方案二：声母+韵母+声调独立编码
  音素不带声调，声调作为独立音素
  示例: "今天" → j in T1 t ian T1
  好处: 减少音素总数，声调信息可以在决策树中灵活处理

方案三：全拼音（每个拼音音节作为一个音素）
  示例: "今天" → jin1 tian1
  音素数量多（约 400+），但每个音素的声学特征更稳定
  适合数据量较大的场景
```

**英文发音词典的构建**

英文的拼写和发音关系不规则（如 "through"、"though"、"thought" 拼写相似但发音不同），因此英文词典通常依赖现有资源或 G2P 工具：

1. 使用现有词典：CMU Pronouncing Dictionary（CMUdict）是最常用的英文发音词典，包含约 13 万词的发音标注，使用 ARPAbet 音素集（39 个音素）。可以从 http://www.speech.cs.cmu.edu/cgi-bin/cmudict 下载。

2. G2P（Grapheme-to-Phoneme）工具处理 OOV 词：对于不在 CMUdict 中的词（Out-of-Vocabulary），需要用 G2P 模型自动预测发音：
   
   - Phonetisaurus：基于 WFST 的 G2P 工具，Kaldi 生态中常用
   - Sequitur G2P：基于联合序列模型的 G2P
   - 神经网络 G2P：如 g2p-en（Python 库），基于 Transformer 的 G2P 模型

```bash
# 使用 Phonetisaurus 训练 G2P 模型并预测 OOV 词的发音
# 1. 用 CMUdict 训练 G2P 模型
phonetisaurus-align --input=cmudict.dict --ofile=aligned.corpus
phonetisaurus-arpa2wfst --lm=g2p_model.arpa --ofile=g2p_model.fst

# 2. 预测新词的发音
phonetisaurus-g2pfst --model=g2p_model.fst --word=kaldi
# 输出: kaldi  K AO1 L D IY0
```

**构建完整的词典目录**

无论中文还是英文，最终都需要准备以下文件：

```bash
# 1. lexicon.txt —— 主词典文件（上面已介绍）
# 必须包含特殊词:
#   <SIL> SIL          （静音）
#   <UNK> SPN          （未知词，映射到 spoken noise 音素）

# 2. nonsilence_phones.txt —— 每行一个非静音音素（或同组音素写在一行）
# 中文示例:
b
p
m
f
...
a1 a2 a3 a4    # 同一韵母的不同声调写在一行（表示它们是同一组）
o1 o2 o3 o4
...

# 3. silence_phones.txt —— 静音类音素
SIL
SPN    # spoken noise（嗯、啊等非语言声音）
NSN    # non-spoken noise（背景噪声）

# 4. optional_silence.txt —— 词间可选静音（通常只有一行）
SIL

# 5. extra_questions.txt —— 额外的决策树聚类问题
# 用于在决策树构建时提供额外的音素分组信息
# 中文中通常按声调分组:
a1 o1 e1 i1 u1 v1 ai1 ei1 ao1 ou1 an1 en1 ang1 eng1
a2 o2 e2 i2 u2 v2 ai2 ei2 ao2 ou2 an2 en2 ang2 eng2
a3 o3 e3 i3 u3 v3 ai3 ei3 ao3 ou3 an3 en3 ang3 eng3
a4 o4 e4 i4 u4 v4 ai4 ei4 ao4 ou4 an4 en4 ang4 eng4
# 这样决策树可以问"当前音素是几声？"来区分声调

# 6. 用 Kaldi 工具编译词典为 L.fst
utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang
# 输出: data/lang/L.fst, L_disambig.fst, phones.txt, words.txt 等
```

> `prepare_lang.sh` 是 Kaldi 词典准备的核心脚本，它会自动完成：验证词典文件格式、生成音素符号表（phones.txt）和词符号表（words.txt）、添加消歧符号（disambiguation symbols，用于处理 WFST 确定化中的歧义）、编译 L.fst（词典 FST）和 L_disambig.fst。消歧符号是 WFST 理论中的重要概念——当两个不同的词共享相同的音素前缀时（如"一"和"一起"），需要消歧符号来保证 L.fst 的确定化正确性。

### 3.7 语言模型（Language Model）

Kaldi 使用 n-gram 统计语言模型，预测词序列的概率：

```
n-gram 语言模型：

Bigram (2-gram):
  P(天气 | 今天) = Count(今天 天气) / Count(今天)

Trigram (3-gram):
  P(真好 | 今天 天气) = Count(今天 天气 真好) / Count(今天 天气)

实际使用中需要平滑（smoothing）处理未见过的 n-gram：
  - Kneser-Ney 平滑（最常用）
  - Good-Turing 平滑
  - Witten-Bell 平滑
```

语言模型训练工具：

- SRILM：经典的 n-gram LM 工具包
- KenLM：更快的 n-gram LM 工具包
- IRSTLM：Kaldi 自带的 LM 工具

## 4. 特征提取

### 4.1 MFCC 特征（Kaldi 传统默认）

Kaldi 早期（GMM-HMM 时代）默认使用 MFCC 特征：

```
提取流程：
WAV → 预加重(0.97) → 分帧(25ms/10ms) → 加窗(Povey窗) → FFT(512点)
    → 功率谱 → Mel滤波器组(23个) → 取对数 → DCT(取前13个系数)
    → 13维 MFCC

加上动态特征：
  静态 MFCC:  13维
  一阶差分Δ:  13维（描述特征的变化速度）
  二阶差分ΔΔ: 13维（描述特征的变化加速度）
  总计:       39维
```

Kaldi 命令行提取 MFCC：

```bash
# 提取 MFCC 特征
compute-mfcc-feats \
  --num-ceps=13 \
  --sample-frequency=16000 \
  --frame-length=25 \
  --frame-shift=10 \
  scp:wav.scp \
  ark:mfcc.ark

# 添加差分特征
add-deltas ark:mfcc.ark ark:mfcc_delta.ark
```

### 4.2 FBank 特征（DNN 时代）

DNN-HMM 和 Chain Model 通常使用 FBank 特征（省略 DCT 步骤，保留更多信息）：

```
提取流程：
WAV → 预加重 → 分帧 → 加窗 → FFT → Mel滤波器组 → 取对数
    → 40维 或 80维 FBank
```

```bash
# Kaldi 命令行提取 FBank
compute-fbank-feats \
  --num-mel-bins=80 \
  --sample-frequency=16000 \
  scp:wav.scp \
  ark:fbank.ark
```

Python 方式提取（使用 kaldi-native-fbank）：

```python
import kaldi_native_fbank as knf
import soundfile as sf
import numpy as np

# 读取音频
waveform, sr = sf.read("test.wav", dtype="float32")

# 配置参数（与 Kaldi 的 compute-fbank-feats 一致）
opts = knf.FbankOptions()
opts.frame_opts.samp_freq = sr
opts.frame_opts.frame_length_ms = 25.0
opts.frame_opts.frame_shift_ms = 10.0
opts.frame_opts.dither = 0.0
opts.mel_opts.num_bins = 80

# 提取特征
fbank = knf.OnlineFbank(opts)
fbank.accept_waveform(sr, waveform.tolist())
fbank.input_finished()

num_frames = fbank.num_frames_ready
features = np.zeros((num_frames, 80), dtype=np.float32)
for i in range(num_frames):
    features[i] = fbank.get_frame(i)

print(f"FBank shape: {features.shape}")  # 例: (998, 80)
```

### 4.3 CMVN（倒谱均值方差归一化）

CMVN 消除不同录音条件（麦克风、环境噪声、信道）带来的特征偏移：

```
归一化公式：
  x_normalized = (x - μ) / σ

其中 μ 和 σ 可以按不同粒度计算：
  - 全局 CMVN: 在整个训练集上计算（最常用）
  - 说话人级 CMVN: 对每个说话人单独计算
  - 句子级 CMVN: 对每条音频单独计算
```

```bash
# Kaldi 计算 CMVN 统计量
compute-cmvn-stats scp:feats.scp ark:cmvn.ark

# 应用 CMVN
apply-cmvn --norm-vars=true ark:cmvn.ark scp:feats.scp ark:feats_cmvn.ark
```

### 4.4 特征变换技术

Kaldi 中有一系列特征变换技术，逐步提升特征质量：

```
原始特征 (39维 MFCC)
    ↓
LDA (线性判别分析)
  - 拼接连续9帧: 9 × 39 = 351维
  - 降维到40维
  - 最大化类间距离，最小化类内距离
    ↓
MLLT (最大似然线性变换)
  - 在 LDA 基础上进一步优化变换矩阵
  - 使特征更适合对角协方差 GMM
    ↓
fMLLR (特征空间最大似然线性回归)
  - 说话人自适应变换
  - 对每个说话人学习一个仿射变换: x' = A·x + b
  - 将不同说话人的特征映射到统一空间
  - 类似于"把每个人的声音都调整成标准发音"
    ↓
i-vector (说话人向量)
  - 将说话人信息压缩为固定维度向量（通常100维）
  - 作为额外特征拼接到声学特征上
  - 帮助模型区分不同说话人
```

下面对每一步变换技术做详细展开。

#### 4.4.1 LDA（线性判别分析，Linear Discriminant Analysis）

LDA 的目标是找到一个线性投影矩阵，将高维特征投影到低维空间，同时让不同类别（不同 senone/音素状态）的特征尽可能分开，同一类别的特征尽可能聚拢。

具体做法：

1. 拼接上下文帧：将当前帧及其前后各 4 帧拼接在一起，形成一个"超级特征向量"。如果原始 MFCC 是 39 维（13 维静态 + 13 维 Δ + 13 维 ΔΔ），拼接 9 帧后变为 9 × 39 = 351 维。拼接的目的是让特征包含时间上下文信息——单独一帧只有 10ms 的信息，拼接 9 帧后覆盖约 105ms，能更好地区分不同音素。

2. 计算类内散布矩阵 S_w 和类间散布矩阵 S_b：
   
   - S_w 衡量同一个 senone 内部特征的分散程度（越小越好，说明同类特征很集中）
   - S_b 衡量不同 senone 之间特征均值的分散程度（越大越好，说明不同类特征分得很开）
   - "类别"就是 senone（由之前的 GMM-HMM 对齐确定每帧属于哪个 senone）

3. 求解广义特征值问题 S_b · v = λ · S_w · v，取最大的 40 个特征值对应的特征向量组成投影矩阵 W（维度 351 × 40）。

4. 对每帧拼接后的 351 维特征做投影：x' = W^T · x，得到 40 维的 LDA 特征。

```
直觉理解：

  拼接前（单帧39维）          拼接后（9帧351维）         LDA投影后（40维）
  ┌───┐                    ┌───────────────┐          ┌────┐
  │39 │  × 9帧拼接 →       │     351       │  × W^T → │ 40 │
  └───┘                    └───────────────┘          └────┘
                           包含时间上下文              去除冗余，保留区分性信息

  351维中有大量冗余（相邻帧高度相关），LDA 找到最有区分力的40个方向
```

> 为什么降到 40 维？这是经验值。实验表明 40 维已经保留了绝大部分区分性信息，再增加维度收益很小，反而增加计算量。而且 40 维与后续 GMM 的对角协方差假设配合较好。

#### 4.4.2 MLLT（最大似然线性变换，Maximum Likelihood Linear Transform）

MLLT（也称为 STC，Semi-Tied Covariance）的目标是在 LDA 的基础上进一步优化变换矩阵，使变换后的特征更适合用对角协方差矩阵的 GMM 来建模。

背景问题：GMM 中每个高斯分量有一个协方差矩阵 Σ。使用全协方差矩阵（full covariance）参数多、计算慢；使用对角协方差矩阵（diagonal covariance）假设各维度独立，参数少、计算快，但如果特征各维度之间存在相关性，对角协方差就不准确。

MLLT 的做法：

1. 在 LDA 变换矩阵的基础上，学习一个额外的方阵 M（40 × 40），使得变换后的特征 x'' = M · x'（x' 是 LDA 后的特征）各维度之间的相关性尽可能小。

2. 优化目标是最大化训练数据在对角协方差 GMM 下的似然（likelihood）。直觉上就是：找一个旋转/缩放变换，让变换后的特征分布尽可能"轴对齐"——每个高斯分量的等概率面尽可能与坐标轴平行，这样对角协方差矩阵就能很好地描述它。

3. 训练过程是迭代的：先用当前 GMM 参数估计变换矩阵 M，再用变换后的特征重新训练 GMM，反复迭代直到收敛。

```
MLLT 的效果示意（二维简化）：

  LDA后的特征分布:              MLLT变换后:
  （各维度有相关性）             （各维度近似独立）

    y                              y
    │  ╱╱                          │  ║
    │╱╱    ← 椭圆倾斜              │  ║   ← 椭圆与坐标轴对齐
    └──────── x                    └──────── x

  对角协方差GMM拟合不好            对角协方差GMM拟合很好
  （需要全协方差才准确）           （对角协方差就够了）
```

> MLLT 的实际效果：在 Kaldi 的训练流程中，LDA+MLLT 通常比单独使用 LDA 降低 1-3% 的相对错误率。计算开销很小（只是一个 40×40 的矩阵乘法），但收益稳定。

#### 4.4.3 fMLLR（特征空间最大似然线性回归，Feature-space MLLR）

fMLLR 是说话人自适应技术的核心，目标是消除不同说话人之间的声学差异，让模型只需要学习"标准化"后的声学模式。

问题背景：不同说话人的声音差异很大——男性和女性的基频（F0）差异可达一倍，不同人的声道长度、共振峰频率、说话习惯都不同。如果模型要同时拟合所有说话人的特征分布，就需要更多的高斯分量，模型更复杂，效果也不一定好。

fMLLR 的做法：

1. 对每个说话人，学习一个仿射变换（affine transform）：x' = A · x + b，其中 A 是 40×40 的变换矩阵，b 是 40 维的偏移向量。合并写成 x' = W · [x; 1]，其中 W 是 40×41 的矩阵。

2. 变换矩阵 W 的估计过程：
   
   - 首先用说话人无关（SI, Speaker Independent）的模型对该说话人的所有语音做解码或对齐，得到每帧对应的 senone 后验概率
   - 然后用这些后验概率和该说话人的特征数据，通过最大化似然的方式求解 W
   - 求解过程使用行迭代（row-by-row iteration）算法，每次优化 W 的一行，交替迭代直到收敛

3. 估计完 W 后，将该说话人的所有特征都做变换 x' = W · [x; 1]，变换后的特征送入模型进行训练或解码。

```
fMLLR 的直觉理解：

  变换前（不同说话人特征分布不同）：

  说话人A（男性）    说话人B（女性）    说话人C（老年人）
    ●●●                ○○○               △△△
    ●●●                ○○○               △△△
  （低频为主）       （高频为主）       （频率偏移）

  fMLLR变换后（所有说话人映射到统一空间）：

    ●○△●○△
    ●○△●○△
    ●○△●○△
  （特征分布趋于一致，模型更容易学习）
```

训练阶段和解码阶段的区别：

- 训练阶段（SAT, Speaker Adaptive Training）：先估计每个训练说话人的 fMLLR 变换，在变换后的特征上训练 GMM-HMM。这样训练出的模型期望看到的是"标准化"后的特征。
- 解码阶段：对测试说话人，先用 SI 模型做一遍初步解码，根据解码结果估计该说话人的 fMLLR 变换，再用变换后的特征做第二遍解码（两遍解码，two-pass decoding）。

```bash
# Kaldi 中的 fMLLR 解码（两遍）
# 第一遍：用 SI 模型解码，生成 lattice
steps/decode.sh --nj 4 exp/tri3/graph data/test exp/tri3/decode_test_si

# 估计 fMLLR 变换
steps/decode_fmllr.sh --nj 4 exp/tri3/graph data/test exp/tri3/decode_test_fmllr
# 内部流程：
#   1. 从第一遍 lattice 中提取对齐信息
#   2. 对每个说话人估计 fMLLR 变换矩阵
#   3. 用变换后的特征重新解码
```

> fMLLR 的效果：在 Kaldi 的标准 recipe 中，SAT + fMLLR 通常比不做说话人自适应降低 10-20% 的相对错误率，是 GMM-HMM 时代最有效的技术之一。

#### 4.4.4 i-vector（说话人向量）

i-vector 是一种将说话人身份信息压缩为固定维度向量的技术，最初用于说话人识别/验证，后来被引入 ASR 作为辅助特征。

核心思想：每个说话人的声学特征分布可以看作是从一个"通用背景模型"（UBM, Universal Background Model）出发，沿着某些方向偏移后得到的。i-vector 就是描述这个偏移方向和幅度的低维向量。

具体做法：

1. 训练 UBM：在所有训练数据上训练一个大的 GMM（通常 256~2048 个高斯分量），作为所有说话人的"平均"声学模型。

2. 训练全变量空间矩阵 T（Total Variability Matrix）：
   
   - 假设每个说话人的 GMM 超向量（将所有高斯分量的均值拼接成一个大向量）可以表示为：M_s = M_ubm + T · w_s
   - M_ubm 是 UBM 的超向量（固定的）
   - T 是全变量空间矩阵（从训练数据中学习，维度为 CF × D，其中 C 是高斯分量数，F 是特征维度，D 是 i-vector 维度）
   - w_s 就是说话人 s 的 i-vector（通常 100 维）
   - T 矩阵通过 EM 算法从训练数据中估计

3. 提取 i-vector：对一段新的语音，先计算其在 UBM 各高斯分量上的零阶和一阶统计量（Baum-Welch 统计量），然后通过后验推断得到该段语音的 i-vector w。

4. 在 ASR 中的使用方式：将 i-vector（100 维）拼接到每帧的声学特征（如 40 维 FBank）上，形成 140 维的输入特征送入 DNN/TDNN。这样神经网络在做声学建模时能同时看到"这帧声音长什么样"和"这是什么样的说话人"，从而隐式地做说话人自适应。

```
i-vector 在 ASR 中的使用流程：

训练阶段：
  全部训练数据 → 训练 UBM (大GMM)
  全部训练数据 + UBM → 训练 T 矩阵
  每个说话人的数据 + UBM + T → 提取 i-vector

使用阶段：
  每帧特征 [40维 FBank]  +  该说话人的 i-vector [100维]
                    ↓ 拼接
            [140维] → 送入 TDNN/DNN
```

```bash
# Kaldi 中 i-vector 的训练和提取
# 1. 训练 UBM
steps/online/nnet2/train_diag_ubm.sh --num-frames 500000 \
  data/train 256 exp/nnet3/diag_ubm

# 2. 训练 i-vector 提取器（T矩阵）
steps/online/nnet2/train_ivector_extractor.sh \
  --ivector-dim 100 \
  data/train exp/nnet3/diag_ubm exp/nnet3/ivector_extractor

# 3. 提取 i-vector
steps/online/nnet2/extract_ivectors_online.sh \
  data/train exp/nnet3/ivector_extractor exp/nnet3/ivectors_train
```

> i-vector vs fMLLR：fMLLR 是在特征空间做显式的线性变换，需要两遍解码；i-vector 是将说话人信息编码为向量拼接到特征上，让神经网络自己学习如何利用说话人信息，只需一遍解码。在 DNN/Chain Model 时代，i-vector 是更主流的说话人自适应方法。在更现代的端到端系统中，i-vector 逐渐被 x-vector（基于 DNN 的说话人嵌入）或模型内部的自适应机制所取代。

## 5. 数据格式与准备

### 5.1 核心数据文件

Kaldi 对数据格式有严格要求，所有文件都是纯文本，按行组织：

```
data/train/
│
├── wav.scp          # 音频路径映射（必需）
│   格式: <utterance_id> <wav_path_or_command>
│   示例:
│     utt_001 /data/audio/001.wav
│     utt_002 /data/audio/002.wav
│     utt_003 sox /data/raw/003.flac -t wav -r 16000 - |
│   注意: 支持管道命令（以 | 结尾），可以在读取时做格式转换
│
├── text             # 转写文本（必需）
│   格式: <utterance_id> <transcription>
│   示例:
│     utt_001 今天天气真好
│     utt_002 我想订一张机票
│     utt_003 语音识别技术
│
├── utt2spk          # 音频到说话人映射（必需）
│   格式: <utterance_id> <speaker_id>
│   示例:
│     utt_001 spk_01
│     utt_002 spk_01
│     utt_003 spk_02
│
├── spk2utt          # 说话人到音频映射（由 utt2spk 自动生成）
│   格式: <speaker_id> <utt_id_1> <utt_id_2> ...
│   示例:
│     spk_01 utt_001 utt_002
│     spk_02 utt_003
│   生成命令: utils/utt2spk_to_spk2utt.pl utt2spk > spk2utt
│
├── segments         # 长音频分段信息（可选，用于长录音切分）
│   格式: <utterance_id> <recording_id> <start_time> <end_time>
│   示例:
│     utt_001 recording_01 0.50 3.20
│     utt_002 recording_01 3.50 6.80
│
└── reco2dur         # 录音时长（可选）
    格式: <recording_id> <duration_seconds>
    示例:
      recording_01 120.5
```

> 所有文件中的 utterance_id 必须按字典序排列，这是 Kaldi 的硬性要求。可以用 `utils/fix_data_dir.sh` 自动修复排序问题。

### 5.2 ark/scp 文件格式

Kaldi 使用独特的 ark/scp 格式存储特征和模型数据：

```
scp 文件（索引文件）：
  指向 ark 文件中每条数据的位置
  格式: <key> <ark_path>:<byte_offset>
  示例:
    utt_001 /data/feats.ark:15
    utt_002 /data/feats.ark:4823

ark 文件（数据文件）：
  实际存储特征矩阵的二进制（或文本）文件
  每条数据 = key + 特征矩阵

读写示例（命令行）：
  # 查看 ark 文件内容（文本格式）
  copy-feats ark:feats.ark ark,t:- | head -20

  # 从 scp 读取，写入 ark
  copy-feats scp:feats.scp ark:feats_copy.ark

  # 管道操作（Kaldi 的精髓）
  compute-fbank-feats scp:wav.scp ark:- | \
    apply-cmvn ark:cmvn.ark ark:- ark:feats_cmvn.ark
```

> Kaldi 大量使用 Unix 管道（pipe），通过 `ark:-` 表示标准输入/输出。这种设计避免了中间文件的磁盘 I/O，非常高效。

### 5.3 数据验证

```bash
# 验证数据目录格式是否正确
utils/validate_data_dir.sh data/train

# 修复常见问题（排序、缺失文件等）
utils/fix_data_dir.sh data/train
```

## 6. 完整训练流程

### 6.1 训练流水线总览

Kaldi 的训练是渐进式的，每个阶段都建立在前一阶段的基础上：

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kaldi 训练流水线                               │
│                                                                  │
│  Stage 1: 数据准备                                               │
│  ├── 生成 wav.scp, text, utt2spk, spk2utt                     │
│  └── 验证数据格式                                               │
│                                                                  │
│  Stage 2: 特征提取                                               │
│  ├── 提取 MFCC 或 FBank 特征                                   │
│  └── 计算 CMVN 统计量                                           │
│                                                                  │
│  Stage 3: 语言模型准备                                           │
│  ├── 构建发音词典 (lexicon.txt)                                 │
│  ├── 编译 L.fst（词典 FST）                                    │
│  ├── 训练 n-gram 语言模型                                       │
│  └── 编译 G.fst（语言模型 FST）                                 │
│                                                                  │
│  Stage 4: 单音素训练 (Monophone)                                 │
│  ├── 均匀分割音频，初始化 GMM 参数                               │
│  ├── EM 迭代训练（约40次）                                      │
│  └── 输出: mono/final.mdl                                       │
│                                                                  │
│  Stage 5: 三音素训练 (Triphone)                                  │
│  ├── 用单音素模型生成对齐                                        │
│  ├── 构建决策树（聚类上下文相关状态）                             │
│  ├── 迭代训练三音素 GMM-HMM                                     │
│  └── 输出: tri1/final.mdl                                       │
│                                                                  │
│  Stage 6: LDA+MLLT 训练                                         │
│  ├── 用 tri1 模型生成对齐                                       │
│  ├── 训练 LDA 和 MLLT 变换矩阵                                 │
│  └── 输出: tri2/final.mdl                                       │
│                                                                  │
│  Stage 7: SAT 训练（说话人自适应）                                │
│  ├── 用 tri2 模型生成对齐                                       │
│  ├── 估计每个说话人的 fMLLR 变换                                │
│  ├── 在变换后的特征上训练 GMM                                    │
│  └── 输出: tri3/final.mdl（最终 GMM-HMM 模型）                 │
│                                                                  │
│  Stage 8: DNN/Chain 训练（可选，现代方法）                        │
│  ├── 用 tri3 生成对齐（作为 DNN 的训练标签）                    │
│  ├── 训练 TDNN / TDNN-LSTM 网络                                 │
│  └── 输出: chain/final.mdl                                      │
│                                                                  │
│  Stage 9: 解码与评估                                             │
│  ├── 编译 HCLG 解码图                                           │
│  ├── 在测试集上解码                                              │
│  └── 计算 WER/CER                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 各阶段详解

#### Stage 4: 单音素训练（Monophone）

单音素模型是最简单的声学模型，不考虑上下文，是后续所有模型的起点。

```bash
# Kaldi 命令
steps/train_mono.sh --nj 10 data/train data/lang exp/mono

# 参数说明:
#   --nj 10: 使用10个并行任务
#   data/train: 训练数据目录
#   data/lang: 语言相关文件（L.fst等）
#   exp/mono: 输出目录
```

训练过程：

1. 将每条音频均匀分割为音素段（因为还没有对齐信息）
2. 用分割后的数据初始化每个音素状态的 GMM 参数
3. EM 迭代：
   - E 步（Expectation）：用前向-后向算法计算每帧属于各状态的后验概率
   - M 步（Maximization）：根据后验概率更新 GMM 参数（均值、方差、权重）
4. 每隔几次迭代，增加高斯分量数（从少到多，逐步细化）
5. 约 40 次迭代后收敛

#### Stage 5: 三音素训练（Triphone）

```bash
# 先用单音素模型生成对齐
steps/align_si.sh --nj 10 data/train data/lang exp/mono exp/mono_ali

# 训练三音素模型
steps/train_deltas.sh 2500 20000 data/train data/lang exp/mono_ali exp/tri1

# 参数说明:
#   2500: 目标叶节点数（senone数量）
#   20000: 总高斯分量数
```

关键步骤：

1. 用单音素模型对训练数据做强制对齐（forced alignment），得到每帧对应的音素状态
2. 构建决策树，将三音素状态聚类为约 2500 个 senone
3. 在对齐数据上训练三音素 GMM-HMM

> 什么是强制对齐（Forced Alignment）？给定音频和已知的文本，找出每个音素在音频中的精确时间位置。这是 Kaldi 训练流程中反复使用的关键操作——每训练完一个更好的模型，就用它重新对齐数据，然后在更准确的对齐上训练下一个模型。

#### Stage 6-7: LDA+MLLT 和 SAT

```bash
# LDA+MLLT
steps/align_si.sh --nj 10 data/train data/lang exp/tri1 exp/tri1_ali
steps/train_lda_mllt.sh 2500 20000 data/train data/lang exp/tri1_ali exp/tri2

# SAT (说话人自适应训练)
steps/align_si.sh --nj 10 data/train data/lang exp/tri2 exp/tri2_ali
steps/train_sat.sh 2500 20000 data/train data/lang exp/tri2_ali exp/tri3
```

SAT 的核心思想：不同说话人的声学特征差异很大（音高、语速、口音等）。fMLLR 为每个说话人学习一个线性变换，将其特征"标准化"后再训练模型。这样模型只需要学习"标准化后"的声学模式，不用同时处理说话人差异。

#### Stage 8: DNN/Chain 训练

Chain Model（LF-MMI）是 Kaldi 最先进的声学模型训练方法：

```bash
# Chain 模型训练（简化版）
local/chain/run_tdnn.sh

# 内部流程：
# 1. 用 tri3 模型生成 lattice 对齐
# 2. 构建 Chain 训练用的分母图（denominator graph）
# 3. 训练 TDNN 网络
# 4. 使用 LF-MMI 准则优化
```

```
Chain Model 的网络结构（TDNN 示例）：

输入: FBank 特征 [T, 40]
    ↓
TDNN层1: context={-1,0,1}, dim=512
    ↓
TDNN层2: context={-1,0,1}, dim=512
    ↓
TDNN层3: context={-3,0,3}, dim=512    ← 感受野扩大
    ↓
TDNN层4: context={-3,0,3}, dim=512
    ↓
TDNN层5: context={-6,-3,0}, dim=512   ← 感受野进一步扩大
    ↓
TDNN层6: context={-6,-3,0}, dim=512
    ↓
输出层: Linear(512, num_senones)
    ↓
输出: 每帧对各 senone 的后验概率
```

> TDNN（Time-Delay Neural Network）通过在不同层使用不同的时间偏移（context），逐层扩大感受野。例如第1层看 {-1,0,1}（3帧），第3层看 {-3,0,3}（跨7帧），最终能覆盖较长的上下文，同时参数量远小于全连接网络。

LF-MMI 训练准则：

```
传统 CE（交叉熵）训练:
  优化每帧的分类准确率
  Loss = -Σ log P(正确状态 | 当前帧)

LF-MMI（Lattice-Free MMI）训练:
  优化序列级别的区分性
  Loss = -log [P(正确词序列 | 音频) / P(所有可能词序列 | 音频)]
       = -log P_numerator / P_denominator

  分子（numerator）: 正确转写对应的所有合法对齐路径的概率
  分母（denominator）: 所有可能转写的所有对齐路径的概率

  直觉：让正确答案的概率尽可能大于所有其他答案的概率
```

## 7. 解码流程

### 7.1 解码全流程

```
┌─────────────────────────────────────────────────────┐
│                  Kaldi 解码流程                       │
│                                                      │
│  测试音频                                            │
│      ↓                                               │
│  特征提取 (MFCC/FBank + CMVN)                       │
│      ↓                                               │
│  [如果是SAT模型] 估计 fMLLR 变换并应用               │
│      ↓                                               │
│  声学模型打分                                        │
│  对每帧计算 log P(state | feature)                   │
│      ↓                                               │
│  WFST Viterbi 解码                                   │
│  在 HCLG 图上搜索最优路径                             │
│      ↓                                               │
│  生成 Lattice（词格）                                │
│  保留多个候选路径及其分数                              │
│      ↓                                               │
│  Lattice 重打分（可选）                               │
│  使用更大的语言模型重新打分                            │
│      ↓                                               │
│  提取最优路径 → 输出文本                              │
└─────────────────────────────────────────────────────┘
```

### 7.2 解码命令示例

```bash
# 编译解码图
utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph

# 解码测试集
steps/decode.sh --nj 4 exp/tri3/graph data/test exp/tri3/decode_test

# 查看最优 WER
cat exp/tri3/decode_test/scoring_kaldi/best_wer
```

单条音频解码的完整命令：

```bash
# 1. 提取特征
compute-fbank-feats --num-mel-bins=80 scp:wav.scp ark:feats.ark

# 2. 应用 CMVN
apply-cmvn --norm-vars=true ark:cmvn.ark ark:feats.ark ark:feats_cmvn.ark

# 3. 解码
latgen-faster-mapped \
  --beam=13 \
  --lattice-beam=6 \
  --acoustic-scale=0.083 \
  --word-symbol-table=words.txt \
  final.mdl \
  HCLG.fst \
  ark:feats_cmvn.ark \
  ark,t:result.txt
```

关键解码参数：

| 参数             | 典型值   | 说明                       |
| -------------- | ----- | ------------------------ |
| beam           | 13    | 搜索束宽，越大搜索越充分但越慢          |
| lattice-beam   | 6     | Lattice 剪枝束宽             |
| acoustic-scale | 0.083 | 声学模型权重（相对于语言模型），约等于 1/12 |
| max-active     | 7000  | 每帧最大活跃状态数                |
| min-active     | 200   | 每帧最小活跃状态数                |

> acoustic-scale 为什么通常远小于 1？因为声学模型的对数概率值通常比语言模型大很多（量级不同），需要缩放声学分数来平衡两者的贡献。

### 7.3 Lattice（词格）

Lattice 是解码过程中保留的多个候选路径的紧凑表示：

```
Lattice 示意图：

        "今天"(0.8)     "天气"(0.9)     "真好"(0.7)
  起点 ──────────── ○ ──────────── ○ ──────────── 终点
    │                │                │
    │  "金天"(0.2)   │  "天汽"(0.1)   │  "真号"(0.3)
    └──────────── ○ ──────────── ○ ──────────┘

每条边上有：
  - 词（word）
  - 声学分数（acoustic score）
  - 语言模型分数（LM score）
  - 时间信息（起止帧）
```

Lattice 的用途：

- 语言模型重打分：用更大的 LM 重新计算路径分数
- 置信度计算：估计每个词的识别置信度
- 关键词检索：在 Lattice 中搜索特定关键词
- 最小贝叶斯风险（MBR）解码：选择期望错误率最低的路径

## 8. Kaldi 常用命令速查

### 8.1 数据准备

```bash
# 验证数据目录
utils/validate_data_dir.sh data/train

# 修复数据目录
utils/fix_data_dir.sh data/train

# utt2spk 转 spk2utt
utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt

# 划分数据子集
utils/subset_data_dir.sh data/train 1000 data/train_1k
```

### 8.2 特征操作

```bash
# 提取 MFCC
steps/make_mfcc.sh --nj 10 data/train exp/make_mfcc mfcc

# 提取 FBank
steps/make_fbank.sh --nj 10 --fbank-config conf/fbank.conf data/train exp/make_fbank fbank

# 计算 CMVN
steps/compute_cmvn_stats.sh data/train exp/make_mfcc mfcc

# 查看特征维度
feat-to-dim scp:data/train/feats.scp -

# 查看特征内容（文本格式）
copy-feats scp:data/train/feats.scp ark,t:- | head -5
```

### 8.3 训练与对齐

```bash
# 单音素训练
steps/train_mono.sh --nj 10 data/train data/lang exp/mono

# 对齐
steps/align_si.sh --nj 10 data/train data/lang exp/mono exp/mono_ali

# 三音素训练
steps/train_deltas.sh 2500 20000 data/train data/lang exp/mono_ali exp/tri1

# LDA+MLLT
steps/train_lda_mllt.sh 2500 20000 data/train data/lang exp/tri1_ali exp/tri2

# SAT
steps/train_sat.sh 2500 20000 data/train data/lang exp/tri2_ali exp/tri3
```

### 8.4 解码与评估

```bash
# 编译解码图
utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph

# 解码
steps/decode.sh --nj 4 exp/tri3/graph data/test exp/tri3/decode_test

# 查看 WER
cat exp/tri3/decode_test/scoring_kaldi/best_wer

# 语言模型重打分
steps/lmrescore.sh data/lang_small data/lang_big data/test exp/tri3/decode_test exp/tri3/decode_test_big_lm
```

## 9. 实战：AISHELL-1 训练示例

以中文 AISHELL-1 数据集为例，展示完整的 Kaldi 训练流程。

[AISHELL-1 语音识别从入门到精通：Kaldi实战指南-CSDN博客](https://blog.csdn.net/weixin_29095049/article/details/159490909)

### 9.1 数据集简介

```
AISHELL-1:
  - 语言: 中文普通话
  - 时长: 178小时
  - 说话人: 400人
  - 录音环境: 安静室内
  - 采样率: 16kHz
  - 划分: train(150h) / dev(20h) / test(8h)
  - 下载: https://www.openslr.org/33/
```

### 9.2 训练脚本结构

```bash
# Kaldi 的 AISHELL recipe 位于 egs/aishell/s5/
egs/aishell/s5/
├── run.sh              # 主训练脚本
├── cmd.sh              # 计算资源配置（本地/集群）
├── path.sh             # 环境变量设置
├── conf/               # 配置文件
│   ├── mfcc.conf       # MFCC 参数
│   └── fbank.conf      # FBank 参数
├── local/              # 数据集特定脚本
│   ├── aishell_data_prep.sh      # 数据准备
│   ├── aishell_prepare_dict.sh   # 词典准备
│   └── chain/
│       └── run_tdnn.sh           # Chain 模型训练
└── steps/ → ../../wsj/s5/steps/  # 通用训练步骤（符号链接）
    utils/ → ../../wsj/s5/utils/  # 通用工具（符号链接）
```

### 9.3 run.sh 核心流程

```bash
#!/bin/bash
# AISHELL-1 训练主脚本（简化版）

stage=0
stop_stage=100

# Stage 0: 数据准备
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "Stage 0: 数据准备"
  local/aishell_data_prep.sh /path/to/aishell data
  # 生成: data/{train,dev,test}/{wav.scp,text,utt2spk,spk2utt}
fi

# Stage 1: 特征提取
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "Stage 1: MFCC 特征提取"
  for x in train dev test; do
    steps/make_mfcc.sh --nj 10 data/$x exp/make_mfcc/$x mfcc
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc
  done
fi

# Stage 2: 语言模型
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "Stage 2: 语言模型准备"
  local/aishell_prepare_dict.sh
  utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang
  local/aishell_train_lms.sh
  # 生成: data/lang/{L.fst, G.fst, ...}
fi

# Stage 3: 单音素
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "Stage 3: 单音素训练"
  steps/train_mono.sh --nj 10 data/train data/lang exp/mono
  # 解码验证
  utils/mkgraph.sh data/lang exp/mono exp/mono/graph
  steps/decode.sh --nj 4 exp/mono/graph data/test exp/mono/decode_test
fi

# Stage 4: 三音素
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "Stage 4: 三音素训练"
  steps/align_si.sh --nj 10 data/train data/lang exp/mono exp/mono_ali
  steps/train_deltas.sh 2500 20000 data/train data/lang exp/mono_ali exp/tri1
fi

# Stage 5: LDA+MLLT
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  echo "Stage 5: LDA+MLLT"
  steps/align_si.sh --nj 10 data/train data/lang exp/tri1 exp/tri1_ali
  steps/train_lda_mllt.sh 2500 20000 data/train data/lang exp/tri1_ali exp/tri2
fi

# Stage 6: SAT
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  echo "Stage 6: SAT (说话人自适应)"
  steps/align_si.sh --nj 10 data/train data/lang exp/tri2 exp/tri2_ali
  steps/train_sat.sh 2500 20000 data/train data/lang exp/tri2_ali exp/tri3
  # 最终 GMM-HMM 解码
  utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph
  steps/decode_fmllr.sh --nj 4 exp/tri3/graph data/test exp/tri3/decode_test
fi

# Stage 7: Chain Model (DNN)
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  echo "Stage 7: Chain TDNN"
  local/chain/run_tdnn.sh
fi
```

### 9.4 典型训练结果

```
AISHELL-1 各阶段 CER（字错误率）参考值：

模型                    CER (%)
─────────────────────────────────
Monophone (单音素)      ~35%
Triphone (三音素)       ~25%
LDA+MLLT               ~20%
SAT (fMLLR)            ~18%
Chain TDNN             ~8%
Chain TDNN-LSTM        ~7%

对比端到端模型：
Conformer (WeNet)      ~4.5%
Paraformer (FunASR)    ~4.2%
```

> 可以看到，Kaldi 的 Chain Model 已经能达到不错的效果（CER ~8%），但与最新的端到端模型仍有差距。不过 Kaldi 在小数据场景（几十小时）下的表现往往优于端到端模型。

## 10. DNN-HMM 深入理解

### 10.1 从 GMM 到 DNN

DNN-HMM 的核心思想：用深度神经网络替代 GMM 来计算声学模型的发射概率。

```
GMM-HMM:
  特征 x → GMM → P(x | state_j)  （生成式模型，直接建模特征的分布）

DNN-HMM:
  特征 x → DNN → P(state_j | x)  （判别式模型，直接预测状态后验概率）

  转换为似然:
  P(x | state_j) ∝ P(state_j | x) / P(state_j)
  其中 P(state_j) 是状态的先验概率（从训练数据统计得到）
```

> 为什么 DNN 比 GMM 好？GMM 用固定数量的高斯分量拟合特征分布，表达能力有限。DNN 是通用函数逼近器，理论上可以拟合任意复杂的决策边界。实验表明，DNN-HMM 的错误率通常比 GMM-HMM 低 20-30%。

### 10.2 TDNN 详解

TDNN 是 Kaldi 中最常用的 DNN 架构：

```
TDNN 的核心：使用不同的时间偏移（context）逐层扩大感受野

层1: context = {-1, 0, 1}
  每个输出位置看3帧输入
  感受野: 3帧 = 30ms

层2: context = {-1, 0, 1}
  每个输出位置看层1的3个位置
  感受野: 5帧 = 50ms

层3: context = {-3, 0, 3}
  跳跃式连接，每个输出位置看层2的3个位置（间隔3）
  感受野: 11帧 = 110ms

层4: context = {-3, 0, 3}
  感受野: 17帧 = 170ms

...逐层扩大，最终覆盖约1秒的上下文

与全连接网络对比：
  全连接: 拼接所有帧 → 参数量巨大
  TDNN: 每层只看少量偏移 → 参数量小，计算快，效果好
```

### 10.3 Chain Model 的训练细节

```
Chain Model 的特殊设计：

1. 帧率降低：
   输入帧率: 100帧/秒（10ms帧移）
   Chain 输出帧率: 33帧/秒（30ms帧移，3倍下采样）
   好处: 减少计算量，加速训练和解码

2. 分母图（Denominator Graph）：
   - 一个小型的 WFST，编码了所有可能的音素序列
   - 用于计算 LF-MMI Loss 的分母部分
   - 通常使用 phone-level 的 bigram 语言模型

3. 正则化：
   - Leaky HMM: 允许从任何状态跳到任何状态（小概率），防止训练不稳定
   - L2 正则化
   - xent 辅助损失: 同时训练一个交叉熵分支，稳定训练

4. 训练配置：
   - 学习率: 0.001 → 0.0001（线性衰减）
   - 优化器: Natural Gradient SGD
   - 训练轮数: 通常 3-4 个 epoch
   - 数据增强: Speed Perturbation (0.9x, 1.0x, 1.1x)
```

## 11. Kaldi 与端到端模型对比

### 11.1 架构对比

| 方面    | Kaldi (传统)                | 端到端 (Conformer等)          |
| ----- | ------------------------- | ------------------------- |
| 架构    | 多模块组合（AM + LM + 词典 + 解码器） | 单一神经网络                    |
| 训练    | 多阶段渐进式，需要对齐               | 端到端，不需要对齐                 |
| 词典    | 需要发音词典（词→音素映射）            | 不需要（使用 BPE/字符）            |
| 语言模型  | 外部 n-gram LM              | 内置于 Decoder               |
| 解码    | WFST 搜索                   | Beam Search               |
| 可解释性  | 高（每个模块可独立分析）              | 低（黑盒）                     |
| 数据需求  | 较少（几十~几百小时可用）             | 较多（千小时以上效果好）              |
| 效果上限  | 较低                        | 较高                        |
| 工程复杂度 | 高（多模块维护）                  | 低（单模型）                    |
| 流式支持  | 天然支持                      | 需要特殊设计                    |
| 热词定制  | 容易（改词典/LM）                | 较难（需要 Contextual Biasing） |
| 部署    | 复杂（多个文件）                  | 简单（一个模型文件）                |

### 11.2 各自的优势场景

```
Kaldi 更适合的场景：
├── 小数据量（< 100小时）
├── 需要精确控制词典和语言模型
├── 热词/专有名词识别要求高
├── 需要高可解释性（如学术研究）
├── 已有成熟的 Kaldi 部署环境
└── 需要 Lattice 做后续处理（关键词检索、置信度等）

端到端模型更适合的场景：
├── 大数据量（> 1000小时）
├── 追求最高识别准确率
├── 快速迭代和部署
├── 多语言/多方言支持
├── 需要简单的训练和部署流程
└── 有充足的 GPU 资源
```

## 12. 为什么还要学 Kaldi

尽管端到端模型已成为主流，学习 Kaldi 仍然有重要价值：

1. 理解 ASR 基础概念：HMM、GMM、对齐、WFST、决策树等概念是 ASR 的基石，理解它们有助于深入理解现代模型
2. 数据格式标准：很多数据集和工具仍使用 Kaldi 格式（wav.scp、text 等）
3. 特征提取：Kaldi 的 FBank/MFCC 实现是事实标准，很多端到端框架也在使用
4. 调试能力：理解传统 pipeline 有助于诊断端到端模型的问题（如对齐问题、语言模型偏差等）
5. 热词/定制：WFST 方法在热词识别等场景仍有独特优势
6. k2/icefall/sherpa：Kaldi 原作者 Daniel Povey 的新项目将 WFST 与端到端结合，代表了 ASR 的未来方向之一

## 13. 从 Kaldi 到现代框架的演进路线

```
技术演进脉络：

声学模型:
  GMM → DNN → TDNN → LSTM → Transformer → Conformer → Zipformer
  (Kaldi)  (Kaldi)  (Kaldi)  (Kaldi)    (ESPnet)   (ESPnet)  (k2/icefall)

训练准则:
  ML(最大似然) → MMI(最大互信息) → LF-MMI → CTC → Transducer → CTC+Attention
  (Kaldi)       (Kaldi)          (Kaldi)  (端到端)  (端到端)    (端到端)

解码方式:
  WFST Viterbi → WFST + DNN → CTC Greedy → Beam Search → CTC+Attention Rescoring
  (Kaldi)       (Kaldi)      (端到端)      (端到端)      (WeNet/FunASR)

特征:
  MFCC(39维) → FBank(40维) → FBank(80维) → 自学习特征(Wav2Vec/HuBERT)
  (Kaldi)      (Kaldi)       (端到端)       (自监督学习)

框架:
  Kaldi(2011) → ESPnet(2018) → WeNet(2021) → FunASR(2023)
                k2/icefall(2020) → sherpa(2022)
```

## 14. 学习资源与推荐路径

### 14.1 推荐学习路径

```
入门阶段（1-2周）：
├── 阅读本文档，理解 Kaldi 的整体架构
├── 安装 kaldi-native-fbank，运行 Python 特征提取 demo
├── 理解 HMM、GMM、WFST 的基本概念
└── 熟悉 Kaldi 数据格式（wav.scp、text 等）

进阶阶段（2-4周）：
├── 完整安装 Kaldi
├── 跑通 AISHELL-1 的 run.sh（从数据准备到解码）
├── 理解每个训练阶段的作用
├── 学习修改训练参数（高斯数、senone 数等）
└── 理解 HCLG 解码图的构建过程

深入阶段（1-2月）：
├── 阅读 Kaldi 源码（从 feat/ 和 gmm/ 开始）
├── 理解 Chain Model / LF-MMI 的训练原理
├── 尝试在自己的数据上训练模型
├── 学习 k2/icefall，理解 WFST + 端到端的结合
└── 对比 Kaldi 与端到端框架的效果差异
```

### 14.2 重要参考资料

```
官方资源：
├── Kaldi 官网: https://kaldi-asr.org/
├── Kaldi GitHub: https://github.com/kaldi-asr/kaldi
├── Kaldi 官方文档: https://kaldi-asr.org/doc/
└── Kaldi 论文: "The Kaldi Speech Recognition Toolkit" (2011)

后续项目：
├── k2: https://github.com/k2-fsa/k2
├── icefall: https://github.com/k2-fsa/icefall
├── sherpa: https://github.com/k2-fsa/sherpa-onnx
└── lhotse: https://github.com/lhotse-speech/lhotse（数据处理工具）

社区资源：
├── Kaldi 帮助论坛: https://groups.google.com/g/kaldi-help
└── Daniel Povey 的讲座和教程（YouTube 可搜索）
```

---

> Kaldi 虽然不再是 ASR 的最前沿，但它是理解语音识别技术的最佳起点。掌握 Kaldi 的核心概念后，再学习端到端模型会事半功倍。正如学习编程要理解计算机底层原理一样，学习 ASR 也需要理解 Kaldi 所代表的经典方法论。