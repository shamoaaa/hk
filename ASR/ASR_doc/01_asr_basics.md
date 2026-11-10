# ASR（自动语音识别）基础知识

## 1. 什么是ASR

ASR（Automatic Speech Recognition，自动语音识别）是一种将人类语音信号转换为文本的技术。简单来说，就是让机器"听懂"人说的话，并把它写下来。

日常生活中的ASR应用：

- 语音助手（Siri、小爱同学、天猫精灵）
- 语音输入法
- 会议实时字幕
- 电话客服系统
- 视频字幕自动生成

## 2. ASR的发展历程

| 时期          | 技术路线            | 代表                      |
| ----------- | --------------- | ----------------------- |
| 1950s-1980s | 基于模板匹配          | DTW（动态时间规整）             |
| 1980s-2010s | 基于统计模型          | GMM-HMM（高斯混合模型+隐马尔可夫模型） |
| 2010s-2018  | 混合深度学习          | DNN-HMM、CTC             |
| 2018-至今     | 端到端（End-to-End） | Transformer、Conformer   |

> **什么是"端到端"？**
> 传统ASR系统由多个独立模块组成（声学模型、语言模型、发音词典等），每个模块需要单独训练和调优。
> 端到端ASR将整个过程合并为一个神经网络，直接从音频输入到文本输出，大大简化了系统架构。

## 3. 语音信号基础

### 3.1 声音的本质

声音是空气中的压力波动。麦克风将这种物理振动转换为电信号，再通过模数转换（ADC）变成数字信号。

关键参数：

- **采样率（Sample Rate）**：每秒采集多少个样本点。常见值：16kHz（语音识别标准）、44.1kHz（CD音质）
- **位深（Bit Depth）**：每个样本点用多少位表示。常见值：16bit
- **声道数（Channels）**：单声道（Mono）或立体声（Stereo）。ASR通常使用单声道

> **为什么ASR常用16kHz？**
> 人类语音的有效频率范围大约在300Hz-8000Hz。根据奈奎斯特采样定理，采样率需要至少是最高频率的2倍，所以16kHz足以覆盖语音信息。更高的采样率会增加计算量但对识别效果提升有限。

### 3.2 时域与频域

- **时域（Time Domain）**：横轴是时间，纵轴是振幅。就是我们看到的波形图
- **频域（Frequency Domain）**：横轴是频率，纵轴是能量。通过傅里叶变换（FFT）从时域转换而来

> **什么是傅里叶变换？**
> 任何复杂的波形都可以分解为多个不同频率的正弦波的叠加。傅里叶变换就是找出这些正弦波各自的频率和强度。就像把一首交响乐分解成每个乐器的声音一样。

### 3.3 语音特征提取

原始音频波形包含大量冗余信息，不适合直接输入模型。我们需要提取更紧凑、更有意义的特征。

#### Mel频率倒谱系数（MFCC）

MFCC是传统ASR中最常用的特征，提取步骤：

```
原始音频 → 预加重 → 分帧 → 加窗 → FFT → Mel滤波器组 → 取对数 → DCT → MFCC
```

每一步的含义：

1. **预加重（Pre-emphasis）**：增强高频部分，补偿语音信号高频衰减
2. **分帧（Framing）**：将连续音频切成短片段（通常25ms一帧，帧移10ms）
3. **加窗（Windowing）**：对每帧乘以窗函数（如汉明窗），减少边缘不连续性
4. **FFT**：将每帧从时域转换到频域
5. **Mel滤波器组**：模拟人耳对频率的非线性感知（低频分辨率高，高频分辨率低）
6. **取对数**：模拟人耳对响度的对数感知
7. **DCT（离散余弦变换）**：去相关，压缩特征维度

> **DCT为什么能去相关和压缩维度？**
> 
> FBank特征的相邻维度之间存在很强的相关性——因为相邻的Mel三角滤波器有大量重叠区域，它们提取的能量值自然高度相似。这种冗余对GMM-HMM不友好（GMM通常假设各维度独立）。
> 
> DCT的本质是把信号分解到一组余弦基函数上。它有一个关键性质：**对于相关性强的信号，DCT变换后的能量会集中在前几个系数上**。这类似于傅里叶变换把时域信号的能量集中到少数频率分量上。
> 
> 具体来说：
> 
> - FBank的80维向量经过DCT后，仍然是80维
> - 但绝大部分有用信息集中在前13个DCT系数中
> - 后面的系数主要是高频细节和噪声，可以直接丢弃
> - 这就实现了从80维压缩到13维，同时各维度之间的相关性大幅降低
> 
> 直觉类比：就像JPEG图片压缩——原始像素之间高度相关（相邻像素颜色相近），DCT变换后能量集中在少数系数上，丢掉小系数就实现了压缩，而人眼几乎看不出差别。MFCC对FBank做的事情本质上一样。
> 
> 但在深度学习时代，神经网络自己就能学到这种去相关和特征选择，所以不再需要DCT这一步，直接用FBank即可。

> **什么是Mel刻度？**
> 人耳对频率的感知不是线性的。比如，我们能轻松区分100Hz和200Hz的差异，但很难区分8000Hz和8100Hz的差异。Mel刻度就是模拟这种非线性感知的数学映射。

#### FBank（Filter Bank）特征

FBank是MFCC的前身，省略了最后的DCT步骤。在深度学习时代，FBank比MFCC更常用。

##### FBank 提取流程

```
原始音频 → 预加重 → 分帧 → 加窗 → FFT → 功率谱 → Mel滤波器组 → 取对数 → FBank
```

与MFCC的区别：FBank在取对数后就停止了，不再做DCT变换。

##### FBank 的计算细节

1. **功率谱计算**：
   
   ```
   对每帧信号做N点FFT（通常N=512），得到复数频谱 X(k)
   功率谱 P(k) = |X(k)|² / N
   只取前 N/2+1 = 257 个点（对称性，后半部分是冗余的）
   ```

2. **Mel滤波器组的构造**：
   
   ```
   频率(Hz)到Mel的转换公式：
     m = 2595 × log₁₀(1 + f/700)
   
   Mel到频率(Hz)的逆转换：
     f = 700 × (10^(m/2595) - 1)
   
   构造过程：
   (1) 确定频率范围：通常 [20Hz, 8000Hz]（对应16kHz采样率）
   (2) 将频率范围转换到Mel域：[20Hz→26.8Mel, 8000Hz→2840.0Mel]
   (3) 在Mel域上均匀放置 M+2 个点（M=80时，放82个点）
   (4) 将这些Mel点转换回Hz频率
   (5) 每3个相邻点构成一个三角形滤波器
   ```
   
   三角形滤波器示意图：
   
   ```
   能量
    ↑
    │    /\      /\      /\      /\      /\
    │   /  \    /  \    /  \    /  \    /  \
    │  /    \  /    \  /    \  /    \  /    \
    │ /      \/      \/      \/      \/      \
    └──────────────────────────────────────────→ 频率(Hz)
      低频区域：滤波器窄（分辨率高）
                          高频区域：滤波器宽（分辨率低）
   ```
   
   > **为什么低频窄、高频宽？**
   > 因为Mel刻度在低频区域是近似线性的，在高频区域是近似对数的。在Mel域上均匀放置的点，映射回Hz后，低频区域的点更密集，高频区域的点更稀疏。这恰好模拟了人耳对低频更敏感的特性。

3. **滤波器应用**：
   
   ```python
   # 伪代码
   for i in range(M):  # M=80个滤波器
       fbank[i] = sum(power_spectrum * mel_filter[i])
   
   # 即：将功率谱与每个三角形滤波器逐点相乘再求和
   # 每个滤波器输出一个标量值，代表该频段的能量
   ```

4. **取对数**：
   
   ```
   log_fbank[i] = log(fbank[i] + ε)
   
   ε 是一个很小的数（如1e-10），防止对0取对数
   ```

##### FBank 的维度与含义

```
一帧FBank特征 = [f₁, f₂, f₃, ..., f₈₀]  （80维向量）

f₁  → 最低频段（约20-85Hz）的对数能量    ← 基频相关
f₂  → 第二低频段的对数能量
...
f₄₀ → 中频段的对数能量                   ← 语音主要信息集中区
...
f₈₀ → 最高频段（约7500-8000Hz）的对数能量 ← 摩擦音、齿音等
```

每个维度代表一个频段的能量，从低频到高频排列。语音的大部分有用信息集中在中低频段。

##### FBank vs MFCC 详细对比

| 特性    | FBank       | MFCC             |
| ----- | ----------- | ---------------- |
| 计算步骤  | 到取对数为止      | 多一步DCT           |
| 维度    | 通常80维       | 通常13维（+Δ+ΔΔ=39维） |
| 特征相关性 | 相邻维度有相关性    | 各维度去相关           |
| 信息保留  | 更完整         | 有信息损失（DCT截断）     |
| 适用模型  | 深度学习模型      | GMM-HMM等传统模型     |
| 为什么   | 神经网络能自己学去相关 | GMM假设特征独立，需要去相关  |

> **为什么深度学习时代更偏好FBank？**
> 
> - GMM-HMM时代：GMM假设各特征维度独立（使用对角协方差矩阵），所以需要DCT去相关。MFCC的13维紧凑表示也减少了GMM的参数量。
> - 深度学习时代：神经网络不需要特征独立的假设，它能自己学到特征之间的关系。FBank保留了更多原始信息（80维 vs 13维），给了网络更多可学习的空间。实验也证明FBank在深度学习模型上效果更好。

##### FBank 特征的可视化

FBank特征通常以频谱图（Spectrogram）的形式可视化：

```
频率(Mel) ↑
          │ ████          ████████    ████
          │ ██████      ████████████  ██████
          │ ████████  ██████████████████████
          │ ██████████████████████████████████
          │ ████████████████████████████████████
          └──────────────────────────────────────→ 时间
            "今"    "天"   "天"  "气"  "真" "好"

颜色深浅代表能量大小：深色=高能量，浅色=低能量
横轴是时间（帧），纵轴是频率（Mel滤波器编号）
```

从频谱图中可以观察到：

- **元音**（如"a"、"o"）：在特定频率上有明显的能量条纹（共振峰/Formant）
- **辅音**（如"s"、"sh"）：高频区域有弥散的能量
- **静音**：整体能量很低
- **基频（F0）**：最低频的能量条纹，与说话人的音高相关

##### 实际代码示例

```python
import numpy as np
import soundfile as sf

def compute_fbank(wav_path, num_mel_bins=80, frame_length_ms=25, 
                  frame_shift_ms=10, sample_rate=16000):
    """
    手动计算FBank特征（教学用途，实际项目请用成熟库）
    """
    # 1. 读取音频
    waveform, sr = sf.read(wav_path, dtype='float32')
    assert sr == sample_rate, f"采样率应为{sample_rate}，实际为{sr}"

    # 2. 预加重
    pre_emphasis = 0.97
    waveform = np.append(waveform[0], waveform[1:] - pre_emphasis * waveform[:-1])

    # 3. 分帧
    frame_length = int(sample_rate * frame_length_ms / 1000)  # 400个采样点
    frame_shift = int(sample_rate * frame_shift_ms / 1000)    # 160个采样点
    num_frames = (len(waveform) - frame_length) // frame_shift + 1

    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        start = i * frame_shift
        frames[i] = waveform[start:start + frame_length]

    # 4. 加窗（汉明窗）
    hamming = np.hamming(frame_length)
    frames *= hamming

    # 5. FFT
    nfft = 512
    magnitude = np.abs(np.fft.rfft(frames, n=nfft))  # [num_frames, 257]
    power_spectrum = magnitude ** 2 / nfft

    # 6. Mel滤波器组
    mel_filters = _create_mel_filterbank(num_mel_bins, nfft, sample_rate)
    # mel_filters shape: [num_mel_bins, 257]

    fbank_features = np.dot(power_spectrum, mel_filters.T)  # [num_frames, 80]

    # 7. 取对数
    fbank_features = np.log(fbank_features + 1e-10)

    return fbank_features  # shape: [num_frames, 80]


def _create_mel_filterbank(num_mel_bins, nfft, sample_rate, 
                           low_freq=20, high_freq=None):
    """构造Mel三角滤波器组"""
    if high_freq is None:
        high_freq = sample_rate / 2  # 8000Hz

    # Hz → Mel
    low_mel = 2595 * np.log10(1 + low_freq / 700)
    high_mel = 2595 * np.log10(1 + high_freq / 700)

    # 在Mel域均匀放置 num_mel_bins+2 个点
    mel_points = np.linspace(low_mel, high_mel, num_mel_bins + 2)

    # Mel → Hz → FFT bin index
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((nfft + 1) * hz_points / sample_rate).astype(int)

    # 构造三角滤波器
    filters = np.zeros((num_mel_bins, nfft // 2 + 1))
    for i in range(num_mel_bins):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        # 上升沿
        for j in range(left, center):
            filters[i, j] = (j - left) / (center - left)
        # 下降沿
        for j in range(center, right):
            filters[i, j] = (right - j) / (right - center)

    return filters


# 使用示例
# features = compute_fbank("test.wav")
# print(f"FBank特征 shape: {features.shape}")
# 输出: FBank特征 shape: (298, 80)  （3秒音频约298帧）
```

##### 常见的FBank配置参数

| 参数           | 常用值     | 说明             |
| ------------ | ------- | -------------- |
| num_mel_bins | 80      | Mel滤波器数量（特征维度） |
| frame_length | 25ms    | 帧长             |
| frame_shift  | 10ms    | 帧移             |
| sample_rate  | 16000   | 采样率            |
| low_freq     | 20Hz    | 最低频率           |
| high_freq    | 8000Hz  | 最高频率（=采样率/2）   |
| nfft         | 512     | FFT点数          |
| pre_emphasis | 0.97    | 预加重系数          |
| window_type  | hamming | 窗函数类型          |
| dither       | 0.0~1.0 | 抖动值（防止静音帧数值问题） |

> **什么是dither（抖动）？**
> 对于完全静音的帧，所有采样点都是0，FFT后功率谱也全是0，取对数会得到负无穷。dither就是给信号加一个极小的随机噪声，避免这个数值问题。Kaldi默认dither=1.0，很多端到端框架默认dither=0.0。

#### 现代端到端模型的输入

大多数现代端到端ASR模型使用：

- **80维FBank特征**（最常见）
- 或直接使用原始波形（如Wav2Vec 2.0），让模型自己学习特征提取

## 4. 端到端ASR的核心架构

### 4.1 CTC（Connectionist Temporal Classification）

CTC是最早的端到端ASR方法之一。

核心思想：

- 引入一个特殊的"空白"符号（blank，记为`-`）
- 模型在每个时间步输出一个字符（包括blank）
- 通过合并规则得到最终文本：先合并连续重复字符，再去掉blank

```
模型输出：h-ee-ll-l-oo → 合并重复 → h-e-l-o → 去blank → hello
```

优点：

- 不需要音频和文本的精确对齐标注
- 解码速度快

缺点：

- 假设各时间步输出独立（条件独立假设），忽略了输出之间的依赖关系
- 容易产生删除错误（漏字）

### 4.2 Attention-based Encoder-Decoder（AED）

也叫Seq2Seq（序列到序列）模型。

结构：

```
音频特征 → Encoder（编码器） → 注意力机制 → Decoder（解码器） → 文本
```

- **Encoder**：将音频特征序列编码为高级表示
- **Attention**：让Decoder在生成每个字符时，关注Encoder输出的不同位置
- **Decoder**：自回归地逐个生成文本字符

> **什么是注意力机制（Attention）？**
> 想象你在听一段话并做笔记。写每个字的时候，你会"注意"到原始语音中对应的那个部分。注意力机制就是让模型学会这种"对应关系"——在生成每个输出字符时，自动找到输入中最相关的部分。

优点：

- 能建模输出之间的依赖关系
- 识别准确率高

缺点：

- 自回归解码速度慢（必须一个字一个字生成）
- 训练时需要teacher forcing，推理时可能出现exposure bias

缺点：

- 自回归解码速度慢（必须一个字一个字生成）
- 训练时需要teacher forcing，推理时可能出现exposure bias

> **什么是Teacher Forcing？**
> 
> Decoder是自回归的——生成第t个字时，需要把前t-1个字作为输入。那训练时前t-1个字从哪来？有两种选择：
> 
> - **用模型自己的预测**：模型先预测第1个字，把预测结果喂回去预测第2个字，以此类推
> - **用真实标签（Teacher Forcing）**：不管模型预测了什么，始终把真实的正确文本喂给Decoder
> 
> 几乎所有AED模型训练时都用Teacher Forcing，原因是：
> 
> - 训练初期模型很弱，预测结果基本是乱的。如果用错误的预测作为下一步输入，错误会不断累积（"雪崩效应"），模型根本学不到东西
> - Teacher Forcing相当于给模型一个"完美的老师"，每一步都告诉它正确答案是什么，让模型专注于学习"给定正确的前文，如何预测下一个字"
> 
> 用一个例子说明：
> 
> ```
> 目标文本: "今天天气真好"
> 
> Teacher Forcing训练（实际使用）：
>   输入: <sos>          → 模型预测: "今" ✓  （输入是正确的<sos>）
>   输入: <sos> 今       → 模型预测: "天" ✓  （输入是正确的"今"，不管上一步预测对没对）
>   输入: <sos> 今 天    → 模型预测: "天" ✓
>   输入: <sos> 今 天 天 → 模型预测: "气" ✓
>   ...
> 
> 不用Teacher Forcing（假设模型预测错了）：
>   输入: <sos>          → 模型预测: "金" ✗
>   输入: <sos> 金       → 模型预测: "属" ✗  （因为输入就是错的，后面全跑偏）
>   输入: <sos> 金 属    → 模型预测: "材" ✗
>   ...完全崩溃
> ```

> **什么是Exposure Bias？**
> 
> Exposure Bias（曝光偏差）是Teacher Forcing带来的副作用，指训练和推理之间的不一致性：
> 
> - **训练时**：Decoder的输入始终是正确的真实文本（Teacher Forcing）
> - **推理时**：没有"老师"了，Decoder只能用自己上一步的预测作为输入
> 
> 问题在于：模型在训练时从未见过"错误的输入"，所以一旦推理时某一步预测错了，模型不知道如何从错误中恢复，错误会像滚雪球一样越来越大。
> 
> ```
> 训练时模型看到的世界（理想）：
>   每一步的输入都是正确的 → 模型学会了"在正确前文下预测下一个字"
> 
> 推理时模型面对的现实：
>   第3步预测错了 → 第4步的输入就是错的 → 第4步大概率也错 → 连锁反应
> 
> 具体例子：
>   真实文本: "今天天气真好"
>   推理过程:
>     "今" ✓ → "天" ✓ → "填" ✗ → "写" ✗ → "表" ✗ → "格" ✗
>     第3步错了之后，模型进入了从未在训练中见过的状态，后面全部崩溃
> ```
> 
> 常见的缓解方法：
> 
> - **Scheduled Sampling**：训练时以一定概率用模型自己的预测替代真实标签，让模型逐渐适应"不完美的输入"
> - **非自回归解码（NAR）**：如Paraformer，一次性并行输出所有字符，不存在逐步累积错误的问题
> - **CTC辅助**：混合CTC/Attention架构中，CTC分支不依赖前一步输出，能提供额外的稳定性
> - **Beam Search**：推理时保留多个候选路径，降低单步错误导致全局崩溃的风险

### 4.3 混合CTC/Attention

将CTC和Attention结合，取长补短。这是目前最主流的端到端ASR架构。

```
                    ┌→ CTC Loss
音频特征 → Encoder ─┤
                    └→ Attention Decoder → Decoder Loss

总Loss = α × CTC_Loss + (1-α) × Decoder_Loss
```

- CTC帮助Encoder学习更好的对齐
- Attention Decoder提供更准确的输出
- α通常取0.3左右

代表模型：WeNet、ESPnet、FunASR中的Paraformer

### 4.4 Transducer（RNN-T / Transformer-Transducer）

Transducer是另一种流行的端到端架构，特别适合流式识别。

```
音频特征 → Encoder → ┐
                      ├→ Joint Network → 输出
上一个输出 → Predictor → ┘
```

- **Encoder**：处理音频特征
- **Predictor**：类似语言模型，处理已输出的文本
- **Joint Network**：融合两者信息，决定当前输出

优点：

- 天然支持流式识别
- 兼顾了CTC的对齐能力和Attention的上下文建模能力

## 5. 主流Encoder架构

### 5.1 Transformer

基于自注意力机制（Self-Attention），能捕获全局依赖关系。

核心组件：

- Multi-Head Self-Attention（多头自注意力）
- Feed-Forward Network（前馈网络）
- Layer Normalization + Residual Connection

### 5.2 Conformer

Conformer = Convolution + Transformer，是目前ASR领域最主流的Encoder架构。

```
输入 → FFN → Multi-Head Self-Attention → Convolution → FFN → 输出
```

为什么加卷积？

- Self-Attention擅长捕获全局依赖
- 卷积擅长捕获局部特征（如音素级别的模式）
- 两者结合效果更好

## 6. ASR的主要痛点与挑战

### 6.1 数据层面

- **标注成本高**：语音数据需要人工逐字转写，成本远高于文本标注
- **数据不平衡**：常见词汇样本多，专业术语/方言样本少
- **噪声问题**：真实环境中的背景噪声、混响、远场拾音等
- **多说话人**：多人同时说话（鸡尾酒会问题）

### 6.2 模型层面

- **长音频处理**：Transformer的自注意力计算量随序列长度平方增长
- **流式 vs 非流式**：流式识别需要在只看到部分音频时就给出结果，准确率通常低于非流式
- **领域适配**：通用模型在特定领域（医疗、法律）表现不佳
- **小语种/方言**：训练数据稀缺

### 6.3 工程层面

- **实时性要求**：语音助手需要低延迟响应
- **端侧部署**：模型需要足够小才能在手机/嵌入式设备上运行
- **热词/关键词**：需要准确识别特定的人名、产品名等

### 6.4 评估指标

#### WER（Word Error Rate，词错误率）

WER是英文ASR最核心的评估指标，衡量识别结果与参考文本之间的差异：

```
WER = (S + D + I) / N × 100%
```

- S（Substitution）：替换错误数——把一个词识别成了另一个词
- D（Deletion）：删除错误数——漏掉了一个词（模型没有输出）
- I（Insertion）：插入错误数——多出了一个词（模型凭空输出）
- N：参考文本的总词数

> **注意**：WER可以超过100%。当插入错误非常多时，S+D+I可能大于N。

##### WER计算示例

```
参考文本（Reference）: the cat sat on the mat
识别结果（Hypothesis）: the cat sit on a mat and

逐词对齐：
  REF:  the  cat  sat   on  the  mat  ---
  HYP:  the  cat  sit   on  a    mat  and
        ✓    ✓    S     ✓   S    ✓    I

S（替换）= 2（sat→sit, the→a）
D（删除）= 0
I（插入）= 1（多了"and"）
N = 6（参考文本共6个词）

WER = (2 + 0 + 1) / 6 × 100% = 50.0%
```

##### 对齐过程的本质

WER的计算依赖于参考文本和识别结果之间的**最小编辑距离（Minimum Edit Distance）**对齐，也叫Levenshtein距离。对齐过程使用动态规划算法，找到将识别结果转换为参考文本所需的最少操作数（替换、删除、插入）。

```
动态规划矩阵示例：

REF: "今 天 天 气 好"
HYP: "今 天 气 很 好"

       ""  今  天  天  气  好
  ""    0   1   2   3   4   5
  今    1   0   1   2   3   4
  天    2   1   0   1   2   3
  气    3   2   1   1   1   2
  很    4   3   2   2   2   2
  好    5   4   3   3   3   2

最小编辑距离 = 2
回溯路径：今(✓) 天(✓) [天被删除D] 气(✓) [很被插入I] 好(✓)

对齐结果：
  REF:  今  天  天  气  ---  好
  HYP:  今  天  ---  气  很   好
        ✓   ✓   D    ✓   I    ✓

S=0, D=1, I=1, N=5
CER = (0+1+1)/5 × 100% = 40.0%
```

#### CER（Character Error Rate，字错误率）

中文ASR通常使用CER，计算方式与WER完全相同，但以**字**为单位而非词：

```
CER = (S + D + I) / N × 100%
```

##### 为什么中文用CER而不是WER？

中文没有天然的词边界（英文用空格分词），如果要计算WER就需要先做分词，而分词本身就会引入误差。以字为单位避免了这个问题，评估结果更客观。

##### CER计算示例

```
参考文本（Reference）: 今天天气真好
识别结果（Hypothesis）: 今天天气很好

逐字对齐：
  REF:  今  天  天  气  真  好
  HYP:  今  天  天  气  很  好
        ✓   ✓   ✓   ✓   S   ✓

S=1（真→很）, D=0, I=0, N=6
CER = (1+0+0)/6 × 100% = 16.7%
```

再看一个更复杂的例子：

```
参考文本: 我想订一张从北京到上海的机票
识别结果: 我想定一张从北京到上海机票

逐字对齐：
  REF:  我 想 订 一 张 从 北 京 到 上 海 的 机 票
  HYP:  我 想 定 一 张 从 北 京 到 上 海 -- 机 票
        ✓  ✓  S  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  D  ✓  ✓

S=1（订→定）, D=1（漏了"的"）, I=0, N=14
CER = (1+1+0)/14 × 100% = 14.3%
```

#### SER（Sentence Error Rate，句错误率）

SER衡量的是整句识别错误的比例：

```
SER = 识别有误的句子数 / 总句子数 × 100%
```

只要一句话中有任何一个字/词识别错误，该句就算错误。SER通常远高于CER/WER。

```
示例（5句话）：
  句1: REF="你好"        HYP="你好"        ✓
  句2: REF="今天天气好"   HYP="今天天气好"   ✓
  句3: REF="我想吃饭"    HYP="我想迟饭"     ✗（1个字错）
  句4: REF="明天见"      HYP="明天见"       ✓
  句5: REF="谢谢你的帮助" HYP="谢谢你的帮住"  ✗（1个字错）

SER = 2/5 × 100% = 40.0%
整体CER = 2/21 × 100% = 9.5%
```

可以看到，虽然CER只有9.5%，但SER高达40%。这说明CER低不代表用户体验好——用户感知的是整句是否正确。

#### RTF（Real-Time Factor，实时率）

RTF衡量ASR系统的处理速度：

```
RTF = 处理时间 / 音频时长
```

- RTF < 1：处理速度快于实时（满足实时需求）
- RTF = 1：刚好实时
- RTF > 1：处理速度慢于实时（不满足实时需求）

```
示例：
  一段10秒的音频，模型用了2秒处理完
  RTF = 2 / 10 = 0.2

  含义：处理速度是实时的5倍，即每秒可以处理5秒的音频
```

对于流式ASR，通常要求RTF < 0.3以保证低延迟体验。

#### 其他常见指标

| 指标                  | 全称                         | 说明                           |
| ------------------- | -------------------------- | ---------------------------- |
| MER                 | Match Error Rate           | 考虑对齐中匹配数量的错误率，与WER略有不同       |
| WIL                 | Word Information Lost      | 衡量丢失的词信息量，基于词的命中率和精确率        |
| WIP                 | Word Information Preserved | WIL的互补指标，衡量保留的词信息量           |
| Latency             | 延迟                         | 从说完一个词到系统输出该词的时间差（流式ASR关键指标） |
| Endpointing Latency | 端点检测延迟                     | 从用户停止说话到系统判定语音结束的时间          |

#### WER/CER计算的Python实现

```python
def compute_wer(reference: str, hypothesis: str) -> dict:
    """
    计算WER/CER（取决于输入是词列表还是字列表）
    使用动态规划求最小编辑距离

    Args:
        reference: 参考文本（空格分隔的词/字序列）
        hypothesis: 识别结果（空格分隔的词/字序列）

    Returns:
        包含S, D, I, N, error_rate的字典
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    n = len(ref_words)
    m = len(hyp_words)

    # 动态规划矩阵
    # dp[i][j] = 将hyp[:j]转换为ref[:i]的最小编辑距离
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # 初始化：空串到非空串的距离
    for i in range(n + 1):
        dp[i][0] = i  # 全部删除
    for j in range(m + 1):
        dp[0][j] = j  # 全部插入

    # 填表
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 匹配，无需操作
            else:
                substitution = dp[i - 1][j - 1] + 1  # 替换
                deletion = dp[i - 1][j] + 1           # 删除（ref有，hyp没有）
                insertion = dp[i][j - 1] + 1           # 插入（hyp多出来的）
                dp[i][j] = min(substitution, deletion, insertion)

    # 回溯统计S, D, I
    i, j = n, m
    substitutions = deletions = insertions = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1  # 匹配
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1  # 替换
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1  # 删除
        else:
            insertions += 1
            j -= 1  # 插入

    error_rate = (substitutions + deletions + insertions) / n * 100 if n > 0 else 0.0

    return {
        "S": substitutions,
        "D": deletions,
        "I": insertions,
        "N": n,
        "error_rate": round(error_rate, 2)
    }


# ===== 英文WER示例 =====
ref_en = "the cat sat on the mat"
hyp_en = "the cat sit on a mat and"
result_en = compute_wer(ref_en, hyp_en)
print(f"英文WER: {result_en}")
# 输出: {'S': 2, 'D': 0, 'I': 1, 'N': 6, 'error_rate': 50.0}

# ===== 中文CER示例（以字为单位，每个字之间加空格）=====
ref_cn = "今 天 天 气 真 好"
hyp_cn = "今 天 天 气 很 好"
result_cn = compute_wer(ref_cn, hyp_cn)
print(f"中文CER: {result_cn}")
# 输出: {'S': 1, 'D': 0, 'I': 0, 'N': 6, 'error_rate': 16.67}

# ===== 更复杂的中文示例 =====
ref_cn2 = "我 想 订 一 张 从 北 京 到 上 海 的 机 票"
hyp_cn2 = "我 想 定 一 张 从 北 京 到 上 海 机 票"
result_cn2 = compute_wer(ref_cn2, hyp_cn2)
print(f"中文CER（复杂）: {result_cn2}")
# 输出: {'S': 1, 'D': 1, 'I': 0, 'N': 14, 'error_rate': 14.29}
```

> **实际项目中的WER/CER计算**
> 实际项目中通常不需要自己实现，可以使用成熟的工具：
> 
> - `jiwer` 库（Python）：`pip install jiwer`，支持WER、MER、WIL等多种指标
> - `sclite`（NIST提供）：Kaldi生态中常用的评估工具，支持详细的对齐报告
> - 各ASR框架自带的评估脚本（如WeNet的`compute-wer.py`、ESPnet的评估模块）

#### 评估中的常见陷阱

1. **文本归一化不一致**：计算WER/CER前，参考文本和识别结果必须做相同的归一化处理（去标点、统一大小写、数字转文字等）。否则"100"和"一百"会被算作错误。

2. **分词标准不统一**：中文如果用WER，不同分词工具的结果会导致WER差异很大。这也是中文更推荐用CER的原因之一。

3. **测试集泄露**：如果训练数据中包含了测试集的内容（或高度相似的内容），WER/CER会虚低，不能反映真实性能。

4. **单一指标的局限性**：WER/CER无法区分错误的严重程度。比如把"北京"识别成"背景"（语义完全不同）和把"的"识别成"地"（影响较小），在WER/CER中权重相同。实际应用中可能需要结合语义级别的评估。

#### 文本归一化（Text Normalization）

文本归一化（TN）是ASR数据准备和评估中的关键步骤，指将文本统一成标准格式，确保训练标签一致性和评估公平性。

##### 为什么需要文本归一化？

如果参考文本中写的是"100"，而模型输出的是"一百"，不做归一化的话会被算作多个字的错误，导致CER/WER虚高。训练时如果同一个意思有多种写法（"100"和"一百"混用），也会干扰模型学习。

##### 常见的归一化操作

| 操作         | 示例                             | 说明            |
| ---------- | ------------------------------ | ------------- |
| 去标点        | `今天天气真好！` → `今天天气真好`           | 去除中英文标点符号     |
| 统一大小写（英文）  | `Hello World` → `hello world`  | 全部转小写         |
| 数字转文字（TN）  | `100` → `一百`，`3.14` → `三点一四`   | 用于训练标签统一      |
| 文字转数字（ITN） | `一百` → `100`，`二零二四年` → `2024年` | 用于推理后处理，提升可读性 |
| 全角/半角统一    | `Ｈｅｌｌｏ` → `Hello`              | 统一字符编码        |
| 去除多余空格     | `你  好` → `你 好`                 | 合并连续空格        |
| 繁简转换       | `語音識別` → `语音识别`                | 按需统一繁简体       |

##### TN与ITN的区别

- **TN（Text Normalization，文本归一化）**：将书面形式转为口语/标准形式。主要用于训练数据准备和评估对齐。方向是 `100` → `一百`。
- **ITN（Inverse Text Normalization，逆文本归一化）**：将口语形式转为书面形式。主要用于推理后处理，让输出对用户更友好。方向是 `一百` → `100`。

```
TN（训练/评估阶段）：
  "他花了100元买了3本书" → "他花了一百元买了三本书"
  "2024年3月15日"       → "二零二四年三月十五日"
  "下午3:30"            → "下午三点三十"

ITN（推理后处理阶段）：
  "他花了一百元买了三本书" → "他花了100元买了3本书"
  "二零二四年三月十五日"   → "2024年3月15日"
  "下午三点三十"          → "下午3:30"
```

##### 评估用的归一化处理

```python
import re

def normalize_for_eval(text: str) -> str:
    """
    评估前的文本归一化
    确保参考文本和识别结果使用完全相同的归一化规则
    """
    # 统一小写（英文）
    text = text.lower()
    # 去除中文标点
    text = re.sub(r'[，。！？、；：""''（）【】《》]', '', text)
    # 去除英文标点
    text = re.sub(r'[,.!?;:\"\'()\[\]{}<>]', '', text)
    # 全角转半角
    text = ''.join(
        chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c
        for c in text
    )
    # 合并多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# 示例
ref = normalize_for_eval("他花了100元，买了3本书。")
hyp = normalize_for_eval("他花了100元买了3本书")
# ref: "他花了100元买了3本书"
# hyp: "他花了100元买了3本书"
# 归一化后完全一致，CER=0
```

##### 常用工具

| 工具                   | 说明                               |
| -------------------- | -------------------------------- |
| WeTextProcessing     | WeNet团队开源，支持中英文TN/ITN，基于规则（WFST） |
| FunTextProcessing    | FunASR配套的文本处理工具                  |
| NeMo Text Processing | NVIDIA NeMo框架的TN/ITN模块           |
| OpenCC               | 繁简转换工具                           |

```python
# WeTextProcessing 使用示例
# pip install WeTextProcessing

from tn.chinese.normalizer import Normalizer as TNNormalizer
from itn.chinese.inverse_normalizer import InverseNormalizer

# 文本归一化（数字→文字，用于训练标签统一）
tn = TNNormalizer()
tn.normalize("他花了100元买了3本书")
# → "他花了一百元买了三本书"

# 逆文本归一化（文字→数字，用于推理后处理）
itn = InverseNormalizer()
itn.inverse_normalize("他花了一百元买了三本书")
# → "他花了100元买了3本书"
```

> **关键原则**：评估时参考文本（ref）和识别结果（hyp）必须使用完全相同的归一化规则。训练时标签也要统一格式，避免同一个意思有多种写法导致模型困惑。

## 7. 数据处理流程

一个完整的ASR数据处理流程如下：

### 7.1 数据采集

```
录音设备 → 原始音频文件（WAV/MP3/FLAC等）
```

注意事项：

- 采样率建议16kHz
- 位深16bit
- 单声道
- 尽量减少背景噪声

### 7.2 数据预处理

```python
# 典型的数据预处理流程
1. 格式转换：统一转为WAV格式，16kHz，16bit，单声道
2. 音量归一化：统一音量水平
3. 静音切除（VAD）：去掉首尾的静音段
4. 分段：将长音频切成适当长度的片段（通常5-30秒）
5. 质量筛选：去掉过短、过长、信噪比过低的音频
```

> **什么是VAD（Voice Activity Detection）？**
> VAD是语音活动检测，用于判断音频中哪些部分包含语音，哪些是静音或纯噪声。它是ASR系统的重要前处理步骤。

### 7.3 数据标注

```
音频片段 + 人工转写文本 → 标注数据对
```

标注格式示例（Kaldi风格）：

```
# wav.scp（音频路径映射）
utt_001 /data/audio/001.wav
utt_002 /data/audio/002.wav

# text（转写文本）
utt_001 今天天气真好
utt_002 我想订一张机票
```

### 7.4 特征提取

```python
# 伪代码
for each audio_file:
    waveform = load_audio(audio_file)          # 加载音频
    frames = framing(waveform, 25ms, 10ms)     # 分帧
    features = fbank(frames, n_mels=80)        # 提取80维FBank特征
    features = cmvn(features)                  # 倒谱均值方差归一化
    save(features)
```

> **什么是CMVN（Cepstral Mean and Variance Normalization）？**
> CMVN是对特征做均值和方差归一化，消除不同录音条件（麦克风、环境）带来的差异。就像把不同光照条件下拍的照片统一调整亮度和对比度一样。

### 7.5 数据增强

为了提高模型的鲁棒性，常用的数据增强方法：

| 方法                 | 说明                             |
| ------------------ | ------------------------------ |
| Speed Perturbation | 改变语速（0.9x, 1.0x, 1.1x），相当于3倍数据 |
| SpecAugment        | 在频谱图上随机遮挡时间/频率区域               |
| 加噪                 | 混入背景噪声（白噪声、环境噪声等）              |
| 混响                 | 模拟不同房间的混响效果                    |
| 音量扰动               | 随机调整音量大小                       |

> **SpecAugment为什么有效？**
> SpecAugment通过在频谱图上随机遮挡一些区域，迫使模型不要过度依赖某些特定的时间或频率信息，从而提高泛化能力。类似于图像领域的随机遮挡（Cutout）。

### 7.6 数据组织

训练数据通常分为：

- **训练集（Train）**：80-90%，用于模型训练
- **验证集（Dev/Validation）**：5-10%，用于调参和早停
- **测试集（Test）**：5-10%，用于最终评估

## 8. 常用开源数据集

| 数据集          | 语言  | 时长      | 说明                |
| ------------ | --- | ------- | ----------------- |
| LibriSpeech  | 英文  | 1000h   | 有声书朗读，最常用的英文ASR基准 |
| AISHELL-1    | 中文  | 178h    | 中文普通话，入门首选        |
| AISHELL-2    | 中文  | 1000h   | 更大规模的中文数据集        |
| WenetSpeech  | 中文  | 10000h+ | 超大规模中文数据集         |
| Common Voice | 多语言 | 各异      | Mozilla开源多语言数据集   |
| GigaSpeech   | 英文  | 10000h+ | 超大规模英文数据集         |

## 9. 常用开源工具/框架

| 框架               | 特点              |
| ---------------- | --------------- |
| FunASR（阿里达摩院）    | 工业级，模型丰富，中文支持好  |
| WeNet（出门问问）      | 生产级，流式支持好       |
| ESPnet           | 学术界主流，功能全面      |
| Kaldi            | 传统ASR经典框架（非端到端） |
| Whisper（OpenAI）  | 多语言，零样本能力强      |
| Sherpa/k2        | 基于FST的现代ASR框架   |
| PaddleSpeech（百度） | 中文支持好，部署方便      |

---

> 下一篇文档将详细讲解ASR模型的数据流、训练流程和推理流程。
