# Sherpa-onnx 详细介绍：从入门到掌握

## 1. Sherpa-onnx 是什么

Sherpa-onnx 是由 k2-fsa 团队（Daniel Povey 等人，即 Kaldi 的原作者）开发的新一代跨平台语音识别推理框架。它基于 ONNX Runtime，专注于将训练好的语音模型高效部署到各种平台上——从服务器到手机、从嵌入式设备到浏览器。

Sherpa-onnx 不是一个训练框架，而是一个纯推理框架。训练在 icefall（基于 PyTorch 和 k2）中完成，训练好的模型导出为 ONNX 格式后，由 sherpa-onnx 负责高效推理和部署。

#### 1.1 ONNX 与 ONNX Runtime 简介

ONNX（Open Neural Network Exchange）是由微软和 Facebook 于 2017 年联合推出的开放式神经网络交换格式。它定义了一套与框架无关的模型表示标准，使得在 PyTorch、TensorFlow 等不同框架中训练的模型可以导出为统一的 `.onnx` 文件，再由任意支持 ONNX 的推理引擎加载运行。

```
ONNX 的核心思想：训练和推理解耦

训练阶段（灵活性优先）              推理阶段（效率优先）
┌──────────────┐                  ┌──────────────────┐
│  PyTorch     │──┐               │  ONNX Runtime    │
│  TensorFlow  │──┼── 导出 .onnx ──→│  TensorRT        │
│  PaddlePaddle│──┘               │  OpenVINO        │
└──────────────┘                  │  CoreML          │
                                  └──────────────────┘
```

ONNX Runtime 是微软开源的高性能推理引擎，专门用于加载和执行 ONNX 模型。它是 sherpa-onnx 的底层推理后端，提供了以下关键能力：

```
ONNX Runtime 核心特性：

1. 跨平台支持
   ├── 操作系统：Windows / Linux / macOS / Android / iOS
   ├── 硬件：CPU / GPU (CUDA, DirectML) / NPU / FPGA
   └── 架构：x86_64 / ARM64 / ARM32 / WebAssembly

2. 高性能优化
   ├── 图优化：算子融合（Conv+BN+ReLU → 单个算子）、常量折叠、冗余节点消除
   ├── 内存优化：内存复用、预分配、零拷贝
   ├── 并行执行：多线程推理、算子间并行、算子内并行
   └── 量化支持：INT8 / FP16 动态量化和静态量化

3. Execution Provider（EP）机制
   ├── CPU EP：默认后端，使用 MLAS（微软线性代数库）加速矩阵运算
   ├── CUDA EP：NVIDIA GPU 加速
   ├── TensorRT EP：NVIDIA TensorRT 深度优化
   ├── CoreML EP：Apple 芯片加速（M1/M2/A 系列）
   ├── NNAPI EP：Android 神经网络 API 加速
   └── WASM EP：浏览器端 WebAssembly 执行
```

sherpa-onnx 选择 ONNX Runtime 作为推理后端的原因：

| 考量     | ONNX Runtime 的优势                                      |
| ------ | ----------------------------------------------------- |
| 跨平台    | 一套代码覆盖服务器、移动端、嵌入式、浏览器                                 |
| 无框架依赖  | 不需要安装 PyTorch / TensorFlow，部署包极小                      |
| CPU 优化 | MLAS 库对 CPU 矩阵运算做了深度优化，端侧推理性能优秀                       |
| 量化友好   | 原生支持 INT8 量化推理，模型体积和速度均有显著提升                          |
| 生态成熟   | 微软长期维护，社区活跃，兼容性好                                      |
| 模型兼容   | 可以加载 PyTorch / TensorFlow / PaddlePaddle 等导出的 ONNX 模型 |

> 正是因为 ONNX Runtime 的这些特性，sherpa-onnx 才能做到"一次导出，到处部署"——同一个 ONNX 模型文件，无需修改即可在 Linux 服务器、Android 手机、树莓派、甚至浏览器中运行。

### 1.2 Sherpa-onnx 的核心定位

```
新一代跨平台语音推理框架
├── 纯推理，不含训练（训练在 icefall 中完成）
├── 基于 ONNX Runtime，无需 PyTorch 依赖
├── 支持多种模型：Zipformer、Paraformer、Whisper、NeMo CTC 等
├── 跨平台：Linux / Windows / macOS / Android / iOS / 嵌入式 / WebAssembly
├── 多语言绑定：C / C++ / Python / Java / Kotlin / Swift / Go / C#
├── 支持流式和非流式识别
├── 支持 VAD、语音合成（TTS）、说话人识别、语种识别等
└── 部署极其轻量，适合端侧场景
```

### 1.3 k2-fsa 生态全景

Sherpa-onnx 并非孤立存在，它是 k2-fsa 生态中的推理部署环节。理解整个生态有助于把握 sherpa-onnx 的定位：

```
┌─────────────────────────────────────────────────────────────────┐
│                       k2-fsa 生态                                │
│                                                                  │
│  ┌──────────┐     ┌──────────────┐     ┌────────────────┐      │
│  │   k2     │────→│   icefall    │────→│  sherpa-onnx   │      │
│  │ (核心库)  │     │  (训练框架)   │     │ (推理部署框架)  │      │
│  │          │     │              │     │                │      │
│  │ FST 运算 │     │ 模型定义     │     │ ONNX Runtime   │      │
│  │ 损失函数  │     │ 数据处理     │     │ 跨平台部署      │      │
│  │ 解码算法  │     │ 训练脚本     │     │ 多语言绑定      │      │
│  └──────────┘     └──────────────┘     └────────────────┘      │
│                                                                  │
│  工作流：                                                        │
│  1. k2 提供高效的 FST 运算和 CTC/Transducer 损失函数             │
│  2. icefall 基于 k2 + PyTorch 训练模型（Zipformer 等）           │
│  3. 训练好的模型导出为 ONNX 格式                                 │
│  4. sherpa-onnx 加载 ONNX 模型，在目标平台上高效推理             │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 Sherpa-onnx 在 ASR 生态中的位置

```
ASR 推理框架对比：

Kaldi (2011)
    │
    ├──→ Kaldi 原生解码器：基于 WFST，C++ 实现，部署复杂
    │
    ├──→ k2/icefall (2020)：Kaldi 原作者新项目，WFST + 端到端训练
    │       │
    │       └──→ sherpa-onnx：icefall 的推理部署方案，ONNX Runtime
    │
    ├──→ FunASR Runtime：阿里达摩院，ONNX/Torch 推理，Paraformer 为主
    ├──→ WeNet Runtime：出门问问，ONNX/TRT 推理，Conformer 为主
    └──→ Whisper.cpp：基于 ggml 的 Whisper 推理，C/C++ 实现

sherpa-onnx 的差异化优势：
├── 跨平台覆盖最广（7+ 语言绑定，含 WebAssembly）
├── 支持模型种类最多（Zipformer、Paraformer、Whisper、NeMo 等）
├── 部署最轻量（无 PyTorch 依赖，纯 ONNX Runtime）
├── Kaldi 原作者团队维护，技术底蕴深厚
└── 活跃的社区和持续更新的预训练模型
```

### 1.5 Sherpa-onnx 与其他推理框架的对比

| 特性          | sherpa-onnx                           | FunASR Runtime   | WeNet Runtime       | Whisper.cpp     |
| ----------- | ------------------------------------- | ---------------- | ------------------- | --------------- |
| 开发者         | k2-fsa (Daniel Povey)                 | 阿里达摩院            | 出门问问                | Georgi Gerganov |
| 推理引擎        | ONNX Runtime                          | ONNX / Torch     | ONNX / TensorRT     | 自研 (ggml)       |
| 模型格式        | ONNX                                  | ONNX / PT        | ONNX                | GGML            |
| 核心模型        | Zipformer (Transducer)                | Paraformer (NAR) | Conformer (CTC/AED) | Whisper (AED)   |
| 流式支持        | 支持                                    | 支持               | 支持                  | 不支持             |
| 移动端         | Android / iOS                         | Android          | Android             | Android / iOS   |
| 嵌入式         | 支持                                    | 有限               | 有限                  | 支持              |
| WebAssembly | 支持                                    | 不支持              | 不支持                 | 支持              |
| 语言绑定        | 7 种+                                  | Python 为主        | Python / C++        | C / C++         |
| 模型种类        | 多种（Zipformer/Paraformer/Whisper/NeMo） | Paraformer 为主    | Conformer 为主        | Whisper 专用      |
| 量化支持        | INT8                                  | INT8             | INT8 / FP16         | INT4 / INT8     |
| PyTorch 依赖  | 无                                     | 可选               | 可选                  | 无               |
| 学习曲线        | 低                                     | 低                | 中等                  | 低               |
| 附加功能        | VAD / TTS / 说话人识别 / 语种识别              | VAD / 标点 / 热词    | VAD / 标点            | 仅 ASR           |

## 2. 安装与环境配置

### 2.1 Python 安装（推荐入门）

sherpa-onnx 的 Python 安装非常简单，一行命令即可：

```bash
pip install sherpa-onnx
```

安装完成后即可使用，无需额外编译或安装 PyTorch。

### 2.2 验证安装

```python
import sherpa_onnx
print(f"sherpa-onnx version: {sherpa_onnx.__version__}")
```

### 2.3 其他平台安装

sherpa-onnx 提供了多种安装方式，覆盖主流开发场景：

```
平台/语言          安装方式
─────────────────────────────────────────────────────
Python            pip install sherpa-onnx
C/C++             从源码编译，或下载预编译库
Android (Java)    Maven / 从源码编译 AAR
Android (Kotlin)  同上
iOS (Swift)       CocoaPods / 从源码编译 Framework
Go                go get github.com/k2-fsa/sherpa-onnx-go
C#                NuGet 包
JavaScript        npm（WebAssembly 版本）
```

### 2.4 从源码编译（C/C++ 开发者）

```bash
# 克隆仓库
git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx

# 编译（Linux/macOS）
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4

# 编译（Windows）
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

### 2.5 环境要求

```
Python >= 3.7
操作系统: Linux / macOS / Windows / Android / iOS
内存: >= 512MB（小模型）/ >= 2GB（大模型）
磁盘: 模型文件约 10MB ~ 500MB（取决于模型大小和量化方式）
GPU: 可选（CPU 即可高效运行，这是 sherpa-onnx 的核心优势）
PyTorch: 不需要
```

> sherpa-onnx 的一大优势是对 CPU 推理做了深度优化。即使在没有 GPU 的设备上（如手机、嵌入式板卡），也能实现实时甚至超实时的识别速度。

## 3. 模型下载与管理

### 3.1 模型来源

sherpa-onnx 的预训练模型统一托管在 GitHub Releases 上：

```
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
```

模型由 icefall 训练后导出为 ONNX 格式，涵盖多种语言和架构。

### 3.2 常用中文模型

| 模型                                                               | 类型  | 架构                       | 大小                           | 适用场景       |
| ---------------------------------------------------------------- | --- | ------------------------ | ---------------------------- | ---------- |
| sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20       | 流式  | Zipformer-Transducer     | ~70MB                        | 实时识别、流式场景  |
| sherpa-onnx-paraformer-zh-2023-09-14                             | 离线  | Paraformer               | ~220MB (FP32) / ~60MB (INT8) | 离线转写、高精度场景 |
| sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16 | 流式  | Zipformer-Transducer (小) | ~15MB                        | 移动端、嵌入式    |

### 3.3 模型下载方式

方式一：使用辅助脚本下载

```python
"""模型下载辅助脚本"""
import os
import subprocess

MODELS_DIR = "models"

# 离线模型（Paraformer）
OFFLINE_MODEL = {
    "name": "sherpa-onnx-paraformer-zh-2023-09-14",
    "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2",
}

# 流式模型（Zipformer）
STREAMING_MODEL = {
    "name": "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
    "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2",
}

def download_model(model_info):
    name = model_info["name"]
    url = model_info["url"]
    target_dir = os.path.join(MODELS_DIR, name)

    if os.path.exists(target_dir):
        print(f"模型已存在: {name}")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)
    archive = os.path.join(MODELS_DIR, f"{name}.tar.bz2")

    print(f"正在下载: {name}")
    # Linux/macOS
    subprocess.run(["wget", "-O", archive, url], check=True)
    subprocess.run(["tar", "xf", archive, "-C", MODELS_DIR], check=True)
    os.remove(archive)
    print(f"模型已就绪: {target_dir}")

download_model(OFFLINE_MODEL)
download_model(STREAMING_MODEL)
```

方式二：手动下载

直接从 GitHub Releases 页面下载 tar.bz2 文件，解压到本地目录即可。

### 3.4 模型文件结构

流式模型（Zipformer-Transducer）的文件结构：

```
sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/
├── encoder-epoch-99-avg-1.onnx       # Encoder 模型（Zipformer）
├── decoder-epoch-99-avg-1.onnx       # Predictor 模型（Embedding + Conv）
├── joiner-epoch-99-avg-1.onnx        # Joint Network 模型
├── tokens.txt                         # 词表（token → id 映射）
├── bpe.model                          # BPE 分词模型（如果使用 BPE）
└── README.md

各文件的输入输出：
  encoder: 音频特征 [B, T, 80] + cache → 编码表示 [B, T', D] + 更新后的 cache
  decoder: token_ids [B, context_size] → predictor_out [B, 1, D_pred]
  joiner:  encoder_out [B, 1, D_enc] + predictor_out [B, 1, D_pred] → logits [B, 1, vocab_size]
```

离线模型（Paraformer）的文件结构：

```
sherpa-onnx-paraformer-zh-2023-09-14/
├── model.onnx              # 完整模型（FP32，约 220MB）
├── model.int8.onnx         # INT8 量化模型（约 60MB，推荐使用）
├── tokens.txt              # 词表
└── README.md
```

> INT8 量化模型在精度几乎无损的情况下，体积缩小约 4 倍，推理速度也更快。生产环境推荐使用 INT8 版本。

## 4. 核心概念与架构

### 4.1 Sherpa-onnx 支持的模型架构

sherpa-onnx 支持多种主流 ASR 模型架构，每种架构有不同的特点和适用场景：

```
sherpa-onnx 支持的模型架构：

┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  流式模型（实时识别）                                             │
│  ├── Zipformer-Transducer  ← k2-fsa 自研，效果最好              │
│  ├── Conformer-Transducer                                       │
│  ├── Paraformer-Streaming  ← 从 FunASR 导入                    │
│  └── NeMo CTC-Streaming    ← 从 NVIDIA NeMo 导入               │
│                                                                  │
│  离线模型（非实时，高精度）                                       │
│  ├── Paraformer             ← 从 FunASR 导入，中文效果最好       │
│  ├── Whisper                ← 从 OpenAI Whisper 导入            │
│  ├── Zipformer-CTC                                              │
│  └── NeMo CTC                                                   │
│                                                                  │
│  其他功能                                                        │
│  ├── VAD（语音活动检测）    ← Silero VAD                         │
│  ├── TTS（语音合成）        ← VITS 等模型                       │
│  ├── 说话人识别/验证                                             │
│  ├── 语种识别                                                    │
│  └── 关键词检测（KWS）                                           │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 核心模型：Zipformer

Zipformer 是 k2-fsa 团队提出的高效 Encoder 架构，是 Conformer 的改进版本，在 sherpa-onnx 的流式模型中广泛使用。

#### Zipformer 相比 Conformer 的改进

| 改进点  | Conformer               | Zipformer               | 效果        |
| ---- | ----------------------- | ----------------------- | --------- |
| 整体结构 | 单一尺度，所有层分辨率相同           | U-Net 式多尺度结构            | 参数更少，效果更好 |
| 归一化  | LayerNorm               | BiasNorm                | 计算更快      |
| 激活函数 | Swish                   | SwooshR / SwooshL       | 训练更稳定     |
| 残差连接 | 标准残差 x + f(x)           | Bypass 模块（可学习缩放）        | 训练效率更高    |
| 注意力  | 标准 Multi-Head Attention | 增加 Non-linear Attention | 表达能力更强    |

#### Zipformer 架构详解

```
┌─────────────────────────────────────────────────────────────────┐
│                      Zipformer Encoder                           │
│                                                                  │
│  输入: FBank [batch, T, 80]                                     │
│      ↓                                                           │
│  Conv Subsampling (4 倍下采样)                                   │
│      [batch, T, 80] → [batch, T/4, d_model]                    │
│      ↓                                                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              多尺度 U-Net 结构                              │  │
│  │                                                            │  │
│  │  Scale 1 (T/4):  Zipformer Block × 2   ← 最高分辨率       │  │
│  │      ↓ (下采样 2x)                                         │  │
│  │  Scale 2 (T/8):  Zipformer Block × 2                      │  │
│  │      ↓ (下采样 2x)                                         │  │
│  │  Scale 3 (T/16): Zipformer Block × 3   ← 最低分辨率       │  │
│  │      ↑ (上采样 2x)                                         │  │
│  │  Scale 2 (T/8):  Zipformer Block × 2                      │  │
│  │      ↑ (上采样 2x)                                         │  │
│  │  Scale 1 (T/4):  Zipformer Block × 2   ← 回到最高分辨率   │  │
│  │                                                            │  │
│  │  每个 Scale 的维度不同：                                    │  │
│  │    Scale 1: d=192（细粒度，捕获局部特征）                   │  │
│  │    Scale 2: d=256（中等粒度）                               │  │
│  │    Scale 3: d=384（粗粒度，捕获全局特征）                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│      ↓                                                           │
│  输出: [batch, T/4, 192]                                        │
└─────────────────────────────────────────────────────────────────┘
```

> 多尺度结构的直觉：类似于图像处理中的 U-Net。低分辨率层（Scale 3）看到更长的时间范围，捕获全局语义信息（如句子结构）；高分辨率层（Scale 1）看到更细的时间粒度，捕获局部声学特征（如音素边界）。两者结合，既有全局视野又有局部精度。

#### Zipformer Block 内部结构

```
输入 x
    ↓
┌─────────────────────────────────────┐
│  Zipformer Block                     │
│                                      │
│  BiasNorm                            │
│      ↓                               │
│  Multi-Head Attention                │
│  (相对位置编码)                       │
│      ↓ + Bypass 残差                 │
│  BiasNorm                            │
│      ↓                               │
│  Conv Module                         │
│  (Depthwise Conv + SwooshR)          │
│      ↓ + Bypass 残差                 │
│  BiasNorm                            │
│      ↓                               │
│  Feed-Forward (SwooshR 激活)         │
│      ↓ + Bypass 残差                 │
│  BiasNorm                            │
│      ↓                               │
│  Non-linear Attention                │
│  (额外的注意力层，增强表达能力)       │
│      ↓ + Bypass 残差                 │
│  BiasNorm                            │
│      ↓                               │
│  Feed-Forward                        │
│      ↓ + Bypass 残差                 │
└─────────────────────────────────────┘
    ↓
输出
```

#### Bypass 模块

Zipformer 使用改进的残差连接（Bypass），替代传统的 `output = x + f(x)`：

```
传统残差连接：
  output = x + module(x)
  → 固定的 1:1 混合比例

Zipformer Bypass：
  bypass_scale = sigmoid(learned_param)   # 可学习参数，值在 0~1 之间
  output = bypass_scale * x + (1 - bypass_scale) * module(x)
  → 模型自己学习最优的混合比例
```

好处：模型可以学习在某些层"跳过"处理（bypass_scale 接近 1），在另一些层充分利用变换（bypass_scale 接近 0）。这比固定的残差连接更灵活，训练效率更高。

### 4.3 Transducer（RNN-T）解码架构

sherpa-onnx 的流式模型主要使用 Transducer 架构进行解码。Transducer 天然支持流式识别，是目前流式 ASR 的主流选择。

#### Transducer 完整结构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Transducer (RNN-T)                             │
│                                                                  │
│  ┌─────────────────────────┐                                    │
│  │      Encoder             │                                    │
│  │   (Zipformer)            │                                    │
│  │                          │                                    │
│  │  音频特征 → 编码表示      │                                    │
│  │  [B, T, 80] → [B, T', D] │                                    │
│  └────────────┬─────────────┘                                    │
│               │                                                  │
│               ↓                                                  │
│  ┌────────────────────────────────────────┐                     │
│  │           Joint Network                 │                     │
│  │                                         │                     │
│  │  encoder_out: [B, T', D_enc]           │                     │
│  │       ↓                                 │                     │
│  │  Linear(D_enc, D_joint)                │                     │
│  │       ↓                                 │                     │
│  │  + ←──── Linear(D_pred, D_joint)       │                     │
│  │       ↓         ↑                       │                     │
│  │  Tanh/ReLU   predictor_out             │                     │
│  │       ↓                                 │                     │
│  │  Linear(D_joint, vocab_size)           │                     │
│  │       ↓                                 │                     │
│  │  LogSoftmax                            │                     │
│  │       ↓                                 │                     │
│  │  输出: [B, T', U+1, vocab_size]        │                     │
│  └────────────────────────────────────────┘                     │
│               ↑                                                  │
│  ┌────────────┴─────────────┐                                    │
│  │      Predictor            │                                    │
│  │   (Embedding + Conv1D)    │                                    │
│  │                           │                                    │
│  │  已输出 token → 预测表示   │                                    │
│  │  [B, U] → [B, U, D_pred] │                                    │
│  └───────────────────────────┘                                    │
│                                                                  │
│  三个组件各自独立，通过 ONNX 分别导出                              │
│  T' = Encoder 输出的时间步数                                      │
│  U = 已输出的 token 数量                                          │
│  vocab_size = 词表大小（包含 blank）                               │
└─────────────────────────────────────────────────────────────────┘
```

#### Transducer 的三个组件

| 组件            | 作用                               | 类比              |
| ------------- | -------------------------------- | --------------- |
| Encoder       | 将音频特征编码为高级表示                     | "听"——理解声音内容     |
| Predictor     | 根据已输出的文本预测下一步                    | "语言模型"——预测下一个字  |
| Joint Network | 融合 Encoder 和 Predictor 的信息，做最终决策 | "裁判"——综合声学和语言信息 |

#### Transducer 解码过程（逐步追踪）

```
假设输入音频内容为"今天"，Encoder 输出 5 个时间步：

时间步 t=0:
  Encoder 输出 h₀, Predictor 输入 <blank>（初始状态）
  Joint(h₀, p₀) → 预测 "blank" → 不输出任何 token，前进到 t=1

时间步 t=1:
  Encoder 输出 h₁, Predictor 输入 <blank>
  Joint(h₁, p₀) → 预测 "今" → 输出 "今"，Predictor 状态更新

时间步 t=1（继续，不前进）:
  Encoder 输出 h₁, Predictor 输入 "今"
  Joint(h₁, p₁) → 预测 "blank" → 不输出，前进到 t=2

时间步 t=2:
  Encoder 输出 h₂, Predictor 输入 "今"
  Joint(h₂, p₁) → 预测 "天" → 输出 "天"，Predictor 状态更新

时间步 t=2（继续）:
  Encoder 输出 h₂, Predictor 输入 "天"
  Joint(h₂, p₂) → 预测 "blank" → 不输出，前进到 t=3

时间步 t=3~4:
  Joint 持续预测 "blank" → 不输出

最终输出: "今天"
```

关键机制：

- 输出 blank 表示"当前时间步没有新 token"，前进到下一个时间步
- 输出非 blank token 后，停留在当前时间步，继续尝试输出更多 token
- 这种机制天然支持流式：每来一帧音频就可以尝试解码，无需等待全部音频

#### Predictor 的实现

sherpa-onnx 中的 Predictor 使用 Embedding + Conv1D（而非传统 RNN-T 中的 LSTM）：

```
已输出 token: [<blank>, 今, 天]
    ↓
Embedding(vocab_size, D_pred)
    → [3, D_pred]
    ↓
Conv1D (context_size=2)
    → 使用最近 2 个 token 的信息
    ↓
输出: [1, D_pred]  (只取最后一个位置)
```

使用 Conv1D 替代 LSTM 的好处：

- 计算更快（无循环依赖，可并行）
- ONNX 导出更方便（LSTM 的 ONNX 导出有兼容性问题）
- 效果相当甚至更好

### 4.4 离线模型：Paraformer (ONNX)

sherpa-onnx 也支持从 FunASR 导出的 Paraformer ONNX 模型，用于离线（非流式）识别场景。

```
Paraformer 在 sherpa-onnx 中的推理流程：

WAV 音频
    ↓
FBank 特征提取 (80 维)
    ↓
Encoder (Conformer, 12 层)
    ↓
CIF 模块（预测输出长度，生成声学嵌入）
    ↓
Decoder (非自回归 Transformer, 6 层)
    ↓
一次性并行输出所有 token → 文本

特点：
├── 非自回归解码，速度比 Transducer 更快
├── 不支持流式（必须看到完整音频才能解码）
├── 中文识别效果最好（阿里 60000+ 小时数据训练）
└── INT8 量化后模型仅约 60MB
```

> Paraformer 的详细架构（CIF 模块、GLM 训练策略等）请参考 `doc/funasr.md` 中的深度解析。

## 5. 使用方法

### 5.1 离线语音识别（Paraformer）

离线识别适用于已录制好的音频文件，追求最高识别精度。

```python
"""
sherpa-onnx 离线语音识别示例
使用 Paraformer ONNX 模型
"""
import sherpa_onnx

# 模型目录（下载后的路径）
model_dir = "models/sherpa-onnx-paraformer-zh-2023-09-14"

# 创建离线识别器
recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
    paraformer=f"{model_dir}/model.int8.onnx",   # INT8 量化模型（推荐）
    tokens=f"{model_dir}/tokens.txt",              # 词表
    num_threads=4,                                  # 推理线程数
    sample_rate=16000,                              # 采样率
    feature_dim=80,                                 # FBank 维度
    decoding_method="greedy_search",                # 解码方式
)

# 读取音频
wave = sherpa_onnx.read_wave("test.wav")

# 创建流并输入音频
stream = recognizer.create_stream()
stream.accept_waveform(wave.sample_rate, wave.samples)

# 执行识别
recognizer.decode_stream(stream)
result = stream.result

print(f"识别结果: {result.text}")
print(f"音频时长: {len(wave.samples)/wave.sample_rate:.2f}秒")
```

#### 离线识别的关键参数

| 参数              | 说明          | 推荐值             |
| --------------- | ----------- | --------------- |
| paraformer      | ONNX 模型文件路径 | model.int8.onnx |
| tokens          | 词表文件路径      | tokens.txt      |
| num_threads     | CPU 推理线程数   | 4（服务端可更大）       |
| sample_rate     | 音频采样率       | 16000           |
| feature_dim     | FBank 特征维度  | 80              |
| decoding_method | 解码方式        | "greedy_search" |

### 5.2 流式语音识别（Zipformer-Transducer）

流式识别适用于实时场景，音频边录制边识别，逐步输出结果。

```python
"""
sherpa-onnx 流式语音识别示例
使用 Zipformer-Transducer 流式模型
"""
import numpy as np
import sherpa_onnx

# 模型目录
model_dir = "models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"

# 创建流式识别器
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    encoder=f"{model_dir}/encoder-epoch-99-avg-1.onnx",
    decoder=f"{model_dir}/decoder-epoch-99-avg-1.onnx",
    joiner=f"{model_dir}/joiner-epoch-99-avg-1.onnx",
    tokens=f"{model_dir}/tokens.txt",
    num_threads=4,
    sample_rate=16000,
    feature_dim=80,
    decoding_method="greedy_search",
)

# 读取音频
wave = sherpa_onnx.read_wave("test.wav")
sample_rate = wave.sample_rate
samples = np.array(wave.samples)

# 创建流
stream = recognizer.create_stream()

# 模拟流式输入：每 320ms 送入一个 chunk
chunk_size = int(sample_rate * 0.32)  # 320ms = 5120 个采样点
last_text = ""

for i in range(0, len(samples), chunk_size):
    chunk = samples[i : i + chunk_size].tolist()
    stream.accept_waveform(sample_rate, chunk)

    # 当有足够数据时进行解码
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    # 获取当前识别结果
    current_text = recognizer.get_result(stream).text
    if current_text and current_text != last_text:
        print(f"  [{i/sample_rate:.1f}s] {current_text}")
        last_text = current_text

# 输入结束：添加尾部静音并标记结束
tail_paddings = [0.0] * int(sample_rate * 0.5)  # 0.5 秒静音
stream.accept_waveform(sample_rate, tail_paddings)
stream.input_finished()

while recognizer.is_ready(stream):
    recognizer.decode_stream(stream)

final_text = recognizer.get_result(stream).text
print(f"\n最终结果: {final_text}")
```

#### 流式识别的关键参数

| 参数                         | 说明                 | 推荐值                                      |
| -------------------------- | ------------------ | ---------------------------------------- |
| encoder / decoder / joiner | 三个 ONNX 模型文件路径     | 对应的 .onnx 文件                             |
| tokens                     | 词表文件路径             | tokens.txt                               |
| num_threads                | CPU 推理线程数          | 4                                        |
| decoding_method            | 解码方式               | "greedy_search" 或 "modified_beam_search" |
| max_active_paths           | beam search 的候选路径数 | 4（仅 modified_beam_search）                |

#### 流式 vs 离线的选择

| 场景       | 推荐方式                      | 原因           |
| -------- | ------------------------- | ------------ |
| 实时语音助手   | 流式 (Zipformer-Transducer) | 需要边说边出结果     |
| 会议录音转写   | 离线 (Paraformer)           | 追求最高精度，不要求实时 |
| 电话客服     | 流式                        | 需要实时响应       |
| 视频字幕生成   | 离线                        | 可以后处理，追求精度   |
| 语音输入法    | 流式                        | 用户期望即时反馈     |
| 播客/音频转文字 | 离线                        | 不要求实时        |

### 5.3 VAD + ASR 组合使用

在实际应用中，通常需要先用 VAD（语音活动检测）找出音频中有人说话的片段，再对这些片段做 ASR。sherpa-onnx 内置了 Silero VAD 支持：

```python
"""
VAD + 离线 ASR 组合示例
先检测语音段，再对每段做识别
"""
import sherpa_onnx

# 创建 VAD
vad_config = sherpa_onnx.VadModelConfig()
vad_config.silero_vad.model = "silero_vad.onnx"
vad_config.silero_vad.threshold = 0.5          # 语音检测阈值
vad_config.silero_vad.min_silence_duration = 0.5  # 最短静音时长（秒）
vad_config.silero_vad.min_speech_duration = 0.25  # 最短语音时长（秒）
vad_config.sample_rate = 16000

vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=60)

# 创建离线识别器
recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
    paraformer="models/sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx",
    tokens="models/sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt",
    num_threads=4,
    sample_rate=16000,
    feature_dim=80,
)

# 读取音频
wave = sherpa_onnx.read_wave("long_audio.wav")

# VAD 处理
window_size = 512  # Silero VAD 的窗口大小
for i in range(0, len(wave.samples), window_size):
    chunk = wave.samples[i : i + window_size]
    if len(chunk) < window_size:
        break
    vad.accept_waveform(chunk)

# 对每个检测到的语音段做 ASR
while not vad.empty():
    speech_segment = vad.front()
    vad.pop()

    stream = recognizer.create_stream()
    stream.accept_waveform(wave.sample_rate, speech_segment.samples)
    recognizer.decode_stream(stream)

    start_time = speech_segment.start / wave.sample_rate
    text = stream.result.text
    print(f"[{start_time:.2f}s] {text}")
```

### 5.4 批量识别（多文件）

sherpa-onnx 支持批量解码多个音频流，提高吞吐量：

```python
"""批量离线识别示例"""
import sherpa_onnx

recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
    paraformer="models/sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx",
    tokens="models/sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt",
    num_threads=4,
    sample_rate=16000,
    feature_dim=80,
)

audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
streams = []

# 为每个音频创建流
for audio_file in audio_files:
    wave = sherpa_onnx.read_wave(audio_file)
    stream = recognizer.create_stream()
    stream.accept_waveform(wave.sample_rate, wave.samples)
    streams.append(stream)

# 批量解码
recognizer.decode_streams(streams)

# 获取结果
for audio_file, stream in zip(audio_files, streams):
    print(f"{audio_file}: {stream.result.text}")
```

### 5.5 使用 Whisper 模型

sherpa-onnx 也支持 OpenAI Whisper 模型（导出为 ONNX 格式）：

```python
"""使用 Whisper 模型进行离线识别"""
import sherpa_onnx

recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
    encoder="models/whisper-tiny/tiny-encoder.onnx",
    decoder="models/whisper-tiny/tiny-decoder.onnx",
    tokens="models/whisper-tiny/tokens.txt",
    num_threads=4,
    language="zh",           # 指定语言
    task="transcribe",       # transcribe（转写）或 translate（翻译为英文）
)

wave = sherpa_onnx.read_wave("test.wav")
stream = recognizer.create_stream()
stream.accept_waveform(wave.sample_rate, wave.samples)
recognizer.decode_stream(stream)

print(f"识别结果: {stream.result.text}")
```

## 6. 流式推理数据流深度追踪

以 Zipformer-Transducer 流式识别"你好世界"为例，逐步追踪数据在系统中的流动：

```
Step 1: 音频输入（模拟流式，每 320ms 一个 chunk）
  ─────────────────────────────────────────────────
  Chunk 1: 320ms 音频 → 5120 个采样点 (16kHz × 0.32s)

Step 2: FBank 特征提取
  ─────────────────────────────────────────────────
  5120 采样点
      ↓ 分帧（25ms 帧长，10ms 帧移）
  31 帧 × 80 维 = [1, 31, 80]

Step 3: Conv Subsampling（4 倍下采样）
  ─────────────────────────────────────────────────
  [1, 31, 80] → [1, 7, 192]
  帧数从 31 减少到 7，特征维度从 80 变为 192

Step 4: Zipformer Encoder（带 cache 的流式推理）
  ─────────────────────────────────────────────────
  输入: 当前 chunk [1, 7, 192] + 历史 cache
  处理: 多尺度 Zipformer Blocks
  输出: [1, 7, 192] + 更新后的 cache

  cache 包含：
  ├── 每层的 attention key/value 缓存
  ├── 卷积模块的左侧 padding 缓存
  └── 不同 scale 之间的状态缓存

  > cache 机制是流式推理的关键。每个 chunk 只处理当前数据，
  > 但通过 cache 保留了历史信息，避免重复计算。

Step 5: Transducer 解码（逐帧 Greedy Search）
  ─────────────────────────────────────────────────
  对 Encoder 输出的 7 帧逐帧解码：

  帧 0: Joint(h₀, p_blank) → blank → 跳过
  帧 1: Joint(h₁, p_blank) → blank → 跳过
  帧 2: Joint(h₂, p_blank) → "你"  → 输出
        Joint(h₂, p_你)    → blank → 下一帧
  帧 3: Joint(h₃, p_你)    → "好"  → 输出
        Joint(h₃, p_好)    → blank → 下一帧
  帧 4~6: blank → 跳过

  当前 chunk 输出: "你好"

Step 6: 下一个 chunk 到来，重复 Step 2~5
  ─────────────────────────────────────────────────
  Chunk 2 处理后输出: "世界"

Step 7: 最终结果
  ─────────────────────────────────────────────────
  拼接所有 chunk 的输出: "你好世界"
```

### 流式推理的延迟分析

```
延迟组成：

1. 算法延迟（固定）:
   ├── chunk 大小: 320ms（必须积累够一个 chunk 才能处理）
   ├── 右侧 context: 约 320ms（Encoder 需要看到未来几帧）
   └── 总算法延迟: 约 640ms

2. 计算延迟（取决于硬件）:
   ├── Encoder 推理: ~10ms (CPU, 4 线程)
   ├── Transducer 解码: ~5ms
   └── 总计算延迟: ~15ms

3. 总延迟: 约 640ms + 15ms ≈ 655ms
   → 用户说完一个字后约 0.65 秒看到结果

对比：
  sherpa-onnx (Zipformer):  ~650ms
  FunASR (Paraformer流式):  ~800ms
  WeNet (Conformer):        ~700ms
  人类反应时间:             ~200-300ms
```

## 7. 模型参数与性能对比

### 7.1 Zipformer-Transducer 不同规模

| 参数             | 小模型             | 中模型             | 大模型             |
| -------------- | --------------- | --------------- | --------------- |
| Encoder 维度     | 192/256/256     | 192/256/384     | 256/384/512     |
| Encoder 层数     | 2/2/3/2/2       | 2/2/3/2/2       | 2/2/4/2/2       |
| Predictor      | Conv, context=2 | Conv, context=2 | Conv, context=2 |
| Joint 维度       | 512             | 512             | 512             |
| 总参数量           | ~12M            | ~30M            | ~70M            |
| 模型大小 (ONNX)    | ~15MB           | ~40MB           | ~70MB           |
| RTF (CPU, 4线程) | ~0.05           | ~0.1            | ~0.2            |
| 适用场景           | 移动端/嵌入式         | 通用              | 服务端             |

### 7.2 不同模型架构对比

| 模型                       | 类型  | 参数量       | RTF (CPU) | 中文 CER | 适用场景  |
| ------------------------ | --- | --------- | --------- | ------ | ----- |
| Zipformer-Transducer (中) | 流式  | ~30M      | ~0.1      | ~5%    | 实时识别  |
| Zipformer-Transducer (小) | 流式  | ~12M      | ~0.05     | ~7%    | 移动端   |
| Paraformer (INT8)        | 离线  | ~220M→60M | ~0.1      | ~4%    | 高精度转写 |
| Whisper-small            | 离线  | ~244M     | ~0.5      | ~6%    | 多语言   |

> RTF < 1 表示实时。RTF = 0.1 意味着处理 1 秒音频只需 0.1 秒，即 10 倍实时速度。

## 8. 部署场景与最佳实践

### 8.1 部署场景推荐

| 场景            | 推荐模型                       | 推荐配置         | 说明      |
| ------------- | -------------------------- | ------------ | ------- |
| 服务端离线转写       | Paraformer (INT8)          | CPU 多线程或 GPU | 高精度，高吞吐 |
| 服务端流式识别       | Zipformer-Transducer (中/大) | CPU 多线程      | 低延迟，高并发 |
| Android / iOS | Zipformer-Transducer (小)   | 4 线程, INT8   | 轻量，省电   |
| 嵌入式设备         | Zipformer-Transducer (小)   | 2 线程, INT8   | 极致轻量    |
| 浏览器           | Zipformer-Transducer (小)   | WebAssembly  | 无需服务端   |
| 多语言场景         | Whisper                    | CPU/GPU      | 99 种语言  |

### 8.2 服务端部署架构

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  客户端      │     │  WebSocket 服务   │     │  sherpa-onnx    │
│  (浏览器/App)│────→│                  │────→│  推理引擎       │
│             │←────│  音频流接收       │←────│                 │
│  实时音频流  │     │  结果推送         │     │  流式/离线识别   │
│  结果展示    │     │  连接管理         │     │  VAD + ASR      │
└─────────────┘     └──────────────────┘     └─────────────────┘

关键设计：
├── 使用 WebSocket 传输音频流和识别结果
├── 每个连接创建独立的 stream（线程安全）
├── recognizer 实例可以在多个连接间共享（模型只加载一次）
├── 根据并发量调整 num_threads 和实例数
└── 长音频建议先 VAD 分段再识别
```

### 8.3 移动端部署注意事项

```
移动端优化要点：

1. 模型选择
   ├── 使用小模型（~12M 参数，~15MB）
   ├── 使用 INT8 量化版本
   └── 避免使用 Paraformer 大模型（太大）

2. 线程配置
   ├── Android: 2~4 线程（取决于 CPU 核心数）
   ├── iOS: 2~4 线程
   └── 不要超过物理核心数

3. 内存管理
   ├── 模型加载后常驻内存（避免反复加载）
   ├── stream 用完及时释放
   └── 小模型内存占用约 50~100MB

4. 电量优化
   ├── 使用 VAD 避免对静音段做 ASR
   ├── 合理设置 chunk 大小（太小会增加调用频率）
   └── 不识别时释放 CPU 资源
```

### 8.4 WebAssembly 部署

sherpa-onnx 支持编译为 WebAssembly，在浏览器中直接运行语音识别，无需服务端：

```
浏览器端 ASR 架构：

┌─────────────────────────────────────────────┐
│  浏览器                                      │
│                                              │
│  ┌──────────┐    ┌────────────────────┐     │
│  │ 麦克风    │───→│ Web Audio API      │     │
│  │ (getUserMedia)│ (采集 + 重采样)     │     │
│  └──────────┘    └────────┬───────────┘     │
│                           ↓                  │
│              ┌────────────────────────┐      │
│              │ sherpa-onnx (WASM)     │      │
│              │ ├── ONNX Runtime WASM  │      │
│              │ ├── Zipformer 模型     │      │
│              │ └── 流式解码           │      │
│              └────────────┬───────────┘      │
│                           ↓                  │
│              ┌────────────────────────┐      │
│              │ 识别结果展示            │      │
│              └────────────────────────┘      │
│                                              │
│  优势：                                      │
│  ├── 数据不出浏览器，隐私安全                 │
│  ├── 无需服务端，零运维成本                   │
│  └── 离线可用（模型缓存后）                   │
│                                              │
│  限制：                                      │
│  ├── 模型大小受限（建议 < 20MB）              │
│  ├── 计算性能不如原生                         │
│  └── 首次加载需要下载模型                     │
└─────────────────────────────────────────────┘
```

## 9. 进阶功能

### 9.1 热词（Hotwords）

sherpa-onnx 支持热词功能，可以提升特定词汇的识别准确率：

```python
# 流式识别 + 热词
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    encoder=f"{model_dir}/encoder-epoch-99-avg-1.onnx",
    decoder=f"{model_dir}/decoder-epoch-99-avg-1.onnx",
    joiner=f"{model_dir}/joiner-epoch-99-avg-1.onnx",
    tokens=f"{model_dir}/tokens.txt",
    num_threads=4,
    sample_rate=16000,
    feature_dim=80,
    hotwords_file="hotwords.txt",    # 热词文件
    hotwords_score=1.5,              # 热词加分权重
)

# hotwords.txt 格式（每行一个热词，可选权重）：
# 人工智能
# 深度学习
# sherpa-onnx :2.0
```

### 9.2 说话人识别

sherpa-onnx 支持说话人嵌入提取和说话人验证：

```python
# 提取说话人嵌入向量
extractor_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
    model="models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
    num_threads=4,
)
extractor = sherpa_onnx.SpeakerEmbeddingExtractor(extractor_config)

# 提取嵌入
wave = sherpa_onnx.read_wave("speaker1.wav")
stream = extractor.create_stream()
stream.accept_waveform(wave.sample_rate, wave.samples)
stream.input_finished()
embedding = extractor.compute(stream)  # 固定维度的向量

# 比较两个说话人是否为同一人
# 计算余弦相似度，阈值通常设为 0.5~0.7
```

### 9.3 语音合成（TTS）

sherpa-onnx 也支持语音合成功能：

```python
# 语音合成示例
tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model="models/vits-zh/model.onnx",
            tokens="models/vits-zh/tokens.txt",
            lexicon="models/vits-zh/lexicon.txt",
        ),
    ),
    max_num_sentences=1,
)
tts = sherpa_onnx.OfflineTts(tts_config)

# 合成语音
audio = tts.generate("你好，欢迎使用语音合成功能。", sid=0, speed=1.0)

# 保存为 WAV
import soundfile as sf
sf.write("output.wav", audio.samples, audio.sample_rate)
```

### 9.4 关键词检测（KWS）

```python
# 关键词检测示例
keyword_spotter = sherpa_onnx.KeywordSpotter(
    encoder=f"{model_dir}/encoder.onnx",
    decoder=f"{model_dir}/decoder.onnx",
    joiner=f"{model_dir}/joiner.onnx",
    tokens=f"{model_dir}/tokens.txt",
    keywords_file="keywords.txt",    # 关键词列表
    num_threads=2,
)

# keywords.txt 格式：
# 你好小助手
# 开始录音
# 停止播放
```

## 10. C/C++ API 使用

sherpa-onnx 的 C API 是所有其他语言绑定的基础，适合嵌入式和高性能场景：

```c
/* C API 离线识别示例 */
#include "sherpa-onnx/c-api/c-api.h"

int main() {
    // 配置
    SherpaOnnxOfflineRecognizerConfig config;
    memset(&config, 0, sizeof(config));

    config.model_config.paraformer.model = "model.int8.onnx";
    config.model_config.tokens = "tokens.txt";
    config.model_config.num_threads = 4;
    config.feat_config.sample_rate = 16000;
    config.feat_config.feature_dim = 80;

    // 创建识别器
    SherpaOnnxOfflineRecognizer *recognizer = 
        SherpaOnnxCreateOfflineRecognizer(&config);

    // 读取音频并识别
    const SherpaOnnxWave *wave = SherpaOnnxReadWave("test.wav");
    SherpaOnnxOfflineStream *stream = 
        SherpaOnnxCreateOfflineStream(recognizer);

    SherpaOnnxAcceptWaveformOffline(
        stream, wave->sample_rate, wave->samples, wave->num_samples);
    SherpaOnnxDecodeOfflineStream(recognizer, stream);

    const SherpaOnnxOfflineRecognizerResult *result = 
        SherpaOnnxGetOfflineStreamResult(stream);

    printf("识别结果: %s\n", result->text);

    // 释放资源
    SherpaOnnxDestroyOfflineRecognizerResult(result);
    SherpaOnnxDestroyOfflineStream(stream);
    SherpaOnnxDestroyOfflineRecognizer(recognizer);
    SherpaOnnxFreeWave(wave);

    return 0;
}
```

编译：

```bash
# 假设 sherpa-onnx 已编译安装
gcc -o offline_asr offline_asr.c \
    -I/path/to/sherpa-onnx/include \
    -L/path/to/sherpa-onnx/lib \
    -lsherpa-onnx-c-api -lonnxruntime
```

## 11. 与 Kaldi 和 FunASR 的关系

### 11.1 技术传承

```
Kaldi (2011, Daniel Povey)
    │
    │  核心贡献：WFST 解码、GMM-HMM/DNN-HMM 训练范式、数据格式标准
    │
    ├──→ k2 (2020, Daniel Povey)
    │       核心贡献：将 FST 运算与 PyTorch 深度集成
    │       ├── 可微分的 FST 运算（支持反向传播）
    │       ├── 高效的 CTC / Transducer 损失函数
    │       └── 基于 FST 的解码算法
    │
    ├──→ icefall (训练框架)
    │       基于 k2 + PyTorch
    │       ├── Zipformer 模型定义和训练
    │       ├── 数据处理（Lhotse）
    │       └── 训练脚本和 recipe
    │
    └──→ sherpa-onnx (推理框架)
            基于 ONNX Runtime
            ├── 加载 icefall 导出的 ONNX 模型
            ├── 也支持 FunASR 的 Paraformer ONNX 模型
            ├── 也支持 OpenAI 的 Whisper ONNX 模型
            └── 跨平台部署
```

### 11.2 三者的分工

| 维度   | Kaldi                | FunASR           | Sherpa-onnx                      |
| ---- | -------------------- | ---------------- | -------------------------------- |
| 定位   | 传统 ASR 训练+推理         | 端到端训练+推理         | 纯推理部署                            |
| 训练   | 支持（GMM-HMM/DNN-HMM）  | 支持（Paraformer 等） | 不支持（依赖 icefall）                  |
| 推理   | 支持（WFST 解码）          | 支持（Python/ONNX）  | 支持（ONNX，跨平台）                     |
| 核心模型 | GMM-HMM / Chain TDNN | Paraformer       | Zipformer / Paraformer / Whisper |
| 部署平台 | Linux 为主             | Linux / Android  | 全平台                              |
| 依赖   | C++ 编译环境             | PyTorch          | 仅 ONNX Runtime                   |
| 适合人群 | ASR 研究者              | 中文 ASR 开发者       | 部署工程师                            |

### 11.3 如何选择

```
决策树：

你需要训练自己的模型吗？
├── 是 → 你的数据主要是中文吗？
│        ├── 是 → FunASR（Paraformer 效果最好）
│        └── 否 → icefall（Zipformer，多语言支持好）
│
└── 否 → 你需要部署到哪里？
         ├── 服务端（Linux）→ FunASR Runtime 或 sherpa-onnx 都可以
         ├── 移动端（Android/iOS）→ sherpa-onnx
         ├── 嵌入式 → sherpa-onnx
         ├── 浏览器 → sherpa-onnx (WebAssembly)
         └── 多平台统一 → sherpa-onnx（一套代码多平台部署）
```

## 12. 常见问题与调优

### 12.1 识别效果不好

```
排查步骤：
1. 检查音频质量
   ├── 采样率是否为 16kHz？（sherpa-onnx 默认 16kHz）
   ├── 是否有严重噪声？（考虑加 VAD 预处理）
   └── 音量是否正常？（过小或过大都影响效果）

2. 检查模型选择
   ├── 中文识别 → 优先用 Paraformer（离线）或 Zipformer 中文模型（流式）
   ├── 英文识别 → Zipformer 英文模型或 Whisper
   └── 多语言 → Whisper

3. 调整解码参数
   ├── 尝试 modified_beam_search（比 greedy_search 效果好，但稍慢）
   └── 增大 max_active_paths（beam search 的候选数）
```

### 12.2 推理速度慢

```
优化方向：
1. 使用 INT8 量化模型（体积减小 4 倍，速度提升 2~3 倍）
2. 调整 num_threads（不要超过物理 CPU 核心数）
3. 使用更小的模型（小模型 RTF ~0.05，大模型 RTF ~0.2）
4. 流式场景调整 chunk 大小（更大的 chunk 计算效率更高，但延迟增加）
5. 批量处理多个音频（decode_streams 比逐个 decode_stream 更高效）
```

### 12.3 内存占用大

```
优化方向：
1. 使用 INT8 量化模型（内存占用减少约 4 倍）
2. 使用小模型（12M 参数 vs 70M 参数）
3. 及时释放不用的 stream
4. 减少 num_threads（每个线程有独立的工作空间）
```

### 12.4 常见错误

| 错误              | 原因         | 解决方案                 |
| --------------- | ---------- | -------------------- |
| 模型文件找不到         | 路径错误或模型未下载 | 检查路径，运行下载脚本          |
| 采样率不匹配          | 音频不是 16kHz | 用 ffmpeg 或 sox 重采样   |
| 输出为空            | 音频太短或全是静音  | 检查音频内容，确保有语音         |
| 内存不足            | 模型太大       | 使用 INT8 量化或更小的模型     |
| ONNX Runtime 错误 | 版本不兼容      | 更新 sherpa-onnx 到最新版本 |

## 13. 实战：构建一个完整的语音识别服务

以下是一个使用 sherpa-onnx 构建 WebSocket 流式语音识别服务的简化示例：

```python
"""
基于 sherpa-onnx 的 WebSocket 流式 ASR 服务（简化版）
客户端通过 WebSocket 发送音频流，服务端实时返回识别结果
"""
import asyncio
import json
import numpy as np
import sherpa_onnx
import websockets

# 全局识别器（所有连接共享，模型只加载一次）
model_dir = "models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    encoder=f"{model_dir}/encoder-epoch-99-avg-1.onnx",
    decoder=f"{model_dir}/decoder-epoch-99-avg-1.onnx",
    joiner=f"{model_dir}/joiner-epoch-99-avg-1.onnx",
    tokens=f"{model_dir}/tokens.txt",
    num_threads=4,
    sample_rate=16000,
    feature_dim=80,
    decoding_method="greedy_search",
)

async def handle_client(websocket):
    """处理单个客户端连接"""
    # 每个连接创建独立的 stream
    stream = recognizer.create_stream()
    last_text = ""

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # 接收音频数据（假设 16kHz, 16bit, 单声道 PCM）
                samples = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                stream.accept_waveform(16000, samples.tolist())

                # 解码
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

                # 发送识别结果
                current_text = recognizer.get_result(stream).text
                if current_text != last_text:
                    await websocket.send(json.dumps({
                        "text": current_text,
                        "is_final": False,
                    }))
                    last_text = current_text

            elif message == "END":
                # 客户端标记音频结束
                tail = [0.0] * 8000  # 0.5 秒静音
                stream.accept_waveform(16000, tail)
                stream.input_finished()

                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

                final_text = recognizer.get_result(stream).text
                await websocket.send(json.dumps({
                    "text": final_text,
                    "is_final": True,
                }))

    except websockets.exceptions.ConnectionClosed:
        pass

async def main():
    async with websockets.serve(handle_client, "0.0.0.0", 6006):
        print("ASR 服务已启动: ws://0.0.0.0:6006")
        await asyncio.Future()  # 永久运行

if __name__ == "__main__":
    asyncio.run(main())
```

> 这是一个简化的示例，生产环境还需要考虑：连接池管理、错误处理、日志记录、负载均衡、音频格式协商、心跳检测等。

## 14. 总结

### 14.1 sherpa-onnx 的核心价值

```
sherpa-onnx 解决的核心问题：
"如何将训练好的语音模型高效部署到任何平台"

核心价值：
├── 极致轻量：无 PyTorch 依赖，纯 ONNX Runtime
├── 全平台覆盖：从服务器到浏览器，一套框架搞定
├── 模型丰富：Zipformer / Paraformer / Whisper / NeMo 等
├── 功能完整：ASR + VAD + TTS + 说话人识别 + 语种识别
└── 开箱即用：预训练模型 + 简洁 API + 丰富示例
```

### 14.2 学习路径建议

```
入门：
1. pip install sherpa-onnx
2. 下载一个预训练模型
3. 跑通离线识别 demo
4. 跑通流式识别 demo

进阶：
5. 理解 Zipformer 架构和 Transducer 解码原理
6. 尝试 VAD + ASR 组合使用
7. 在 Android/iOS 上部署
8. 搭建 WebSocket 流式服务

深入：
9. 用 icefall 训练自己的模型并导出 ONNX
10. 优化推理性能（量化、线程调优）
11. 定制热词和领域适配
12. 贡献代码到 sherpa-onnx 开源社区
```

### 14.3 相关资源

```
官方资源：
├── GitHub: https://github.com/k2-fsa/sherpa-onnx
├── 文档: https://k2-fsa.github.io/sherpa/onnx/
├── 预训练模型: https://github.com/k2-fsa/sherpa-onnx/releases
├── icefall (训练): https://github.com/k2-fsa/icefall
└── k2 (核心库): https://github.com/k2-fsa/k2

相关项目：
├── FunASR: https://github.com/modelscope/FunASR
├── WeNet: https://github.com/wenet-e2e/wenet
└── Whisper: https://github.com/openai/whisper
```
