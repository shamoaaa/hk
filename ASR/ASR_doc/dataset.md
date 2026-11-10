# 中文语音识别（ASR）开源数据集介绍

本文档汇总了常用的中文普通话语音识别开源数据集的详细信息，包括数据规模、来源、许可证、数据结构、音频格式、标签格式及官方链接等。

---

## 1. THCHS-30

| 项目         | 说明                                                      |
| ---------- | ------------------------------------------------------- |
| 全称         | THCHS-30 (Tsinghua Continuous Mandarin Speech Database) |
| 发布机构       | 清华大学语音与语言技术中心 (CSLT)                                    |
| OpenSLR 编号 | SLR18                                                   |
| 数据规模       | 约 30 小时（训练集约 25 小时、开发集约 2.5 小时、测试集约 2.5 小时）             |
| 说话人数       | 约 40 人                                                  |
| 许可证        | Apache License v2.0                                     |
| 发布时间       | 2015 年                                                  |

### 简介

THCHS-30 是清华大学 CSLT 实验室发布的免费中文语音数据库，原始录音于 2002 年由王东在清华大学计算机系智能技术与系统国家重点实验室完成，原名 TCMSD。该数据集旨在为语音识别领域的新研究者提供一个入门级数据库，完全免费供学术使用。数据集还提供了噪声环境下的测试数据，可用于评估鲁棒性。

### 音频格式

| 属性   | 值                      |
| ---- | ---------------------- |
| 文件格式 | WAV (PCM)              |
| 采样率  | 16,000 Hz              |
| 位深度  | 16 bit                 |
| 声道数  | 单声道 (Mono)             |
| 编码   | 线性 PCM (Little-Endian) |

### 数据结构

```
thchs30/
├── data_thchs30/
│   ├── data/                       # 所有音频和转写的原始存放目录
│   │   ├── A2_0.wav                # 音频文件（说话人A2，第0句）
│   │   ├── A2_0.wav.trn            # 对应的转写标注文件
│   │   ├── A2_1.wav
│   │   ├── A2_1.wav.trn
│   │   ├── B2_100.wav
│   │   ├── B2_100.wav.trn
│   │   └── ...
│   ├── train/                      # 训练集（符号链接指向 data/ 中的文件）
│   │   ├── A2_0.wav -> ../data/A2_0.wav
│   │   ├── A2_0.wav.trn -> ../data/A2_0.wav.trn
│   │   └── ...
│   ├── dev/                        # 开发集
│   │   └── ...
│   ├── test/                       # 测试集
│   │   └── ...
│   ├── lm_word/                    # 词级语言模型
│   │   └── lexicon.txt             # 发音词典
│   └── lm_phone/                   # 音素级语言模型
├── test-noise/                     # 带噪声的测试集（白噪声、车内噪声、餐厅噪声等）
│   ├── 0db/                        # 0dB 信噪比
│   ├── 5db/                        # 5dB 信噪比
│   └── ...
└── resource/                       # 发音词典等资源文件
```

音频文件命名规则：`<说话人ID>_<语句序号>.wav`

- `A2` / `B11` / `C4` 等为说话人编号，字母表示录音批次，数字表示说话人序号
- `_0` / `_100` / `_374` 等为语句序号

### 标签格式

每条音频对应一个 `.wav.trn` 转写文件，包含三行内容：

```
# 文件：A2_0.wav.trn

# 第1行：汉字文本（空格分隔的词/字）
他 仅 凭 腰部 的 力量 在 弹跳 之间 完成 了 两 个 L = 转体 七百 二十 度

# 第2行：拼音标注（声调用数字 1-4 表示）
ta1 jin3 ping2 yao1 bu4 de5 li4 liang4 zai4 tan2 tiao4 zhi1 jian1 wan2 cheng2 le5 liang3 ge4 zhuan3 ti3 qi1 bai3 er4 shi2 du4

# 第3行：声韵母级音素标注
t a1 j in3 p ing2 ii iao1 b u4 d e5 l i4 l iang4 z ai4 t an2 t iao4 zh ix1 j ian1 uu uan2 ch eng2 l e5 l iang3 g e4 zh uan3 t i3 q i1 b ai3 ee er4 sh ix2 d u4
```

标签说明：

- 第 1 行为中文文本转写，词之间用空格分隔
- 第 2 行为拼音标注，声调以数字后缀表示（1=阴平, 2=阳平, 3=上声, 4=去声, 5=轻声）
- 第 3 行为声韵母级音素标注，用于声学模型训练

### 官方链接

- OpenSLR: https://www.openslr.org/18/
- 论文: [THCHS-30: A Free Chinese Speech Corpus](https://arxiv.org/abs/1512.01882)
- CSLT 主页: http://cslt.org

### 引用

```bibtex
@misc{THCHS30_2015,
  title={THCHS-30 : A Free Chinese Speech Corpus},
  author={Dong Wang, Xuewei Zhang, Zhiyong Zhang},
  year={2015},
  url={http://arxiv.org/abs/1512.01882}
}
```

---

## 2. AISHELL (AISHELL 系列总览)

AISHELL 是由北京希尔贝壳科技有限公司（Beijing Shell Shell Technology Co., Ltd.）发布的一系列中文语音数据集，涵盖语音识别（ASR）和语音合成（TTS）等多个任务。主要包括：

| 数据集       | 规模       | 说话人    | 用途     | OpenSLR |
| --------- | -------- | ------ | ------ | ------- |
| AISHELL-1 | ~178 小时  | 400 人  | ASR    | SLR33   |
| AISHELL-2 | ~1000 小时 | 1991 人 | ASR    | 需申请     |
| AISHELL-3 | ~85 小时   | 218 人  | TTS    | SLR93   |
| AISHELL-4 | 会议场景     | 多人     | 会议 ASR | SLR111  |

### 官方链接

- 公司主页: http://www.aishelltech.com
- AISHELL-1 OpenSLR: https://www.openslr.org/33/
- AISHELL-3 OpenSLR: https://openslr.org/93/
- AISHELL-4 OpenSLR: https://www.openslr.org/111/
- Hugging Face: https://huggingface.co/datasets/AISHELL/AISHELL-1

---

## 3. AISHELL-1

| 项目         | 说明                                               |
| ---------- | ------------------------------------------------ |
| 全称         | AISHELL-1: An Open-Source Mandarin Speech Corpus |
| 发布机构       | 北京希尔贝壳科技有限公司                                     |
| OpenSLR 编号 | SLR33                                            |
| 数据规模       | 约 178 小时（训练集 ~170h / 开发集 ~10h / 测试集 ~5h）         |
| 说话人数       | 400 人（来自中国不同口音区域）                                |
| 许可证        | Apache License v2.0                              |
| 发布时间       | 2017 年                                           |

### 简介

AISHELL-1 是目前最广泛使用的开源中文普通话语音识别数据集之一。录音在安静的室内环境中使用高保真麦克风完成，采样率降至 16kHz。人工转写准确率在 95% 以上，经过专业语音标注和严格质量检查。该数据集免费供学术使用，是中文 ASR 研究的标准基准数据集。Kaldi、ESPnet、WeNet 等主流框架均提供了现成的训练 recipe。

### 音频格式

| 属性   | 值                      |
| ---- | ---------------------- |
| 文件格式 | WAV (PCM)              |
| 采样率  | 16,000 Hz              |
| 位深度  | 16 bit                 |
| 声道数  | 单声道 (Mono)             |
| 编码   | 线性 PCM (Little-Endian) |
| 录音设备 | 高保真麦克风（室内安静环境）         |

### 数据结构

```
data_aishell/
├── wav/                                    # 音频目录（按数据集划分 → 说话人子目录）
│   ├── train/                              # 训练集（约 340 位说话人）
│   │   ├── S0002/                          # 说话人 S0002 的目录
│   │   │   ├── BAC009S0002W0122.wav        # 单条语音文件
│   │   │   ├── BAC009S0002W0123.wav
│   │   │   ├── BAC009S0002W0124.wav
│   │   │   └── ...                         # 每位说话人约 300-500 条语音
│   │   ├── S0003/
│   │   ├── S0004/
│   │   └── ...
│   ├── dev/                                # 开发集（约 40 位说话人）
│   │   ├── S0723/
│   │   └── ...
│   └── test/                               # 测试集（约 20 位说话人）
│       ├── S0764/
│       └── ...
├── transcript/
│   └── aishell_transcript_v0.8.txt         # 全部转写文本（单一文件）
└── resource_aishell/
    └── lexicon.txt                         # 发音词典
```

音频文件命名规则：`BAC009S0002W0122.wav`

- `BAC009` — 录音批次编号
- `S0002` — 说话人编号
- `W0122` — 语句编号

### 标签格式

转写文件为单一纯文本文件 `aishell_transcript_v0.8.txt`，每行一条语音的转写：

```
# 格式：<utterance_id> <空格分隔的汉字序列>

BAC009S0002W0122 甚 至 出 现 交 易
BAC009S0002W0123 在 这 期 间 她 始 终 没 有 放 弃 寻 找 自 己 的 亲 生 父 母
BAC009S0002W0124 这 一 次 我 们 是 受 到 了 公 司 的 委 托
BAC009S0764W0001 广 州 市 房 地 产 中 介 协 会 发 布 统 计 数 据 显 示
```

标签说明：

- 每行第一个字段为语音文件名（不含 `.wav` 后缀），与音频文件一一对应
- 后续字段为空格分隔的单个汉字序列
- 不包含标点符号
- 全角英文字母（如 `ａ`、`ｂ`）需在预处理时转为半角

### 官方链接

- OpenSLR: https://www.openslr.org/33/
- 公司主页: http://www.aishelltech.com/kysjcp
- Hugging Face: https://huggingface.co/datasets/AISHELL/AISHELL-1
- LDC: https://catalog.ldc.upenn.edu/LDC2018S14

### 引用

```bibtex
@inproceedings{aishell_2017,
  title={AIShell-1: An Open-Source Mandarin Speech Corpus and A Speech Recognition Baseline},
  author={Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, Hao Zheng},
  booktitle={Oriental COCOSDA 2017},
  year={2017}
}
```

---

## 4. aidatatang_200zh

| 项目         | 说明                                                 |
| ---------- | -------------------------------------------------- |
| 全称         | aidatatang_200zh                                   |
| 发布机构       | 北京数据堂科技有限公司（Beijing DataTang Technology Co., Ltd.） |
| OpenSLR 编号 | SLR62                                              |
| 数据规模       | 约 200 小时，共 237,265 条语音                             |
| 说话人数       | 600 人（来自中国不同口音区域）                                  |
| 转写准确率      | 每句大于 98%                                           |
| 许可证        | CC BY-NC-ND 4.0                                    |
| 发布时间       | 2018 年                                             |

### 简介

aidatatang_200zh 是由数据堂（DataTang）发布的开源中文普通话语音数据集，包含 200 小时的语音数据。600 位来自中国不同口音区域的说话人参与了录音，每句转写准确率大于 98%。该数据集适用于语音识别系统的训练和评估，免费供学术使用。数据集按照训练集、验证集和测试集进行了划分。Kaldi 在 `egs/aidatatang_200zh/` 下提供了完整的训练 recipe，基准实验字识别准确率可达 94.41%（Chain 模型）。

> 注意：该数据集在 OpenSLR 上已被数据所有者撤回（Resource retracted as per the data owner wish），但部分镜像仍可获取。

### 音频格式

| 属性   | 值                      |
| ---- | ---------------------- |
| 文件格式 | WAV (PCM)              |
| 采样率  | 16,000 Hz              |
| 位深度  | 16 bit                 |
| 声道数  | 单声道 (Mono)             |
| 编码   | 线性 PCM (Little-Endian) |

### 数据结构

```
aidatatang_200zh/
├── corpus/                                 # 音频数据主目录
│   ├── train/                              # 训练集
│   │   ├── G0002/                          # 说话人目录
│   │   │   ├── T0055G0002S0001.wav         # 单条语音文件
│   │   │   ├── T0055G0002S0002.wav
│   │   │   └── ...                         # 每位说话人数百条语音
│   │   ├── G0003/
│   │   ├── G0004/
│   │   └── ...
│   ├── dev/                                # 开发集
│   │   ├── G0500/
│   │   └── ...
│   └── test/                               # 测试集
│       ├── G0550/
│       └── ...
└── transcript/
    └── aidatatang_200_zh_transcript.txt     # 全部转写文本
```

音频文件命名规则：`T0055G0002S0001.wav`

- `T0055` — 录音任务/批次编号
- `G0002` — 说话人编号
- `S0001` — 语句序号

### 标签格式

转写文件为单一纯文本文件 `aidatatang_200_zh_transcript.txt`，每行一条语音的转写：

```
# 格式：<utterance_id> <转写文本>

T0055G0002S0001 他 在 学校 里 表现 非常 好
T0055G0002S0002 今天 天气 怎么样
T0055G0002S0003 请 帮 我 查 一下 明天 的 航班
```

标签说明：

- 每行第一个字段为语音文件名（不含 `.wav` 后缀）
- 后续为空格分隔的中文文本转写
- 转写经过人工精心标注和质量检查

Kaldi recipe 数据准备阶段会生成以下标准映射文件：

- `text` — `<utterance-id> <transcription>`
- `wav.scp` — `<utterance-id> <wav-path>`
- `utt2spk` — `<utterance-id> <speaker-id>`
- `spk2utt` — `<speaker-id> <utterance-id1> <utterance-id2> ...`

### 官方链接

- OpenSLR: https://openslr.org/62/ （已撤回）
- 数据堂官网: https://www.datatang.ai
- GitHub 参考: https://github.com/fantasyyyy/aidatatang_200zh

---

## 5. ST-CMDS (STCMDS)

| 项目         | 说明                                                   |
| ---------- | ---------------------------------------------------- |
| 全称         | Free ST Chinese Mandarin Corpus (ST-CMDS-20170001_1) |
| 发布机构       | Surfingtech（www.surfing.ai）                          |
| OpenSLR 编号 | SLR38                                                |
| 数据规模       | 约 110 小时，102,600 条语音                                 |
| 说话人数       | 855 人                                                |
| 每人语句数      | 120 条                                                |
| 许可证        | CC BY-NC-ND 4.0                                      |

### 简介

ST-CMDS 是由 Surfingtech 发布的免费中文普通话语音数据集。所有录音在安静的室内环境中使用手机完成，共有 855 位说话人，每人 120 条语音。所有语音均经过人工仔细转写和检查，转写准确率有保障。该数据集是一个更大规模数据集的子集。该数据集不提供官方的 train/dev/test 划分，通常由使用者自行划分。

### 音频格式

| 属性   | 值                      |
| ---- | ---------------------- |
| 文件格式 | WAV (PCM)              |
| 采样率  | 16,000 Hz              |
| 位深度  | 16 bit                 |
| 声道数  | 单声道 (Mono)             |
| 编码   | 线性 PCM (Little-Endian) |
| 录音设备 | 手机（室内安静环境）             |

### 数据结构

```
ST-CMDS-20170001_1-OS/
├── 20170001P00001A0001.wav         # 音频文件
├── 20170001P00001A0001.txt         # 对应转写文本（一对一）
├── 20170001P00001A0002.wav
├── 20170001P00001A0002.txt
├── 20170001P00001A0003.wav
├── 20170001P00001A0003.txt
├── ...
├── 20170001P00002A0001.wav         # 第二位说话人
├── 20170001P00002A0001.txt
├── ...
├── 20170001P00855A0120.wav         # 最后一位说话人的最后一条
├── 20170001P00855A0120.txt
└── (共 102,600 对 .wav + .txt 文件，平铺在同一目录下)
```

音频文件命名规则：`20170001P00001A0001.wav`

- `20170001` — 数据集批次编号（固定前缀）
- `P00001` — 说话人编号（P00001 ~ P00855）
- `A0001` — 语句编号（A0001 ~ A0120，每人 120 条）

### 标签格式

每条音频对应一个同名的 `.txt` 文件，内容为单行中文转写文本：

```
# 文件：20170001P00001A0001.txt
绿 是 阳春 烟 景 大块 文章 的 底色 四月 的 林 峦 更是 绿 得 鲜活 诗意 盎然

# 文件：20170001P00001A0002.txt
他 的 想法 很 有 道理
```

标签说明：

- 每个 `.txt` 文件仅包含一行文本
- 文本为中文汉字，可能包含中文逗号（`，`）等标点
- 预处理时通常需要去除标点并转为大写（针对可能出现的英文字母）
- 说话人 ID 可从文件名中提取（第 9~15 位字符，即 `P00001` 部分）

### 官方链接

- OpenSLR: https://openslr.org/38/
- Surfingtech: https://www.surfing.ai

### 引用

请引用为：`ST-CMDS-20170001_1, Free ST Chinese Mandarin Corpus`

---

## 6. Primewords (Primewords Chinese Corpus Set 1)

| 项目         | 说明                                                                  |
| ---------- | ------------------------------------------------------------------- |
| 全称         | Primewords Chinese Corpus Set 1                                     |
| 发布机构       | 上海普力蒙信息技术有限公司（Shanghai Primewords Information Technology Co., Ltd.） |
| OpenSLR 编号 | SLR47                                                               |
| 数据规模       | 约 100 小时，约 50,000 条语音                                               |
| 说话人数       | 296 人（中文母语者）                                                        |
| 转写准确率      | 大于 98%（95% 置信水平）                                                    |
| 许可证        | CC BY-NC-ND 4.0                                                     |
| 发布时间       | 2018 年                                                              |

### 简介

Primewords 中文语音数据集由上海普力蒙信息技术有限公司发布，包含 100 小时的语音数据。录音使用智能手机完成，由 296 位中文母语者参与。转写准确率在 95% 置信水平下大于 98%，免费供学术使用。该数据集的特点是转写与音频的映射关系以 JSON 格式提供，而非传统的纯文本格式。数据集不提供官方的 train/dev/test 划分。

### 音频格式

| 属性   | 值                      |
| ---- | ---------------------- |
| 文件格式 | WAV (PCM)              |
| 采样率  | 16,000 Hz              |
| 位深度  | 16 bit                 |
| 声道数  | 单声道 (Mono)             |
| 编码   | 线性 PCM (Little-Endian) |
| 录音设备 | 智能手机                   |

### 数据结构

```
primewords_md_2018_set1/
├── audio_files/                            # 音频文件目录（多级子目录组织）
│   ├── 0/                                  # 第一级目录（按 ID 前缀分桶）
│   │   ├── 00/                             # 第二级目录
│   │   │   ├── 0000a001.wav                # 单条语音文件
│   │   │   ├── 0000a002.wav
│   │   │   └── ...
│   │   ├── 01/
│   │   │   └── ...
│   │   └── ...
│   ├── 1/
│   │   └── ...
│   ├── 2/
│   │   └── ...
│   └── ...
└── set1_transcript.json                    # JSON 格式转写文件（全部转写集中在此）
```

音频文件通过多级目录组织以避免单目录下文件过多，目录层级基于文件名前缀进行分桶。

### 标签格式

转写信息集中存储在单一 JSON 文件 `set1_transcript.json` 中：

```json
{
  "0000a001": {
    "file": "audio_files/0/00/0000a001.wav",
    "text": "今天天气真好适合出去走走",
    "user_id": "U0001"
  },
  "0000a002": {
    "file": "audio_files/0/00/0000a002.wav",
    "text": "请帮我订一张明天去北京的机票",
    "user_id": "U0001"
  },
  "0100b015": {
    "file": "audio_files/1/01/0100b015.wav",
    "text": "我想听一首周杰伦的歌",
    "user_id": "U0042"
  }
}
```

JSON 字段说明：

- 键（key）：语音 ID（与音频文件名对应，不含 `.wav` 后缀）
- `file`：音频文件的相对路径
- `text`：中文转写文本（连续汉字，无空格分隔）
- `user_id`：说话人编号

标签特点：

- 转写文本为连续汉字字符串，不使用空格分隔
- 包含说话人信息（`user_id`），可用于说话人相关实验
- JSON 格式便于程序化解析和处理

### 官方链接

- OpenSLR: https://www.openslr.org/47/
- 公司主页: https://www.primewords.cn

### 引用

```bibtex
@misc{primewords_201801,
  title={Primewords Chinese Corpus Set 1},
  author={Primewords Information Technology Co., Ltd.},
  year={2018},
  note={\url{https://www.primewords.cn}}
}
```

---

## 7. MAGICDATA (Magic Data)

| 项目         | 说明                                                |
| ---------- | ------------------------------------------------- |
| 全称         | MAGICDATA Mandarin Chinese Read Speech Corpus     |
| 发布机构       | 北京魔数智擎科技有限公司（Magic Data Technology Co., Ltd.）     |
| OpenSLR 编号 | SLR68                                             |
| 数据规模       | 755 小时（训练集 ~712h / 验证集 ~14h / 测试集 ~29h，比例 51:1:2） |
| 说话人数       | 1080 人（来自中国不同口音区域）                                |
| 转写准确率      | 大于 98%                                            |
| 许可证        | CC BY-NC-ND 4.0                                   |
| 发布时间       | 2019 年                                            |

### 简介

MAGICDATA 是由魔数智擎科技有限公司发布的大规模中文普通话朗读语音数据集，包含 755 小时的语音数据，主要为手机录制。1080 位来自中国不同口音区域的说话人参与录音，在安静的室内环境中完成。数据集按 51:1:2 的比例划分为训练集、验证集和测试集。录音文本领域多样化，涵盖交互问答、音乐搜索、社交消息、智能家居控制等场景。该数据集是一个更大规模数据集（10,566.9 小时）的子集，同时提供了分词后的转写文本。

### 音频格式

| 属性   | 值                      |
| ---- | ---------------------- |
| 文件格式 | WAV (PCM)              |
| 采样率  | 16,000 Hz              |
| 位深度  | 16 bit                 |
| 声道数  | 单声道 (Mono)             |
| 编码   | 线性 PCM (Little-Endian) |
| 录音设备 | 手机（室内安静环境）             |

### 数据结构

```
magicdata/
├── train/                                  # 训练集（约 712 小时）
│   ├── 16_3865/                            # 说话人目录
│   │   ├── 16_3865_20170916222802.wav      # 单条语音文件
│   │   ├── 16_3865_20170916222810.wav
│   │   └── ...
│   ├── 17_0291/
│   │   └── ...
│   └── ...
├── dev/                                    # 验证集（约 14 小时）
│   ├── 53_1266/
│   │   └── ...
│   └── ...
├── test/                                   # 测试集（约 29 小时）
│   ├── 38_0045/
│   │   └── ...
│   └── ...
├── TRANS.txt                               # 全部转写文本（或按子集拆分）
├── METADATA.txt                            # 说话人元数据信息
└── README.txt                              # 数据集说明文档
```

音频文件命名规则：`16_3865_20170916222802.wav`

- `16_3865` — 说话人编号（同时作为目录名）
- `20170916222802` — 录音时间戳（年月日时分秒）

### 标签格式

#### 转写文件 (TRANS.txt)

制表符分隔的纯文本文件，每行一条语音的转写：

```
# 格式：<音频文件名>\t<说话人ID>\t<转写文本>

16_3865_20170916222802.wav    16_3865    这个 酒店 的 位置 非常 好
16_3865_20170916222810.wav    16_3865    帮 我 查 一下 附近 有 什么 好吃 的
17_0291_20170801103408.wav    17_0291    今天 下午 三点 有 一个 会议
```

转写说明：

- 第 1 列：音频文件名（含 `.wav` 后缀）
- 第 2 列：说话人 ID
- 第 3 列：转写文本（空格分隔的词语）
- 同时提供分词后的转写版本

#### 元数据文件 (METADATA.txt)

包含说话人的详细信息：

```
# 格式：<说话人ID>\t<性别>\t<口音区域>\t<年龄段>\t<录音设备>

16_3865    F    北方    20-30    iPhone
17_0291    M    南方    30-40    Android
53_1266    F    西南    20-30    iPhone
```

元数据字段说明：

- 说话人 ID
- 性别（M=男, F=女）
- 口音区域（北方、南方、西南等）
- 年龄段
- 录音设备类型

### 官方链接

- OpenSLR: https://openslr.org/68/
- 公司主页: http://www.imagicdatatech.com
- 联系邮箱: business@magicdatatech.com

### 引用

请引用为：`Magic Data Technology Co., Ltd., "http://www.imagicdatatech.com/index.php/home/dataopensource/data_info/id/101", 05/2019`

---

## 8. TAL_ASR (TALCS)

| 项目   | 说明                               |
| ---- | -------------------------------- |
| 全称   | TALCS: TAL Code-Switching Corpus |
| 发布机构 | 好未来教育集团（TAL Education Group）     |
| 数据规模 | 约 587 小时                         |
| 语言   | 中英文混合（Code-Switching）            |
| 许可证  | 开源许可                             |
| 发布时间 | 2022 年（Interspeech 2022）         |

### 简介

TALCS（TAL Code-Switching）是由好未来教育集团（TAL Education Group）发布的中英文混合语音识别数据集。数据来源于真实的在线一对一英语教学场景，包含约 587 小时的语音数据。据发布者称，TALCS 是目前全球最大的标注良好的中英文代码切换（Code-Switching）开源 ASR 数据集。该数据集适用于训练和评估中英文混合语音识别系统，已在 ESPnet 和 WeNet 两个主流框架上建立了基线系统。

### 音频格式

| 属性   | 值                      |
| ---- | ---------------------- |
| 文件格式 | WAV (PCM)              |
| 采样率  | 16,000 Hz              |
| 位深度  | 16 bit                 |
| 声道数  | 单声道 (Mono)             |
| 编码   | 线性 PCM (Little-Endian) |
| 录音来源 | 在线一对一英语教学场景（真实录音）      |

### 数据结构

```
TALCS/
├── train/                                  # 训练集
│   ├── wav/                                # 音频目录
│   │   ├── <utterance_id_1>.wav            # 单条语音文件
│   │   ├── <utterance_id_2>.wav
│   │   └── ...
│   └── text                                # 转写文本文件
├── dev/                                    # 开发集
│   ├── wav/
│   │   └── ...
│   └── text
└── test/                                   # 测试集
    ├── wav/
    │   └── ...
    └── text
```

### 标签格式

转写文件 `text` 为纯文本格式，每行一条语音的转写：

```
# 格式：<utterance_id> <中英文混合转写>

utt_00001 老师 can you explain this again
utt_00002 我 不太 understand 这个 sentence
utt_00003 the answer is 三百 二十 五
utt_00004 请 read the first paragraph
```

标签特点：

- 中文部分使用汉字书写
- 英文部分使用英文单词书写
- 中英文在同一句中自然混合（Code-Switching）
- 评估指标使用混合错误率（Mixture Error Rate, MER），综合考虑中英文识别效果

### 官方链接

- 论文: [TALCS: An Open-Source Mandarin-English Code-Switching Corpus and a Speech Recognition Baseline](https://arxiv.org/abs/2206.13135)
- ISCA Archive: https://www.isca-archive.org/interspeech_2022/li22j_interspeech.html
- 好未来 GitHub: https://github.com/tal-tech

### 引用

```bibtex
@inproceedings{li2022talcs,
  title={TALCS: An Open-Source Mandarin-English Code-Switching Corpus and a Speech Recognition Baseline},
  author={Li, et al.},
  booktitle={Interspeech 2022},
  year={2022}
}
```

---

## 数据集对比总览

| 数据集              | 规模    | 说话人  | 采样率   | 位深    | 许可证             | OpenSLR | 标签格式                          |
| ---------------- | ----- | ---- | ----- | ----- | --------------- | ------- | ----------------------------- |
| THCHS-30         | ~30h  | ~40  | 16kHz | 16bit | Apache 2.0      | SLR18   | .wav.trn（三行：汉字/拼音/音素）         |
| AISHELL-1        | ~178h | 400  | 16kHz | 16bit | Apache 2.0      | SLR33   | 单一 txt（utterance_id + 空格分隔汉字） |
| aidatatang_200zh | ~200h | 600  | 16kHz | 16bit | CC BY-NC-ND 4.0 | SLR62   | 单一 txt（utterance_id + 文本）     |
| ST-CMDS          | ~110h | 855  | 16kHz | 16bit | CC BY-NC-ND 4.0 | SLR38   | 逐条 .txt（每条音频对应一个 txt）         |
| Primewords       | ~100h | 296  | 16kHz | 16bit | CC BY-NC-ND 4.0 | SLR47   | JSON（含 file/text/user_id）     |
| MAGICDATA        | ~755h | 1080 | 16kHz | 16bit | CC BY-NC-ND 4.0 | SLR68   | TSV（文件名 + 说话人 + 文本）+ 元数据      |
| TAL_ASR (TALCS)  | ~587h | -    | 16kHz | 16bit | 开源              | -       | text 文件（中英混合转写）               |

### 音频格式统一说明

以上所有数据集的音频格式均为：

- WAV 容器格式，PCM 线性编码
- 采样率 16,000 Hz（16kHz）
- 量化位深 16 bit
- 单声道（Mono）
- 字节序 Little-Endian

这是中文语音识别领域的事实标准格式，各主流 ASR 框架（Kaldi、ESPnet、WeNet、FunASR 等）均默认支持。

### 使用建议

- 入门学习：推荐 THCHS-30，规模小、结构清晰、标注丰富（含拼音和音素），适合快速上手
- 标准基准：推荐 AISHELL-1，社区使用最广泛，各框架均有现成 recipe
- 大规模训练：推荐 MAGICDATA（755h）或多数据集联合训练
- 中英混合场景：推荐 TAL_ASR (TALCS)
- 手机录音场景：推荐 ST-CMDS 或 Primewords
