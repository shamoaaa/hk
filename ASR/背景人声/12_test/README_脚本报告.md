# 12_test 目录脚本报告

本目录负责**独立测试集的构建与预处理**。核心是把散落在各日期文件夹的人工标注整合成统一的 `text` + `wav.scp` 测试集（即 10/11 阶段反复用到的"周报独立测试集"），并提供一套**唤醒词裁剪流水线**——在物理音频层面切掉"你好萤石"唤醒词，得到去唤醒词的干净测试集（与 11 阶段在文本层面剥离唤醒词形成互补）。

## 目录结构

```
12_test/
  gen_test_text.py          从各日期 label.txt 整合 text + wav.scp
  周报/                      周报独立测试集
    text                     合并后的标注
    wav.scp                  音频路径映射
    trim_wakeword.py         【服务器端】FunASR 时间戳检测唤醒词位置
    trim_wakeword_local.py   【本地端】按检测结果物理裁剪音频
    verify_trim_energy.py    基于能量低谷校验裁剪边界是否合理
    wavs/ wavs_trimmed2/ output/   音频与输出
```

## 唤醒词裁剪流水线（核心）

这是本目录最有特色的设计——**服务器检测 + 本地裁剪 + 能量校验**三段式，把 GPU 推理和音频 IO 解耦：

```
       服务器（有 GPU）              本地（有原始音频）
  ┌──────────────────────┐     ┌──────────────────────────┐
  │ trim_wakeword.py     │     │ trim_wakeword_local.py   │
  │ FunASR 时间戳识别     │ ──→ │ 按时间戳边界物理裁剪音频  │
  │ 检测唤醒词边界        │ trim│ 输出裁剪后 wav + wav.scp │
  │ → trim_log.jsonl     │_log │                          │
  └──────────────────────┘.jsonl└──────────────────────────┘
                                            │
                                            ▼
                              verify_trim_energy.py
                              （能量低谷校验边界合理性）
```

为什么拆两端：GPU 推理在服务器跑，但原始音频量大放在本地，只传一个轻量的 `trim_log.jsonl`（时间戳边界）回本地裁剪，避免大批音频在服务器/本地间来回拷贝。

---

## 1. `gen_test_text.py` — 测试集整合

| 项 | 说明 |
|---|---|
| **输入** | `wavs/<日期范围>/labelled/label/label.txt`（`utt_id text`）+ `labelled/source/*.wav` |
| **输出** | `12_test/text`（合并标注）+ `12_test/wav.scp`（`utt_id wav_path`） |

**逻辑**：遍历 `wavs/` 下所有日期文件夹（排序），读取每个的 `label/label.txt`，逐条收集标注；同时检查 `source/` 下对应 wav 是否存在，存在才写入 wav.scp。最后统计总标注数、匹配音频数、缺失数。路径硬编码，无参数。这是测试集的入口脚本。

## 2. `trim_wakeword.py` — 【服务器端】唤醒词位置检测

| 项 | 说明 |
|---|---|
| **输入** | `--input-scp`（默认 `/data1/12_test/wav.scp`）+ ASR 模型 |
| **输出** | `trim_log.jsonl`（每行 `{uid, wav_path, recognized_text, head_boundary, tail_boundary}`） |
| **关键** | 只检测、不裁剪 |

**逻辑**：
1. 加载 wav.scp，**过滤损坏文件**（0KB / 读不出 / 时长<0.1s）
2. 加载 SeACo-Paraformer（带时间戳输出），用唤醒词做**热词增强**提高识别率
3. 批量 `model.generate` 拿到逐字 `timestamp`
4. `detect_wakeword_boundaries` 用**模糊匹配**定位唤醒词边界：
   - 精确匹配"你好萤石"
   - 前两字"你好"+任意 2 字（覆盖"你好岩石/英石"等识别变体）
   - 开头"好"+模式（"你"被吞掉的情况）
   - 结尾同理检测
   返回开头/结尾唤醒词的 `(start_ms, end_ms)` 时间戳
5. 统计开头/结尾检测率、无唤醒词比例，写出 trim_log.jsonl

## 3. `trim_wakeword_local.py` — 【本地端】物理裁剪音频

| 项 | 说明 |
|---|---|
| **输入** | `trim_log.jsonl` + 原始音频 + 原始 `text` |
| **输出** | `trimmed/wavs/*.wav`（裁剪后音频）+ `trimmed/wav.scp` + `trimmed/text` + `trim_result.jsonl` |

**逻辑**：读 trim_log 的边界，用 `soundfile` 切音频：
- 裁开头：从 `head_boundary[1] + margin` 开始保留（margin 默认 100ms 安全边距，避免切到正文）
- 裁结尾：保留到 `tail_boundary[0] - margin`
- 无边界则直接复制
支持 `--trim-head/--trim-tail` 开关、`--margin` 调边距、`--sample` 采样。统计四类动作（trim_head/tail/both/copy）和总节省时长，生成可直接喂给 `eval_on_testset.py` 的新 wav.scp + text。

## 4. `verify_trim_energy.py` — 能量校验裁剪边界

| 项 | 说明 |
|---|---|
| **输入** | `trim_log.jsonl` + 本地音频目录 |
| **输出** | `verify_result.jsonl` + 统计报告 + 可选能量曲线 PNG |

**原理**：唤醒词和正文之间通常有一个能量低谷（微停顿/换气）。本脚本验证 ASR 检测的 `head_boundary[1]`（唤醒词结束时间）是否真的落在能量低谷处。

**逻辑**：
1. 计算短时能量曲线（帧长 20ms / 帧移 10ms，转 dB）
2. 在 boundary 附近 ±300ms 找能量最低点（valley）
3. 算 valley 与 boundary 的偏差 `offset_ms`，以及 boundary 处的能量下降 `energy_drop_db`
4. 判定：偏差≤容差(150ms) 且 能量下降>3dB → `pass`；偏差稍大 → `suspect`；找不到低谷 → `fail`
5. `--plot` 可生成波形 + 能量曲线 + 边界标注的可视化图（前 20 条）
6. 输出偏差/能量下降的统计分布（均值/中位数/90 分位）

用途：质检 ASR 时间戳检测的可靠性，发现切错位置的样本（比如能量下降不明显说明边界可能不准）。

---

## 几个值得注意的点

1. **两层去唤醒词的互补**：
   - **12_test（物理层）**：直接切掉音频里的唤醒词，得到真正不含唤醒词的测试集——模型听不到唤醒词
   - **11_analyse（文本层）**：保留原音频，只在 CER 计算时从文本里剥离唤醒词——模型仍听到唤醒词，只是评分时不算
   
   两者目的都是消除唤醒词对业务指标的干扰，但物理裁剪更彻底（也能验证模型在无唤醒词输入下的真实表现）。

2. **服务器/本地解耦的工程考量**：GPU 在服务器、音频在本地，只传轻量 trim_log.jsonl，是处理大规模音频数据集时很务实的设计。trim_log.jsonl 也成了两端的标准接口。

3. **模糊匹配应对识别变体**：`detect_wakeword_boundaries` 不死磕"你好萤石"四字精确匹配，而是用"你好"+2字、"好"+模式等启发式——因为模型本身就常把唤醒词识别成"你好岩石"等，精确匹配会漏掉大量样本。这与 11_analyse 的唤醒词变体列表是同一问题的两种应对。

4. **能量校验是质检兜底**：纯靠 ASR 时间戳可能切错（模型时间戳不准、唤醒词与正文连读无停顿），用独立的能量低谷检测交叉验证，是对裁剪质量的二次把关。

5. **路径硬编码 + Windows/Linux 混用**：`gen_test_text.py`/`trim_wakeword_local.py`/`verify_trim_energy.py` 是 Windows 本地路径（`E:\...`），`trim_wakeword.py` 是 Linux 服务器路径（`/data1/...`），与服务器检测+本地裁剪的工作流一致。

6. **产出对应 steering 规范**：本目录生成的"周报独立测试集"正是规范里 `testset_results/`（~3335 条独立测试集）的数据来源；裁剪后的去唤醒词版本支撑了规范里"testset 剔除唤醒词 CER"的评估。
