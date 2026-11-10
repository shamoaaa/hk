# 背景人声 根目录脚本报告

本报告覆盖项目**根目录下的独立脚本**（不含 06~12 等编号子目录、utils、funasr 等）。这些脚本分两大类：

1. **流水线编号脚本（01~05、13）+ 编排器** — 数据准备 → 质量筛查 → 模型精简的主干流程
2. **临时分析脚本（analyze_* / extract_* / calc_* / check_* / find_*）** — 围绕特定版本评估、儿童语音退化、唤醒词残留等专题的一次性深度分析，消费 `eval_output/` 下的评估结果

---

## 一、数据准备与质量筛查流水线

### `01_prepare_funasr_data.py` — S10 数据整理

| 项 | 说明 |
|---|---|
| **输入** | `label/ready/*.txt`（标签）+ `wav/source/` 下的音频 |
| **输出** | `labeled/`（wav/ + train.jsonl + wav.scp + text.txt）+ `unlabeled/`（无标签音频 + wav.scp） |

**逻辑**：递归建立 wav 文件名→路径索引 → 读所有标签文件（`normalize_label_id` 去掉 `_agi_stream_data_sequence_*` 后缀，去重）→ 按 ID 匹配标签与音频 → 把匹配上的复制/移动到 `labeled/`（生成 ChatML jsonl + Kaldi scp + text），未匹配的音频归入 `unlabeled/`。`COPY_MODE` 控制复制还是移动。是整条流水线的入口。

### `01_prepare_funasr_data_sp4.py` — SP4 数据整理

与上一个同目的，但针对 **SP4 数据集**的差异：
- SP4 已有现成的 `text` / `wav.scp`（含 `_add` 增量文件），无需从 `label/*.txt` 扫描
- wav.scp 里是 **Linux 服务器路径**，`remap_wav_path` 重映射到本地 `sp4_data/` 下并校验存在性
- 用 `wav_all.scp`（全量列表）识别未标注音频
其余输出格式（labeled/ + unlabeled/）一致。

### `02_audio_quality_check.py` — 音频质量扫描（只采集）

| 项 | 说明 |
|---|---|
| **输入** | `labeled/wav` + `unlabeled` 下所有 wav |
| **输出** | `quality_check/quality_detail.jsonl`（逐条原始指标） |

**逻辑**：`ProcessPoolExecutor` 多进程，对每个 wav 用标准库 `wave`+`struct` 采集时长/采样率/声道/位深/RMS/峰值/削波比例/直流偏移/SNR。**只采数值不判断**——阈值判断留给下一步。SNR 用帧能量排序法（最低 20% 帧当噪声，最高 20% 当信号）估算。

### `03_quality_report.py` — 质量报告（阈值判断）

| 项 | 说明 |
|---|---|
| **输入** | `quality_detail.jsonl` |
| **输出** | `quality_report.txt` + `quality_summary.json` + `bad_files.txt` + `good_files.txt` |

**逻辑**：对每条记录套用阈值（时长 0.3~30s、RMS<50 静音、削波>1%、DC 偏移>500、SNR<5dB、采样率≠16k、非单声道、非 16bit）标记 issues → 输出问题/合格清单 → 生成统计报告（时长/问题类型/采样率/声道/RMS/SNR/DC 分布分桶带条形图 + 按目录统计）。**所有阈值集中在此**，调整后只需重跑本脚本不用重新扫描——扫描与判断解耦的设计。

### `04_sort_bad_files.py` — 问题音频按原因归类

| 项 | 说明 |
|---|---|
| **输入** | `quality_check/bad_files.txt` |
| **输出** | `quality_check/bad_sorted/{原因}/` 下的音频副本 |

按每个文件的问题类型（`too_short`、`silence_or_low_energy`、`clipping` 等）拷贝到对应子目录，一个文件有多个问题会出现在多个目录，方便人工抽检各类问题。

### `05_remove_bad_files.py` — 删除问题音频

| 项 | 说明 |
|---|---|
| **输入** | `bad_files.txt` |
| **输出** | 就地删除 wav + 清理 text.txt/wav.scp/train.jsonl 对应行 |

从 `labeled/`、`unlabeled/` 删除问题音频，并同步清理 `labeled/` 下三个索引文件里的对应条目。支持 `--dry-run` 预览。属于破坏性操作，默认先预览。

### `13_slim_model_for_inference.py` — 模型瘦身

| 项 | 说明 |
|---|---|
| **输入** | 训练 checkpoint（如 `model.pt.ep35`） |
| **输出** | 精简后的 `model.pt`（只留模型权重） |

`torch.load` 读 checkpoint，识别并剥离优化器/调度器/epoch/step 等训练状态，只保留 `model_state_dict`（兼容多种 key 命名），重新保存。打印体积缩减比例。用于部署前减小模型体积、加快加载。

### `run_pipeline.py` — 流水线编排器

把 01→02→03→04→05→06(FireRedASR 预标注) 串成一条可控流水线：`subprocess` 顺序调用各脚本。支持 `--start/--end` 指定范围、`--skip` 跳步、`--dry-run` 预览、`--list` 列步骤、`--remove-bad` 启用可选删除步、`--continue-on-error` 忽错继续。删除步（5）默认跳过且需交互确认。

---

## 二、专题分析脚本（消费 eval_output/）

> 这批脚本都是围绕某次具体的模型评估/版本对比临时编写的一次性工具，路径硬编码到 `E:\8.音频\03 s10\eval_output\`，输出多为控制台打印。唤醒词列表和剥离逻辑与 11_analyse 目录一致。

### `check_eval_data.py` — 评估数据完整性检查

快速巡检各版本（v0.1.0~v0.6.0 最佳权重）的 `combined_summary.json`、base 基线（reflow/speech/augment × seaco/online_model）、全量业务结果（s10/sp4/testset）等关键文件是否存在，打印 OK/MISSING。是做分析前的前置体检（对应 steering 规范"确认数据完整性"）。

### `extract_performance.py` — 全版本性能对比表（主力）

| 项 | 说明 |
|---|---|
| **输入** | `base/`、各版本 `combined_summary.json`、`s10_results/sp4_results/comparison_summary.json` |
| **输出** | 控制台三张表 |

生成 steering 规范要求的核心对比表：S10 全量 CER、SP4 全量 CER、**19 个数据集（3 业务 + 8 公开 + 8 增强）的 test CER 横向对比**（8 个模型列：seaco/online/v0.1~v0.6），每行标注最优模型并统计各模型最优数量。缺失数据标 N/A 并汇报。严格基于实际数据，是规范"全数据集 CER 对比表"的实现。

### `extract_best_model_cer.py` — 性能对比表（早期版）

与 `extract_performance.py` 几乎同功能，但 v0.5.0 最佳权重取 `ep23`（`extract_performance` 用 `ep35`），key 映射逻辑更绕、缺失处理较粗。是 `extract_performance.py` 的前身/早期版本。

### `calc_recall_and_stripped_cer.py` — SP4 召回率 + testset 剔唤醒词 CER

| 项 | 说明 |
|---|---|
| **输入** | `sp4_results/`、`testset_results/` 各模型 `eval_detail.jsonl` |
| **输出** | 控制台两张表 |

计算 8 个模型的 **SP4 指令召回率**（剥离唤醒词后 `ref_stripped in hyp_stripped`）和 **独立测试集剔除唤醒词前后的 CER 对比**。对应 steering 规范的"指令召回率计算"和"testset 剔除唤醒词 CER"。

### `find_wakeword_residual.py` — 唤醒词残留样本定位

专门找出 testset 中 v0.5.0/ep35 剔除唤醒词后**仍残留唤醒词**的样本，并列出标签、seaco 识别、v0.5.0 识别，用于排查剥离逻辑没覆盖到的唤醒词变体（即剥离规则的漏网之鱼）。

### `analyze_testset_gap.py` — testset seaco vs v0.5.0 退化归因

剔除唤醒词后逐条对比 seaco 和 v0.5.0/ep35，统计谁更差/更好/持平，按"插入过多/删除过多/替换错误"归类退化原因，分析唤醒词残留数和输出长度差异。用于理解 v0.5.0 在独立测试集上为何 CER 偏高。

### `analyze_child_gap.py` — 儿童语音 ChildMandarin 退化分析

| 项 | 说明 |
|---|---|
| **输入** | `child/ChildMandarin/` 下 seaco 和 v0.5.0 的 `eval_detail.jsonl` |
| **输出** | 控制台多维度报告 |

逐条对比 seaco vs v0.5.0，并从 uid 解析说话人维度（年龄/性别/方言/城市/设备），按**年龄、性别、方言、句长**分组统计 CER 差距，输出错误类型对比和退化最严重的 case。定位微调模型在儿童语音上退化的具体人群。

### `analyze_child_error_patterns.py` — 儿童语音退化场景分类

进一步把 ChildMandarin 上 v0.5.0 退化的样本按**场景**归类：长句截断、唤醒词插入、重复/结巴语音、语气词/拟声词、含英文、模型重复输出（幻觉）、口语化多填充词、一般识别退化。统计各场景的样本数和多出错误数，附典型 case。回答"v0.5.0 在儿童语音上退化在哪类内容"。

### `analyze_audio_distribution.py` — 音频音量/SNR/距离分布

| 项 | 说明 |
|---|---|
| **输入** | 任意 wav 目录（默认 sp4 labeled_check/wav） |
| **输出** | 控制台统计 + `audio_analysis.csv` + `audio_distribution.png` |

多进程扫描 wav，统计 RMS 音量(dBFS)、SNR(dB)，并用 RMS+SNR 的简单加权模型**推测录音距离**（<0.3m/0.3-1m/1-3m/>3m）。输出百分位统计、文本直方图、CSV 明细，以及 matplotlib 四联图（音量分布/SNR 分布/音量-SNR 散点/距离饼图）。用于了解数据集的录音距离/信号质量构成，支撑数据增强的场景设计。

---

## 几个值得注意的点

1. **流水线脚本 vs 分析脚本泾渭分明**：编号脚本（01~05/13）是可复用的工程化主干，有完整参数、日志、断点；而 `analyze_*`/`extract_*` 是为具体版本评估临时写的一次性脚本，路径硬编码、输出到控制台，用完即弃。

2. **`extract_performance.py` 和 `extract_best_model_cer.py` 重复**：两者功能基本相同，区别仅在 v0.5.0 最佳权重（ep35 vs ep23）和实现细节。`extract_performance.py` 是更干净的最终版，`extract_best_model_cer.py` 可视为废弃前身。

3. **质量筛查的解耦设计很好**：`02`（只采集数值）和 `03`（只判断阈值）分离，调整过滤标准时无需重新扫描几十万音频，只重跑轻量的报告脚本即可。这是处理大规模数据集的务实设计。

4. **唤醒词处理逻辑在多处重复**：根目录的 `calc_recall_and_stripped_cer.py`、`find_wakeword_residual.py`、`analyze_testset_gap.py`、`analyze_child_error_patterns.py` 各带一份 `WAKEWORDS` 列表和 `strip_wakewords`，和 11_analyse、12_test 里的实现重复。整个项目的唤醒词剥离逻辑可考虑统一成一个公共模块。

5. **分析脚本紧扣 steering 规范**：`check_eval_data`（数据完整性）、`extract_performance`（19 数据集对比表）、`calc_recall_and_stripped_cer`（召回率 + 剔唤醒词 CER）直接对应 `model-eval-analysis.md` 规范的分析流程和历史基准；儿童语音和 testset 退化归因脚本则是针对规范中记录的"v0.5.0 在儿童/testset 上退化"现象的深挖工具。

6. **`run_pipeline.py` 引用的步骤 6 路径已过时**：它指向 `06_eval_and_pretag/FireRedASR_pretag.py`，但实际目录已改名为 `06_pretag_and_compare/`，这条编排会失效（同样的旧路径也出现在 07 的 `07_SeACo_Paraformer_eval.py` 里）。
