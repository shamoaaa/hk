# 06_pretag_and_compare 目录脚本报告

本目录实现了一套完整的 **ASR 多模型预标注 → 评估 → 交叉验证 → 人工复核 → 导出** 流水线。
核心思路：用三个不同的 ASR 模型（FunASR、FireRedASR、Qwen3-ASR）分别给音频打标签，再通过 CER 比对找出分歧样本，重点定位人工标注可能出错的地方。

## 整体流水线

```
                 ┌─ FunASR_pretag_unlabeled.py ─┐
   未标注音频 ───┼─ FireRedASR_pretag.py ───────┼──→ 各自 text.txt / pretag.jsonl
                 └─ QwenASR_pretag.py ──────────┘
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                 ▼
    compare_pretag.py   cross_validate_     FunASR_eval_labeled.py
    (两两比对)          labels.py           (对已标注数据评估 CER)
                        (人工+两模型三方比对)
                              │
                              ▼
                   validate_detail.jsonl
                        │           │
                        ▼           ▼
       cross_validate_review_     export_labeled_check.py
       server.py (Web 人工复核)   (导出分歧样本+三方标签)

公共依赖: pretag_utils.py (扫描/进度/写入/速度监控/GPU 选择)
```

---

## 一、预标注脚本

### 1. `pretag_utils.py` — 公共工具模块

其他预标注脚本的共享底座，本身不可独立运行。

**提供的能力：**

- **日志**：`setup_logging()` 同时输出到控制台和带时间戳的日志文件；`format_duration()` 秒转时分秒。
- **WAV 扫描**：`get_wav_duration()` 优先用 `soundfile` 读时长，失败回退手动解析 44 字节 WAV 头部；`scan_wav_files()` 用 8 线程并发扫描目录，结果缓存到 `wav_scan_cache.json`，返回 `[(uid, path, duration)]`。
- **进度管理**：`load_progress()` 从 `progress.txt` 读取已完成 uid 集合，支持断点续跑。
- **结果写入器 `PretagWriter`**：封装 4 个文件（`pretag.jsonl` / `wav.scp` / `text.txt` / `progress.txt`）的同步写入。jsonl 为 ChatML 格式（system/user/assistant，user 内容含 `<|startofspeech|>!路径<|endofspeech|>`）。内置挂起检测：批次间隔超过 120s 视为系统休眠，从耗时统计里扣除。
- **速度监控 `SpeedMonitor`**：用"每秒音频的处理耗时"做归一化指标，前 5 批建立基线，性能跌到基线 60% 判定为慢批次，连续慢批次达阈值时触发模型重启（解决长跑显存泄漏/降速）。
- **GPU 选择 `select_gpus()`**：支持 `cuda:0`、`cuda:0,1`、`cuda:auto`（自动选最空闲单卡）、`cuda:auto:N`（选最空闲 N 卡）、`cpu`。通过 `nvidia-smi` 查询空闲显存，失败回退 `torch.cuda`，并设置 `CUDA_VISIBLE_DEVICES`。
- **流程封装**：`prepare_pretag()` 统一"扫描+断点续跑"；`run_inference_loop()` 统一推理主循环，封装 writer 生命周期、重启逻辑、超长音频额外处理（`extra_items`）、汇总打印。

### 2. `FunASR_pretag_unlabeled.py` — FunASR 预标注未标注数据

| 项 | 说明 |
|---|---|
| **输入** | `unlabeled/` 目录下的 `*.wav`（注意：此脚本未用 `pretag_utils`，自带扫描逻辑，仅扫描顶层目录） |
| **输出** | `pretag_output/` 下的 `pretag.jsonl`、`wav.scp`、`text.txt`、`progress.txt`、日志 |
| **模型** | `paraformer-zh`（默认，带 `ct-punc` 标点模型）或 `iic/SenseVoiceSmall`，均配 `fsmn-vad` |

**逻辑**：扫描 wav → 断点续跑过滤 → 加载 FunASR `AutoModel` → 按 `file-batch`（默认 500 个文件/批）分批 `model.generate`，动态 batch 时长 `batch_size`（默认 300s）→ SenseVoice 结果走 `rich_transcription_postprocess` 后处理 → 写四文件 → 实时打印进度/速度/ETA。批次失败则整批跳过计入 failed。

**参数**：`--model` `--device`（默认 `cuda:0`）`--batch-size` `--file-batch` `--resume`。

### 3. `FireRedASR_pretag.py` — FireRedASR-LLM 预标注

| 项 | 说明 |
|---|---|
| **输入** | `--wav-dir`（默认 `/data/share/dataset/s10_data_reflow/unlabeled`）下的 wav |
| **输出** | `firered_pretag_output/` 下的 `pretag.jsonl`/`wav.scp`/`text.txt`/`progress.txt`/日志 |
| **模型** | `FireRedASR-LLM-L`（8.3B，旗舰，≤30s 音频）或 `FireRedASR-AED-L`（1.1B，≤60s） |

**逻辑**（基于 `pretag_utils` 的通用流程）：

- `create_duration_buckets()` 按 5s 宽度**分桶排序**——这是 FireRedASR-LLM 的关键约束，要求同批次音频时长相近，否则短音频会出现重复输出；超过时长上限的音频单独拎出来逐条处理。
- `get_decode_params()` 按模型类型给不同解码参数（llm: beam=3、repetition_penalty=3.0；aed: beam=3、softmax_smoothing=1.25 等）。
- `process_batches()` 批量 `model.transcribe`，**整批失败时回退到逐条推理**，最大限度保住结果。
- 通过 `run_inference_loop` 接入速度监控与自动重启；`reload_fn` 释放显存后重新加载模型；超长音频用 `process_extra_fn` 逐条补处理。

**参数**：`--asr-type llm/aed`、`--device cuda:auto`、`--batch-size`（LLM 建议 1-4）、`--resume`、`--no-cache`、`--max-slow-batches`。

### 4. `QwenASR_pretag.py` — Qwen3-ASR 预标注

| 项 | 说明 |
|---|---|
| **输入** | `--wav-dir`（默认 `/data1/s10/wavs`），**递归**扫描，支持按 `labelled`/`unlabelled` 子目录过滤 |
| **输出** | `qwen_pretag_output/` 下的 `pretag.jsonl`/`wav.scp`/`text.txt`/`progress.txt`/日志 |
| **模型** | `Qwen3-ASR-1.7B`（默认，开源 SOTA，52 语种）或 0.6B；三种后端 |

**逻辑**：

- `scan_wav_files_recursive()` 递归扫描，按目录路径里的 `/labelled/` `/unlabelled/` 做过滤，缓存文件名按过滤类型区分。
- `load_model()` 支持三种后端：`transformers`（`Qwen3ASRModel.from_pretrained`）、`vllm`（加速）、`raw`（直接用 `AutoModelForCausalLM`+`AutoProcessor`，不依赖 qwen-asr 包，逐条用 librosa 加载音频+chat template 推理）。
- `transcribe_batch()` 统一封装三种后端的推理接口；`process_batches()` 同样有整批失败→逐条回退。
- 支持 `--sample N` 随机采样快速验证（seed=42）、`--language` 强制语种、`--max-new-tokens`。同样接入速度监控/自动重启（默认 `max-slow-batches=9`）。

**参数**：`--wav-dir` `--backend transformers/vllm/raw` `--model` `--batch-size`（默认 128）`--filter labelled/unlabelled` `--sample` `--language` `--device cuda:auto` `--resume` `--no-cache`。

---

## 二、评估脚本

### 5. `FunASR_eval_labeled.py` — 对已标注数据评估

| 项 | 说明 |
|---|---|
| **输入** | `labeled/text.txt`（人工标签 `uid text`）+ `labeled/wav/*.wav` |
| **输出** | `eval_output/` 下的 `eval_detail.jsonl`（逐条）、`eval_report.txt`（报告）、`eval_summary.json`、`eval_progress.txt`、日志 |

**逻辑**：加载人工标签 → 匹配存在的 wav（统计缺失）→ 可选 `--sample N` 随机采样 → 加载 FunASR 模型 → 分批推理得到 hyp → 文本规范化（去标点、转小写、去空格）后用**自实现的编辑距离**算 CER（区分替换/插入/删除）。

**关键指标**：

- **CER**（字符错误率）= (Sub+Ins+Del)/参考字符数，整体累计统计
- **SER**（句错率）= 1 − 完全正确句数/总句数
- **CER 分布分桶**（0% / 0-5% / 5-10% / 10-20% / 20-50% / >50%）带条形图
- **Bad Cases**：CER>30% 的条目按降序列出（报告里展示前 30）

`generate_report()` 可独立从已有 `eval_detail.jsonl` 重新生成报告（断点续跑全部完成时直接调用）。

**参数**：`--model paraformer/sensevoice`、`--device`、`--batch-size`、`--file-batch`、`--sample`、`--resume`。

---

## 三、比对与交叉验证脚本

### 6. `compare_pretag.py` — 两模型标签两两比对

| 项 | 说明 |
|---|---|
| **输入** | `--a` 和 `--b` 两份标签文件（支持 `text.txt` 或 ChatML `jsonl`，`load_labels` 自动识别格式） |
| **输出** | `compare_output/` 下的 `compare_detail.jsonl`（逐条）、`diff_list.txt`（超阈值差异清单）、`compare_report.txt`、`compare_summary.json` |

**逻辑**：按 uid 求交集，统计共同/仅 A/仅 B → 逐条规范化后算 CER → 完全一致 vs 有差异 vs 超阈值差异（`--threshold` 默认 0.1，即 CER≥10% 列入差异清单）→ CER 分布分桶 + 平均 CER → 报告展示差异最大的前 30 条。

**用途**：例如比对 FireRedASR vs Qwen 谁和谁更接近。

### 7. `cross_validate_labels.py` — 三方交叉验证（核心）

| 项 | 说明 |
|---|---|
| **输入** | `--human`（人工 `labeled/text.txt`）、`--firered`、`--qwen` 三份标签 |
| **输出** | `cross_validate_output/` 下的 `validate_detail.jsonl`（逐条，含分类）、按类别的 `models_agree.txt`/`all_differ.txt`/`human_firered.txt`/`human_qwen.txt`、`validate_report.txt`、`validate_summary.json` |

**核心判定逻辑**（`classify_disagreement`，阈值 `--threshold` 默认 0.05）：

| 类别 | 含义 | 结论 |
|---|---|---|
| `all_agree` | 三方一致 | 无需复核 |
| `models_agree` | 两模型一致但与人工不同 | **人工大概率错了**（最有价值的发现） |
| `human_firered` | 人工=FireRed，Qwen 不同 | Qwen 可能错 |
| `human_qwen` | 人工=Qwen，FireRed 不同 | FireRed 可能错 |
| `all_differ` | 三方都不同 | 需人工复核 |

报告重点展示 `models_agree`（前 30）和 `all_differ`（前 20），帮助快速修正人工标注错误。`load_labels` 对只有 uid 没文本的行记为空字符串（处理空音频）。

### 8. `cross_validate_review_server.py` — Web 人工复核工具

| 项 | 说明 |
|---|---|
| **输入** | `--detail validate_detail.jsonl` + `--wav-dir`（音频目录）；可选 `--category` 只复核某类 |
| **输出** | `review_decisions.jsonl`（追加写，每条 `{uid, choice, text}`） |
| **依赖** | Flask（`pip install flask`），默认端口 5001 |

**逻辑**：基于 Flask 的单页 Web 应用。加载分歧条目（默认排除 `all_agree`），按最大 CER 降序排列，浏览器里播放音频、并排展示人工/FireRed/Qwen 三方标签和 CER。

**复核选项**：`human`/`firered`/`qwen`（采用对应标签）、`edit`（手动编辑）、`bad`（信号差剔除）。支持键盘快捷键（1/2/3/E/D/←/→/S），自动跳到下一个未复核条目，进度持久化可续。选 Qwen 时自动去除末尾标点。

**接口**：`/api/item`（取条目）、`/api/audio`（播放 wav）、`/api/decide`（提交决策）、`/api/next_unreviewed`。

### 9. `export_labeled_check.py` — 导出分歧样本

| 项 | 说明 |
|---|---|
| **输入** | `--detail validate_detail.jsonl` + `--wav-dir`（源 wav）；可选 `--category` |
| **输出** | `labeled_check/` 目录（见下） |

**逻辑**：筛选所有非 `all_agree` 的分歧条目 → 拷贝对应 wav 到 `labeled_check/wav/`（已存在则跳过，统计缺失，可用 `--no-copy-wav` 跳过拷贝）→ 为人工/FireRed/Qwen 三方各生成 `*_text.txt`（`uid text`）和 `*_train.jsonl`（ChatML 格式）→ 生成 `wav.scp` 和完整的 `validate_detail.json`。

**输出结构**：

```
labeled_check/
  wav/                 分歧样本音频
  wav.scp              Kaldi 格式路径（用 --wav-prefix 指定 Linux 路径前缀）
  human_text.txt / firered_text.txt / qwen_text.txt
  human_train.jsonl / firered_train.jsonl / qwen_train.jsonl
  validate_detail.json
```

Qwen 标签会去除末尾标点（`strip_trailing_punct`）。jsonl 中 wav 路径用 `--wav-prefix`（默认 `/data1/sp4_data/labeled_check/wav`）拼接，方便迁移到 Linux 训练机。

---

## 几个值得注意的点

1. **`BASE_DIR` 路径不统一**：脚本里硬编码的基础路径各异（有 `E:\8.音频\03 s10\s10_data_reflow`、`E:\...\sp4_data`、`/data/share/...`、`/data1/s10/`），看起来是 Windows 本地分析 + Linux 服务器推理混用，实际运行时基本都靠命令行参数覆盖。

2. **编辑距离/CER 计算重复实现**：`compare_pretag.py`、`cross_validate_labels.py`、`FunASR_eval_labeled.py` 各自实现了一份几乎相同的 `edit_distance`/`compute_cer`/`normalize_text`，可以考虑抽到 `pretag_utils.py` 复用。

3. **设计意图清晰**：三方交叉验证 + "两模型一致则人工可能错"的判定，是整套流程的灵魂——用模型共识反过来质检人工标注，再通过 Web 工具人工裁决，最终导出干净的训练数据。
