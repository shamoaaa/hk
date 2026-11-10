# S10/SP4 ASR 全流程详解（数据 + 增强 重点）

本文梳理整个项目从**原始音频**到**部署模型**的端到端流水线，重点详述**数据准备**和**数据增强**两个环节。

## 全景流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│  阶段 0: 原始数据                                                       │
│  自有业务音频(S10/SP4) + 公开数据集 + 背景语料(AliMeeting)             │
└─────────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                       ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ 01 数据整理   │    │ 08 公开数据集     │    │ (背景语料直接供   │
│ labeled/     │    │ gen_scp_txt      │    │  07 增强使用)     │
│ unlabeled/   │    │ → speech_label/  │    │                  │
└──────────────┘    └──────────────────┘    └──────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│ 02-05 质量筛查                              │
│ 扫描指标 → 阈值判断 → 分类 → 删除问题音频   │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│ 06 预标注 + 交叉验证（针对 unlabeled）       │
│ FunASR/FireRed/Qwen 三模型打标 → 交叉质检   │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│ 07 数据增强（核心）                          │
│ 干净音频 → 模拟真实声学场景 → augment_label/ │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│ 08 切分 + 汇总                              │
│ train/dev/test 划分 → merge → all_label/   │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│ 09 训练                                     │
│ fast_scp2jsonl → torchrun 微调 SeacoParaformer │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│ 10 评估 → 11 分析 → 12 测试集 → 13 模型瘦身  │
└──────────────────────────────────────────┘
```

三类数据源最终在 `08_split/merge_all_labels.py` 汇入 `all_label/`：
- `reflow_label/` — 自有业务数据（S10/SP4 回流）
- `speech_label/` — 公开数据集（8 个）
- `augment_label/` — 增强数据（07 阶段产出）

---

# 第一部分：数据准备（阶段 01–06、08）

## 阶段 01：原始数据整理 → FunASR 格式

`run_pipeline.py` 的第 1 步，是整条流水线的入口。

### 01_prepare_funasr_data.py（S10）

**输入**：`label/ready/*.txt`（标注）+ `wav/source/` 下海量原始音频

**执行过程**：

1. **建立 wav 索引** `build_wav_index()`：`os.walk` 递归扫描 `wav/source/`，建立 `文件名(去.wav) → 完整路径` 字典。几十万文件，每 5 万条打印进度。
2. **读取所有标签** `load_all_labels()`：遍历 `label/ready/*.txt`，每行 `<id> <text>`。关键一步是 `normalize_label_id()`——去掉 `_agi_stream_data_sequence_数字_end_True/False` 这类流式数据后缀，让标签 ID 能和 wav 文件名对上。同 ID 去重。
3. **匹配** `match_labels_to_wav()`：标签 ID 与 wav 索引求交集，输出匹配成功/未找到音频的统计。
4. **整理输出** `organize_data()`：把匹配上的音频**复制或移动**（`COPY_MODE` 控制）到 `labeled/wav/`，同时写三个文件：
   - `labeled/wav.scp`：`<uid> <绝对路径>`（Kaldi 格式）
   - `labeled/text.txt`：`<uid> <文本>`
   - `labeled/train.jsonl`：FunASR ChatML 格式，每行：
     ```json
     {"messages":[
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":"语音转写：<|startofspeech|>!<wav路径><|endofspeech|>"},
        {"role":"assistant","content":"<文本>"}
     ],"text_length":<字数>}
     ```
5. **分离未标注** `move_unlabeled()`：wav 索引里没匹配上标签的，全部归入 `unlabeled/`（这些后续走 06 预标注）。
6. 为 `unlabeled/` 生成 wav.scp。

**输出**：`labeled/`（带标签训练数据）+ `unlabeled/`（待预标注音频）

### 01_prepare_funasr_data_sp4.py（SP4）

目的相同，但 SP4 数据特点不同：
- SP4 **已有现成的** `text` / `wav.scp`（还有 `_0924_add`、`_20241204_add` 等增量文件），无需从 `label/*.txt` 扫描，直接 `load_text_files()` 合并。
- wav.scp 里是 **Linux 服务器路径**，`remap_wav_path()` 找到 `sp4_data/` 前缀后重映射为本地路径并校验存在性。
- 用 `wav_all.scp`（全量列表）识别未标注音频。

输出结构与 S10 版一致（`labeled/` + `unlabeled/`）。

## 阶段 02–05：音频质量筛查

这是一条"扫描 → 判断 → 分类 → 删除"的子流水线，**扫描和判断刻意解耦**。

### 02_audio_quality_check.py — 只采集，不判断

**输入**：`labeled/wav` + `unlabeled` 下所有 wav

`ProcessPoolExecutor` 多进程，对每个 wav 用标准库 `wave`+`struct` 采集原始指标（不做任何阈值判断）：
- 文件完整性、时长、采样率、声道数、位深
- RMS 能量、峰值
- 削波比例（|采样值|≥32000 的占比）
- 直流偏移（采样均值）
- SNR：帧能量排序法——切 25ms 帧算能量，**最低 20% 帧当噪声、最高 20% 帧当信号**，`10*log10(信号/噪声)`

**输出**：`quality_check/quality_detail.jsonl`（逐条原始数值）

### 03_quality_report.py — 阈值判断 + 报告

**输入**：`quality_detail.jsonl`

对每条记录套用阈值（可命令行覆盖）标记 issues：
- 时长 < 0.3s 或 > 30s
- RMS < 50（静音/低能量）
- 削波比例 > 1%
- |直流偏移| > 500
- SNR < 5dB
- 采样率 ≠ 16000、非单声道、非 16bit

**输出**：
- `quality_report.txt`：统计报告（时长/问题类型/采样率/RMS/SNR/DC 分布分桶带条形图 + 按目录统计）
- `quality_summary.json`、`bad_files.txt`（问题清单，带原因）、`good_files.txt`

> **设计要点**：所有阈值集中在 03，调整后只重跑 03（秒级），无需重新扫描几十万音频（02 很慢）。

### 04_sort_bad_files.py — 按原因归类

读 `bad_files.txt`，把问题音频按原因（`too_short`/`silence_or_low_energy`/`clipping` 等）拷贝到 `bad_sorted/{原因}/` 子目录，一个文件多个问题会出现在多个目录，方便人工抽检。

### 05_remove_bad_files.py — 删除（可选，破坏性）

从 `labeled/`、`unlabeled/` 删除问题音频，并同步清理 `labeled/` 下 text.txt/wav.scp/train.jsonl 的对应条目。支持 `--dry-run` 预览。在 `run_pipeline.py` 里默认跳过，需 `--remove-bad` 显式启用且交互确认。

## 阶段 06：未标注数据预标注 + 交叉验证

针对 `unlabeled/` 的音频，用多个 ASR 模型自动打标，再交叉质检。详见 `06_pretag_and_compare/README_脚本报告.md`，核心流程：

1. **三模型预标注**：FunASR(paraformer)、FireRedASR-LLM、Qwen3-ASR 各自对未标注音频生成标签
2. **交叉验证** `cross_validate_labels.py`：人工 vs FireRed vs Qwen 三方比对，分类——
   - `models_agree`（两模型一致但与人工不同 → 人工大概率错）
   - `all_differ`（三方都不同 → 需复核）等
3. **Web 人工复核** + **导出分歧样本**做精修

产出干净的扩充标签，最终也进入训练数据池。

## 阶段 08：公开数据集格式归一化

`08_split/public_dataset_gen_scp_txt.py` 把 8 个公开数据集各自的转写格式统一成 `wav.scp` + `text`，输出到 `speech_label/{train,dev,test}/`。8 个数据集：aidatatang_200zh、AISHELL-1、AISHELL-1 iOS、MagicData、PrimeWords、ST-CMDS、TAL_ASR、THCHS-30。各自有格式陷阱（分词需去空格、tab 分隔、trn 相对引用等），详见 08 报告。

---

# 第二部分：数据增强（阶段 07，核心）

增强的目标：把**干净的近场录音**模拟成**真实智能音箱在家庭/会议环境收到的声音**，扩充训练集、提升鲁棒性。设计原则是 **"先分析再增强"**——增强参数要基于真实业务数据的声学特征标定，不靠拍脑袋。

## 7.1 增强引擎 AudioAugmentor（augment_engine.py）

所有增强脚本共享的核心类，配置驱动（YAML），按 `chain` 顺序对每条音频执行一串策略。

### 16 种策略

| 策略 | 作用 |
|---|---|
| gaussian_noise | 按 SNR 加高斯白噪 |
| **speaker_mix** | 多说话人混合（核心，下详） |
| reverb | 合成 RIR 混响（直达声+早期反射+晚期尾巴） |
| real_rir | 真实 RIR 卷积 |
| speed_perturb | 变速（重采样） |
| volume_perturb | 音量增益 |
| noise_from_file | 真实环境噪声叠加 |
| pitch_shift | 变调（保时长） |
| eq_random | 随机均衡（频段独立增益） |
| random_crop_pad | 随机裁剪/填充 |
| time_stretch | 变速保调（Phase Vocoder） |
| random_silence | 插入静音 |
| codec_simulate | G.711 编解码失真 |
| bandpass_filter | 带通滤波 |
| clipping | 削波失真 |
| concatenate | 多条短音频拼接 |

### 统一入口 augment_single()

对每条音频，执行流程：
1. 根据 `wav_path` 匹配 `folder_chains` 决定执行链（默认用全局 `chain`）
2. **确定性随机种子** `seed + file_idx*augment_factor + aug_idx`——保证可复现，且每条增强副本不同
3. 按链顺序逐个执行启用的策略
4. 末尾 `np.isfinite` 检查，含 NaN/Inf 则丢弃（返回 None）
5. `normalize_peak` 峰值归一化

### 关键设计：正态采样而非均匀采样

```python
def _sample_normal_clipped(rng, low, high):
    mu = (low + high) / 2.0       # 区间中点为均值
    sigma = (high - low) / 4.0    # 约 95% 落在区间内
    return float(np.clip(rng.normal(mu, sigma), low, high))
```

几乎所有数值参数（SNR、RT60、音量、削波阈值）都用它采样。原因：**真实场景里典型值集中、极端值少见**。均匀分布会让"超强混响""超大衰减"和正常值一样频繁出现，增强音频听起来就假。正态采样让中间值多、极端值少，更真实。

## 7.2 距离驱动的多说话人混合（最核心的设计）

`speaker_mix` 策略模拟"主说话人对音箱说话，周围有背景人在不同距离交谈"。难点在于背景人的各个声学参数要**物理自洽**。

### 问题：独立采样会产生假音频

背景人离麦克风的**距离**是个根因变量，同时决定：
- 距离远 → 声音小（SNR 高）、混响重（RT60 大）、高频衰减（低通截止低）
- 距离近 → 声音大、混响轻、高频保留

如果各参数独立随机采，会采出矛盾组合（如"声音很小但高频很亮"），一听就假。

### 解法：bg_distance_correlated —— 一个距离因子驱动多参数

对**每条**背景人声：

**① 采样距离因子**（正态分布）
```python
t = self._sample_normal_clipped(aug_rng, 0.0, 1.0)   # t∈[0,1], 0=最近 1=最远
```

**② 用同一个 t 线性映射出三个物理相关参数**（注意方向）

| 参数 | 公式 | t=0（近/1米） | t=1（远/4米） | 方向 |
|---|---|---|---|---|
| RT60 混响 | `min + t*(max-min)` | 0.1s 轻混响 | 0.15s 重混响 | 正向↑ |
| 低通截止 | `max - t*(max-min)` | 7800Hz 高频亮 | 6500Hz 高频闷 | **反向↓** |
| SNR | `min + t*(max-min)` | 5dB 背景较响 | 15dB 背景很弱 | 正向↑ |

低通是减号（远→截止频率降低→高频被砍）；SNR 是加号（这里 SNR = 主说话人/背景人声，背景在分母，远→背景小→SNR 高）。

代码片段：
```python
for bp in bg_paths:
    bg_audio_tmp = audio_cache[bp][0].copy()
    t = self._sample_normal_clipped(aug_rng, 0.0, 1.0)

    # 音色：与距离无关，独立采样（不同人的嗓音差异）
    if bg_pitch_shift:
        semitones = aug_rng.uniform(bg_semitone_range[0], bg_semitone_range[1])
        bg_audio_tmp = self.pitch_shift(bg_audio_tmp, sr, semitones)

    # 混响：t 正向
    if bg_reverb:
        bg_rt60 = bg_rt60_range[0] + t * (bg_rt60_range[1] - bg_rt60_range[0])
        bg_audio_tmp = self.apply_reverb(bg_audio_tmp, sr, bg_rt60)

    # 低通：t 反向
    if has_lp_range:
        lp_freq = bg_lowpass_range[1] - t * (bg_lowpass_range[1] - bg_lowpass_range[0])
        bg_audio_tmp = self.bandpass_filter(bg_audio_tmp, sr, low_freq=60, high_freq=lp_freq)

    bg_audios.append(bg_audio_tmp)
    bg_snrs.append(snr_range[0] + t * (snr_range[1] - snr_range[0]))   # SNR：t 正向

# 逐个按各自 SNR 混入主音频
for bg_a, bg_s in zip(bg_audios, bg_snrs):
    aug_audio = self.mix_speakers(aug_audio, sr, [bg_a], bg_s, overlap_mode=..., rng=aug_rng)
```

**解耦原则**：距离相关量（响度/混响/高频）由 t 锁死；声源属性（音色 pitch）独立采样。符合声学第一性原理。

### 强制背景人声更小

`mix_speakers` 里 SNR 换算成增益：
```python
target_bg_power = signal_power / (10 ** (snr_db / 10))   # 背景目标功率
gain = np.sqrt(target_bg_power / bg_power)
mixed = mixed + bg_placed * gain
```
配置 `snr_range: [5, 15]` 全为正值 → 主说话人始终比背景响 5~15dB，不会出现背景盖过主说话人的假场景。

### 时间对齐 6 模式

`mix_speakers` 支持背景声在时间轴上的不同出现方式：`full`（全程）、`random`（随机位置保证 ≥80% 重叠）、`mid_insert`（说一半插入）、`bg_first`（背景先出现）、`bg_tail`（说完后出现）、`bg_longer`（背景更长），`auto` 每次随机选一种（推荐，最大多样性）。

## 7.3 业务场景标定的参数（augment_config_speaker_multi.yaml）

参数范围都按智能音箱实际使用场景标注了对应米数：

```yaml
speaker_mix:
  bg_distance_correlated: true     # 开启距离关联
  snr_range: [5, 15]               # 背景 1-4米
  bg_rt60_range: [0.1, 0.15]       # 1米(0.15s)~4米(0.6s)
  bg_lowpass_range: [6500, 7800]   # 4米(5.5kHz)~1米(7.8kHz)
  bg_pitch_shift: true
  bg_semitone_range: [-2, 2]       # ±2半音，男女老幼音色
  overlap_mode: "auto"

reverb:                            # 主说话人房间混响
  rt60_range: [0.08, 0.1]          # 主说话人 0.5-2.5米
volume_perturb:
  gain_range: [-14, 0]             # 0.5米≈0dB, 2.5米≈-14dB
gaussian_noise:
  snr_range: [35, 40]              # 家庭底噪很轻微

chain:                             # 链顺序刻意安排
  - reverb         # 1. 主说话人房间混响（在混背景之前，避免双重混响）
  - speaker_mix    # 2. 混入背景说话人（各自独立距离/音色/音量）
  - speed_perturb  # 3. 语速多样性
  - eq_random      # 4. 音箱麦克风频响
  - volume_perturb # 5. 主说话人距离/音量
  - gaussian_noise # 6. 家庭底噪
  - clipping       # 7. 偶发过载
```

链顺序遵循真实信号传播路径，且 `reverb` 在 `speaker_mix` 之前避免双重混响。

## 7.4 三种增强调度器

| 脚本 | 数据来源 | 内存策略 | 适用 |
|---|---|---|---|
| **augment_dir.py** | 目录扫描（无标签） | 全量预加载 | 小数据集 |
| **augment_manifest.py** | manifest.jsonl（带标签） | LRU 缓存(2000)懒加载，**按 split 隔离** | 大数据集 |
| **batch_augment_public.py** | 公开数据集标签目录 | 多进程 + 单任务缓存 + progress.jsonl 断点 | 多个大型公开数据集 |

### batch_augment_public.py 的核心约束

1. 每个数据集只**采样 10%** 作为主说话人
2. 背景说话人池 = **同一 split 所有数据集合并后再采样 10%**
3. **split 完全隔离**：train 的主+背景都来自 train，dev/test 同理，避免数据泄漏
4. 多进程 worker 各自创建 `AudioAugmentor`，背景池作为只读 `initargs` 共享

### 7.5 专项场景：短音频嵌入会议背景（augment_short_in_meeting.py）

独立场景脚本（不复用 engine），把 ≤2s 短指令嵌入 AliMeeting 远场会议背景：
- `MeetingBgProvider` 懒加载 + 持久化索引 + 活跃段优先（按 RMS 选含语音的片段）
- 主说话人筛选：≤2s + RMS≥0.02 + 削波率≤5%
- 混合：随机 SNR（正值保证主突出）+ 独立 `main_gain_db`/`bg_gain_db`
- 归一化 `mode="main_region"`：以主说话人区域峰值归一化，**不抬高背景**，保证主说话人始终清晰

## 7.6 增强的"先分析再增强"闭环（设计中）

`design.md` 规划了完整的四阶段闭环：
1. **Analyze** `analyze_real_data.py` → 从真实数据提取声学基线 `acoustic_baseline.json`
2. **Recommend** `recommend_params.py` → 据基线自动推荐 YAML 参数
3. **Augment** → 带场景多样性的增强（房间预设/说话场景/距离控制）
4. **Verify** `compare_distributions.py` → 增强后分布 vs 真实分布对比 + A/B 试听

> 注：当前 `07_augment_used/` 已落地引擎+三调度器+会议场景脚本；analyze/recommend/verify 这套验证工具链仍是 design.md 的规划，尚未实现成代码。

---

# 第三部分：切分、训练、评估（阶段 08–13）

## 阶段 08：切分 + 汇总

1. **自有数据切分**：`s10_split_train_dev_test.py` / `sp4_split_train_dev_test.py` 把 reflow 数据按 90/5/5 切成 train/dev/test，输出到 `reflow_label/`
2. **汇总** `merge_all_labels.py`：把 `reflow_label/` + `speech_label/` + `augment_label/` 三类按 split 拼接到 `all_label/`，按 key 去重 + 文件存在性校验（用预扫描 set 加速，220w 条秒级）

最终 `all_label/{train,dev,test}/wav.scp + text` 即训练用数据。

## 阶段 09：训练

1. `fast_scp2jsonl.py`：把 `all_label` 的 scp+text 转成 train.jsonl（**只读 WAV 头算时长，比官方 scp2jsonl 快 100~500 倍**）
2. `finetune_fast.sh`：torchrun 8 卡 DDP 微调 SeacoParaformer，从预训练权重出发，冻结热词分支，每 4000 步存 checkpoint（`model.pt.epN`/`best`/`avgN`）

## 阶段 10–13

- **10 评估**：`batch_eval_models.py` 等对各 checkpoint 在 19 个数据集 + 业务全量上算 CER/SER，与 seaco/online 基线对比
- **11 分析**：指令召回率、唤醒词剥离后 CER、跨版本错误对比、儿童语音退化归因
- **12 测试集**：构建独立测试集 + 唤醒词物理裁剪（服务器检测 + 本地裁剪 + 能量校验）
- **13 瘦身**：剥离训练状态，导出部署用精简 `model.pt`

---

# 附：数据流转的目录全景

```
原始音频 + 标签
    │ 01
    ▼
labeled/ (wav + scp + text + jsonl)   unlabeled/
    │ 02-05 质量筛查                       │ 06 预标注+交叉验证
    ▼                                     ▼
干净的带标签业务数据  ◄───────────── 扩充的标签数据
    │ 08 切分
    ▼
reflow_label/{train,dev,test}
    │
    ├── 07 增强 ──► augment_label/{train,dev,test}
    │
公开数据集 ─ 08 gen_scp ─► speech_label/{train,dev,test}
    │                          │
    └──────────┬───────────────┘
               │ 08 merge_all_labels
               ▼
        all_label/{train,dev,test}/wav.scp + text
               │ 09 fast_scp2jsonl + finetune
               ▼
        微调模型 model.pt.epN
               │ 10 评估 → 11 分析 → 12 测试 → 13 瘦身
               ▼
        部署模型 model.pt
```

## 关键设计思想小结

1. **数据准备**：标签与音频用规范化 ID 匹配；带标签走训练、无标签走预标注；质量筛查"扫描与判断解耦"便于调参。
2. **数据增强**：核心是**距离驱动多参数关联**——一个正态分布的距离因子统一驱动背景人的 SNR/混响/高频，保证物理自洽不失真；参数范围按业务场景（1-4 米）标定；强制背景人声小于主说话人。
3. **数据隔离**：train/dev/test 严格隔离（增强和切分都不跨 split），避免数据泄漏。
4. **可复现**：增强用确定性种子，切分用固定 seed=42。
