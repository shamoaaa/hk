# 07_augment_used 目录脚本报告

本目录实现 ASR 训练的**音频数据增强**模块，将干净的近场录音模拟成各类真实声学场景（智能音箱、家庭多人、会议远场等），用于扩充训练集、提升模型鲁棒性。

整体由三部分组成：

1. **共享引擎**：`augment_engine.py` 提供 16 种增强策略，按 YAML 配置组成增强链。
2. **顶层调度器**：`augment_dir.py`（无标签按目录）、`augment_manifest.py`（带标签按 manifest，split 隔离）。
3. **场景化扩展**：
   - `s10数据集增强/` — 短音频嵌入 AliMeeting 远场会议背景
   - `公开数据集增强/` — 多进程批量增强多个公开数据集

```
                   ┌──────────────────────────────┐
                   │  augment_engine.py           │
                   │  AudioAugmentor (16 策略)    │
                   │  augment_single() 入口       │
                   └──────────────┬───────────────┘
                                  │ import
       ┌──────────────────────────┼─────────────────────────────────────────┐
       ▼                          ▼                                         ▼
augment_dir.py            augment_manifest.py            公开数据集增强/batch_augment_public.py
(目录扫描，无标签)        (manifest，按 split 隔离)      (多进程，按 split/数据集隔离 + 10% 采样)

                                  ▼ 独立场景脚本
                       s10数据集增强/augment_short_in_meeting.py
                       (短音频嵌入 AliMeeting 远场背景，自带混合逻辑)
```

---

## 一、共享引擎

### 1. `augment_engine.py` — 数据增强核心引擎

| 项 | 说明 |
|---|---|
| **角色** | 类库，被其他脚本 import；本身不可独立运行 |
| **核心类** | `AudioAugmentor(config_path)` |

**配置驱动**：从 YAML 读取 `seed`、`num_workers`、`augment_factor`、`strategies`（每个策略的开关和参数）、`chain`（执行顺序）、`folder_chains`（按路径匹配不同的链）、`resume`。

**16 种增强策略**（每个对应一个独立方法）：

| 策略 | 方法 | 说明 |
|---|---|---|
| `gaussian_noise` | `add_gaussian_noise` | 按 SNR 加高斯白噪声 |
| `speaker_mix` | `mix_speakers` | 多说话人混合（核心，下文详述） |
| `reverb` | `apply_reverb` | 合成 RIR 混响：直达声脉冲 + 5–50ms 稀疏早期反射（3–8 个）+ 50ms 后指数衰减晚期尾巴 |
| `real_rir` | `apply_real_rir` | 真实 RIR 卷积（FFT），需要外部 RIR 目录 |
| `speed_perturb` | `speed_perturb` | 速度扰动（重采样改长度） |
| `volume_perturb` | `volume_perturb` | 音量扰动（dB 增益） |
| `noise_from_file` | `add_noise_from_file` | 真实环境噪声叠加（MUSAN/DEMAND） |
| `pitch_shift` | `pitch_shift` | 两次重采样实现保时长改音调 |
| `eq_random` | `eq_random` | 按频段切分 + 各段独立增益（`butter` 带通 SOS） |
| `random_crop_pad` | `random_crop_pad` | 随机裁剪/填充（zero 或 noise 填充） |
| `time_stretch` | `time_stretch` | Phase Vocoder（FFT/IFFT + 相位累积），保音调改时长 |
| `random_silence` | `random_silence_insert` | 在随机位置插入静音段 |
| `codec_simulate` | `codec_simulate` | G.711 μ-law / A-law 量化失真 |
| `bandpass_filter` | `bandpass_filter` | 带通滤波（默认 300–3400Hz 模拟电话信道） |
| `clipping` | `apply_clipping` | 削波失真（按 peak 比例硬限幅） |
| `concatenate` | `concatenate_audios` | 多条短音频拼接（中间随机插静音） |

**`mix_speakers` 时间对齐 6 种模式**（多说话人场景的关键）：
- `full`：背景声循环填充全程覆盖
- `random`：随机位置插入，保证 `min_overlap`（默认 0.8）
- `mid_insert`：主音频说一半时插入背景
- `bg_first`：先出现背景，主音频稍后加入
- `bg_tail`：主音频说完后才出现背景
- `bg_longer`：背景比主音频长，从主音频之前就开始
- `auto`：每次随机选一种（推荐，最大多样性）

**`augment_single()` 统一入口**：

- 接受 `audio, sr, wav_path, aug_idx, file_idx, audio_cache, candidate_paths`，根据 `wav_path` 匹配 `folder_chains` 决定执行链
- **确定性随机种子** `seed + file_idx * augment_factor + aug_idx`，保证可复现且每条增强不同
- 数值采样普遍用 `_sample_normal_clipped`（正态分布 mu=(low+high)/2、sigma=(high-low)/4，再 clip 到区间），比均匀分布更贴近真实场景的"典型值集中、极端值少见"
- `speaker_mix` 中支持 `bg_distance_correlated`：用同一个距离因子 `t∈[0,1]` 同时驱动背景人 SNR、RT60、低通频率，物理上一致（远 → 弱+混响重+高频衰减）
- 末尾用 `np.isfinite` 检查 NaN/Inf，无效返回 `None`；最后 `normalize_peak` 峰值归一化

### 2. `augment_config_speaker.yaml` — 单说话人智能音箱场景配置

定义一条声学传播链：`speed_perturb → pitch_shift → reverb → eq_random → volume_perturb → gaussian_noise → clipping`。`augment_factor=3`，每条原始音频生成 3 条。关闭了 `speaker_mix`（单说话人场景）。

家庭客厅/卧室典型参数：RT60=0.2–0.7s（房间）、`gain_range=[-10, 3]`（偏衰减，远距离更常见）、`snr_range=[15, 35]`（卧室到开空调客厅）、`clipping=[0.6, 0.95]`（轻微削波）。

---

## 二、顶层调度器

### 3. `augment_dir.py` — 基于目录的数据增强（无标签）

| 项 | 说明 |
|---|---|
| **输入** | `--input_dir` 下递归扫描的所有 `*.wav`（无标签） |
| **输出** | `--output_dir/<stem>_aug<i>.wav`（每个原始文件生成 `augment_factor` 个副本） |
| **配置** | `--config` YAML，可用 `--factor` 覆盖增强倍数 |

**逻辑**：
1. `Path.rglob("*.wav")` 递归扫描
2. **预加载所有音频到内存**（`audio_cache`）以便 `speaker_mix`/`concatenate` 用——注意：内存敏感
3. 断点续跑：`augmentor.resume=true` 时扫描输出目录，跳过已存在的 `*_augN.wav`
4. 对每条音频循环 `augment_factor` 次，调 `augment_single()` → 写入
5. 每完成 ~5% 打印一次进度（`done/total` + 速度 + ETA）

### 4. `augment_manifest.py` — 基于 manifest 的增强（带标签）

| 项 | 说明 |
|---|---|
| **输入** | `--input_manifest` JSONL，每行 `{key, wav, txt}` |
| **输出** | `--output_dir/{split}/{key}_aug{i}.wav` + `wav_{split}.scp` + `text_{split}.txt` |
| **关键参数** | `--splits train,dev,test` 指定哪些 split 参与增强（其余只输出原始） |

**逻辑**：

1. 读 manifest，用 `augmentor._detect_split()` 从路径里抓 `train/dev/test` 划分
2. 对每个 split 分别处理（**split 严格隔离**，不会跨 split 用音频混合）
3. 用 `LRUAudioCache(maxsize=2000)` **懒加载**音频（OrderedDict 实现 LRU），避免 OOM；先逐文件 `register()` 验证可读
4. 跳过 `--splits` 指定之外的 split（仅复制原始 manifest 条目）
5. 断点续跑：扫描已有 `*_augN.wav`，跳过的同时把对应条目仍写入新 manifest
6. 写出 `wav_{split}.scp`（`key path`）和 `text_{split}.txt`（`key text`）——按 split 分开输出，方便后续训练分别加载

---

## 三、场景化扩展

### 5. `公开数据集增强/augment_config_speaker_multi.yaml` — 多说话人场景配置

为多人家庭场景设计，启用 `speaker_mix` 并精细控制：

- `max_speakers=1`（实际是 1 个背景说话人，名字保留为 max_speakers）
- `overlap_mode="auto"`（每次随机选时间对齐模式）
- `bg_distance_correlated=true`：SNR/RT60/lowpass 由同一距离因子驱动
- `bg_rt60_range=[0.1, 0.15]`（背景 1m 近 → 4m 远）
- `bg_lowpass_range=[6500, 7800]`（远距离高频衰减）
- `bg_pitch_shift=true ±2 半音`（覆盖男女老幼音色）

链顺序刻意把 `reverb` 放在 `speaker_mix` **之前**，避免双重混响。

### 6. `公开数据集增强/batch_augment_public.py` — 多数据集多进程批量增强

| 项 | 说明 |
|---|---|
| **输入** | `--label-root`（默认 `/data1/code/label`，约定子目录 `train/dev/test/`，含 `<dataset>_wav.scp` + `<dataset>_text`） |
| **输出** | `--output-root/{split}/{dataset}/<key>_aug<i>.wav` + `{split}/<dataset>_wav.scp` + `{split}/<dataset>_text` + 每数据集的 `progress.jsonl` |

**核心约束**：
1. 每个数据集只**采样 10%**（`--sample-ratio 0.1`）作为主说话人
2. 背景说话人池来自**同一 split 所有数据集合并后再采样 10%**
3. **split 完全隔离**：train 主说话人和背景都来自 train，dev 来自 dev，test 来自 test

**多进程并行**：

- `mp.Pool` 的 worker 通过 `_worker_init` 各自创建 `AudioAugmentor`，并接收只读的 `bg_pool`（背景路径列表）作为 `initargs`
- 任务粒度：单条音频 + 该条目的所有 `aug_factor` 副本
- worker 内用 `_SimpleAudioCache`（无界，单任务内）按需加载主音频和背景
- `pool.imap_unordered` 流式收集结果

**断点机制（progress.jsonl）**：
- 每条增强结果写入 `progress.jsonl`（追加模式，进程安全）：`{key, wav, txt, status}`
- 重跑时 `_load_progress` 加载已完成 key，**完整覆盖该原始条目所有副本**才跳过；否则提交剩余 `aug_idx` 任务
- 最终从 progress + 新增结果重建 `*_wav.scp` 和 `*_text`

`discover_datasets` 自动扫描 `<split>/<dataset>_wav.scp` 配对 `<dataset>_text`，无需手写数据集列表。

### 7. `s10数据集增强/augment_config_meeting_bg.yaml` — 会议背景嵌入场景配置

针对短音频嵌入 AliMeeting 远场会议背景的场景，定义所有参数（不复用 `augment_engine`）：

- 主说话人筛选：`max_duration=2.0s`、`min_rms=0.02`（排除静音）、`max_clip_ratio=0.05`（排除严重削波）
- 背景：`crop_min/max=5/7s`、`prefer_active=true`（优先含活跃语音段，候选数 5）
- 混合：`snr_min/max=5/20 dB`、`main_gain_db=-6`、`bg_gain_db=-20`（独立增益）
- 归一化：`mode="main_region"`（以主说话人区域峰值归一化，不抬背景）、`target_peak=0.05`
- 后处理（默认关）：`reverb`、`gaussian_noise`

### 8. `s10数据集增强/augment_short_in_meeting.py` — 短音频嵌入会议背景增强

| 项 | 说明 |
|---|---|
| **输入** | `--main_dir`（≤2s 短音频）+ `--bg_dir`（AliMeeting 远场，8 声道，14–42 分钟/条）+ 可选 `text_file` `scp_file` |
| **输出** | `output_dir/wav/<stem>_meetbg<i>.wav` + 可选 `text` 和 `wav.scp` |
| **配置** | `--config` YAML，CLI 参数可覆盖 |

**`MeetingBgProvider` 背景管理器**：

- **持久化索引** `.bg_index.json`（含目录指纹 MD5），首次扫描后缓存每个 wav 的 frames/sr/channels；目录变化时自动重建
- **懒加载** + **声道随机**：每次只用 `sf.read(start, frames)` 截取 4–8s 片段，避免读完整 14–42 分钟文件
- **活跃段优先** (`prefer_active=true`)：随机采 5 个候选位置，按 RMS 能量选最高的（更可能含语音）；并随机选一个声道
- 重采样到 16kHz

**主流程 `run_augment`**：

1. `scan_short_wavs` 扫描主音频，过滤 ≤2s + RMS≥0.02 + 削波率≤5%；同样写 `.main_index.json` 指纹缓存
2. 加载可选的 `text_file` 和 `scp_file`，建立 `wav_abs_path → key → 文本` 映射
3. 对每条主音频循环 `factor` 次：
   - 截背景片段，独立施加 `bg_gain_db`
   - 重采样主音频到 16kHz，独立施加 `main_gain_db`
   - 随机 SNR 调主说话人增益，`mix_main_into_bg` 在背景的随机偏移位置叠加
   - 可选后处理：`_apply_simple_reverb`（与 engine 类似的合成 RIR）、`_add_gaussian_noise`
   - **峰值归一化**：`normalize_mode="main_region"` 时只看主说话人区域峰值，避免把背景拉高（重要！保证主说话人始终突出）
   - NaN/Inf 检查后写文件
4. 同步写 `text` 和 `wav.scp`，统计未匹配标签的条目
5. 断点续跑：扫描输出目录已有 `*_meetbg<i>.wav` 并跳过

---

## 几个值得注意的点

1. **三种顶层脚本的差异**：
   - `augment_dir.py` 简单粗暴**全量预加载**到内存，适合小数据集
   - `augment_manifest.py` 用 `LRUAudioCache(maxsize=2000)` **按需懒加载**，扛大数据集
   - `batch_augment_public.py` 多进程 + 每进程 `_SimpleAudioCache` 单任务级缓存 + `progress.jsonl` 断点，扛多个大型公开数据集

2. **断点续跑机制有两套**：
   - `augment_dir.py` / `augment_manifest.py`：扫输出目录里已存在的 `*_augN.wav` 文件名
   - `batch_augment_public.py`：维护 `progress.jsonl`（更稳，可记录 failed 状态，支持多进程并发追加）

3. **采样策略的物理建模思路**：`_sample_normal_clipped`（正态采样）+ `bg_distance_correlated`（同一距离因子驱动 SNR/RT60/lowpass）让多说话人场景的背景人在物理上自洽——离得近就大声、混响小、高频好；离得远就反过来。这比独立均匀采样真实得多。

4. **split 严格隔离**：`augment_manifest.py` 和 `batch_augment_public.py` 都强制 train/dev/test 不互相污染，避免增强引入数据泄漏。

5. **`augment_short_in_meeting.py` 不复用 engine**：这个脚本完全独立，自己实现了背景管理 + 混合 + 简单混响 + 噪声，因为它的核心逻辑（懒加载多声道远场背景、活跃段检测）不属于 `AudioAugmentor` 的策略范畴。
