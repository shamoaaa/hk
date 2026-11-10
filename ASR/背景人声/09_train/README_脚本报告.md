# 09_train 目录脚本报告

本目录是整套流水线的**模型训练阶段**，基于 FunASR 对 `SeacoParaformer`（带热词偏置的 Paraformer）预训练模型做微调（finetune）。训练数据来自 08 阶段汇总的 `all_label/`（自有 reflow + 公开数据集 + 增强数据）。

## 目录结构

```
09_train/
  config.yaml              训练超参与模型结构配置（SeacoParaformer 完整定义）
  fast_scp2jsonl.py        ★ 自研：快速 scp+text → train.jsonl（只读音频头，不解码）
  finetune.sh             微调启动脚本（单卡，aishell 实验版）
  finetune_fast.sh        ★ 微调启动脚本（8 卡，全量 all_label，从头训练）
  finetune_fast_5.sh       微调启动脚本（8 卡，断点续训版）
  README.MD               一句话说明
  tools/
    create_wavscp.py       自研：扫描 wav 目录生成 wav.scp（按标签 key 过滤）
    filter_reduplicate_line.py  自研：检测 wav.scp 中重复 key
  funasr/                  ← 本地 vendored FunASR 源码（训练引擎，含 bin/train.py）
  examples/                ← FunASR 官方示例 recipe（aishell/wenetspeech 等，参考用）
  data/                    ← 本地数据列表样例（wav.scp / text / train.jsonl 等）
```

> 说明：`funasr/` 和 `examples/` 是 vendored 的 FunASR 库及其官方示例（第三方代码），本报告聚焦项目自研脚本和配置，对它们只描述在流程中的角色。

## 训练流水线

```
08 阶段产出                      本目录处理
all_label/{train,dev}/          
  wav.scp + text     ──┐
                       ▼
            fast_scp2jsonl.py（自研，替代 funasr 自带 scp2jsonl）
                       │  只读 wav 头取时长，100~500x 加速
                       ▼
            {train,dev}/*.jsonl  （key/source/source_len/target/target_len）
                       │
                       ▼
            finetune_fast.sh → torchrun → funasr/bin/train.py
                       │         （8 卡 DDP，加载 config.yaml + 预训练 model.pt）
                       ▼
            /data1/train_outputs/  （model.pt.best / model.pt.epN / model.pt.avgN）
```

---

## 1. `config.yaml` — SeacoParaformer 训练配置

完整定义模型结构和训练超参（FunASR 的 Hydra 风格配置）。

| 模块 | 关键配置 |
|---|---|
| **model** | `SeacoParaformer`，`ctc_weight=0`（不用 CTC），`sampling_ratio=0.75`，`NO_BIAS=8377`（热词偏置占位 id） |
| **encoder** | `SANMEncoder`，512 维，4 头，**50 层**，kernel=11 |
| **decoder** | `ParaformerSANMDecoder`，16 层 |
| **seaco_decoder** | 额外的热词解码器，4 层（热词偏置分支） |
| **predictor** | `CifPredictorV3`（CIF 连续积分发放，做声学边界预测） |
| **frontend** | `WavFrontend`，16kHz，80 维 fbank，LFR（m=7,n=6），引用预训练的 `am.mvn` 做 CMVN |
| **specaug** | `SpecAugLFR`，频率掩码 + 时间掩码（不做 time warp） |
| **train_conf** | `max_epoch=40`，按 `valid/acc` 选最优，保留 nbest=5，每 4000 步验证/存档 |
| **optim** | adam，`lr=5e-5`，`warmuplr` 调度（warmup 3000 步） |
| **dataset** | `AudioDatasetHotword`，按 token 动态组 batch（`batch_size=60000` tokens），`max_token_length=2048` |
| **tokenizer** | `CharTokenizer`，引用预训练的 `tokens.json` + `seg_dict` |
| **init_param** | 从预训练 `model.pt` 初始化（vocab8404 的 seaco_paraformer large） |
| **freeze_param** | 冻结 `hotword_output_layer` 和 `seaco_decoder`（只调主干，保留热词能力） |
| **数据** | `train_data_set_list=all_label/train/train.jsonl`，`valid=all_label/dev/dev.jsonl` |

注意 `device: cpu` 是配置文件里的默认值，实际训练时由 shell 脚本的 `CUDA_VISIBLE_DEVICES` + torchrun 覆盖到 GPU。

## 2. `fast_scp2jsonl.py` — 快速 jsonl 生成（自研，核心工具）

| 项 | 说明 |
|---|---|
| **输入** | `--wav_scp`（`key path`）+ `--text`（`key text`） |
| **输出** | `--output` 的 JSONL，每行 `{key, source, source_len, target, target_len}` |
| **参数** | `--workers`（默认 8 线程）、`--frame_shift`（默认 10ms）、`--resume`（断点续跑） |

**为什么存在**：FunASR 自带的 `scp2jsonl` 用 `librosa.load()` **完整解码音频**来算时长，在 NFS 上奇慢（14 万条要 15~25 小时）。本脚本改用 `soundfile.info()` **只读 WAV 文件头**拿时长，不加载采样数据，提速 **100~500 倍**。

**逻辑**：
1. 优先用 `soundfile`，缺库时回退到标准库 `wave` 模块读头
2. `source_len` = 音频时长(ms) / 帧移(ms)，即帧数；`target_len` = 文本字符数（有空格按词数）
3. 文件不存在时 `source_len=1` 占位（与原版行为一致）
4. 读 wav.scp 和 text 取**交集**，对只在一边的 key 打警告并跳过
5. `ThreadPoolExecutor` 多线程读头，`as_completed` 边处理边写、`flush`，配 tqdm 进度条
6. `--resume`：先扫已有输出收集 done_keys，追加模式跳过已完成条目

## 3. `finetune_fast.sh` — 主训练脚本（8 卡全量，自研）★

| 项 | 说明 |
|---|---|
| **输入** | `all_label/{train,dev}/` 的 `wav.scp` + `text`（脚本内 jsonl 生成步骤已注释，假定已生成） |
| **输出** | `/data1/train_outputs/`（checkpoints + `log.txt`） |
| **环境** | `conda activate asr`，`CUDA_VISIBLE_DEVICES=0~7`（8 卡） |

**逻辑**：
1. 激活 conda 环境，设置 8 张可见 GPU，`gpu_num` 自动算卡数
2. 指向本地预训练模型目录（vocab8404 seaco_paraformer large）
3. （注释掉的）调用 `merge_all_labels.py` 和 `fast_scp2jsonl.py` 预处理步骤——表明数据准备已单独完成
4. **`export PYTHONPATH=/data1/code/09_train`**：让 Python 优先加载本地 vendored 的 funasr 代码而非 pip 安装版（关键，因为模型代码可能有改动）
5. `torchrun --nproc_per_node=$gpu_num funasr/bin/train.py` 启动 DDP 分布式训练，命令行 `++` 覆盖 config：
   - `max_epoch=50`、`lr=2e-5`、`warmup_steps=30000`、`resume=false`（从预训练权重重新开始微调）
   - `batch_size=60000` tokens，每 4000 步验证/存档，保留 nbest=5 并做权重平均
   - `freeze_param=[hotword_output_layer, seaco_decoder]`（冻结热词分支）
6. 输出 tee 到日志文件。后台运行：`nohup 09_train/finetune_fast.sh > finetune.out 2>&1 &`

## 4. `finetune_fast_5.sh` — 断点续训版（自研）

与 `finetune_fast.sh` 几乎相同，**差异**：

| 项 | finetune_fast.sh | finetune_fast_5.sh |
|---|---|---|
| `max_epoch` | 50 | **60** |
| `train_conf.resume` | false（重新开始） | **true（断点续训）** |
| `freeze_param` | `hotword_output_layer, seaco_decoder` | 额外多冻结 **`bias_encoder`** |
| `warmup_steps` | 30000（命令行覆盖） | 未覆盖（用 config 默认 3000） |

用途：在 `finetune_fast.sh` 训练基础上继续训练到 60 epoch，并多冻结一个热词偏置编码器。

## 5. `finetune.sh` — 早期单卡实验脚本

| 项 | 说明 |
|---|---|
| **输入** | `data_aishell/augment` 下的 `wav_{train,val}.scp` + `text_{train,val}.txt` |
| **输出** | `data_aishell/outputs` |

早期在 **aishell 增强数据**上的单卡（`CUDA_VISIBLE_DEVICES=0`）实验脚本。与 fast 版的区别：
- 用 FunASR **自带的 `scp2jsonl`**（慢）现场生成 jsonl
- 训练入口是 `asr_offline/funasr/bin/train.py`（旧路径）
- `max_epoch=20`，`freeze_param` 还包含 `bias_encoder`
- 末尾有注释掉的模型导出 + CER 评测命令

属于历史/实验脚本，正式训练用的是 `finetune_fast.sh`。

## 6. `tools/create_wavscp.py` — 生成 wav.scp（自研）

| 项 | 说明 |
|---|---|
| **输入** | 标签文件 `label_s10_1215_1231.txt`（取每行首列 key）+ wav 目录 `20241215_20250115/` |
| **输出** | `wav_s10_20241215_20241231.scp`（`key abspath`） |

**逻辑**：先读标签文件收集所有合法 key（顺带检测重复并打 WARN）；`rglob("*.wav")` 递归扫描音频目录；只为 key 在标签列表里的 wav 写入 scp，写完从列表移除（用于最后打印**有标签但找不到对应 wav 的 key**）。一次性脚本，路径硬编码。

## 7. `tools/filter_reduplicate_line.py` — 重复 key 检测（自研）

极简工具：读一个 wav.scp，逐行取首列 key，已出现过就打 `[WARN]`。仅做重复键检测，不产出文件。

---

## 几个值得注意的点

1. **`fast_scp2jsonl.py` 是性能关键的自研替换件**：用读文件头代替完整解码，把 jsonl 生成从"小时级"降到"分钟级"，对 14 万+ 条 NFS 数据是刚需。这是本目录最有价值的自研代码。

2. **三个 shell 脚本是同一训练任务的演进**：
   - `finetune.sh`（早期，单卡，aishell 小实验，用慢版 scp2jsonl）
   - `finetune_fast.sh`（正式，8 卡，全量 all_label，从预训练权重微调到 50 epoch）
   - `finetune_fast_5.sh`（续训，断点恢复到 60 epoch，多冻结 bias_encoder）

3. **vendored funasr + PYTHONPATH 注入**：通过 `export PYTHONPATH=/data1/code/09_train` 强制使用本地 funasr 源码，意味着项目对 funasr 训练引擎做过定制改动，不能用 pip 版替代。

4. **冻结策略反映微调意图**：始终冻结 `seaco_decoder` 和 `hotword_output_layer`（保留预训练模型的热词能力），只微调声学主干（encoder/decoder/predictor），用很小的学习率（2e-5）——典型的"保留通用能力 + 适配业务数据"微调思路。

5. **路径全部硬编码到 `/data1/...`**：所有脚本面向特定 Linux 训练机布局，跨机器使用需改路径。`config.yaml` 里 `device: cpu` 仅是占位，实际由 torchrun 接管 GPU。

6. **`tools/` 下两个脚本是早期一次性工具**：`create_wavscp.py` / `filter_reduplicate_line.py` 路径写死、功能单一，属于数据准备早期的临时脚本，已被 08 阶段更完善的脚本取代。
