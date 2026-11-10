# 服务器端目录结构

> 服务器: `10.6.217.118`，端口: `20028`，用户: `root`，工作目录: `/data/shamo/project/asr`
>
> GPU: 单卡 **NVIDIA GeForce RTX 4090 D**（训练走 `CUDA_VISIBLE_DEVICES=1`）。连接信息以 `configs/remote.yaml` 为准。

## 完整目录树

```
/data/shamo/project/asr/                    ← remote.yaml 中的 workspace
│
├── FunASR/                                 ← （可选）自定义 FunASR 框架源码
│   ├── funasr/bin/train.py                ← torchrun 调用的训练脚本
│   └── ...                                 ← 注：当前训练默认用 pip 版 funasr（python3 -m funasr.bin.train）
│
├── models/                                 ← 预训练模型
│   └── iic/                                ← ModelScope 命名空间
│       └── speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/
│           ├── model.pt                    ← 预训练权重（约 944M，247M 参数）
│           ├── config.yaml                 ← 模型配置
│           ├── tokens.txt                  ← 词表（vocab8404）
│           └── seaco_hotwords.txt          ← 热词列表（推理时用）
│
├── data/                                   ← 数据目录
│   ├── aishell_small/                     ← AISHELL 小数据集（骨架验证）
│   │   ├── raw/                           ← 原始音频 + 标签
│   │   │   ├── S0001/                     ← 按说话人/设备组织
│   │   │   │   ├── BAC009S0001W0001.wav
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── label/                         ← 人工标注文本
│   │       └── aishell_transcript.txt
│   │
│   ├── s10_reflow/                        ← 业务数据集 s10（萤石设备录音）
│   │   ├── raw/
│   │   │   ├── DEV001/                    ← 按设备 ID 组织
│   │   │   ├── DEV002/
│   │   │   └── ...
│   │   └── label/
│   │       └── s10_transcript.txt
│   │
│   ├── sp4_reflow/                        ← 业务数据集 sp4
│   │   ├── raw/
│   │   └── label/
│   │
│   └── pipeline/                          ← 流水线产物目录（由 asr-evolve 生成）
│       ├── aishell_small/                 ← 按数据集组织
│       │   ├── labeled/                   ← [prepare] 产物
│       │   │   ├── wav.scp               ← "utt_id /path/to/audio.wav"
│       │   │   └── text                   ← "utt_id 文本内容"
│       │   ├── unlabeled/                 ← [prepare] 产物
│       │   │   └── wav.scp
│       │   ├── quality/                   ← [quality] 产物
│       │   │   ├── metrics.jsonl          ← 每条音频的质量指标
│       │   │   ├── good.lst              ← 合格 utt_id 列表
│       │   │   └── bad.lst               ← 问题 utt_id + 原因
│       │   ├── split/                     ← [split] 产物
│       │   │   ├── train/
│       │   │   │   ├── wav.scp
│       │   │   │   └── text
│       │   │   ├── dev/
│       │   │   │   ├── wav.scp
│       │   │   │   └── text
│       │   │   └── test/
│       │   │       ├── wav.scp
│       │   │       └── text
│       │   └── augmented/                 ← [augment] 产物
│       │       └── train/
│       │           ├── wav.scp            ← 含 _aug 后缀的增强条目
│       │           └── text
│       │
│       ├── all_label/                     ← [merge] 产物（跨数据集合并）
│       │   ├── train/
│       │   │   ├── wav.scp               ← 原始 + 增强，去重后
│       │   │   └── text
│       │   ├── dev/
│       │   │   ├── wav.scp
│       │   │   └── text
│       │   └── test/
│       │       ├── wav.scp
│       │       └── text
│       │
│       ├── train.jsonl                    ← [jsonl] 产物（FunASR 训练用）
│       └── dev.jsonl                      ← [jsonl] 产物
│
├── output/                                ← 训练产出（按实验 tag 组织）
│   ├── baseline_1782982510/               ← 实验 1 的输出（scripts/run_train.py 用 baseline_<ts>）
│   │   ├── model.pt.ep0.4000             ← 按步保存的检查点（save_checkpoint_interval=4000）
│   │   ├── model.pt.best                 ← 最优检查点（按 loss 选）
│   │   ├── model.pt.avg5                 ← 最优 5 个检查点的平均（keep_nbest=5, avg_nbest=5）
│   │   ├── train_stdout.log             ← 训练日志（scripts 重定向输出）
│   │   └── config.yaml                   ← 本次训练配置快照
│   ├── exp_1719129600/                    ← 实验 2（打包 pipeline 用 exp_<ts>）
│   │   └── ...
│   └── latest -> baseline_1782982510      ← 软链接到最新实验（如脚本创建）
│
├── results/                               ← 推理/评估结果
│   ├── v0.1.0/                            ← 按版本号组织
│   │   ├── s10_test_best.jsonl           ← 推理结果（每行 ref + hyp）
│   │   ├── sp4_test_best.jsonl
│   │   └── evaluation_report.json        ← DomainEvaluator 输出
│   └── baselines/
│       ├── seaco_s10_test.jsonl           ← SeaCo 原始基线
│       └── online_s10_test.jsonl          ← 线上基线
│
├── scripts/                               ← 辅助脚本
│   ├── decode.sh                          ← 推理脚本（调 FunASR infer）
│   ├── eval.sh                            ← 评估脚本
│   └── prepare_aishell.sh                 ← AISHELL 数据准备
│
└── configs/                               ← 服务器端配置（可选，与本地同步）
    └── seaco_hotwords.txt                 ← 热词文件（推理时挂载）
```

## 关键文件格式

### wav.scp
```
BAC009S0001W0001 /data/shamo/project/asr/data/aishell_small/raw/S0001/BAC009S0001W0001.wav
BAC009S0001W0002 /data/shamo/project/asr/data/aishell_small/raw/S0001/BAC009S0001W0002.wav
```

### text
```
BAC009S0001W0001 你好萤石打开客厅的灯
BAC009S0001W0002 萤石关闭空调
```

### train.jsonl
```json
{"key": "BAC009S0001W0001", "source": "/data/shamo/project/asr/data/aishell_small/raw/S0001/BAC009S0001W0001.wav", "target": "你好萤石打开客厅的灯"}
{"key": "BAC009S0001W0002", "source": "/data/shamo/project/asr/data/aishell_small/raw/S0001/BAC009S0001W0002.wav", "target": "萤石关闭空调"}
```

### metrics.jsonl
```json
{"utt_id": "BAC009S0001W0001", "path": "/data/.../W0001.wav", "duration_s": 3.2, "sample_rate": 16000, "channels": 1, "rms": 0.045}
```

### evaluation_report.json
```json
{
  "version": "v0.1.0",
  "test_set": "s10_test",
  "cer": 0.0092,
  "stripped_cer": 0.0085,
  "instruction_recall": 0.94,
  "sample_count": 500,
  "timestamp": "2026-06-22T15:30:00Z"
}
```

## 本地 ↔ 服务器文件同步

| 操作 | 方向 | 什么文件 |
|------|:---:|---------|
| 上传配置 | 本地→服务器 | `configs/*.yaml`, `seaco_hotwords.txt` |
| 上传数据清单 | 本地→服务器 | `data/pipeline/*/labeled/{wav.scp,text}`, `train.jsonl`, `dev.jsonl` |
| 下载结果 | 服务器→本地 | `results/v0.X.0/*.jsonl`, `output/*/train_stdout.log` |
| 下载检查点 | 服务器→本地 | `output/*/model.pt.best`（仅需要部署时） |

Pipeline 的 `prepare` → `merge` → `jsonl` 阶段在本地完成数据处理，产物通过 SSH 上传到服务器的 `data/pipeline/` 下。`train` 阶段在服务器远程执行。评估结果下载回本地分析。

> **训练输入路径说明**：打包的 `train` 阶段读取 `data/pipeline/train.jsonl`；当前运维脚本 `scripts/run_train.py` / `scripts/check_log.py` 读取的是 `data/merged/{train,dev}/{split}.jsonl`（合并后的最终清单）。两者是同一数据的不同布局，迁移时以脚本实际使用的 `data/merged/` 为准。

## 首次部署检查清单

```bash
# 在服务器上执行（注意端口 20028、用户 root）
ssh -p 20028 root@10.6.217.118

# 1. 确认目录存在
ls /data/shamo/project/asr/{models,data,output,results,scripts}

# 2. 确认 pip 版 funasr 可用
python3 -c "import funasr; print(funasr.__version__)"

# 3. 确认 GPU 可用（应显示 NVIDIA GeForce RTX 4090 D）
nvidia-smi

# 4. 确认基座模型存在
ls -lh /data/shamo/project/asr/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.pt

# 5. 确认业务数据存在（或 AISHELL 骨架验证用）
ls /data/shamo/project/asr/data/merged/train/train.jsonl

# 6. 确认训练入口可用
python3 -m funasr.bin.train --help
```

> 上面这些检查可直接用本地 `scripts/diag_gpu.py`（诊断 GPU 显存/模型加载）自动完成，无需手动 SSH。
