# ASR 训练优化系统 — 执行流程文档

> 更新时间：2026-05-21
> 基于当前项目实际代码和配置

---

## 系统概述

基于 FunASR SeACo-Paraformer 的 ASR 模型训练优化闭环系统。

- **服务器**（Docker 容器，8×RTX 3090）：数据增强、训练、推理
- **本地**（Windows）：分析、诊断、反馈、策略管理、迭代管理
- **唯一主配置**：`configs/funasr_config.yaml`
- **数据库**：`metadata.db`（仅本地，SQLite）

---

## 目录结构与产物

```
workspace/
├── configs/
│   ├── funasr_config.yaml          # 唯一主配置
│   ├── remote.yaml                 # SSH 连接配置
│   ├── post_process/hotwords.txt   # 热词表
│   ├── model/                      # 模型配置版本快照
│   ├── training/                   # 训练配置版本快照
│   └── augmentation/               # 增强配置版本快照
├── data/
│   ├── train.jsonl / val.jsonl / test.jsonl
│   ├── musan/                      # MUSAN 噪声数据集
│   ├── RIRS_NOISES/                # 真实/仿真 RIR 数据集（reverb 用）
│   └── augmented/                  # 增强后数据（服务器生成）
├── models/iterations/iter_xxx/     # 训练产物
├── logs/training/training_metrics.jsonl
├── reports/evaluation/
│   ├── inference_iter_xxx.jsonl    # 推理逐样本结果
│   ├── eval_report_iter_xxx.json   # 结构化评估报告
│   └── eval_report_iter_xxx.txt    # 人类可读报告
├── metadata.db                     # SQLite
└── scripts/                        # CLI 脚本
```

---

## 远程连接配置

```yaml
# configs/remote.yaml
host: kiro-docker-remote
username: root
remote_path: /data2/shamo/project/test_skill
remote_python: /data2/shamo/project/test_skill/workspace/.venv310/bin/python
```

**同步命令**：

| 命令 | 方向 | 内容 |
|------|------|------|
| `push-configs` | 本地→服务器 | `funasr_config.yaml` + `configs/post_process/` |
| `push-data` | 本地→服务器 | 原始数据 jsonl（首次部署） |
| `sync-down` | 服务器→本地 | logs/ + reports/ + 增强 jsonl |
| `push <file>` | 本地→服务器 | 单个文件 |
| `pull <file>` | 服务器→本地 | 单个文件 |

---

## 完整迭代流程（逐步详解）

### Step 1: 开始迭代（本地）

```powershell
.venv_win\Scripts\python -m scripts.manage_iterations start --description "启用SpecAugment+SpeedPerturb"
```

**stdout 输出**：
```json
{
  "status": "success",
  "iteration_id": "iter_a1b2c3d4",
  "description": "启用SpecAugment+SpeedPerturb",
  "created_at": "2026-05-21T10:00:00"
}
```

**副作用**：`metadata.db` → `iterations` 表插入一行（status=in_progress）。

---

### Step 2: 数据增强（服务器）

```powershell
.venv_win\Scripts\python -m scripts.remote_runner run "python -m scripts.run_augmentation --all"
```

**内置质量检查**：
- **增强前**（pre_check）：自动扫描原始 train/val/test，检测 empty_label、corrupted_audio、too_short、too_long 等异常。发现严重异常会 warning（不阻断，但提示清洗）。
- **增强后**（post_check）：自动校验增强产物，检测增强失败的样本（路径不存在、时长异常等）。

**stdout 输出**：
```json
{
  "mode": "all",
  "augmented_dir": "data/augmented",
  "results": {
    "train": { "status": "completed", "original_count": 5000, "augmented_count": 8500, "total_count": 13500 },
    "val": { "status": "completed", "total_count": 1200 },
    "test": { "status": "completed", "total_count": 1800 }
  },
  "pre_check": {
    "train": { "total_anomalies": 3, "critical": 1, "types": {"empty_label": 1, "too_short": 2} }
  },
  "post_check": {
    "train": { "total_anomalies": 0, "types": {} }
  }
}
```

**生成文件**（服务器）：`data/augmented/train_aug.jsonl`、`val_aug.jsonl`、`test_aug.jsonl` + 对应 wav 文件。

**增强 JSONL 格式**（每行）：
```json
{"key": "BAC009S0002W0122", "source": "/path/original.wav", "source_len": 48000, "target": "而对楼市成交抑制作用最大的限购", "target_len": 14, "scene_type": "clean"}
{"key": "BAC009S0002W0122_aug_speed", "source": "/path/augmented/xxx_aug_speed.wav", "source_len": -1, "target": "而对楼市成交抑制作用最大的限购", "target_len": 14, "scene_type": "speed_perturb"}
{"key": "BAC009S0002W0122_aug_babble", "source": "/path/augmented/xxx_aug_babble.wav", "source_len": -1, "target": "而对楼市成交抑制作用最大的限购", "target_len": 14, "scene_type": "background_voice"}
```

**当前增强策略**：Speed(1.0) + Babble(0.4) + Music(0.2) + Noise(0.2) + Reverb(0.15, 真实RIR卷积)

**增强前自动过滤**：empty_label、missing_source、short_text(<2字) 的样本自动跳过，不进入增强输出（输出中 `filtered_count` 字段报告过滤数量）。

---

### Step 3: 训练（服务器）

```powershell
.venv_win\Scripts\python -m scripts.remote_runner run "python -m scripts.run_training --tag iter_a1b2c3d4"
```

**stdout 输出**：
```json
{
  "status": "completed",
  "tag": "iter_a1b2c3d4",
  "model_dir": "models/iterations/iter_a1b2c3d4",
  "final_loss": 0.342,
  "total_steps": 2500,
  "metrics_path": "logs/training/training_metrics.jsonl"
}
```

**生成文件**（服务器）：

| 文件 | 说明 |
|------|------|
| `models/iterations/iter_a1b2c3d4/` | 模型权重目录 |
| `logs/training/training_metrics.jsonl` | 逐步训练指标 |
| `configs/training/training_*.yaml` | 训练配置快照 |

**training_metrics.jsonl 格式**（每行）：
```json
{"step": 100, "epoch": 1, "train_loss": 2.15, "val_loss": 2.30, "learning_rate": 5e-05, "timestamp": "2026-05-21T10:15:00"}
{"step": 200, "epoch": 1, "train_loss": 1.85, "val_loss": 2.10, "learning_rate": 5e-05, "timestamp": "2026-05-21T10:18:00"}
```

**当前训练参数**：batch_type=length, batch_size=16000, lr=5e-5, epoch=10, spec_aug=true, warmup=50

---

### Step 4: 推理（服务器）

```powershell
# 推理一次，默认启用后处理，输出同时含 hypothesis_text 和 hypothesis_raw
.venv_win\Scripts\python -m scripts.remote_runner run "python -m scripts.run_inference --model-dir models/iterations/iter_a1b2c3d4 --tag iter_a1b2c3d4"
```

**stdout 输出**（一次推理，两组指标）：
```json
{
  "mode": "inference",
  "status": "completed",
  "num_samples": 1800,
  "postprocess_applied": true,
  "cer": 0.0098,
  "wer": 0.125,
  "cer_raw": 0.0115,
  "wer_raw": 0.140,
  "postprocess_cer_delta": 0.0017,
  "postprocess_wer_delta": 0.015
}
```

**生成文件**：`reports/evaluation/inference_iter_a1b2c3d4.jsonl`

**推理 JSONL 格式**（每行同时含两列，无需推理两次）：
```json
{"audio_path": "/path/test.wav", "reference_text": "十宗最大的土地", "hypothesis_text": "十宗最大的土地", "hypothesis_raw": "时松最大的土地", "scene_type": "clean"}
```

- `hypothesis_text`：后处理后（热词纠正 + regex）
- `hypothesis_raw`：模型原始输出
- `postprocess_cer_delta`：后处理带来的 CER 改善（正值 = 后处理有效）

---

### Step 5: 下载产物（本地）

```powershell
.venv_win\Scripts\python -m scripts.remote_runner sync-down
```

下载 `logs/`、`reports/`、配置快照到本地。

---

### Step 6: 分析评估（本地）

```powershell
.venv_win\Scripts\python -m scripts.analyze_results --test-set reports/evaluation/inference_iter_a1b2c3d4.jsonl --tag iter_a1b2c3d4

# 用原始 hypothesis 分析（对比后处理效果）
.venv_win\Scripts\python -m scripts.analyze_results --test-set reports/evaluation/inference_iter_a1b2c3d4.jsonl --tag iter_a1b2c3d4_raw --use-raw
```

**stdout 输出**：
```json
{
  "status": "success",
  "tag": "iter_a1b2c3d4",
  "total_samples": 1800,
  "overall_cer": 0.0098,
  "overall_wer": 0.125,
  "scene_metrics": [
    {"scene_type": "clean", "cer": 0.005, "wer": 0.08, "sample_count": 600},
    {"scene_type": "background_voice", "cer": 0.015, "wer": 0.18, "sample_count": 400},
    {"scene_type": "background_music", "cer": 0.012, "wer": 0.15, "sample_count": 300}
  ],
  "error_type_distribution": {"substitution": 45, "deletion": 12, "insertion": 8},
  "badcase_count": 320,
  "report_json": "reports/evaluation/eval_report_iter_a1b2c3d4.json",
  "report_txt": "reports/evaluation/eval_report_iter_a1b2c3d4.txt"
}
```

**生成文件**：

| 文件 | 说明 |
|------|------|
| `reports/evaluation/eval_report_iter_a1b2c3d4.json` | 结构化报告（含 badcases 列表） |
| `reports/evaluation/eval_report_iter_a1b2c3d4.txt` | 人类可读报告 |

**eval_report JSON 格式**：
```json
{
  "test_set_path": "reports/evaluation/inference_iter_a1b2c3d4.jsonl",
  "tag": "iter_a1b2c3d4",
  "total_samples": 1800,
  "overall_cer": 0.0098,
  "overall_wer": 0.125,
  "scene_metrics": [...],
  "error_type_distribution": {"substitution": 45, "deletion": 12, "insertion": 8},
  "badcase_threshold": 0.0,
  "badcase_count": 320,
  "badcases": [
    {"audio_path": "...", "reference_text": "十宗最大", "hypothesis_text": "时松最大", "scene_type": "clean", "cer": 0.5}
  ]
}
```

**评测文本统一化**：CER/WER 计算前自动做 NFKC + 全角半角 + 去标点 + 去空格，保证口径一致。

---

### Step 7: 自动诊断 + 策略匹配（本地）

```powershell
.venv_win\Scripts\python -m scripts.auto_diagnose --verbose
```

**stdout 输出**：
```json
{
  "status": "success",
  "diagnosis": {
    "cer": 0.0098,
    "error_ratios": {"substitution": 0.69, "deletion": 0.18, "insertion": 0.12},
    "repetition_ratio": 0.02,
    "training": {"loss_reduction_pct": 42.5, "still_decreasing": true, "overfitting": false},
    "matched_strategies": ["hotword_extraction"]
  },
  "recommendations": [
    {"action": "more_epochs", "priority": 3, "description": "Loss 仍在下降"},
    {"action": "fix_substitution", "priority": 4, "description": "Substitution 占比 69%，优先提取热词"}
  ],
  "top_action": {"action": "fix_substitution", "priority": 4}
}
```

**诊断规则**：

| 条件 | 推荐动作 |
|------|---------|
| loss 下降 < 5% | 增加 epoch 或提高学习率 |
| loss 仍在下降 | 增加 epoch |
| 出现 NaN | 降低学习率至 1/10 |
| val/train gap > 100% | 过拟合 → 增加 SpecAugment/数据 |
| deletion > 40% | 模型吃字 → 检查 max_token_length |
| substitution > 70% | 同音混淆 → 提取热词 |
| 重复率 > 10% | Attention 问题 → 增加 CTC 权重 |
| 连续 3 轮无改善 | 切换优化维度 |

---

### Step 8: 执行优化 + 推送（本地→服务器）

```powershell
# 训练参数变更
.venv_win\Scripts\python -m scripts.apply_training_config apply --max-epoch 15

# 热词提取
.venv_win\Scripts\python -m scripts.extract_hotwords --eval-report reports/evaluation/eval_report_iter_a1b2c3d4.json --merge

# 推送到服务器
.venv_win\Scripts\python -m scripts.remote_runner push-configs
```

**hotwords.txt 格式**（tab 分隔）：
```
时松	十宗	5
做用	作用	3
```

---

### Step 9: 重新训练与评估

回到 Step 3 → 4 → 5 → 6，使用新 tag。

---

### Step 10: 记录反馈与反思（本地）

```powershell
# 基线（第一轮，不传 --cer-before）
.venv_win\Scripts\python -m scripts.record_feedback label --iteration-id iter_a1b2c3d4 --cer-after 0.0098 --wer-after 0.125

# 后续轮次（含对比 + 反思）
.venv_win\Scripts\python -m scripts.record_feedback label --iteration-id iter_e5f6g7h8 --cer-before 0.0098 --cer-after 0.007 --wer-before 0.125 --wer-after 0.10 --reflection "增加epoch到15有效，热词修复了时松→十宗等高频替换。"
```

**stdout 输出**：
```json
{
  "status": "success",
  "iteration_id": "iter_e5f6g7h8",
  "cer_delta": -0.0028,
  "wer_delta": -0.025,
  "is_positive": true,
  "reflection": "增加epoch到15有效，热词修复了时松→十宗等高频替换。"
}
```

**副作用**：`metadata.db` → `effect_feedbacks` 表插入一行（含 reflection_text）。

---

## 后处理管线

推理时默认启用后处理，管线顺序：`hotword → regex → lm_rerank`

**当前 regex 规则**：
1. 去不可见 Unicode 字符（BOM、零宽）
2. 全角→半角
3. 去空格
4. 去中文标点

**热词纠正**：精确匹配 wrong_text 子串才替换（防过度偏置）。

---

## 数据质量检查

```powershell
.venv_win\Scripts\python -m scripts.analyze_dataset --dataset-path data/train.jsonl --check-audio-exists
```

**检测的异常类型**：

| 类型 | 说明 | 阈值 |
|------|------|------|
| empty_label | 文本为空 | — |
| too_short | 音频 < 0.5s | 0.5s |
| too_long | 音频 > 30s | 30s |
| short_text | 文本 < 2 字 | 2 |
| duration_mismatch | 字/秒超范围 | 1-25 字/秒 |
| corrupted_audio | 时长 ≤ 0 | — |
| missing_audio | 文件不存在 | — |

---

## 系统重置

```powershell
.venv_win\Scripts\python scripts/reset_system.py --confirm --include-remote
```

**清除**：数据库、日志、报告、增强数据、配置快照、模型产物。
**保留**：原始数据、主配置、脚本代码、Skill 文件、虚拟环境。

---

## 设计原则

1. **增强数据是前提**：训练/推理必须用增强数据
2. **推理和分析分离**：服务器只做推理输出 JSONL，本地做所有分析
3. **本地改配置，push-configs 同步，远程跑计算**
4. **评测口径统一**：CER/WER 计算前自动 NFKC + 去标点 + 去空格
5. **策略动态进化**：边用边改，发现新模式立即记录
6. **配置单源**：`funasr_config.yaml` 是唯一可修改源
7. **数据库仅本地**：`metadata.db` 不同步到服务器
8. **后处理可对比**：推理输出同时含 raw 和 processed，量化后处理贡献
