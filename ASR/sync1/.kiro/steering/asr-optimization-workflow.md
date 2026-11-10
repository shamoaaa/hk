---
inclusion: manual
---

# ASR 训练优化工作流

## 概述

本文件引导执行 ASR 模型训练优化闭环流程。基于 FunASR 框架下的 `speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` 模型，聚焦于背景人声场景下的语音识别率优化。

**运行环境**：所有命令在 `workspace/` 目录下执行，需 `PYTHONPATH=. .venv/bin/python`。

## 触发条件

当用户请求进行 ASR 模型优化时，按以下流程执行。

---

## 工作流步骤

### Step 1: 开始迭代

```bash
PYTHONPATH=. .venv/bin/python scripts/manage_iterations.py start --description "描述本次优化目标"
```

### Step 2: 确保增强数据就绪

**业务规则**：训练和推理必须使用增强后的数据（`require_augmentation: true`）。

- 如果 `data/augmented/train_aug.jsonl` 已存在 → **跳过此步**，直接用已有数据
- 如果不存在、或需要调整增强策略 → 执行增强：

```bash
# 全量增强（使用配置中所有策略：babble/music/noise/reverb）
PYTHONPATH=. .venv/bin/python scripts/run_augmentation.py --all

# 或指定单一策略
PYTHONPATH=. .venv/bin/python scripts/run_augmentation.py --all --strategy babble
```

增强时自动标注 `scene_type`（babble→background_voice, noise→background_noise, 原始→clean）。

### Step 3: 训练

```bash
# 正式训练（自动使用 data/augmented/train_aug.jsonl）
PYTHONPATH=. .venv/bin/python scripts/run_training.py --tag <iter_tag>

# 可覆盖参数
PYTHONPATH=. .venv/bin/python scripts/run_training.py --tag <iter_tag> \
  --learning-rate 5e-5 --max-epoch 10 --gpu-ids 4

# 预览命令（不执行）
PYTHONPATH=. .venv/bin/python scripts/run_training.py --dry-run --tag <iter_tag>
```

训练过程中指标实时写入 `logs/training/training_metrics.jsonl`。

### Step 4: 评估

```bash
# 推理 + 后处理 + CER/WER 计算（自动使用 data/augmented/test_aug.jsonl）
PYTHONPATH=. .venv/bin/python scripts/run_inference.py \
  --model-dir models/iterations/<iter_tag> --evaluate --tag <iter_tag>
```

产出：`reports/evaluation/eval_report_<iter_tag>.json`

### Step 5: 分析

根据需要执行一项或多项分析：

```bash
# 训练过程分析（loss 趋势、停滞检测、异常检测）
PYTHONPATH=. .venv/bin/python scripts/collect_metrics.py --log-dir logs/training/

# Badcase 与错误模式分析
PYTHONPATH=. .venv/bin/python scripts/analyze_results.py \
  --test-set data/augmented/test_aug.jsonl

# 数据集质量分析（基于增强后数据）
PYTHONPATH=. .venv/bin/python scripts/analyze_dataset.py \
  --dataset-path data/augmented/train_aug.jsonl
```

### Step 6: 综合诊断与建议生成

基于 Step 5 的分析结果，结合知识库历史经验，从四个维度生成优化建议：

```bash
# 查询知识库中的历史经验
PYTHONPATH=. .venv/bin/python scripts/manage_knowledge.py query --type scene_solution
PYTHONPATH=. .venv/bin/python scripts/manage_knowledge.py query --type param_effect
```

**四个优化维度**：

| 维度 | 分析依据 | 执行脚本 |
|------|---------|---------|
| 模型结构 | SeACo 参数、注意力配置 | `apply_model_config.py` |
| 训练参数 | 学习率、batch size、调度策略 | `apply_training_config.py` |
| 数据增强 | badcase 错误模式 → `badcase_strategy_map` 映射 | `run_augmentation.py` |
| 后处理 | 高频替换错误 → 热词/正则 | `extract_hotwords.py` + `apply_postprocess.py` |

**决策优先级**：数据问题 > 训练参数 > 模型结构 > 后处理

### Step 7: 执行优化（用户确认后）

根据建议调用对应脚本：

```bash
# 模型结构变更
PYTHONPATH=. .venv/bin/python scripts/apply_model_config.py apply \
  --seaco-layers 3 --attention-weight 0.7

# 训练参数变更
PYTHONPATH=. .venv/bin/python scripts/apply_training_config.py apply \
  --learning-rate 1e-4 --lr-schedule cosine_annealing

# 数据增强策略调整（需要删除旧增强数据后重新生成）
# rm -rf data/augmented/*
PYTHONPATH=. .venv/bin/python scripts/run_augmentation.py --all --strategy babble

# 后处理：从 badcase 挖掘热词并合并
PYTHONPATH=. .venv/bin/python scripts/extract_hotwords.py \
  --eval-report reports/evaluation/eval_report_<iter_tag>.json --merge
PYTHONPATH=. .venv/bin/python scripts/apply_postprocess.py update-config \
  --hotword-file configs/post_process/hotwords.txt
```

每个 apply 脚本自动：(1) 更新主配置 (2) 保存版本快照 (3) 记录到 SQLite。

### Step 8: 重新训练与评估

配置变更后，回到 Step 3 重新训练和评估，对比前后效果。

### Step 9: 记录反馈

```bash
# 标记本次迭代效果（自动判断正面/负面）
PYTHONPATH=. .venv/bin/python scripts/record_feedback.py label \
  --iteration-id <iter_id> --cer-before <val> --cer-after <val>

# 查看反馈历史
PYTHONPATH=. .venv/bin/python scripts/record_feedback.py list
```

### Step 10: 迭代对比与停滞检测

```bash
# 对比两次迭代
PYTHONPATH=. .venv/bin/python scripts/manage_iterations.py compare \
  --iter-a <id_a> --iter-b <id_b>

# 检测优化是否停滞（连续 N 次无改善）
PYTHONPATH=. .venv/bin/python scripts/manage_iterations.py plateau --window 3

# 回滚到某次迭代
PYTHONPATH=. .venv/bin/python scripts/manage_iterations.py rollback --iteration-id <iter_id>
```

---

## 快速参考：单次迭代命令序列

```bash
cd /data2/shamo/project/test_skill/workspace

# 1. 开始
PYTHONPATH=. .venv/bin/python scripts/manage_iterations.py start --description "xxx"

# 2. 增强（首次或策略变更时，否则跳过）
PYTHONPATH=. .venv/bin/python scripts/run_augmentation.py --all

# 3. 训练
PYTHONPATH=. .venv/bin/python scripts/run_training.py --tag iter_001

# 4. 评估
PYTHONPATH=. .venv/bin/python scripts/run_inference.py \
  --model-dir models/iterations/iter_001 --evaluate --tag iter_001

# 5. 分析
PYTHONPATH=. .venv/bin/python scripts/collect_metrics.py --log-dir logs/training/
PYTHONPATH=. .venv/bin/python scripts/analyze_results.py --test-set data/augmented/test_aug.jsonl

# 6. 反馈
PYTHONPATH=. .venv/bin/python scripts/record_feedback.py label \
  --iteration-id iter_001 --cer-before 0.15 --cer-after 0.09
```

---

## 决策原则

1. **增强数据是前提**：训练/推理必须用增强数据，但增强本身只在需要时执行
2. **优先级排序**：数据质量 > 训练参数 > 模型结构 > 后处理
3. **渐进式优化**：每次迭代只调整一个主要维度，便于归因
4. **可回滚性**：所有变更通过版本化配置实现，支持任意版本回滚
5. **数据驱动**：所有建议基于量化分析结果（CER/WER），避免主观判断
6. **闭环映射**：badcase 错误模式通过 `badcase_strategy_map.DATA_STRATEGIES` 自动映射到优化手段
