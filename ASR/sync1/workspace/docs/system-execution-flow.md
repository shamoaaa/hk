# ASR 训练优化系统 — 执行流程文档

## 系统概述

本系统是一个基于 FunASR Paraformer 的 ASR 模型训练优化闭环系统。用户通过与 Kiro 对话触发优化流程，Kiro 通过 **Skill 文件**（`.kiro/skills/`）和 **Steering 文件**（`.kiro/steering/`）引导决策，调用 Python CLI 脚本执行具体操作，形成"增强 → 训练 → 评估 → 分析 → 优化 → 反馈"的自动化迭代闭环。

**运行环境**：所有脚本在 `workspace/` 目录下执行，需设置 `PYTHONPATH=.`，使用 `.venv/bin/python`。

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                         用户与 Kiro 对话                              │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Kiro Skill / Steering 层                          │
│  .kiro/skills/                        .kiro/steering/               │
│  ├── asr-training-execution.md        ├── asr-optimization-        │
│  ├── asr-training-diagnostics.md      │   workflow.md              │
│  ├── asr-inference-evaluation.md      ├── training-analysis-       │
│  ├── asr-badcase-analysis.md          │   guide.md                 │
│  ├── asr-data-optimization.md         ├── badcase-analysis-        │
│  ├── asr-model-tuning.md             │   guide.md                  │
│  └── asr-postprocess-rules.md         └── data-quality-guide.md   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ 调用 CLI 脚本
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Python CLI 脚本层                                │
│  scripts/                                                           │
│  ├── run_augmentation.py  → execution/funasr_noise_augment.py      │
│  ├── run_training.py      → execution/funasr_trainer.py            │
│  ├── run_inference.py     → execution/funasr_evaluator.py          │
│  ├── collect_metrics.py   → analysis/training_monitor.py           │
│  ├── analyze_results.py   → analysis/result_analyzer.py            │
│  ├── analyze_dataset.py   → analysis/dataset_analyzer.py           │
│  ├── preprocess_metadata.py (音频元数据)                             │
│  ├── extract_hotwords.py  (热词挖掘)                                │
│  ├── apply_model_config.py / apply_training_config.py              │
│  ├── apply_postprocess.py                                          │
│  ├── record_feedback.py / manage_iterations.py                     │
│  ├── manage_knowledge.py / seed_strategies.py                      │
│  └── decision/badcase_strategy_map.py (闭环映射表)                   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      数据与持久化层                                    │
│  configs/funasr_config.yaml (唯一主配置)                             │
│  configs/{model,training,augmentation,post_process}/ (版本快照)      │
│  data/ (原始jsonl + augmented/ + musan/)                            │
│  logs/training/training_metrics.jsonl (实时训练指标)                  │
│  models/iterations/<tag>/ (模型产物)                                 │
│  reports/evaluation/ (评估报告)                                      │
│  metadata.db (SQLite 元数据)                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 完整自动化训练流程

### 前置条件

```
workspace/
├── data/
│   ├── train.jsonl          # FunASR 格式，source 为绝对路径指向 wav
│   ├── val.jsonl
│   ├── test.jsonl
│   ├── musan/               # MUSAN 噪声数据集 (speech/music/noise)
│   └── augmented/           # 空目录，等待增强产出
├── configs/funasr_config.yaml  # 已配置好的主配置
└── .venv/                   # 虚拟环境 (torch + funasr + scipy + soundfile)
```

---

### Step 1: 数据增强（首次必须，后续按需）

**业务规则**：`augmentation.require_augmentation: true` — 训练和推理脚本强制要求使用增强后的数据。但增强本身不需要每轮迭代都执行——只要 `data/augmented/train_aug.jsonl` 已存在，后续训练直接使用即可。

**何时需要重新增强**：
- 首次运行（`data/augmented/` 为空）
- 分析结果表明需要调整增强策略（如 badcase 分析发现插入错误多，需要加大 babble 增强比例）
- 原始数据发生变化（新增了训练样本）

```bash
cd workspace
PYTHONPATH=. .venv/bin/python scripts/run_augmentation.py --all
```

**内部流程**：

```
run_augmentation.py
  → 读取 configs/funasr_config.yaml
  → 创建 NoiseAugmentor(config, base_dir=".")
  → augmentor.prepare_noise_data()
      → 加载 data/musan/{speech,music,noise}/*.wav 索引
  → 对 train/val/test 三份 jsonl 分别执行 augment_dataset():
      → 逐条读取原始 jsonl
      → 对每条样本按 strategies 列表中的 probability 抽样:
          babble(0.4) / music(0.2) / noise(0.2) / reverb(0.15)
      → 生成增强 wav → data/augmented/<key>_aug_<type>.wav
      → 原始条目标记 scene_type="clean"
      → 增强条目标记 scene_type=对应类型 (background_voice/background_noise/...)
      → 写入 data/augmented/{split}_aug.jsonl
```

**产物**：

```
data/augmented/
├── train_aug.jsonl    # 原始 2000 条 + 增强 ~1500 条 ≈ 3500 条
├── val_aug.jsonl
├── test_aug.jsonl
└── *_aug_babble.wav / *_aug_noise.wav / *_aug_reverb.wav ...
```

---

### Step 2: 启动训练

```bash
PYTHONPATH=. .venv/bin/python scripts/run_training.py --tag iter_001
```

**内部流程**：

```
run_training.py
  → 读取 configs/funasr_config.yaml
  → 路径校验: train_data 必须是 data/augmented/train_aug.jsonl（或位于 augmented_dir 下）
  → 创建 Trainer(config, base_dir=".")
  → trainer.train(train_jsonl, val_jsonl, tag="iter_001"):
      → 构建 funasr.bin.train_ds 命令:
          --train_set data/augmented/train_aug.jsonl
          --valid_set data/augmented/val_aug.jsonl
          --output_dir models/iterations/iter_001
          --gpu_id 4
          --max_epoch 3
          --batch_size 32 (token mode)
          --learning_rate 5e-5
          ...
      → subprocess.Popen 启动训练
      → 逐行读取 stdout:
          → 写入 logs/training/iter_001/log.txt (原始日志)
          → FunASRLogParser.parse_line(line):
              正则匹配 epoch/step/loss/lr/grad_norm
              → TrainingMetrics dataclass
          → append_jsonl(metrics, "logs/training/training_metrics.jsonl")
      → 等待训练完成
```

**产物**：

```
models/iterations/iter_001/       # 模型 checkpoint
logs/training/iter_001/log.txt    # FunASR 原始日志
logs/training/training_metrics.jsonl  # 结构化训练指标（实时追加）
```

**训练日志实时解析流**：

```
funasr.bin.train_ds (subprocess stdout)
    │
    ├──→ logs/training/iter_001/log.txt (raw copy)
    │
    └──→ FunASRLogParser.parse_line()
            │
            └──→ TrainingMetrics {step, epoch, train_loss, val_loss, lr, grad_norm, cer, wer}
                    │
                    └──→ logs/training/training_metrics.jsonl (append)
```

---

### Step 3: 推理与评估

```bash
PYTHONPATH=. .venv/bin/python scripts/run_inference.py \
  --model-dir models/iterations/iter_001 --evaluate --tag iter_001
```

**内部流程**：

```
run_inference.py
  → 路径校验: test_data 必须是增强后数据
  → 创建 Evaluator(config, base_dir=".")
      → _build_post_processor(config):
          从 config["post_process"] 构造 PostProcessor
          加载热词表 + 正则规则 + LM rerank 配置
  → evaluator.evaluate(model_dir, test_jsonl, tag):
      → FunASR AutoModel 加载模型
      → 逐条推理:
          raw_hyp = model.generate(audio)
          hypothesis = post_processor.process(raw_hyp)  # 后处理
          保留 hypothesis_raw 和 hypothesis 两个字段
      → 委托 ResultAnalyzer 计算:
          - 整体 CER/WER
          - 按 scene_type 分组的 CER/WER
          - badcase 列表 (CER > threshold)
          - 错误类型分布 (substitution/insertion/deletion)
      → 写入 reports/evaluation/eval_report_iter_001.json
```

**关键设计**：推理结果**先经过后处理再计算 CER/WER**，同时保留 `hypothesis_raw` 用于对比后处理效果。CER/WER 计算逻辑集中在 `result_analyzer.py`，不在 evaluator 中重复实现。

---

### Step 4: 训练过程分析

```bash
PYTHONPATH=. .venv/bin/python scripts/collect_metrics.py --log-dir logs/training/
```

**内部流程**：

```
collect_metrics.py
  → 读取 logs/training/training_metrics.jsonl
  → TrainingMonitor:
      → compute_epoch_metrics(): 每 epoch 平均 CER/WER/loss
      → check_stagnation(threshold, window=5): 连续 N epoch loss 未下降
      → check_numerical_anomaly(): NaN/Inf 检测
      → get_training_curves(): loss/lr/cer/wer 曲线数据
  → 输出 JSON 到 stdout
```

---

### Step 5: 数据集质量分析（基于增强后数据）

```bash
PYTHONPATH=. .venv/bin/python scripts/analyze_dataset.py \
  --dataset-path data/augmented/train_aug.jsonl
```

或追加元数据后分析：

```bash
PYTHONPATH=. .venv/bin/python scripts/preprocess_metadata.py --all
# 产出 data/augmented/train_aug_meta.jsonl（追加 duration/snr/energy/zcr）
```

---

### Step 6: Badcase 分析 → 策略映射（闭环核心）

Kiro 读取评估报告后，调用 `badcase_strategy_map.match_strategies(report)` 自动匹配优化手段：

| 错误模式 | 触发策略 | 推荐命令 |
|---------|---------|---------|
| 插入错误 > 替换错误 | `babble_augment_for_insertion` | `run_augmentation.py --all --strategy babble` |
| 替换错误为主 | `hotwords_from_substitution` | `extract_hotwords.py --merge` |
| CER > 0.10 + 替换错误 | `specaug_for_substitution` | 启用 `training.spec_aug.enabled` |
| 删除错误为主 | `reverb_speed_for_deletion` | `run_augmentation.py --all --strategy reverb/speed` |
| 整体 CER > 0.20 | `full_augment_for_high_cer` | `run_augmentation.py --all` |
| 某场景样本 < 20 | `balance_scene_augment` | 针对性增强 |

这张映射表定义在 `scripts/decision/badcase_strategy_map.py` 的 `DATA_STRATEGIES` 列表中，可通过 `seed_strategies.py` 注入知识库。

---

### Step 7: 执行优化（配置变更）

根据分析结果，Kiro 调用对应的 apply 脚本：

```bash
# 模型结构变更
PYTHONPATH=. .venv/bin/python scripts/apply_model_config.py apply \
  --attention-heads 8 --seaco-layers 3

# 训练参数变更
PYTHONPATH=. .venv/bin/python scripts/apply_training_config.py apply \
  --learning-rate 1e-4 --lr-schedule cosine_annealing

# 后处理规则变更
PYTHONPATH=. .venv/bin/python scripts/apply_postprocess.py update-config \
  --hotword-file configs/post_process/hotwords.txt
```

**每个 apply 脚本的三重一致性**：
1. 更新 `configs/funasr_config.yaml` 对应段落
2. 保存带版号的快照到 `configs/{model|training|augmentation|post_process}/v00N_*.yaml`
3. 记录变更到 SQLite

---

### Step 8: 效果反馈与 Skill 进化

```bash
# 记录反馈
PYTHONPATH=. .venv/bin/python scripts/record_feedback.py label \
  --iteration-id iter_001 --cer-before 0.12 --cer-after 0.09

# 触发 Skill 学习
PYTHONPATH=. .venv/bin/python scripts/manage_knowledge.py trigger-learning \
  --start-time 2026-01-01 --end-time 2026-12-31
```

**OptimizerSystem 自动审查**：
- 每次迭代完成 → `_auto_review_skills()`: 更新反馈标记、知识条目生命周期转换、更新速查表
- 每 5 次迭代 → `_deep_review_skills()`: 合并相似条目、淘汰无效规则、重算策略权重

---

## 单次完整迭代命令序列

```bash
cd workspace

# 0. 开始迭代
PYTHONPATH=. .venv/bin/python scripts/manage_iterations.py start --description "优化背景人声 CER"

# 1. 数据增强（仅首次或策略变更时需要，已有增强数据则跳过）
# PYTHONPATH=. .venv/bin/python scripts/run_augmentation.py --all

# 2. 训练（直接使用已有的增强数据）
PYTHONPATH=. .venv/bin/python scripts/run_training.py --tag iter_001

# 3. 评估
PYTHONPATH=. .venv/bin/python scripts/run_inference.py \
  --model-dir models/iterations/iter_001 --evaluate --tag iter_001

# 4. 分析
PYTHONPATH=. .venv/bin/python scripts/collect_metrics.py --log-dir logs/training/
PYTHONPATH=. .venv/bin/python scripts/analyze_results.py --test-set data/augmented/test_aug.jsonl

# 5. 反馈
PYTHONPATH=. .venv/bin/python scripts/record_feedback.py label \
  --iteration-id iter_001 --cer-before 0.15 --cer-after 0.09

# 6. 根据分析结果决定下一步：
#    - 需要调整增强策略 → 重新 run_augmentation.py（回到 Step 1）
#    - 需要调参/改模型 → apply_training_config.py / apply_model_config.py → 重新训练
#    - 需要后处理优化 → extract_hotwords.py + apply_postprocess.py → 重新评估
```

---

## 数据目录结构

```
workspace/data/
├── train.jsonl              # 原始训练集索引（2000 条，只读）
├── val.jsonl                # 原始验证集索引（200 条，只读）
├── test.jsonl               # 原始测试集索引（200 条，只读）
├── original/                # → 软链接到 aishell wav 目录
├── musan/                   # → 软链接到 MUSAN 噪声数据集
│   ├── speech/ (426 wav)    #   用于 babble 增强
│   ├── music/  (660 wav)    #   用于 music 增强
│   └── noise/  (930 wav)    #   用于 noise 增强
└── augmented/               # 增强产物（训练/推理实际使用）
    ├── train_aug.jsonl      #   原始 + 增强条目，每条带 scene_type
    ├── val_aug.jsonl
    ├── test_aug.jsonl
    └── *_aug_*.wav          #   增强后的音频文件
```

**jsonl 格式**（FunASR 标准 + 扩展字段）：

```json
{"key": "BAC009S0002W0122", "source": "/abs/path/to.wav", "source_len": 100, "target": "文本", "target_len": 15, "scene_type": "clean"}
{"key": "BAC009S0002W0122_aug_babble", "source": "/abs/path/augmented/xxx_aug_babble.wav", "source_len": -1, "target": "文本", "target_len": 15, "scene_type": "background_voice"}
```

---

## 配置架构

**唯一主配置**：`configs/funasr_config.yaml`

```yaml
model:       # 模型结构（attention_heads, hidden_dim, seaco_params）
data:        # 数据路径（train/val/test jsonl + augmented 路径）
augmentation: # 增强策略（strategies 列表 + require_augmentation 开关）
training:    # 训练超参（lr, batch_size, max_epoch, gpu_ids, spec_aug）
evaluation:  # 评估配置（metrics, badcase_threshold）
post_process: # 后处理（hotword_file, regex_rules, pipeline_order）
```

**版本快照**（只读，用于审计和回滚）：

```
configs/
├── model/v001_model.yaml ... v00N_model.yaml
├── training/v001_train.yaml ... v00N_train.yaml
├── augmentation/v001_augment.yaml ... v00N_augment.yaml
└── post_process/v001_postprocess.yaml ... v00N_postprocess.yaml
```

---

## 脚本目录结构

```
scripts/
├── [CLI 入口 - 16 个]         # 直接被 Kiro Skill 调用
├── core/                      # 基础设施
│   ├── models.py              #   全部数据类
│   ├── database.py            #   SQLite 封装
│   ├── config_utils.py        #   配置读写 + 版本快照
│   └── error_handlers.py      #   异常处理器
├── analysis/                  # 分析层
│   ├── training_monitor.py    #   训练指标采集 + 异常检测
│   ├── result_analyzer.py     #   CER/WER/badcase（唯一计算点）
│   ├── dataset_analyzer.py    #   数据质量分析
│   └── funasr_log_analyzer.py #   FunASR 日志行解析器
├── execution/                 # 执行层
│   ├── funasr_trainer.py      #   subprocess 启动训练 + 实时日志解析
│   ├── funasr_evaluator.py    #   推理 + 后处理 + 委托 ResultAnalyzer
│   ├── funasr_noise_augment.py #  MUSAN 真实音频增强（6 种策略）
│   ├── model_optimizer.py     #   模型结构配置管理
│   ├── param_tuner.py         #   训练超参管理
│   └── post_processor.py      #   热词/正则/LM rerank
└── decision/                  # 决策层
    ├── optimizer_system.py    #   优化闭环协调器 + Skill 自动审查
    ├── llm_skill_engine.py    #   数据准备 + 方案解析
    ├── skill_evolution.py     #   Skill 进化引擎
    ├── skill_knowledge_base.py #  知识库管理
    └── badcase_strategy_map.py #  badcase → 优化手段映射表
```

---

## 设计原则

1. **增强数据是训练/推理的前提**：所有训练和推理必须使用增强后数据（业务规则强制），但增强本身只在首次或策略变更时执行，已有增强数据则直接复用
2. **场景标签在增强时确定**：babble→background_voice, noise→background_noise, 原始→clean
3. **CER/WER 计算唯一入口**：`result_analyzer.py`，evaluator 不重复实现
4. **推理自动应用后处理**：evaluator 构造时加载 PostProcessor，每条 hypothesis 都经过处理
5. **配置单源**：`funasr_config.yaml` 是唯一可修改源，版本快照只读
6. **日志实时解析**：训练时 stdout 逐行经 FunASRLogParser 转为 TrainingMetrics，无需事后转换
7. **闭环映射驱动**：badcase 错误模式通过 `DATA_STRATEGIES` 表自动映射到优化手段
8. **Skill 自我进化**：每次迭代后自动审查，每 5 次深度审查，知识条目有生命周期管理
