# ASR 推理与评估

## 能力描述

使用 FunASR AutoModel 对训练好的模型进行推理，生成识别结果，并通过 CER/WER 指标评估模型性能。支持场景级别的性能分析。

## 业务强制约束（❗必读）

评估数据必须是**增强后的数据**。`run_inference.py` 内置 `require_augmentation` 校验，传原始 `test.jsonl` 会被拒绝。

正确的执行顺序：

```bash
# 1. 先生成增强数据
python scripts/run_augmentation.py --all

# 2. 推理/评估（默认读取 data/augmented/test_aug.jsonl）
python scripts/run_inference.py --model-dir models/iterations/iter_001 --evaluate --tag iter_001
```

## 推理流程

### 1. 运行推理

```bash
# 基本推理（test-data 默认从 funasr_config.yaml 读取）
python scripts/run_inference.py --model-dir models/iterations/iter_001 --output data/test_set.jsonl

# 指定测试数据
python scripts/run_inference.py --model-dir models/iterations/iter_001 --test-data data/test.jsonl --output data/test_set.jsonl
```

### 2. 运行评估（推理 + CER/WER 计算 + 报告生成）

```bash
python scripts/run_inference.py --model-dir models/iterations/iter_001 --evaluate --tag iter_001
```

### 3. 分析结果（badcase + 场景分析）

```bash
python scripts/analyze_results.py --test-set data/test_set.jsonl
```

## 完整评估流水线

```bash
# Step 1: 推理
python scripts/run_inference.py --model-dir models/iterations/iter_001 --output data/test_set.jsonl --evaluate --tag iter_001

# Step 2: Badcase 分析
python scripts/analyze_results.py --test-set data/test_set.jsonl

# Step 3: 查看训练指标
python scripts/collect_metrics.py --log-dir logs/training/
```

## 数据格式

### FunASR jsonl 格式（输入：train.jsonl / val.jsonl / test.jsonl）

```json
{"key": "BAC009S0002W0122", "source": "/abs/path/to/audio.wav", "source_len": 100, "target": "识别文本", "target_len": 5}
```

### 推理结果格式（输出：test_set.jsonl）

```json
{"audio_path": "/abs/path/to/audio.wav", "reference_text": "参考文本", "hypothesis_text": "识别结果", "scene_type": "unknown"}
```

evaluator 自动检测输入格式（FunASR jsonl 或 project test_set.jsonl）。

## 评估指标

- **CER**：字符错误率，中文 ASR 的主要指标
- **WER**：词错误率，中文按字分词计算
- **场景级 CER/WER**：按 scene_type 分组，用于定位薄弱场景

## 报告输出

评估报告保存在 `reports/evaluation/`：
- `eval_report_<tag>.json`：结构化 JSON 报告
- `eval_report_<tag>.txt`：可读文本报告（含错误样本列表）
