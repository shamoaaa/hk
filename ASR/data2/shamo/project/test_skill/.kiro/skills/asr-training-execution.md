# ASR 训练执行

## 能力描述

使用 FunASR 框架执行 ASR 模型训练。基于 SeACo-Paraformer 模型进行微调，聚焦于背景人声场景优化。

## 业务强制约束（❗必读）

训练数据必须是**增强后的数据**。`run_training.py` 内置 `require_augmentation` 校验，传原始 `train.jsonl` 会直接报错并要求你先运行 `run_augmentation.py --all`。

正确的执行顺序：

```bash
# 1. 先生成增强数据（train/val/test 三份）
python scripts/run_augmentation.py --all

# 2. 再启动训练（默认读取 data/augmented/train_aug.jsonl + val_aug.jsonl）
python scripts/run_training.py --tag iter_001
```

如需临时跳过校验（仅限开发调试），在 `configs/funasr_config.yaml` 中设置：
```yaml
augmentation:
  require_augmentation: false
```

## 训练执行

```bash
# 基本训练
python scripts/run_training.py --train-data data/train.jsonl --val-data data/val.jsonl --tag iter_001

# 自定义参数训练
python scripts/run_training.py --train-data data/train.jsonl --val-data data/val.jsonl \
    --learning-rate 5e-5 --max-epoch 10 --batch-size 5000 --gpu-ids 4 --tag iter_002

# 查看训练命令（不执行）
python scripts/run_training.py --dry-run --train-data data/train.jsonl --val-data data/val.jsonl --tag iter_001
```

## 参数调优规则

### 学习率调整

| 情况 | 建议 |
|------|------|
| 训练 loss 不下降 | 提高学习率至 1e-4 ~ 5e-4 |
| 训练 loss 震荡 | 降低学习率至 1e-5 ~ 5e-5 |
| 训练 loss 下降后停滞 | 切换为 cosine annealing 调度 |
| 出现 NaN/Inf | 降低学习率至当前值的 1/10 |

### Batch Size 调整

| 情况 | 建议 |
|------|------|
| GPU 内存不足 | 减小 batch_size（如 5000 → 2000） |
| 训练速度过慢 | 增大 batch_size（如 5000 → 10000） |
| 泛化能力差 | 减小 batch_size 以增加梯度噪声 |

### Epoch 数调整

| 情况 | 建议 |
|------|------|
| 正常训练 | 10-50 epoch |
| 数据量小 | 减少 epoch 防止过拟合 |
| 数据增强后 | 适当增加 epoch |

## 配置文件

所有配置统一存储在 `configs/funasr_config.yaml`，包含：
- 模型配置（名称、结构参数、SeACo 参数）
- 训练超参数（学习率、batch size、epoch 数等）
- 数据增强配置（策略、噪声目录）
- 评估配置
- 后处理配置
- GPU 配置

修改参数后，系统会自动在 `configs/model/` 和 `configs/training/` 下保存版本快照。

## 输出

- 模型文件: `models/iterations/<tag>/`
- 训练日志: `logs/training/<tag>/log.txt`
- JSON 状态输出到 stdout

## 训练日志

训练过程中，FunASR 的输出会被实时解析并写入 `logs/training/training_metrics.jsonl`（项目统一格式）。原始 FunASR 日志保存在 `logs/training/<tag>/log.txt`。

训练完成后，使用 `collect_metrics.py` 分析指标：

```bash
python scripts/collect_metrics.py --log-dir logs/training/
```
