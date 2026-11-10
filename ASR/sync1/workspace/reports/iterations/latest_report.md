# 迭代执行报告

**生成时间**: 2026-05-09 08:01:43

## 1. 迭代信息

| 字段 | 值 |
|------|-----|
| 迭代 ID | `iter_b81a1b84` |
| 描述 | CPU test iteration - full pipeline validation |
| 状态 | pending |
| 创建时间 | 2026-05-07T06:57:27.470154+00:00 |
| 完成时间 | None |

## 2. 训练指标

| 指标 | 起始值 | 最终值 | 变化 |
|------|--------|--------|------|
| train_loss | 5.200000 | 3.330000 | -1.870000 |
| val_loss | 5.500000 | 3.630000 | -1.870000 |
| learning_rate | 0.000100 | 0.000050 | -0.000050 |

总步数: 10, 总 epoch: 6

## 3. 评估结果

| 指标 | 值 |
|------|-----|
| 整体 CER | N/A |
| 整体 WER | N/A |

### 场景级指标

| 场景 | CER | WER | 样本数 |
|------|-----|-----|--------|
| unknown | 0.0 | 0.0 | 5 |

### 错误类型分布

| 类型 | 数量 |
|------|------|
| substitution | 0 |
| insertion | 0 |
| deletion | 0 |


## 4. 前后对比

*迭代记录中无 CER before/after 数据*

## 5. 执行的优化操作

### 当前配置版本

| 维度 | 最新快照 |
|------|---------|
| model | `无快照` |
| training | `无快照` |
| augmentation | `v005_augment.yaml` |
| post_process | `v002_postprocess.yaml` |

## 6. 知识库状态

### 最近更新的知识条目（Top 5）

| ID | 类型 | 置信度 | 状态 | 来源迭代 | 更新时间 |
|-----|------|--------|------|---------|---------|
| ke_76d41ff0 | data_strategy | 0.7 | active | seed | 2026-05-09T02:27 |
| ke_4972f848 | data_strategy | 0.65 | active | seed | 2026-05-09T02:27 |
| ke_6a14ff3a | data_strategy | 0.6 | active | seed | 2026-05-09T02:27 |
| ke_ea632dac | data_strategy | 0.75 | active | seed | 2026-05-09T02:27 |
| ke_4b4dc10a | param_effect | 0.7 | active | seed | 2026-05-09T02:27 |

### 知识条目统计

| 状态 | 数量 |
|------|------|
| active | 8 |

## 7. Skill 进化状态

*尚无策略权重数据（需要至少完成一轮反馈后触发学习）*

### 反馈统计

- 总反馈数: 1
- 正面反馈: 1 (100%)
- 负面反馈: 0 (0%)
- 本次迭代反馈: 1 条

---
*报告由 `generate_iteration_report.py` 自动生成*