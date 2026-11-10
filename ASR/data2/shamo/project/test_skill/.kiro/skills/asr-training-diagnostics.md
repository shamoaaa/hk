# ASR 训练诊断

## 能力描述

分析训练过程异常和瓶颈，基于 `auto_diagnose.py` 的输出生成优化建议。

## 何时使用

- 训练完成后评估训练质量
- CER 不降或恶化时排查原因
- 需要决定下一步优化方向时

## 执行步骤

```powershell
# 自动诊断（读取 training_metrics.jsonl + 最新 eval_report）
.venv_win\Scripts\python -m scripts.auto_diagnose --verbose

# 训练指标分析
.venv_win\Scripts\python -m scripts.collect_metrics --log-dir logs/training/
```

## 诊断规则表

| 条件 | 诊断 | 推荐动作 |
|------|------|---------|
| loss 下降 < 5% | 训练不充分 | 增加 epoch 或提高 lr |
| loss 仍在下降（尾部 20%） | 可继续训练 | 增加 epoch |
| 出现 NaN | 梯度爆炸/数据异常 | lr 降至 1/10 |
| val/train loss gap > 100% | 过拟合 | 增加 SpecAugment/数据增强/减 epoch |
| deletion 占比 > 40% | 模型吃字 | 检查 max_token_length / 增加 epoch |
| substitution 占比 > 70% | 同音混淆 | 提取热词 |
| 重复率 > 10% | Attention 对齐问题 | 增加 CTC 权重 |
| 连续 3 轮 CER 无改善 | 优化停滞 | 切换优化维度 |
| Loss 降但 CER 不降 | 过拟合或 TN 不一致 | 检查评测统一化 |

## 训练参数参考

| 参数 | 当前值 | 调整建议 |
|------|--------|---------|
| batch_type | length | 按 token 总数分桶，减少 padding |
| batch_size | 16000 | token 总数（非样本数） |
| learning_rate | 5e-5 | 精细调优阶段 |
| warmup_steps | 50 | 建议 = 总步数 × 5% |
| spec_aug | enabled | 防过拟合标配 |
| gradient_clip | 3.0 | NaN 时降至 1.0 |

## 常见问题

- **NaN 排查顺序**：先查数据（空值/长度违例）→ 再查 lr → 最后查梯度裁剪
- **Loss 降 CER 不降**：最常见原因是评测 TN 不一致（已通过 normalize_for_eval 修复）
- **OOM**：切换 batch_type=length 后通常解决；仍 OOM 则减小 max_token_length

## 知识库引用

查询历史诊断经验，了解类似问题的解决方案：

```powershell
# 查询训练参数相关的历史经验
.venv_win\Scripts\python -m scripts.manage_knowledge query --type param_effect

# 查询特定场景的解决方案
.venv_win\Scripts\python -m scripts.manage_knowledge query --type scene_solution

# 诊断完成后记录有效经验
.venv_win\Scripts\python -m scripts.manage_knowledge add --type param_effect --content "{\"diagnosis\":\"overfitting\",\"action\":\"enable_specaug\",\"cer_delta\":-0.005}" --confidence 0.8 --source-iteration iter_xxx
```
