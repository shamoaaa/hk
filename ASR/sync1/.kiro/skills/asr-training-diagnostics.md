# ASR 训练诊断

## 能力描述

分析 ASR 模型训练过程中的异常和瓶颈，生成诊断报告和优化建议。基于训练指标数据（loss、学习率、梯度范数、CER/WER）进行模式识别和原因诊断。

## 数据格式说明

本项目训练数据使用 **FunASR jsonl 格式**，仅含 `key`、`source`、`source_len`、`target`、`target_len` 字段。因此诊断逻辑**不依赖 scene_type 字段**，无法直接按场景（纯净语音/背景人声/背景噪声）分组诊断。场景相关诊断需要通过 badcase 文本特征间接推断。

## 数据来源

训练指标实时写入 `logs/training/training_metrics.jsonl`。通过 CLI 脚本分析：

```bash
python scripts/collect_metrics.py --log-dir logs/training/
```

## 分析规则

### 训练停滞诊断

当训练 loss 连续 5 个 epoch 未下降超过阈值时：

1. **检查学习率是否过小**（< 1e-6）
   - 建议：提高学习率至 1e-5 ~ 1e-4 范围
   - 或切换学习率调度策略（如从 step_decay 切换到 cosine_annealing）
   - 优先级：5，置信度：0.8

2. **检查梯度范数是否过小**（< 1e-4）
   - 可能陷入局部最优
   - 建议：增加学习率 warmup 步数，或增加梯度噪声
   - 优先级：4，置信度：0.7

3. **检查验证 loss 是否同步停滞**
   - 如果验证 loss 仍在下降：可能是训练集过拟合
   - 如果验证 loss 也停滞：可能是模型容量不足或数据质量问题
   - 优先级：4，置信度：0.75

4. **检查 batch size 是否过大**
   - 过大的 batch size 可能导致泛化能力下降
   - 建议：减小 batch size 以增加梯度噪声
   - 优先级：3，置信度：0.6

### 背景人声优化诊断

**检测方式**：由于训练数据中没有 scene_type 字段，无法直接按场景分组计算 CER。此诊断通过**评估报告中的 badcase 文本特征**间接触发：

当整体 CER 高且 badcase 中存在较多插入/替换错误时（可能由背景人声干扰导致）：

1. **检查训练数据是否使用了 babble 增强**
   - 查看 `data/augmented/augmented.jsonl` 是否存在
   - 如果未使用：建议执行 `python scripts/run_augmentation.py --dataset-path data/train.jsonl --strategy babble`
   - 优先级：5，置信度：0.75

2. **检查 SeACo 模块参数**
   - 建议调整 co-decoder 的注意力权重（参见 `configs/funasr_config.yaml` 中 `seaco_params`）
   - 增加 SeACo 模块的层数（如从 2 层增加到 3 层）
   - 优先级：4，置信度：0.7

3. **检查是否启用语音增强前端**
   - 如果未启用：建议添加 speech enhancement 模块（如 Conv-TasNet）
   - 配置建议：channels=64, layers=4
   - 优先级：4，置信度：0.75

### 数值异常诊断

当检测到 NaN/Inf 值时：

1. **立即暂停训练**，保存当前状态
2. **检查最近的学习率变化**：是否在学习率跳变后出现
3. **检查梯度范数历史**：是否有梯度爆炸趋势
4. **建议操作**：
   - 降低学习率至当前值的 1/10
   - 增加梯度裁剪阈值（如设为 1.0）
   - 回滚到上一个有效 checkpoint
   - 优先级：5，置信度：0.9

### 过拟合诊断

当训练 loss 持续下降但验证 loss 开始上升时：

1. **增加正则化**：提高 weight_decay（如从 0.01 增加到 0.05）
2. **数据增强**：增加训练数据多样性
3. **减小模型容量**：减少注意力头数或隐藏层维度
4. **早停策略**：设置 patience 参数

## 知识库引用

分析时参考 `knowledge/knowledge_base.json` 中的历史诊断经验，特别是：
- 相似训练停滞场景的成功解决方案
- 学习率调整与 CER 改善的历史关联
- 背景人声场景优化的有效策略

查询知识库中的历史诊断经验：

```bash
# 查询训练参数相关的历史经验
python scripts/manage_knowledge.py query --type param_effect

# 查询特定场景的解决方案
python scripts/manage_knowledge.py query --type scene_solution --content-match '{"scene":"background_voice"}'
```

诊断完成后，将有效的诊断经验写入知识库：

```bash
python scripts/manage_knowledge.py add --type param_effect --content '{"param":"learning_rate","action":"increase_to_1e-4","effect":"CER improved 3%"}' --confidence 0.8 --source-iteration <iter_id>
```

## 输出格式

诊断报告应以 JSON 格式输出，包含：
```json
{
  "iteration_id": "iter_xxx",
  "diagnosis_summary": "诊断摘要",
  "suggestions": [
    {
      "dimension": "param|model|data|post_process",
      "title": "建议标题",
      "description": "详细描述",
      "priority": 1-5,
      "confidence": 0.0-1.0,
      "config_changes": {},
      "rationale": "建议理由"
    }
  ]
}
```
