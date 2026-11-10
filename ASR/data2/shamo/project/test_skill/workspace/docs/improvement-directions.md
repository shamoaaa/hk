# 系统改进方向（基于 ASR 教程对比分析）

> 基于 asr-tutorial-summary.md（21 章）与当前系统架构的差距分析
> 更新时间：2026-05-21

---

## 改进分层

```
P0 — 改配置即可，零代码或极少代码，预期收益大
P1 — 需要开发新功能，中等工作量，明确收益
P2 — 架构级改进，需要较多开发，长期收益
P3 — 远期方向，当前阶段记录备查
```

---

## P0：立即可做

### 1. 启用 SpecAugment（Ch6/Ch9）

**现状**：配置中 `spec_aug.enabled: false`
**教程依据**：现代 ASR 标配，频率遮挡 2×27bins + 时间遮挡 2×40帧，等效正则化
**动作**：修改 `funasr_config.yaml`

```yaml
spec_aug:
  enabled: true
  num_t_mask: 2
  num_f_mask: 2
  max_t: 40
  max_f: 27    # 80-dim Fbank 的 ~33%
  max_w: 80
```

**预期**：相对 CER 改善 5-15%，防过拟合

---

### 2. 评测文本统一化（Ch4/Ch13/Ch20）

**现状**：`analyze_results.py` 直接对比 reference 和 hypothesis，无预处理
**教程依据**：Ch13 明确指出"TN 是隐形杀手，同一模型不同规范 CER 可差很多"；Ch20 给出评测前统一化清单
**动作**：在 `result_analyzer.py` 的 CER/WER 计算前加入统一化

```python
def normalize_for_eval(text: str) -> str:
    """评测前统一化"""
    text = unicodedata.normalize("NFKC", text)   # Unicode 标准化
    # 全角→半角
    text = ''.join(
        chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c
        for c in text
    )
    text = text.lower()                           # 英文小写
    text = re.sub(r'[\s\u3000]+', '', text)       # 去空格（中文场景）
    text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)  # 去标点
    return text
```

**预期**：锁定评测口径，后续迭代对比才有意义

---

### 3. 切换 Dynamic Batching（Ch14）

**现状**：`batch_type: example`，固定 batch_size=16
**教程依据**：Ch14 推荐按长度分桶，减少 padding 浪费，GPU 利用率提升 20-40%
**动作**：修改 `funasr_config.yaml`

```yaml
training:
  batch_type: length
  max_token_length: 2000
  batch_size: 16000    # token 总数
```

**预期**：训练速度提升，OOM 风险降低

---

## P1：需要开发

### 4. Speed Perturbation 增强（Ch5）

**现状**：增强策略只有 babble/music/noise/reverb
**教程依据**：Ch5 称 Speed Perturb（0.9/1.0/1.1）是"波形增广三件套"之一，等效 3 倍数据量
**动作**：
- `funasr_config.yaml` 增加 speed 策略
- `NoiseAugmentor` 实现 speed perturbation（torchaudio/sox）

```yaml
augmentation:
  strategies:
    - type: speed
      probability: 1.0
      speed_factors: [0.9, 1.0, 1.1]
```

**预期**：相对 CER 改善 10-20%

---

### 5. 数据质量检查增强（Ch3/Ch5）

**现状**：`analyze_dataset.py` 存在但覆盖不全
**教程依据**：Ch3 强调"数据准备 40%"，Ch5 列出音频清洗清单
**动作**：增加检查项

| 检查项 | 来源 | 阈值 |
|--------|------|------|
| 音频时长分布 | Ch3 | <0.5s 或 >30s 标记 |
| 采样率一致性 | Ch5/Ch6 | 必须全部 16kHz |
| 削波检测 | Ch5 | >1% 采样点削波则丢弃 |
| 文本为空/极短 | Ch3 | 空或 <2 字标记 |
| source_len 与实际时长一致性 | Ch3 | 偏差 >10% 标记 |
| DC Offset | Ch5 | 均值偏移 >0.01 标记 |

---

### 6. 训练诊断增强（Ch7/Ch9/Ch21）

**现状**：`auto_diagnose.py` 只检查 loss 下降幅度和停滞
**教程依据**：Ch21 FAQ 给出完整诊断树
**动作**：增加诊断规则

| 现象 | 诊断 | 推荐动作 |
|------|------|---------|
| Loss 降但 CER 不降 | 过拟合 / TN 不一致 | 检查 val_loss gap；检查评测统一化 |
| hypothesis 出现重复片段 | Attention 对齐问题 | 增加 CTC 权重 / Length Penalty |
| deletion 占比突增 | 模型吃字 | 检查 max_token_length / 增加 epoch |
| substitution 集中在特定字 | 同音混淆 | 提取热词 / 增加该类数据 |
| train_loss vs val_loss gap > 2x | 过拟合 | 增加 SpecAugment / 减小模型 / 增加数据 |

---

### 7. 后处理管线完善（Ch4/Ch19）

**现状**：post_process 只有 hotword + 一条去空格 regex + lm_rerank（未启用）
**教程依据**：Ch19 给出完整 ITN 管线
**动作**：分阶段增加

Phase 1（regex 规则）：
- NFKC 标准化
- 全角→半角
- 连续空格合并
- 基础数字规范化（一二三→123，可选）

Phase 2（独立模块）：
- 标点恢复（Punctuation Model，FunASR 有现成的 ct-punc）
- 繁简统一（OpenCC t2s）

---

### 8. RIR 真实混响数据（Ch5/Ch15）

**现状**：reverb 策略用合成 RT60
**教程依据**：Ch5/Ch15 推荐 RIR_noises 数据集做真实卷积，"MUSAN+RIR 不可跳过"
**动作**：
- 下载 RIR_noises 数据集
- 修改 reverb 策略为 RIR 卷积模式

---

### 9. 热词防过度偏置（Ch17）

**现状**：热词直接替换，无置信度门控
**教程依据**：Ch17 强调"必须有置信度门控"，Over-biasing 需 Negative Sampling
**动作**：
- 热词匹配增加置信度阈值（音素编辑距离 < N 才替换）
- 评测增加 B-WER（热词相关 WER）和 FPR（误触发率）

---

## P2：架构级改进

### 10. Warmup 策略自适应（Ch9）

**现状**：固定 `warmup_steps: 50`
**教程依据**：Ch9 "无 Warmup → NaN"，建议 warmup = 总步数的 5-10%
**动作**：`apply_training_config.py` 增加自动计算逻辑

```python
total_steps = num_samples / batch_size * max_epoch
warmup_steps = max(50, int(total_steps * 0.05))
```

---

### 11. 无真值监控指标（Ch18）

**现状**：只有 CER/WER（需要 reference）
**教程依据**：Ch18 生产环境需要无真值监控
**动作**：在 `analyze_results.py` 增加辅助指标

| 指标 | 计算方式 | 异常阈值 |
|------|---------|---------|
| 空结果率 | hypothesis 为空的比例 | > 5% |
| 时长比 | 字数 / 音频秒数 | < 1 或 > 8 字/秒 |
| 重复率 | hypothesis 中重复 3+ 字片段 | > 10% 样本 |
| 平均置信度 | 模型输出 confidence（如有） | < 0.5 |

---

### 12. 过拟合检测（Ch2/Ch21）

**现状**：只看 train_loss，不对比 val_loss
**教程依据**：Ch21 Q2 "Loss 降 WER 不降 = 过拟合"
**动作**：
- `run_training.py` 同时记录 val_loss 到 `training_metrics.jsonl`
- `auto_diagnose.py` 增加 train/val gap 检测

---

### 13. 强制对齐验证（Ch3/Ch5）

**现状**：无
**教程依据**：Ch3 "强烈建议做强制对齐"，Ch5 "MFA 工业界首选"
**动作**：
- 对 badcase 中 CER > 50% 的样本，用 CTC forced alignment 验证标注是否正确
- 标注错误的样本标记为 dirty，从训练集移除或修正

---

## P3：远期方向

| 方向 | 教程章节 | 触发条件 |
|------|---------|---------|
| LM Shallow Fusion | Ch7 | substitution 为主且热词无法覆盖时 |
| SSL 微调（WavLM） | Ch9/Ch16 | CER 卡住，模型能力瓶颈 |
| 向量记忆检索 | Ch17 + improvement-ideas.md | 迭代 >10 轮，历史经验足够多 |
| Speaker Diarization | Ch10/Ch11 | 业务需要多说话人场景 |
| 流式部署 | Ch8/Ch18 | 产品需要实时转写 |
| MoE / Adapter | Ch12 | 多领域/多语种扩展 |
| MLLM 集成 | Ch16/Ch17 | 需要语义理解能力 |
| 数据闭环 | Ch18 | 有线上流量，Hard Example Mining |

---

## 执行优先级总结

```
本轮迭代立即做：
  [P0-1] 启用 SpecAugment
  [P0-2] 评测文本统一化
  [P0-3] 切换 Dynamic Batching

下一轮迭代前完成：
  [P1-4] Speed Perturbation
  [P1-7] 后处理管线 Phase 1（NFKC + 全角半角 + 基础 regex）

持续改进（边用边做）：
  [P1-5] 数据质量检查
  [P1-6] 训练诊断增强
  [P1-9] 热词置信度门控

CER 卡住时启动：
  [P2-12] 过拟合检测
  [P2-13] 强制对齐验证
  [P1-8] RIR 真实混响
```

---

## 教程核心法则（贯穿所有改进）

> "数据质量提升的收益永远高于模型结构微调；WER 卡住时去听 Bad Cases 检查清洗脚本。"
> — Chapter 21 最终法则

> "遇到问题先查数据（空值/长度/TN）→ 再查配置（梯度裁剪/LR）→ 最后看模型结构。"
> — Chapter 21 诊断顺序
