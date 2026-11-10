# ASR 数据优化策略

## 能力描述

基于 badcase 分析和数据质量检查，决定数据增强策略和数据清洗方向。

## 何时使用

- 迭代开始前决定增强策略
- badcase 分析后确定针对性增强方向
- 数据质量检查发现异常时

## 业务强制约束

train/val/test 必须经过增强后才能用于训练和评估。原始 jsonl 仅作为增强器输入。

## 当前增强策略

| 策略 | 概率 | 参数 | 适用场景 |
|------|------|------|---------|
| speed | 1.0 | 0.9/1.0/1.1 | 等效 3 倍数据，通用 |
| babble | 0.4 | SNR 5-20dB, 1-3人 | 插入错误多、背景人声 |
| music | 0.2 | SNR 5-25dB | 背景音乐干扰 |
| noise | 0.2 | SNR 5-25dB | 环境噪声 |
| reverb | 0.15 | 真实 RIR 卷积（60000+ RIR 文件） | 删除错误多、室内环境 |

reverb 策略优先使用 `data/RIRS_NOISES/` 中的真实/仿真 RIR 做卷积，无 RIR 文件时回退到合成指数衰减。

## 执行步骤

### 全量增强
```powershell
.venv_win\Scripts\python -m scripts.remote_runner run "python -m scripts.run_augmentation --all"
```

### 单策略增强
```powershell
.venv_win\Scripts\python -m scripts.remote_runner run "python -m scripts.run_augmentation --all --strategy babble"
```

### 数据质量检查
```powershell
# 独立运行（手动）
.venv_win\Scripts\python -m scripts.analyze_dataset --dataset-path data/train.jsonl --check-audio-exists
```

检测异常类型：empty_label、too_short(<0.5s)、too_long(>30s)、short_text(<2字)、duration_mismatch、missing_audio。

**注意**：`run_augmentation --all` 已内置增强前/后质量检查，输出中含 `pre_check` 和 `post_check`。独立运行 `analyze_dataset` 适用于首次部署数据或深入排查时。

## badcase → 增强策略映射

| 错误模式 | 推荐策略 | 优先级 |
|---------|---------|--------|
| insertion > substitution | babble + noise | 5 |
| substitution 为主 | 热词 + SpecAugment | 5 |
| deletion 为主 | reverb + speed | 4 |
| 整体 CER > 20% | 全量 --all | 4 |
| 某场景样本不足 | 针对性单策略增强 | 4 |

## 常见问题

- **增强数据已存在时**：`run_augmentation` 会复用已有文件（跳过），需删除旧文件才能重新生成
- **Speed Perturb 改变时长**：0.9x 变长、1.1x 变短，source_len 设为 -1（训练时重新计算）
- **增强后数据量**：原始 N 条 → 约 N×(1 + sum(probabilities)) 条（含原始 clean 样本）
- **增强前自动过滤**：empty_label、missing_source、short_text(<2字) 的样本会被自动跳过，不进入增强输出
- **RIR 数据**：`data/RIRS_NOISES/` 含 417 条真实 RIR + 60000 条仿真 RIR，reverb 策略自动加载

## 知识库引用

查询历史数据优化经验，了解哪些策略对哪类 badcase 有效：

```powershell
# 查询数据策略历史效果
.venv_win\Scripts\python -m scripts.manage_knowledge query --type data_strategy

# 查询特定策略的历史记录
.venv_win\Scripts\python -m scripts.manage_knowledge query --type data_strategy --content-match "{\"name\":\"babble_augment_for_insertion\"}"

# 迭代完成后记录策略效果
.venv_win\Scripts\python -m scripts.manage_knowledge add --type data_strategy --content "{\"name\":\"babble_augment\",\"cer_delta\":-0.003}" --confidence 0.85 --source-iteration iter_xxx
```
