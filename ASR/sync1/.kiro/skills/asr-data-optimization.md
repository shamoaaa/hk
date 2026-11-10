# ASR 数据优化策略

## 能力描述

基于数据集分析和 badcase 分析结果，生成"前置数据优化手段"和"后处理手段"的建议。本 Skill 是闭环的关键节点，将 evaluation 发现的问题自动对应到下一轮迭代的数据/配置改动。

## 业务强制约束（❗必读）

本项目**强制要求 train / val / test 都必须经过增强后才能用于训练和评估**。原始 jsonl（`data/train.jsonl` 等）仅作为增强器的输入，**不得直接喂给 `run_training.py` 或 `run_inference.py`**。

- 训练和推理脚本内置校验（`augmentation.require_augmentation: true`），直接传原始数据会报错
- 每次迭代开始前，必须先执行：`python scripts/run_augmentation.py --all`
- 增强后的输出会自动写入 `data/augmented/{train,val,test}_aug.jsonl`

执行顺序：

```bash
python scripts/run_augmentation.py --all                    # 1. 生成增强数据
python scripts/run_training.py --tag iter_xxx               # 2. 训练（自动用增强数据）
python scripts/run_inference.py --model-dir ... --evaluate  # 3. 评估（自动用增强数据）
```

## 数据格式说明

原始数据使用 FunASR jsonl 格式，每条仅含 `key / source / source_len / target / target_len`。如果需要 SNR 过滤、场景分布分析、语速分析，必须先运行元数据预处理：

```bash
# 预处理，生成带 snr/duration/scene_type/zcr/energy 的增强 manifest
python scripts/preprocess_metadata.py --all
```

输出 `data/{train,val,test}_meta.jsonl`，包含字段：`duration`, `snr`, `energy_mean`, `energy_std`, `zero_crossing_rate`, `scene_type`。

## 前置数据优化手段（6 大类）

### 1. 噪声增强（基于 MUSAN）

```bash
python scripts/run_augmentation.py --all --strategy babble   # 背景人声
python scripts/run_augmentation.py --all --strategy music    # 背景音乐
python scripts/run_augmentation.py --all --strategy noise    # 环境噪声
python scripts/run_augmentation.py --all --strategy reverb   # 混响（scipy 生成）
```

| 策略 | 噪声源 | 默认 SNR 范围 | 默认概率 | 适用场景 |
|------|--------|--------------|---------|----------|
| babble | `data/musan/speech/` 1-3 人混合 | 5-20 dB | 0.4 | **插入错误多、背景人声场景首选** |
| music | `data/musan/music/` | 5-25 dB | 0.2 | 抗音乐干扰 |
| noise | `data/musan/noise/` | 5-25 dB | 0.2 | 提升噪声鲁棒性 |
| reverb | scipy 算法生成 | RT60 0.1-1.0s | 0.15 | 删除错误多、室内环境 |

### 2. 纯算法增强（不需要 MUSAN）

```bash
python scripts/run_augmentation.py --all --strategy speed    # 语速扰动 ±10%
python scripts/run_augmentation.py --all --strategy volume   # 音量扰动 ±6dB
```

| 策略 | 参数范围 | 适用场景 |
|------|---------|---------|
| speed | 0.9x / 1.0x / 1.1x | 删除错误多（字被吞）、跨语速鲁棒性 |
| volume | -6dB 到 +6dB | 对音量变化敏感（罕见，通常不是首选） |

### 3. SpecAugment（训练期频谱遮挡）

不在 run_augmentation 阶段执行，而是训练时由 FunASR 内建支持。在 `configs/funasr_config.yaml` 中启用：

```yaml
training:
  spec_aug:
    enabled: true     # 打开 SpecAugment
    num_t_mask: 2     # 时间域遮挡块数
    num_f_mask: 2     # 频率域遮挡块数
    max_t: 50         # 每块最大帧数
    max_f: 10         # 每块最大频点
```

适用场景：整体 CER 偏高，模型过拟合特定频谱模式。

### 4. 数据清洗（基于可用字段）

| 异常类型 | 检测方式 | 处理建议 |
|---------|---------|---------|
| 空标注 `empty_label` | `target` 为空或纯空白 | 严重，必须移除 |
| 缺失 source_len | `source_len = 0` | 严重，检查数据生成流程 |
| 长度不匹配 `length_mismatch` | `target_len / source_len` 超出 [0.03, 0.5] | 警告，人工复核 |
| 音频文件缺失 | `source` 路径不存在 | 严重，必须移除 |
| 低 SNR（需预处理） | `snr < 10 dB` | 警告，根据任务决定保留/移除 |

### 5. 场景分布分析 + 针对性增强

场景标签（`scene_type`）在增强时由 `NoiseAugmentor` 自动标注：
- 原始样本 → `clean`
- babble 增强 → `background_voice`
- noise 增强 → `background_noise`
- music 增强 → `background_music`
- reverb/speed/volume → 对应类型

```bash
# Step 1: 对增强后数据追加 SNR / 能量 / 过零率元数据
python scripts/preprocess_metadata.py --all

# Step 2: 用 analyze_dataset 查看场景分布是否均衡
python scripts/analyze_dataset.py --dataset-path data/augmented/train_aug.jsonl

# Step 3: 如果某场景样本不足，针对性增强
python scripts/run_augmentation.py --all --strategy babble
```

### 6. 热词自动提取（后处理范畴，但属于前置闭环）

```bash
# 从评估报告的 badcase 自动挖掘高频替换错误对，生成热词
python scripts/extract_hotwords.py \
    --eval-report reports/evaluation/eval_report_iter_xxx.json \
    --merge \
    --output configs/post_process/hotwords.txt

# 应用到后处理
python scripts/apply_postprocess.py update-config --hotword-file configs/post_process/hotwords.txt
```

---

## badcase 模式 → 前置手段映射（闭环关键）

下列映射**固化在 `scripts/badcase_strategy_map.py` 中**，同时会通过 `seed_strategies.py` 写入知识库。Kiro 分析 badcase 时必须参考此映射。

| 错误模式 | 推荐前置手段 | 优先级 | 初始置信度 |
|---------|-------------|--------|----------|
| 插入错误占主导（> 替换） | **babble + noise 增强** | 5 | 0.80 |
| 替换错误占主导 | **热词提取 + SpecAugment** | 5 | 0.90（热词）/ 0.70（specaug） |
| 删除错误占主导 | **reverb + speed 增强** | 4 | 0.75 |
| 整体 CER > 20% | **全量综合增强** `--all` | 4 | 0.70 |
| 某场景样本不足（< 20） | **预处理 + 分类 + 针对性增强** | 4 | 0.65 |
| 对音量敏感 | volume 增强 | 2 | 0.60 |

使用流程：

```bash
# 1. 系统初始化时（一次性）把映射注入知识库
python scripts/seed_strategies.py

# 2. 评估完成后，Kiro 调用 match_strategies() 得到针对本轮 badcase 的推荐手段
# （代码层自动完成，无需手动触发）

# 3. 按推荐执行对应命令，完成闭环
```

## 增强策略的自动选择（Skill 驱动）

Kiro 的决策优先级：

1. **先查知识库**：`manage_knowledge.py query --type data_strategy` 看历史哪些策略对当前 badcase 模式有效
2. **再查 strategy_map**：`badcase_strategy_map.match_strategies(eval_report)` 基于规则匹配
3. **给出建议**：综合两方面输出 OptimizationPlan

## 建议生成模板

```json
{
  "dimension": "data",
  "title": "基于 badcase 的前置数据优化",
  "description": "插入错误占主导（62%），推荐 babble 增强",
  "config_changes": {
    "augmentation.strategies[babble].probability": 0.5
  },
  "commands": [
    "python scripts/run_augmentation.py --all --strategy babble"
  ],
  "rationale": "历史经验：babble 增强在插入错误场景的成功率为 0.80",
  "source": "badcase_strategy_map:babble_augment_for_insertion"
}
```

## 知识库引用

```bash
# 查询历史数据优化经验
python scripts/manage_knowledge.py query --type data_strategy

# 查询特定策略的历史效果
python scripts/manage_knowledge.py query --type data_strategy --content-match '{"name":"babble_augment_for_insertion"}'

# 迭代完成后写入新的经验（置信度由自动反馈机制调整）
python scripts/manage_knowledge.py add --type data_strategy \
    --content '{"name":"babble_augment_for_insertion","iter":"iter_xxx","cer_delta":-0.03}' \
    --confidence 0.85 --source-iteration iter_xxx
```
