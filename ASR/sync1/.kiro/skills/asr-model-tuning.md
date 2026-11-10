# ASR 模型调优建议

## 能力描述

基于训练分析和评估结果，生成模型结构优化和训练参数调优建议。针对 FunASR Paraformer 模型的特点，提供注意力机制、SeACo 模块、语音增强前端和训练超参数的调优方案。

## 适用模型

`speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`

关键特性：
- 非自回归（NAR）端到端架构，支持并行推理
- SeACo（Semantic-Aware Co-decoder）模块支持热词定制
- 基于 Transformer 的编码器-解码器结构

## 分析规则

### 模型结构优化

#### 注意力机制调整

**触发条件**：模型在复杂场景下表现不佳（背景人声 CER > 15%）

调整参数：
- `attention_heads`：注意力头数（必须能整除 `hidden_dim`）
- `hidden_dim`：隐藏层维度（必须为正整数）

建议策略：
| 当前配置 | 问题 | 建议调整 |
|---------|------|---------|
| heads=4, dim=256 | 容量不足 | heads=8, dim=512 |
| heads=8, dim=512 | 标准配置 | heads=16, dim=512（更细粒度注意力） |
| heads=16, dim=512 | 计算开销大 | 保持不变，优化其他维度 |

**验证规则**：
- `hidden_dim % attention_heads == 0`
- `hidden_dim > 0`
- `attention_heads > 0`

#### SeACo 模块参数调整

**触发条件**：热词识别率低或背景人声干扰严重

可调参数：
- `co_decoder_layers`：Co-decoder 层数（建议范围：1-4）
- `attention_weight`：注意力权重（建议范围：0.1-1.0）
- `vocab_size`：词表大小（通常固定为 8404）

建议策略：
| 问题 | 建议调整 |
|------|---------|
| 热词识别率低 | 增加 co_decoder_layers（+1），提高 attention_weight |
| 背景人声干扰 | 提高 attention_weight 至 0.7-0.9 |
| 推理速度慢 | 减少 co_decoder_layers（-1） |

#### 语音增强前端

**触发条件**：背景人声场景 CER 显著高于纯净语音（差值 > 8%）

配置建议：
```json
{
  "speech_enhance_config": {
    "type": "conv_tasnet",
    "channels": 64,
    "layers": 4,
    "kernel_size": 3
  }
}
```

注意事项：
- 添加语音增强前端会增加推理延迟
- 建议先在验证集上评估增强效果
- 如果纯净语音场景 CER 上升，考虑条件性启用

### 训练参数调优

#### 学习率调整

| 训练阶段 | 建议学习率范围 | 调度策略 |
|---------|-------------|---------|
| 初始微调 | 1e-4 ~ 5e-4 | warmup_decay |
| 精细调优 | 1e-5 ~ 5e-5 | cosine_annealing |
| 最终收敛 | 1e-6 ~ 5e-6 | step_decay |

学习率调度策略选择：
- **warmup_decay**：适合初始训练阶段，先升后降
- **cosine_annealing**：适合中后期，平滑衰减
- **step_decay**：适合最终阶段，阶梯式衰减

#### Batch Size 调整

| 问题 | 建议调整 |
|------|---------|
| 训练不稳定（loss 震荡） | 增大 batch size |
| 泛化能力差 | 减小 batch size |
| 显存不足 | 减小 batch size + 梯度累积 |

建议范围：8-64（取决于 GPU 显存）

#### Warmup 步数

- 建议值：总训练步数的 5%-10%
- 如果训练初期 loss 不稳定：增加 warmup 步数
- 如果训练收敛太慢：减少 warmup 步数

#### 权重衰减

- 默认值：0.01
- 过拟合时：增加到 0.05-0.1
- 欠拟合时：减少到 0.001-0.005

#### 梯度裁剪

- 默认值：5.0
- 梯度爆炸时：降低到 1.0-2.0
- 训练稳定时：可适当放宽到 10.0

### 配置回滚规则

1. **模型配置回滚**：如果新配置导致模型无法初始化，自动回滚
2. **训练参数回滚**：如果前 100 步出现 loss 异常，自动回滚
3. **版本化存储**：所有配置变更保存为版本化文件（v001, v002, ...）

## 建议生成模板

### 模型结构建议
```json
{
  "dimension": "model",
  "title": "模型结构优化",
  "config_changes": {
    "attention_heads": 16,
    "hidden_dim": 512,
    "seaco_params": {
      "co_decoder_layers": 3,
      "attention_weight": 0.7
    },
    "speech_enhance_config": {
      "type": "conv_tasnet",
      "channels": 64,
      "layers": 4
    }
  }
}
```

### 训练参数建议
```json
{
  "dimension": "param",
  "title": "训练参数调优",
  "config_changes": {
    "learning_rate": 5e-5,
    "batch_size": 32,
    "warmup_steps": 2000,
    "weight_decay": 0.01,
    "lr_schedule": "cosine_annealing",
    "gradient_clip": 5.0
  }
}
```

## 知识库引用

分析时参考 `knowledge/knowledge_base.json` 中的：
- 参数调整与效果的经验规律
- 不同训练阶段的最优参数组合
- 模型结构变更对各场景 CER 的影响

查询知识库中的调优经验：

```bash
# 查询参数调整效果
python scripts/manage_knowledge.py query --type param_effect

# 查询模型结构相关的场景解决方案
python scripts/manage_knowledge.py query --type scene_solution --content-match '{"dimension":"model"}'
```

调优完成后，将有效经验写入知识库：

```bash
python scripts/manage_knowledge.py add --type param_effect --content '{"dimension":"model","change":"seaco_layers_2_to_3","effect":"background_voice CER -3%"}' --confidence 0.75 --source-iteration <iter_id>
```
