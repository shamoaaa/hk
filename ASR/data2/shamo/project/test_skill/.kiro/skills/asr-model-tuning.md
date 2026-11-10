# ASR 模型调优

## 能力描述

基于训练分析和评估结果，生成模型结构和训练参数的调优建议。

## 何时使用

- 数据增强和后处理已优化但 CER 仍不满意
- 需要调整 SeACo 参数改善热词识别
- 训练参数需要精细调整时

## 当前模型配置

```yaml
model:
  name: speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
  attention_heads: 8
  hidden_dim: 512
  seaco_params:
    co_decoder_layers: 3
    attention_weight: 0.6
  speech_enhance_config:
    type: conv_tasnet
    channels: 64
```

## 执行步骤

### 模型结构变更
```powershell
.venv_win\Scripts\python -m scripts.apply_model_config apply --seaco-layers 3 --attention-weight 0.7
```

### 训练参数变更
```powershell
.venv_win\Scripts\python -m scripts.apply_training_config apply --learning-rate 1e-4 --max-epoch 15
```

### 推送 + 重新训练
```powershell
.venv_win\Scripts\python -m scripts.remote_runner push-configs
.venv_win\Scripts\python -m scripts.remote_runner run "python -m scripts.run_training --tag iter_yyy"
```

## 调优参考

### SeACo 参数

| 问题 | 调整 |
|------|------|
| 热词识别率低 | co_decoder_layers +1, attention_weight ↑ |
| 背景人声干扰 | attention_weight → 0.7-0.9 |
| 推理速度慢 | co_decoder_layers -1 |

### 训练参数

| 阶段 | 学习率 | 调度策略 |
|------|--------|---------|
| 初始微调 | 1e-4 ~ 5e-4 | warmup_decay |
| 精细调优 | 1e-5 ~ 5e-5 | cosine_annealing |
| 最终收敛 | 1e-6 ~ 5e-6 | step_decay |

### SpecAugment（已启用）

```yaml
spec_aug:
  enabled: true
  num_t_mask: 2   # 时间遮挡块数
  num_f_mask: 2   # 频率遮挡块数
  max_t: 40       # 每块最大帧数
  max_f: 27       # 80-dim Fbank 的 ~33%
```

## 常见问题

- **改了参数 CER 反而变差**：回滚配置（版本快照在 configs/model/ 或 configs/training/）
- **每次只改一个维度**：便于归因，避免多变量混淆
- **warmup 不够导致 NaN**：warmup_steps 建议 = 总步数 × 5%
- **batch_type=length 后 OOM**：减小 batch_size（token 总数）或 max_token_length

## 知识库引用

查询历史参数调整效果，避免重复踩坑：

```powershell
# 查询参数调整历史效果
.venv_win\Scripts\python -m scripts.manage_knowledge query --type param_effect

# 查询模型结构相关经验
.venv_win\Scripts\python -m scripts.manage_knowledge query --type scene_solution --content-match "{\"dimension\":\"model\"}"

# 记录有效的调优经验
.venv_win\Scripts\python -m scripts.manage_knowledge add --type param_effect --content "{\"param\":\"learning_rate\",\"from\":\"5e-5\",\"to\":\"1e-4\",\"cer_delta\":-0.002}" --confidence 0.8 --source-iteration iter_xxx
```
