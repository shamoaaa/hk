---
name: skill-pipeline
description: ASR 数据增强与管理。远程执行数据增强（speed perturbation/noise/volume），管理训练数据版本。当需要对训练数据做增强时激活。
---

# ASR 数据增强 Skill

## 命令

```bash
# 执行远程数据增强（读 augment.yaml，在远程用 torchaudio 处理）
asr-evolve augment run

# 查看远程增强产物状态（当前 train.jsonl 行数、是否有增强版本）
asr-evolve augment status

# 恢复原始训练数据（撤销增强）
asr-evolve augment restore
```

## 工作原理

增强全部在**远程 GPU 服务器**执行（原始 wav 在远程）：

```
本地: 上传 augment_engine.py + augment.yaml → 远程
远程: AudioAugmentor 加载配置 → 对每条 wav 执行完整增强链 → 输出增强 wav + jsonl
远程: 替换 train.jsonl（原始自动备份）
```

完整增强链（按 augment.yaml 的 chain 定义）：
```
reverb → speaker_mix → speed_perturb → eq_random → volume_perturb → gaussian_noise → clipping
```

- **speaker_mix** 使用训练集自身音频互相作为背景声候选
- 每个效果由 `augment.yaml` 中 `strategies.<name>.enabled` 控制开关
- 增强引擎来自原始项目 `augment_engine.py`（完整实现，非简化版）

## 增强配置 (configs/augment.yaml)

完整配置包含增强链和各策略参数。核心结构：

```yaml
chain:              # 增强执行顺序
  - reverb
  - speaker_mix
  - speed_perturb
  - eq_random
  - volume_perturb
  - gaussian_noise
  - clipping

strategies:         # 各策略的参数
  reverb:
    enabled: true
    rt60_range: [0.08, 0.1]
  speaker_mix:
    enabled: true
    max_speakers: 3
    snr_range: [-3, 15]
  speed_perturb:
    enabled: true
    factor_range: [0.9, 1.1]
  # ... 其他策略
```

每个策略有 `enabled: true/false` 开关。Kiro 调参时只需改对应策略的参数值。

## 使用场景

| 场景 | 操作 |
|------|------|
| 首次增强 | 编辑 augment.yaml → `augment run` → `train submit` |
| 调整增强参数 | 编辑 augment.yaml → `augment restore` → `augment run` → `train submit` |
| 撤销增强（对照实验） | `augment restore` → `train submit` |
| 查看当前状态 | `augment status` |

## 注意事项

- 增强后 train.jsonl 行数翻倍（原始 + 增强版各一份）
- 原始数据自动备份为 `train_original.jsonl`，不会丢失
- 如果要换增强参数，**先 restore 再 run**（否则增强会叠加）
- 完整增强链（含 speaker_mix）预计 5-15 分钟
- 远程需要 numpy + scipy + soundfile（已确认有）

## 命名空间与复用

增强 wav 按配置参数哈希保存在独立目录：

```
远程 data/augmented/
├── aug_3f2a1b8c/       ← 某组参数的增强结果
│   └── aug_audio_xxx.wav ...
├── aug_9e7d4c21/       ← 另一组参数
│   └── ...
```

- **命名空间 = augment.yaml 配置的 SHA256 前 8 位**
- 相同参数 → 相同目录 → 直接复用已有 wav，不重新生成
- 不同参数 → 新目录 → 重新生成
- **重置时不删除 augmented 目录**（保留复用），只恢复 train.jsonl
