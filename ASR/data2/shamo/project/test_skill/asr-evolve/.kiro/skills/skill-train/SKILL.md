---
name: skill-train
description: ASR 远程训练操作。提交训练、检查进度、停止训练、实时监控、健康检查。当闭环流程进入 EXECUTE 阶段需要训练时激活。
---

# ASR 训练 Skill

## 命令

```bash
# 提交训练（完整模式，nohup 后台）
asr-evolve train submit [--tag TAG] [--mode full]

# 提交 Probe 快速验证（2 epoch，判断方向是否值得全量训练）
asr-evolve train submit --mode probe [--tag TAG]

# 检查训练状态（进程/GPU/日志/检查点）
asr-evolve train check

# 阻塞等待训练完成（不轮询，进程退出时立即返回）
asr-evolve train wait

# 实时监控训练 telemetry（loss/梯度/健康告警）
asr-evolve train monitor [--lines 50]

# 快速健康检查（基于本地状态，不连接远程）
asr-evolve train health

# 停止训练
asr-evolve train stop
```

## Probe 模式（改进方向 #3）

在全量训练前先跑 2 epoch 快速验证方向是否有效：

```
asr-evolve train submit --mode probe --tag probe_lr5e5
```

- 自动覆盖 max_epoch=2, save/validate_interval=9999
- 完成后 `asr-evolve train check` 输出 `PROBE_COMPLETED`
- 通过（loss 有下降趋势）→ `asr-evolve train submit --mode full`
- 未通过 → 剪枝此方向，不浪费全量训练资源

## 训练监控（改进方向 #1 #2）

训练运行中查看 telemetry 数据：

```bash
asr-evolve train monitor --lines 100
```

输出纯数据（loss 值/趋势/grad norm/NaN 标记），你自行判断是否需要干预。

参考判断框架（不是硬规则）：
- Loss 单步跳变很大 → 可能需要降 lr 或停止
- Grad norm 持续走高 → 可能需要梯度裁剪
- NaN 出现 → 必须停止
- Loss 长期不下降 → 考虑 early stop 或降 lr

`configs/training.yaml` 中的 `health_monitors` 和 `bounds` 段定义了参考阈值。

## 训练配方（configs/training.yaml）

当前配方：
- batch_size=64, batch_type=example, accum_grad=1
- max_epoch=5, lr=5e-5, warmup=20 steps
- GPU 2 (RTX 4090D), peak ~29 GB, 训练时间 ~10 分钟
- freeze: hotword_output_layer, seaco_decoder
- seaco_id=8377（必须，缺失会静默崩溃）

## 执行流程

1. 确认 GPU 可用（先激活 `skill-gpu`）
2. 当前阶段是否要求 probe？→ `asr-evolve strategy status` 查看
3. 提交：`asr-evolve train submit --mode <probe|full> --tag <标签>`
4. **立即执行** `asr-evolve train wait`（timeout=1800s，阻塞等待完成）
5. wait 返回后自动输出 TRAIN_COMPLETED 或 TRAIN_FAILED

**重要：提交后必须紧接着 `train wait`，不要暂停等待用户指令。**

## 健康检查判定

- `Status: DONE` + `Checkpoints > 0` → PASS
- `Status: DONE` + `Checkpoints: 0` → FAIL（静默崩溃，不进入评估）
- OOM → 降低 batch_size 或换 GPU
