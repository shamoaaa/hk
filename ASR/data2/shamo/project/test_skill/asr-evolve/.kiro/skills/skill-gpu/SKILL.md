---
name: skill-gpu
description: ASR 远程 GPU 状态查询和诊断。检查 GPU 显存、利用率、运行中的进程。训练前确认 GPU 可用时激活。
---

# GPU 诊断 Skill

## 命令

```bash
# 查看所有 GPU 状态
asr-evolve gpu
```

## 输出格式

```
0, NVIDIA GeForce RTX 4090 D, 25082 MiB, 49140 MiB, 0 %
1, NVIDIA GeForce RTX 4090 D, 3074 MiB, 49140 MiB, 0 %
2, NVIDIA GeForce RTX 4090 D, 53 MiB, 49140 MiB, 0 %      ← 训练用这张
3, NVIDIA GeForce RTX 4090 D, 36770 MiB, 49140 MiB, 97 %
```

列含义：index, name, memory_used, memory_total, gpu_utilization

## 服务器信息

- 地址: root@10.6.217.118:20028
- GPU: 4× RTX 4090 D (49 GB each)
- 当前训练使用: **GPU 2**（configs/training.yaml 中 `cuda_visible_devices: "2"`）
- 训练峰值显存: ~26 GB

## 判定规则

- 目标 GPU 空闲（used < 5 GB）→ 可以训练
- 目标 GPU 被占（used > 30 GB）→ 等待或换卡
- GPU utilization 99-100% → 有训练在跑
- GPU utilization 0% + 高显存占用 → 可能在做 avg/保存
