---
name: skill-reset
description: 系统初始化/重置。将本地和远程状态恢复到全新（无迭代历史）。当用户要求"重置系统"、"从头开始"、"清理所有实验"时激活。
---

# 系统重置 Skill

将 ASR-Evolve 系统（本地 + 远程）恢复到全新状态，清除所有迭代历史。

## 命令（推荐方式）

```bash
# 全部重置（远程 + 本地，一条命令搞定）
asr-evolve reset all

# 或分步执行
asr-evolve reset remote    # 只清远程（output/脚本/恢复jsonl）
asr-evolve reset local     # 只清本地（记忆文件/logs/reports）
```

## 执行步骤

**必须按顺序执行：**

### Step 1: 停止远程训练（如有）

```bash
asr-evolve train stop
```

如果报 "No active training" 则跳过。

### Step 2: 执行重置

```bash
asr-evolve reset all
```

这一条命令会自动完成：
- 清理远程 output 目录（所有训练产出）
- 删除远程脚本文件
- 恢复远程 train.jsonl 为原始版本（增强数据 wav 保留复用）
- 重置本地所有记忆文件（STATE/DEAD_ENDS/INSIGHTS/实验树/hpo/performance）
- 清理 logs/ 和 reports/
- search_strategy.yaml current_phase 回到 "coarse"

### Step 3: 验证

```bash
asr-evolve status
asr-evolve gpu
```

预期输出：
- status 显示"尚未开始优化"
- GPU 空闲

## 注意事项

- **不会删除** `configs/` 下的配方文件（training.yaml/augment.yaml/model.yaml）
- **不会删除** 远程原始数据 `data/wav/` `data/merged/train/train_original.jsonl`
- **不会删除** 远程增强缓存 `data/augmented/`（保留复用，只恢复 jsonl 指向）
- **不会删除** `.kiro/`（steering/skills/hooks 是系统本身）
- **不会删除** `asr_evolve/`（代码）

## 重置后状态

```
本地：
  STATE.md          → 初始（无历史）
  experiment_tree   → 空 ({next_id:1, nodes:{}})
  DEAD_ENDS.md      → 空
  INSIGHTS.md       → 空
  loop-run-log.md   → 空
  configs/hpo_history.json     → 空 (trials=[])
  configs/search_strategy.yaml → current_phase="coarse"
  memory/           → 清零
  logs/             → 空
  reports/          → 空

远程：
  output/           → 空（所有训练产出已删）
  train.jsonl       → 恢复为原始版本
  data/augmented/   → 保留（增强 wav 缓存，可复用）
  data/wav/         → 保留（原始音频不动）
  models/           → 保留（预训练模型不动）
```
