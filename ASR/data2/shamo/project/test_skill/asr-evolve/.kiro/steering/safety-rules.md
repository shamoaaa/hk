---
inclusion: always
---

# 安全规约

## 急停

如果 STATE.md 有 `## ⚠️ LOOPS PAUSED`：
- 立即停止所有操作
- 输出："⏸️ 环路已暂停"

## 禁止区域

不得自动修改：
- `configs/remote.yaml` 中的凭证信息
- `data/` 下的原始数据文件
- `.kiro/hooks/*.json`
- 任何 `.env` / 密钥文件

## 训练安全

- L2 模式下训练启动需人工确认
- seaco_id 必须在启动前校验
- 训练结束后必须做健康检查
- 静默崩溃（rc=0 无 .pt）→ 不进入评估

## 预算约束

- 每日最多 3 次训练
- 连续 2 次 CER 无改善 → 暂停并报告
- 遵守 loop-budget.md 中的限额

## 死胡同硬拦截

DEAD_ENDS.md 中的条目是**硬性禁止**。
EXECUTE 前必须检查冲突。冲突 → 不执行。
