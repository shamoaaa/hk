# AGENTS.md — ASR-Evolve 协作规约

给 Kiro Agent 和人类开发者的共同规则。

## 构建与验证

```bash
cd asr-evolve
pip install -e .
asr-evolve status                              # 查看系统状态
asr-evolve pipeline run --mode aishell --dry-run  # 预览流水线
```

## Agent 操作规则

### THINK 阶段
- 先读 STATE.md + DEAD_ENDS.md + INSIGHTS.md
- 分析最近评估报告（reports/）
- 决定下一步优化方向

### EXECUTE 阶段
- 通过 CLI 调用具体阶段：`asr-evolve pipeline stage --only <stage>`
- 训练前必须确认 GPU 可用：`asr-evolve gpu`
- L2 模式下训练启动需人工确认

### RECORD 阶段
- 评估结果与之前对比
- 判定内置在 pipeline 中（健康检查 + CER 阈值）
- 更新 STATE.md、追加 loop-run-log.md

## 禁止区域（Denylist）

以下操作 Agent 不得自动执行：
- 修改 `configs/remote.yaml` 中的 SSH 凭证
- 删除 `data/` 下的原始数据
- 修改 `.kiro/hooks/*.json`（环路不改自己的触发器）
- 跳过训练健康检查
- 在 DEAD_ENDS.md 已记录的方向上重试

## 死胡同硬拦截

DEAD_ENDS.md 中记录的方向是**硬性禁止**，不是建议。
Agent 在 EXECUTE 前必须检查提议的操作是否与死胡同冲突。
冲突 → 不执行，换方向。

## 决策点记录

以下决策必须持久化到 loop-run-log.md：
- 质量阈值选择
- 训练配方调整
- 是否继续迭代
- 切分比例变更

## 评审规范

- 每次训练后必须跑评估（eval）
- 评估结果必须与基线对比
- CER 改善 < 0.1% 视为"无显著改善"
- 连续 2 次无改善 → 记入 DEAD_ENDS + 换方向
