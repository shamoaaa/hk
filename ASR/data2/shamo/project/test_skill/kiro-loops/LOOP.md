# LOOP.md — Kiro 多环路项目

本文件描述这个项目正在运行的环路、各自的触发方式、技能和状态管理。

## 设计原则

- **调度器 = Kiro Agent Hooks**：利用 SessionStart / PostFileSave / PostTaskExec 等事件驱动环路
- **技能 = .kiro/steering/*.md**：持久化的项目知识与规约
- **记忆 = STATE.md**：跨会话的持久状态脊柱
- **渐进信任**：L1 仅报告 → L2 辅助修复 → L3 无人值守

## 正在运行的环路

### 1. 会话分诊环路（Session Triage）
- **触发**：SessionStart（每次新会话开始）
- **级别**：L1（仅报告）
- **做什么**：读取 STATE.md，向开发者简报当前待办、上次进展、需要关注的事项
- **技能**：`steering/loop-triage.md`
- **Hook**：`.kiro/hooks/session-triage.json`

### 2. 代码质量守卫环路（Code Quality Guard）
- **触发**：PostFileSave（保存 .ts/.py/.js 文件时）
- **级别**：L1（报告问题）→ L2（建议修复）
- **做什么**：检查保存的文件是否有常见问题（类型错误、未使用变量、安全隐患），报告发现
- **技能**：`steering/code-quality.md`
- **Hook**：`.kiro/hooks/code-quality-guard.json`

### 3. 任务验证环路（Task Verification）
- **触发**：PostTaskExec（Spec 任务完成后）
- **级别**：L2（验证 + 更新状态）
- **做什么**：验证刚完成的任务是否真正达标（测试通过、产物存在），更新 STATE.md
- **技能**：`steering/task-verifier.md`
- **Hook**：`.kiro/hooks/task-verify.json`

## 多环路协调

优先级：Task Verification > Code Quality Guard > Session Triage

三者天然不冲突：
- Session Triage 只在会话开始时跑一次
- Code Quality Guard 只在保存文件时触发
- Task Verification 只在 Spec 任务完成时触发

## 预算与急停

- Token 上限见 `loop-budget.md`
- 运行历史见 `loop-run-log.md`
- 急停：在 STATE.md 顶部加 `## ⚠️ LOOPS PAUSED` 即暂停所有环路

## 安全闸门

- Code Quality Guard **不自动修改文件**（L1），只报告
- Task Verification 只更新 STATE.md，不改业务代码
- 所有环路在 `## ⚠️ LOOPS PAUSED` 标记存在时停止工作
