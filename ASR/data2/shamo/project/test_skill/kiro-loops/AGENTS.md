# AGENTS.md — Kiro 多环路项目协作规约

给人类开发者和 Kiro agent 的共同规则。

## 构建与验证

```bash
# 项目无特定构建步骤，环路通过 Kiro hooks 自动运行
# 验证环路状态：
cat STATE.md
cat loop-run-log.md
```

## 环路操作规则

### Session Triage（会话分诊）
- 每次会话开始时**读取 STATE.md 并简报**
- 只报告，不修改任何文件（除 STATE.md 的 Last run 时间戳）
- 如果 STATE.md 顶部有 `## ⚠️ LOOPS PAUSED`，跳过简报

### Code Quality Guard（代码质量守卫）
- 仅在 .ts / .py / .js / .tsx / .jsx 文件保存时触发
- **不自动修改源文件**——只在聊天中报告发现的问题
- 不对测试文件 (*test* / *spec*) 报告风格问题

### Task Verification（任务验证）
- Spec 任务完成后自动运行
- 验证：相关测试是否通过、预期产物是否存在
- 验证通过 → 更新 STATE.md（移入"已完成"）
- 验证失败 → 在 STATE.md 高优先级区标注"验证未通过"

## 人工闸门

以下情况必须等人工决定：
- 删除文件或目录
- 修改 .env / 密钥 / 认证相关代码
- 任何涉及数据库 schema 变更的操作
- STATE.md 中标记为"需要讨论"的事项

## 评审规范

- 环路产出的建议，人类可以忽略（环路不会因此"生气"）
- 如果环路连续 3 次在同一文件报告相同问题，应人工介入检查是否是误报
