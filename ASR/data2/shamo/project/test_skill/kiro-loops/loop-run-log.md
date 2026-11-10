# Loop Run Log — Kiro 多环路项目

每次环路运行追加一条记录。超过 30 天的条目可清理。

## 格式

```json
{
  "timestamp": "ISO 时间",
  "loop": "session-triage | code-quality-guard | task-verify",
  "trigger": "触发事件",
  "outcome": "report | silent | pass | fail | need-human | paused",
  "details": "简要描述",
  "tokens_estimate": 0
}
```

## outcome 含义

| 值 | 含义 |
|----|------|
| `report` | 产出了报告/简报 |
| `silent` | 检查了但没问题，静默通过 |
| `pass` | 验证通过 |
| `fail` | 验证未通过 |
| `need-human` | 无法自动判断，标记等人工 |
| `paused` | 检测到 LOOPS PAUSED，跳过 |

## 运行记录

<!-- 环路在下方追加 -->
