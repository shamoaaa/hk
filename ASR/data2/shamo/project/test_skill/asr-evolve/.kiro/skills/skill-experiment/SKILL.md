---
name: skill-experiment
description: ASR 实验树与记忆系统管理。增/查/记录/标记实验节点，UCB1 搜索建议，搜索策略，HPO 历史，Agent 效能。当闭环进入 THINK/DECIDE/RECORD 阶段或需要查状态时激活。
---

# ASR 实验树 Skill

## 实验树命令

```bash
# 查看实验树
asr-evolve tree view

# 摘要
asr-evolve tree summary

# 添加新实验节点
asr-evolve tree add --parent-id <父节点> --action-desc "操作描述" --hypothesis "假设"

# 记录实验结果
asr-evolve tree record --node-id <节点> --cer <CER值> --status completed

# 标记死胡同
asr-evolve tree dead-end --node-id <节点> --reason "原因"

# UCB1 推荐下一步
asr-evolve tree suggest-next [--top-n 5]
```

## 搜索策略命令

```bash
# 查看当前搜索阶段（coarse/fine/exploit）
asr-evolve strategy status

# 检查是否满足阶段切换条件（只报告，不自动修改）
asr-evolve strategy check-transition [--cer 0.08]

# 查看当前阶段允许的参数范围
asr-evolve strategy params
```

## HPO 历史统计

```bash
# 输出各方向统计数据（纯数据，你自行分析选方向）
asr-evolve suggest
```

## 状态管理命令

```bash
# 检查所有状态文件一致性
asr-evolve state validate

# 结构化状态摘要
asr-evolve state summary [--json]
```

## Agent 效能命令

```bash
# 生成 Agent 效能指标（纯数据）
asr-evolve agent-report [--last 10] [--update] [--json]
```

## 记忆系统命令

```bash
# 查看全部记忆
asr-evolve memory full

# 查看死胡同
asr-evolve memory dead-ends

# 查看洞察
asr-evolve memory insights

# 查看系统状态
asr-evolve status
```

## 记忆文件

| 文件 | 作用 | 何时更新 |
|------|------|---------|
| `STATE.md` | 当前状态、高优先级、最优模型 | 每轮 RECORD |
| `experiment_tree.json` | 实验搜索空间树 | tree add/record/dead-end |
| `DEAD_ENDS.md` | 硬性禁止方向 | 连续 N 次无效时 |
| `INSIGHTS.md` | 已验证经验（正面） | 每轮有效实验即记录 |
| `loop-run-log.md` | 追加式运行日志 | 每次操作 |
| `loop-budget.md` | 每日配额定义 | 不修改（只读） |
| `configs/hpo_history.json` | 超参变更追踪 | 每次实验后追加 |
| `configs/search_strategy.yaml` | 搜索策略（含当前阶段） | 阶段切换时 |
| `memory/agent_performance.json` | Agent 效能数据 | agent-report --update |

## THINK 阶段标准流程

1. `asr-evolve state validate` — 确认状态一致
2. 直接读 `STATE.md` — 了解当前状况和高优先级
3. 直接读 `INSIGHTS.md` — 参考已验证经验，避免重复犯错
4. 直接读 `DEAD_ENDS.md` — 确认哪些方向不能碰
5. `asr-evolve tree suggest-next` — UCB1 推荐方向
6. `asr-evolve suggest` — HPO 各方向统计
7. `asr-evolve strategy status` — 当前搜索阶段约束
8. `asr-evolve agent-report` — 效能指标，自行分析

### Agent 效能指标分析指引

`agent-report` 输出纯数据，你需要自己分析。以下是思考框架（不是硬规则）：

**决策准确率**
- 高（>60%）：当前决策模式有效，可以继续
- 中等（40-60%）：正常水平，关注哪些类别拖后腿
- 低（<40%）：决策模式可能有问题，考虑：更多用 probe 验证、收窄参数变动幅度、换个大方向

**累积效率 vs 边际效率**
- 边际效率 ≈ 累积效率：改善速度稳定
- 边际效率 < 累积效率的一半：进入瓶颈区，当前方向的低垂果实可能已摘完
- 边际效率 > 累积效率：最近加速了，方向很好

**分类统计**
- 某方向 0% 准确率 + 多次尝试：可能该方向的根本假设有问题，考虑写入 DEAD_ENDS
- 某方向 100% 准确率：继续深挖
- 某方向从未尝试：未知潜力，值得探索

**最近趋势**
- 连续 effective：势头好，延续
- 连续 ineffective：该换方向了
- 交替出现：搜索空间噪声大，考虑用 probe 降低风险

## DECIDE 阶段标准流程

1. 选定方向
2. 读 `DEAD_ENDS.md` — 确认不与死胡同方向重复
3. 读 `INSIGHTS.md` — 确认不违反已知经验
4. 读 `loop-budget.md` — 确认配额

## RECORD 阶段标准流程

0. 读取 `reports/analysis_*.json`（如果本轮跑了 analyse）— 提取关键发现写入 STATE.md
1. `asr-evolve tree record --node-id <n> --cer <值> --status completed`
2. 更新 `STATE.md`（当前最优、高优先级、观察列表）— 直接编辑文件
3. 追加 `configs/hpo_history.json`（记录本次参数变更+结果）— 直接编辑文件
4. 追加 `loop-run-log.md`（一行 JSON 摘要）— 直接编辑文件
5. `asr-evolve strategy check-transition` — 检查阶段切换
6. `asr-evolve agent-report --update` — 更新效能数据
7. **追加 `INSIGHTS.md`** — 有效实验即记录经验（不需要多次验证）
8. 如果连续 N 次同方向无效 → `asr-evolve tree dead-end` + 写 `DEAD_ENDS.md`

## RECORD 阶段写入格式参考

### STATE.md 更新格式

```markdown
Last run: <日期>（<tag>, CER=<值>%, 召回=<值>%）

## 高优先级
- <本轮发现的关键信息>
- <下一步方向建议>

## 当前最优
- 模型: <tag> (model.pt.best)
- CER: <值>%
- 召回率: <值>% (<n>/<total>)
```

### hpo_history.json 追加格式

在 `trials` 数组末尾追加一条：

```json
{
  "node_id": "<实验节点 ID>",
  "date": "<YYYY-MM-DD>",
  "params_changed": {"<改了什么参数>": "<改成什么值>"},
  "params_unchanged": {"<没动的关键参数>": "<值>"},
  "result": {"cer": <float>, "recall": <float or null>},
  "category": "<training_recipe|augmentation|hotwords|data_addition>",
  "reasoning": "<为什么这么做>",
  "verdict": "<effective|ineffective|baseline>"
}
```

### loop-run-log.md 追加格式

追加一行 JSON：

```json
{"timestamp":"<ISO>","loop":"iterate","action":"<做了什么>","outcome":"<effective|ineffective>","cer_before":<float>,"cer_after":<float>,"gpu_hours":<float>,"node_id":"<id>","details":"<一句话摘要>"}
```

### INSIGHTS.md 追加格式

**每轮有效实验必须追加**。在 `## 记录` 下追加：

```markdown
### [<日期>] <一句话标题>
- 证据: <哪个实验证明了这一点>
- 适用条件: <什么情况下适用>
- 应用方式: <具体怎么用>
```

### DEAD_ENDS.md 追加格式

只在连续 N 次同方向无效时追加：

```markdown
### [<日期>] <方向名称>
- 尝试: <做了什么>
- 结果: <CER 没改善 / 退化>
- 结论: <为什么这个方向不行>
```
