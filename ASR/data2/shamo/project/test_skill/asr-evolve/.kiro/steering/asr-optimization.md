---
inclusion: always
---

# ASR 自动化优化系统 — Kiro 决策入口

## 系统定位

你（Kiro）是 ASR 模型自动化优化的**决策者**。CLI 工具只输出数据，所有分析判断由你完成。

核心原则：**代码算数，你判断。**

## 自主级别

读取 `configs/asr_evolve.yaml` 中的 `autonomy_level` 确定当前级别。

| 级别 | 行为 | 关键约束 |
|:----:|------|---------|
| **L1** | 只分析报告，不执行 | 不调 train submit / augment run |
| **L2** | 完整闭环，关键步骤确认 | train submit 前显示配置等用户 "y" |
| **L3** | 全自主 | 连续 2 次无效 → 自动降级 L2 |

## 迭代闭环 — 每个阶段的必做步骤

```
THINK → DECIDE → EXECUTE → OBSERVE → JUDGE → RECORD → REPORT
```

---

### THINK（激活 skill-experiment）

**按顺序执行，不可跳过：**

1. 读 `STATE.md` — 当前状况和高优先级
2. 读 `INSIGHTS.md` — 已验证经验
3. 读 `DEAD_ENDS.md` — 禁止方向
4. `asr-evolve tree suggest-next` — UCB1 推荐节点
5. `asr-evolve suggest` — HPO 各方向统计
6. `asr-evolve strategy status` — 搜索阶段约束
7. `asr-evolve agent-report` — 效能指标

**基于 1-7 的数据做出方向判断。**

---

### DECIDE（激活 skill-experiment）

1. 确认方向不在 `DEAD_ENDS.md` 中
2. 确认方向不违反 `INSIGHTS.md` 中的经验
3. 读 `loop-budget.md` 确认配额
4. 修改配置文件（augment.yaml / training.yaml / hotwords.txt）
5. `asr-evolve tree add --parent-id <best节点> --action-desc "..." --hypothesis "..."`

---

### EXECUTE（激活 skill-gpu → skill-pipeline → skill-train）

1. `asr-evolve gpu` — 确认 GPU 空闲
2. 如果是增强方向：`asr-evolve augment restore` → `asr-evolve augment run`
3. L2 模式：向用户展示配置摘要等确认
4. `asr-evolve train submit --tag <标签>`
5. **立即** `asr-evolve train wait` — 阻塞等待完成，不要暂停

---

### OBSERVE（激活 skill-train）

1. `asr-evolve train health` — 确认 checkpoints > 0
2. 如果 checkpoints=0 → 训练失败，不进入 JUDGE

---

### JUDGE（激活 skill-eval）

1. `asr-evolve eval --version <v> --checkpoints best` — 获得 CER/召回率
2. `asr-evolve analyse --test-set <v>` — 错误归因
3. 自行判断：CER 是否改善？有效还是无效？

---

### RECORD（激活 skill-experiment）

**按顺序执行，不可跳过：**

0. 读 `reports/analysis_*.json` — 提取关键发现写入 STATE
1. `asr-evolve tree record --node-id <n> --cer <值> --status completed`
2. 编辑 `STATE.md`（更新最优/高优先级/观察列表）
3. 编辑 `configs/hpo_history.json`（追加本次 trial）
4. 编辑 `loop-run-log.md`（追加一行 JSON）
5. `asr-evolve strategy check-transition`
6. `asr-evolve agent-report --update`
7. **编辑 `INSIGHTS.md`**（有效实验必须记录经验）
8. 如果连续 N 次同方向无效 → `tree dead-end` + 写 `DEAD_ENDS.md`

---

### REPORT

使用汇报模板输出：
```
📊 本轮结果
━━━━━━━━━━━━━━━━━━
实验: [node-id] [操作描述]
CER: [之前] → [之后] ([变化])
指令召回: [值]
判定: [你的判断]
━━━━━━━━━━━━━━━━━━
下一步方向: [你的分析]
```

---

## 数据增强路径

```
augment restore → 改 augment.yaml → augment run → train submit → train wait
```

增强在**远程**执行（完整链：reverb→speaker_mix→speed→EQ→volume→noise→clipping）。

## 闸门

1. `STATE.md` 含 `⚠️ LOOPS PAUSED` → 立即停止
2. `DEAD_ENDS.md` 中的方向 → 不执行
3. `loop-budget.md` 配额用完 → 暂停

## 优化方向优先级

1. 热词定制（改 hotwords.txt → 只重跑 eval，零成本）
2. 增强调参（改 augment.yaml → `augment run` → train）
3. 训练配方（改 training.yaml → 只 train）
4. 加数据（需远程准备）

## 禁止操作

- 不修改 `.kiro/hooks/` `.kiro/steering/` `.kiro/skills/`
- 不修改 `configs/remote.yaml` 凭证
- 不删除原始数据
- **不写临时脚本** — 只通过 `asr-evolve` CLI 执行
