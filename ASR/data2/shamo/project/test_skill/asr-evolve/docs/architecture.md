# ASR-Evolve v3 — 架构文档

## 一、项目定位

Kiro 驱动的 ASR 模型自动化优化平台。远程 GPU 训练，Kiro Agent 做策略决策。

**核心设计哲学：代码算数，LLM 判断。**

- CLI 工具只做数学计算和数据 CRUD
- 所有分析、诊断、决策由 Kiro 自行推理
- Skill 文档是思考框架，不是硬编码执行逻辑

---

## 二、自主级别（L1/L2/L3）

| 级别 | 行为 | 关键约束 |
|:----:|------|---------|
| **L1** | 只分析报告，不执行训练/增强 | 不调 train submit / augment run |
| **L2** | 完整闭环，关键步骤前确认用户 | train submit 前显示配置等用户确认 |
| **L3** | 全自主，异常时才告警 | 连续 2 次无效 → 自动降级为 L2 |

**升级路径**：L1（稳定 3 轮）→ L2（准确率>60% + 5 轮无退化）→ L3

当前级别在 `configs/asr_evolve.yaml` 的 `autonomy_level` 字段定义。

---

## 三、编排架构

```
┌─────────────────────────────────────────────────────┐
│  .kiro/steering/asr-optimization.md                 │
│  （始终加载，闭环流程 + 分级行为 + Skill 调度）        │
└──────────────────────┬──────────────────────────────┘
                       │ 按阶段激活对应 Skill
         ┌─────────────┼─────────────┬──────────────┐
         ▼             ▼             ▼              ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐ ┌─────────────────┐
│ skill-train  │ │skill-eval│ │skill-pipeline│ │skill-experiment │
│ 训练/监控    │ │评估/分析 │ │远程数据增强  │ │搜索/策略/效能   │
│ /Probe/wait  │ │          │ │             │ │/实验树/记忆     │
└──────────────┘ └──────────┘ └──────────────┘ └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  asr-evolve CLI   │
                    │ （纯数据工具）      │
                    └───────────────────┘
```

---

## 四、目录结构

```
asr-evolve/
├── .kiro/
│   ├── steering/asr-optimization.md     ← 决策入口 + 分级行为
│   ├── skills/
│   │   ├── skill-train/SKILL.md         ← 训练/监控/Probe/wait
│   │   ├── skill-eval/SKILL.md          ← 评估/分析
│   │   ├── skill-pipeline/SKILL.md      ← 远程数据增强
│   │   ├── skill-experiment/SKILL.md    ← 搜索/策略/效能/记忆
│   │   ├── skill-gpu/SKILL.md           ← GPU 诊断
│   │   └── skill-reset/SKILL.md         ← 系统重置（CLI）
│   └── hooks/session-triage.json        ← 会话开始自动分诊
│
├── asr_evolve/                          ← Python 包
│   ├── cli.py                           ← CLI 入口
│   ├── commands/
│   │   ├── status.py                    ← 系统概览
│   │   ├── gpu.py                       ← 远程 GPU
│   │   ├── pipeline.py                  ← 本地流水线（备用）
│   │   ├── augment_remote.py            ← 远程数据增强（完整链）
│   │   ├── train.py                     ← 训练 submit/check/stop/wait
│   │   ├── monitor.py                   ← telemetry 采集 + 健康摘要
│   │   ├── evaluate.py                  ← 评估
│   │   ├── analyse.py                   ← 错误分析
│   │   ├── tree.py                      ← 实验树 CRUD
│   │   ├── suggest.py                   ← UCB1 计算 + HPO 统计
│   │   ├── strategy.py                  ← 搜索策略状态
│   │   ├── state_validate.py            ← 状态一致性检查
│   │   ├── agent_report.py              ← Agent 效能指标
│   │   ├── memory.py                    ← 记忆查看
│   │   └── reset.py                     ← 系统重置
│   ├── pipeline/
│   │   ├── orchestrator.py              ← 编排器（本地备用）
│   │   ├── augment_engine.py            ← 完整增强引擎（上传到远程执行）
│   │   └── stages/                      ← 本地流水线阶段
│   ├── environment/remote.py            ← SSH 远程执行
│   └── memory/
│       ├── experiment_tree.py           ← 实验树数据结构
│       └── dead_ends.py                 ← 死胡同守卫
│
├── configs/
│   ├── asr_evolve.yaml                  ← 主配置（分级/pipeline/预算）
│   ├── remote.yaml                      ← SSH 凭证（禁区）
│   ├── training.yaml                    ← 训练配方 + 健康监控 + Probe
│   ├── augment.yaml                     ← 增强参数（7 效果链）
│   ├── model.yaml                       ← 模型配置
│   ├── evaluation.yaml                  ← 评估配置
│   ├── hotwords.txt                     ← 热词列表
│   ├── search_strategy.yaml             ← 三阶段搜索策略
│   ├── agent_metrics.yaml               ← 效能参考标准（Kiro 阅读）
│   └── hpo_history.json                 ← 超参变更追踪
│
├── memory/agent_performance.json        ← Agent 效能累计数据
├── logs/train_state.json                ← 训练状态
│
├── STATE.md                 ← 环路状态（当下快照）
├── experiment_tree.json     ← 实验搜索树
├── DEAD_ENDS.md             ← 硬性禁止（方向级）
├── INSIGHTS.md              ← 已验证经验（正面，每轮有效即记录）
├── loop-budget.md           ← 每日配额
└── loop-run-log.md          ← 运行日志
```

---

## 五、迭代优化闭环

```
THINK → DECIDE → EXECUTE → OBSERVE → JUDGE → RECORD → REPORT
```

| 阶段 | 做什么 | 数据来源 |
|:--:|--------|---------|
| THINK | 找瓶颈选方向 | 读 STATE/INSIGHTS/DEAD_ENDS + CLI 数据工具 |
| DECIDE | 确认可行性 | 读 DEAD_ENDS + INSIGHTS + loop-budget |
| EXECUTE | 训练 | gpu + augment(如需) + train submit + **train wait** |
| OBSERVE | 确认完成 | train wait 返回 / train health |
| JUDGE | 评估效果 | eval + analyse |
| RECORD | 写回记忆 | 读 analyse 报告 + CLI 工具 + 直接编辑文件 |
| REPORT | 汇报用户 | 纯输出 |

---

## 六、CLI 命令总览

| 命令 | 输出什么 |
|------|---------|
| `status` | 系统概览 |
| `gpu` | GPU 显存/利用率 |
| `train submit [--mode probe/full]` | 提交确认 |
| `train check` | 进程状态 + checkpoint 数 |
| `train wait` | 阻塞等待完成 |
| `train stop` | 停止确认 |
| `train monitor` | telemetry 数据 |
| `train health` | 本地状态摘要 |
| `augment run/status/restore` | 远程数据增强 |
| `eval` | CER/召回率 |
| `analyse` | 混淆对/设备CER/bad cases |
| `tree view/summary/add/record/dead-end` | 实验树 CRUD |
| `tree suggest-next` | UCB1 分数排序 |
| `strategy status/check-transition/params` | 搜索阶段状态 |
| `suggest` | HPO 各方向统计 |
| `state validate/summary` | 文件一致性 |
| `agent-report [--update] [--json]` | 效能指标 |
| `memory` | 记忆文件展示 |
| `reset all/remote/local` | 系统重置 |

---

## 七、环路记忆系统

| 文件 | 读写方式 | 作用 |
|------|---------|------|
| `STATE.md` | Kiro 直接读写 | 当下快照（每轮覆盖） |
| `INSIGHTS.md` | Kiro 直接读写 | 正面经验（每轮有效即追加） |
| `DEAD_ENDS.md` | Kiro 直接写 + CLI `tree dead-end` | 方向级禁止 |
| `loop-run-log.md` | Kiro 直接追加 | 运行日志 |
| `configs/hpo_history.json` | Kiro 直接追加 | 超参变更记录 |
| `experiment_tree.json` | CLI (`tree` 命令) | 实验搜索树 |
| `configs/search_strategy.yaml` | Kiro 手动修改 | 搜索阶段 |
| `memory/agent_performance.json` | CLI (`agent-report --update`) | 效能统计 |
| `loop-budget.md` | 只读 | 配额定义 |

> **STATE** = 今天的快照（每轮覆盖）
> **INSIGHTS** = 什么有效（正面经验，累积）
> **DEAD_ENDS** = 什么方向不行（负面，方向级封死）

---

## 八、闸门机制

1. `STATE.md` 含 `⚠️ LOOPS PAUSED` → 立即停止
2. `DEAD_ENDS.md` 中的方向 → 不执行
3. `loop-budget.md` 配额用完 → 暂停
4. `INSIGHTS.md` 中的经验 → 不违反

---

## 九、设计原则

1. **代码算数，LLM 判断** — CLI 不输出建议/诊断，只输出数字
2. **Skill 是思考框架** — 分析指引不是硬编码，Kiro 可灵活变通
3. **有界性** — 干预动作有边界（lr 范围、probe epoch 数）
4. **可审计** — 所有操作写入日志
5. **持久状态** — 不依赖对话历史，每次从文件重新 ground
6. **分级自主** — L1→L2→L3 渐进升级，退化时降级
7. **INSIGHTS 低门槛** — 有效实验即记录，不需要多次验证


---

## 十、完整迭代示例

> 场景：用户说"帮我把 CER 继续降下来"。当前 CER=14.90%，已有 baseline + 一轮增强实验。

### 前提状态

```
当前最优: node-2 (aug_snr_ep10) CER=14.90%, 召回=78%
搜索阶段: COARSE
今日训练: 0/3
自主级别: L2（训练前需确认）
```

---

### 0. Hook — SessionStart 自动分诊

**模块**：`.kiro/hooks/session-triage.json`

**读取**：STATE.md, experiment_tree.json, DEAD_ENDS.md, INSIGHTS.md, loop-budget.md

**输出**：
```
🔄 ASR 优化简报 | 2026-07-10
━━━━━━━━━━━━━━━━━━
📊 当前最优: node-2 CER=14.90% 召回=78%
🔴 高优先级: 插入错误31%/个别设备CER>1.0
💀 死胡同: 无
💡 洞察: 数据增强有效(CER -1.24%)
📦 今日预算: 已用 0/3 次训练
🌿 活跃分支: 1
━━━━━━━━━━━━━━━━━━
```

---

### 1. THINK — 找瓶颈选方向

**激活**：`skill-experiment`

#### Kiro 直接读文件

| 文件 | 获得信息 | 用于判断 |
|------|---------|---------|
| `STATE.md` | "插入错误31%，背景人声碎片被误识别" | 增强方向可能需要更好的 speaker_mix |
| `INSIGHTS.md` | "数据增强有效(CER -1.24%)" | 继续增强方向 |
| `DEAD_ENDS.md` | （暂无） | 无限制 |
| `loop-budget.md` | max_trains=3 | 今天还有 3 次 |

#### CLI 数据采集

```bash
# → commands/state_validate.py
asr-evolve state validate
# 输出: "状态一致性检查通过"

# → commands/suggest.py (UCB1)
asr-evolve tree suggest-next
# 输出:
#   1. [node-2] aug_snr_ep10 — UCB1=2.16, CER=0.1490 (expandable)

# → commands/suggest.py (HPO)
asr-evolve suggest
# 输出:
#   augmentation: 1/1 (100%) — 最近 effective, 参数 {epochs:10, augment_data:5000}
#   training_recipe: 0/1 — baseline
#   hotwords: 未尝试

# → commands/strategy.py
asr-evolve strategy status
# 输出: Phase=COARSE, C=2.0, max_params=3, probe不强制

# → commands/agent_report.py
asr-evolve agent-report
# 输出: 准确率 100% (1/1), 累积效率 4.13%/h
```

#### Kiro 分析（自行推理）

> INSIGHTS 说增强有效。analyse 报告说插入错误 31% 来自背景人声碎片。
> 方案：改 augment.yaml 增大 speaker_mix SNR 范围到 [-5, 20]，让模型见到更极端的背景声。
> 同时尝试 label_smoothing=0.1 防过拟合。

---

### 2. DECIDE — 确认可行性

**激活**：`skill-experiment`

#### 闸门检查

- 读 `DEAD_ENDS.md` → 空，无限制
- 读 `INSIGHTS.md` → "数据增强有效" → 与计划方向一致
- 读 `loop-budget.md` → 0/3 → 可以训练

#### 修改配置

Kiro 编辑 `configs/augment.yaml`：
```yaml
speaker_mix:
  snr_range: [-5, 20]    # 从 [-3, 15] 扩大
```

Kiro 编辑 `configs/training.yaml`：
```yaml
train_conf:
  label_smoothing: 0.1   # 新增，防过拟合
```

#### 登记实验节点

```bash
# → commands/tree.py → memory/experiment_tree.py
asr-evolve tree add --parent-id node-2 \
  --action-desc "speaker_mix SNR[-5,20] + label_smoothing=0.1" \
  --hypothesis "极端背景声+正则化降低插入错误"
# 输出: "node-3 added"
```

**文件变更**：`configs/augment.yaml`, `configs/training.yaml`, `experiment_tree.json`

---

### 3. EXECUTE — 增强 + 训练

**激活**：`skill-gpu` → `skill-pipeline` → `skill-train`

#### 3a. GPU 确认

```bash
# → commands/gpu.py → environment/remote.py
asr-evolve gpu
# 输出: "GPU 2: 15 MiB, 0%"
```

#### 3b. 数据增强

```bash
# → commands/augment_remote.py → environment/remote.py
# 上传 augment_engine.py + augment.yaml → 远程执行完整增强链
asr-evolve augment restore   # 先恢复原始数据
asr-evolve augment run       # 用新参数重新增强
# 输出:
#   [OK] augment_engine.py uploaded
#   [OK] augment.yaml uploaded
#   Running remote augmentation (namespace: aug_7f3a2b1c)...
#   Namespace: aug_7f3a2b1c
#   Chain: [reverb, speaker_mix, speed_perturb, eq_random, volume_perturb, gaussian_noise, clipping]
#   Samples: 2500
#   Augmenting 2500 samples...
#   Done. Original: 2500, Total: 4800
#   MERGED: 4800 lines
#   [OK] Augmentation complete.
```

#### 3c. L2 确认（向用户展示配置）

```
训练确认（L2）：
  方向: speaker_mix SNR[-5,20] + label_smoothing=0.1
  数据: 4800 条（原始+增强）
  配方: epoch=5, lr=1e-4, batch=64
  GPU 2 空闲，预计 ~10 分钟
  今日配额 0/3
  是否提交？
```

用户: "y"

#### 3d. 训练 + 等待

```bash
# → commands/train.py → environment/remote.py
asr-evolve train submit --tag aug_snr_wide_ls01
# 输出: "TRAIN_SUBMITTED: pid=315000"

# → commands/train.py (wait) → 每 15 秒检查进程
asr-evolve train wait
# 输出 (~10 分钟后): "TRAIN_COMPLETED: 8 checkpoints (615s)"
```

**文件变更**：`logs/train_state.json` (running → completed)

---

### 4. OBSERVE — 确认训练健康

**激活**：`skill-train`

```bash
# → commands/monitor.py (本地)
asr-evolve train health
# 输出:
#   TAG: aug_snr_wide_ls01
#   Status: completed
#   Mode: full
#   Checkpoints: 8
```

**Kiro 判断**：8 checkpoints > 0，训练成功，进入评估。

---

### 5. JUDGE — 评估 + 错误分析

**激活**：`skill-eval`

```bash
# → commands/evaluate.py → environment/remote.py
asr-evolve eval --version v0.3.0 --checkpoints best
# 输出:
#   CER: 0.1312 | Stripped CER: 0.1445 | 指令召回率: 0.8100 (243/300)

# → commands/analyse.py
asr-evolve analyse --test-set v0.3.0
# 输出: 混淆对/设备CER/bad cases
```

**Kiro 判断**：
- 基线 14.90% → 新 13.12%，改善 1.78% → **有效**
- 召回 78% → 81%，也有改善
- 插入错误从 31% 降到 25%（speaker_mix 扩大范围有效果）

---

### 6. RECORD — 写回记忆

**激活**：`skill-experiment`

#### 6a. 消费 analyse 报告

Kiro 读 `reports/analysis_v0.3.0.json`，提取关键发现。

#### 6b. CLI 操作

```bash
asr-evolve tree record --node-id node-3 --cer 0.1312 --status completed
asr-evolve strategy check-transition
# 输出: "未满足切换条件: CER=0.1312 >= 0.1"
asr-evolve agent-report --update
# 输出: "已更新。准确率 100% (2/2)"
```

#### 6c. Kiro 直接编辑文件

**STATE.md**：
```markdown
Last run: 2026-07-10（aug_snr_wide_ls01, CER=13.12%, 召回=81%）

## 高优先级
- 插入错误从31%降至25%，仍有改善空间
- 召回81%距目标90%还差9%

## 当前最优
- 模型: aug_snr_wide_ls01
- CER: 13.12%
- 召回率: 81%
```

**INSIGHTS.md** 追加：
```markdown
### [2026-07-10] speaker_mix SNR 范围扩大有效
- 证据: node-3 SNR[-5,20] + label_smoothing=0.1, CER 14.90%→13.12%
- 适用条件: 背景人声干扰严重、插入错误占比高时
- 应用方式: augment.yaml speaker_mix.snr_range 扩到 [-5, 20]
```

**hpo_history.json** 追加：
```json
{
  "node_id": "node-3",
  "date": "2026-07-10",
  "params_changed": {"speaker_mix.snr_range": "[-5,20]", "label_smoothing": 0.1},
  "params_unchanged": {"lr": "1e-4", "epochs": 5},
  "result": {"cer": 0.1312, "recall": 0.81},
  "category": "augmentation",
  "reasoning": "扩大背景声 SNR 范围+正则化降低插入错误",
  "verdict": "effective"
}
```

**loop-run-log.md** 追加：
```json
{"timestamp":"2026-07-10T10:30:00Z","loop":"iterate","action":"augment SNR扩大+label_smoothing","outcome":"effective","cer_before":0.1490,"cer_after":0.1312,"gpu_hours":0.17,"node_id":"node-3","details":"CER -1.78%, 召回+3%, 插入错误-6%"}
```

---

### 7. REPORT — 汇报用户

```
📊 本轮结果
━━━━━━━━━━━━━━━━━━
实验: [node-3] speaker_mix SNR[-5,20] + label_smoothing=0.1
CER: 14.90% → 13.12% (-1.78%)
指令召回: 81% (243/300)
判定: 有效（插入错误 31%→25%，增强方向继续）

Agent 效能: 准确率 100% (2/2) | 累积效率 3.8%/h
搜索阶段: COARSE (距 FINE 还差 CER=0.031)
━━━━━━━━━━━━━━━━━━
下一步方向:
  - 继续增强：尝试 snr_range[-10,25] 更极端
  - 或尝试 hotwords 方向（未探索，零成本）
```

---

### 模块参与矩阵

```
                     HOOK  THINK  DECIDE  EXECUTE  OBSERVE  JUDGE  RECORD
session-triage        x
state_validate              x
suggest (UCB1+HPO)          x
strategy                    x                                       x
agent_report                x                                       x
experiment_tree                    x                                x
gpu                                         x
augment_remote                              x
train (submit+wait)                         x
monitor (health)                                     x
evaluate                                                    x
analyse                                                     x
STATE.md (读)         x     x      x
STATE.md (写)                                                       x
INSIGHTS.md (读)      x     x
INSIGHTS.md (写)                                                    x
DEAD_ENDS.md (读)     x            x
hpo_history (写)                                                    x
loop-run-log (写)                                                   x
reports/*.json (读)                                                 x
```
