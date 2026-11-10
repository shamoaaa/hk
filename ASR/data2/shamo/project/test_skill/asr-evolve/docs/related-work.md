# Related Work — Agent 驱动的模型训练优化

> 本文档梳理与 asr-evolve 设计理念相近的前沿论文和开源项目，逐一分析其设计思路，并提出对本项目的具体改进方向。

---

## 目录

1. [AI Training Manager — Bounded Closed-Loop Control](#1-ai-training-manager)
2. [AiScientist — File-as-Bus 长周期编排](#2-aiscientist)
3. [The AI Scientist v2 — Agentic Tree Search](#3-the-ai-scientist-v2)
4. [AgentHPO — LLM 做超参优化](#4-agenthpo)
5. [OptiMindTune — 多 Agent 协作 HPO](#5-optimindtune)
6. [Gödel Agent — 递归自我改进](#6-gödel-agent)
7. [EMERALD-O — Multi-Agent RL 调参](#7-emerald-o)
8. [LBW-Guard — Bounded Autonomous Training](#8-lbw-guard)
9. [MLAgentBench / MLE-Bench — Agent ML 评估基准](#9-mlagentbench--mle-bench)
10. [综合改进路线图](#10-综合改进路线图)

---

## 1. AI Training Manager

**论文**: Bounded Closed-Loop Control of Adaptive Training Recipes (arXiv:2606.29871, 2026.06)

### 设计思路

该论文提出 LLM 作为"有界监督控制器"（bounded supervisory controller）来自适应调整训练过程。核心设计：

- **Sense → Interpret → Plan → Act 控制环路**：每隔 N 步读取训练 telemetry（loss 曲线、梯度范数、验证指标），LLM 解读当前状态是否异常（overfitting / loss imbalance / exploration collapse），生成干预动作
- **有界性**（boundedness）：干预动作被限制在预定义的"安全动作空间"内（如 lr 只能乘 0.5~2.0，不能改架构），防止 LLM 做出灾难性修改
- **审计日志**：所有干预都有完整的 reasoning trace 记录，可事后审查
- **实验验证**：在 TinyStories 任务上检测并纠正 overfitting，validation loss 比固定 recipe 低 60%

### 与 asr-evolve 的关系

你的三道闸门（急停 / DEAD_ENDS / 预算）就是一种 bounded control。但你目前是**训练间**控制（一轮训练完成后再决策），而该论文是**训练中**控制（mid-run intervention）。

### 改进方向

**引入训练中监控与干预机制**

```
当前 asr-evolve 闭环:
  EXECUTE(整轮训练) → OBSERVE(训练完毕后看结果) → JUDGE

升级后的闭环:
  EXECUTE {
    每 N steps → 读 telemetry → LLM 判定 → 微调 lr/停止
  } → OBSERVE → JUDGE
```

具体实现思路：

1. **扩展 `skill-train` 的 `check` 命令**：增加 `asr-evolve train monitor --interval 50` 命令，定期拉取远程训练日志（loss、grad norm）
2. **定义安全动作空间**：在 `configs/training.yaml` 中新增 `bounds` 字段

```yaml
bounds:
  lr_multiplier: [0.3, 3.0]       # lr 可调范围
  early_stop_patience: 3           # 连续 N epoch 无改善可提前停止
  max_grad_norm_threshold: 10.0    # 超过此值告警
```

3. **审计日志**：干预记录写入 `logs/interventions.jsonl`

```json
{"timestamp": "...", "step": 120, "signal": "val_loss_plateau", "action": "lr *= 0.5", "reasoning": "..."}
```

4. **分级干预**：
   - L0（自动）：early stop、梯度裁剪
   - L1（通知）：lr 调整、batch 策略变更
   - L2（人工确认）：停止训练、回退 checkpoint

---

## 2. AiScientist

**论文**: Toward Autonomous Long-Horizon Engineering for ML Research (arXiv:2604.13018, 2026.04)

### 设计思路

该系统的核心原则是"**strong long-horizon performance requires both structured orchestration and durable state continuity**"（强长周期性能需要结构化编排 + 持久状态连续性）。

关键设计模式：

- **File-as-Bus 架构**：所有 agent 间通信通过文件系统完成。状态文件就是"消息总线"——agent 写文件发出信号，其他 agent 读文件获取上下文
- **持久化项目状态**：不依赖对话历史（会退化），而是维护 persistent project artifacts（配置、实验记录、代码产物），agent 每次重新 ground on 这些文件
- **分层编排**：Manager Agent 不直接写代码，而是将任务分解后 dispatch 给 Worker Agent，Worker 完成后写回结果文件
- **Re-grounding**：长周期任务中，agent 定期重新读取全部状态文件，而不是依赖上下文窗口中的历史记忆

### 与 asr-evolve 的关系

你的设计已经非常接近 File-as-Bus 模式：
- STATE.md / experiment_tree.json / DEAD_ENDS.md / INSIGHTS.md = "durable state"
- Kiro 每次 session 先读这些文件 = "re-grounding"
- Skill 之间不直接通信，都通过 CLI + 文件 = "file as bus"

### 改进方向

**强化 File-as-Bus 的一致性和正式性**

1. **统一状态 schema**：当前部分状态在 markdown（STATE.md），部分在 JSON（experiment_tree.json）。建议引入正式的 state protocol：

```
memory/
├── state.json          ← 结构化状态（取代 STATE.md 中的数据部分）
├── STATE.md            ← 保留为人类可读摘要（由 state.json 生成）
├── experiment_tree.json
├── dead_ends.json      ← 结构化死胡同（可查询）
├── insights.json       ← 结构化洞察（可索引）
└── budget.json         ← 机器可读预算
```

2. **引入 state version / checksum**：每次状态更新带版本号，agent 可以检测是否有并发修改

```json
{
  "version": 42,
  "last_modified": "2026-07-03T15:00:00Z",
  "last_actor": "kiro-session-xyz",
  "checksum": "sha256:..."
}
```

3. **显式的 Agent → File → Agent 通信协议**：定义哪个 stage 写什么文件、哪个 stage 读什么文件

```
DECIDE → writes → experiment_tree.json (pending node)
EXECUTE → writes → logs/train_state.json
OBSERVE → reads → logs/train_state.json
JUDGE → writes → results/{version}/eval_report.json
RECORD → reads → results/, writes → experiment_tree.json + STATE.md
```

4. **Re-grounding 协议**：在 session-triage hook 中加入 state consistency check

```bash
asr-evolve state validate   # 检查所有状态文件一致性
```

---

## 3. The AI Scientist v2

**论文**: The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search (arXiv:2504.08066, 2025.04)
**GitHub**: [github.com/SakanaAI/AI-Scientist-v2](https://github.com/sakanaai/ai-scientist-v2) (6.5k stars)
**发表**: Nature 2026，首个通过同行评审的 AI 生成论文

### 设计思路

核心创新是 **Progressive Agentic Tree Search**：

- **Experiment Manager Agent**：专职管理实验树，决定展开哪个分支、剪枝哪个方向
- **无模板依赖**：v1 需要人类写代码模板，v2 完全自主生成实验代码
- **渐进式搜索**：不是一次性生成完整实验，而是先做小规模验证（cheap probe），通过后再全量执行
- **多步回溯**：实验失败时可以回退到祖先节点尝试不同分支
- **自动论文写作**：实验结果自动整合为 LaTeX 论文

Tree Search 策略：
```
Root(研究问题)
├── Hypothesis A → Experiment A1 (probe) → [pass] → A1-full → [fail]
│                                                   └→ A2 (变体)
├── Hypothesis B → Experiment B1 (probe) → [fail, pruned]
└── Hypothesis C → ...
```

### 与 asr-evolve 的关系

你的 `experiment_tree.json` + `DEAD_ENDS.md` 已经实现了基础版的 tree search + pruning。但缺少：
- Cheap probe（廉价探测）机制
- 自动回溯策略
- 搜索优先级排序

### 改进方向

**升级实验树为正式的 Agentic Tree Search**

1. **引入 probe 机制**：训练前先做 1-2 epoch 的快速验证

```yaml
# configs/training.yaml 新增
probe:
  enabled: true
  max_epoch: 2
  pass_threshold:
    val_loss_decrease: 0.01    # 2 epoch 内 loss 需下降 1%
  on_pass: "submit full training"
  on_fail: "prune this branch, try next"
```

对应 CLI：
```bash
asr-evolve train submit --mode probe --tag <tag>    # 快速验证
asr-evolve train submit --mode full --tag <tag>     # 全量训练（probe pass 后）
```

2. **树搜索策略**：在 `experiment_tree.json` 中加入搜索元数据

```json
{
  "search_strategy": "ucb1",    // Upper Confidence Bound
  "nodes": {
    "node-3": {
      "parent": "node-1",
      "priority_score": 0.85,    // 基于预期改善量
      "probe_result": "pass",    // probe 结果
      "cost_estimate_hours": 0.5,
      "expected_cer_reduction": 0.02,
      "exploration_count": 0     // UCB1 的探索计数
    }
  }
}
```

3. **自动回溯**：当前分支连续失败时，自动选择祖先节点的其他子方向

```
asr-evolve tree suggest-next    # 基于 UCB1 或 Thompson Sampling 推荐下一个节点
```

4. **搜索预算分配**：根据预期回报分配训练资源

```
高预期 + 低成本 → 优先
高预期 + 高成本 → 先 probe
低预期 + 低成本 → 填充
低预期 + 高成本 → 跳过
```

---

## 4. AgentHPO

**论文**: Large Language Model Agent for Hyper-Parameter Optimization (arXiv:2402.01881, ICML 2025)
**GitHub**: [github.com/michaelrzhang/LLM-HyperOpt](https://github.com/michaelrzhang/LLM-HyperOpt)

### 设计思路

AgentHPO 让 LLM 模拟人类 ML 工程师做超参搜索：

- **观察**（Observe）：读取训练日志、loss 曲线、指标变化
- **分析**（Analyze）：用 LLM 理解训练动态（过拟合？欠拟合？学习率太大？）
- **建议**（Suggest）：生成下一组超参配置，附带 reasoning
- **执行**（Execute）：调用训练脚本
- **历史利用**（History）：维护所有已尝试配置及其结果，避免重复

与传统 HPO（Bayesian Optimization、Grid Search）的区别：
- 不需要预定义搜索空间（LLM 自己决定调什么）
- 可以利用语义知识（"val loss 抖动说明 lr 太大"）
- 可以做跨任务迁移（从其他任务的经验推断）

### 与 asr-evolve 的关系

你的 THINK→DECIDE 阶段本质就是 AgentHPO 的 Observe→Analyze→Suggest。你已有的 INSIGHTS.md 就是"历史利用"。

### 改进方向

**引入结构化的超参搜索记录和跨实验知识迁移**

1. **超参变更追踪表**：在每次实验中明确记录"改了什么 → 效果如何"

```json
// configs/hpo_history.json
{
  "trials": [
    {
      "node_id": "node-1",
      "params_changed": {"lr": "1e-4", "max_epoch": 5},
      "params_unchanged": {"batch_size": 64, "accum_grad": 1},
      "result": {"cer": 0.1595, "recall": 0.7933},
      "reasoning": "baseline, first attempt",
      "verdict": "baseline established"
    },
    {
      "node_id": "node-2",
      "params_changed": {"augment.babble.prob": 0.6},
      "params_unchanged": {"lr": "1e-4"},
      "result": {"cer": 0.1490},
      "reasoning": "增强嘈杂鲁棒性",
      "verdict": "effective, CER -0.98%"
    }
  ]
}
```

2. **自动生成调参建议**：在 THINK 阶段，基于 hpo_history 生成结构化建议

```bash
asr-evolve suggest --budget 1    # 考虑预算，推荐 1 个最优方向
```

输出：
```
建议: lr 从 1e-4 降至 5e-5
依据: 历史显示 epoch10 后 loss 仍在下降但速度变慢，可能 lr 过大导致震荡
置信度: 0.7
预期改善: CER -0.5%~-1.5%
成本: 1 次训练 (~10min)
```

3. **效果归因**：每次实验后自动分析"哪个参数变更贡献了多少"

```bash
asr-evolve analyse attribution --node node-2
```

---

## 5. OptiMindTune

**论文**: A Multi-Agent Framework for Intelligent Hyperparameter Optimization (arXiv:2505.19205, 2025.05)

### 设计思路

三个专职 Agent 协作：

| Agent | 职责 | 类比 |
|-------|------|------|
| **Recommender Agent** | 基于历史和领域知识推荐下一组超参 | 你的 THINK+DECIDE |
| **Evaluator Agent** | 运行实验并评估性能 | 你的 EXECUTE+OBSERVE |
| **Decision Agent** | 编排搜索过程，决定终止/继续/切方向 | 你的 JUDGE+RECORD |

关键技术：
- **Dynamic Knowledge Fusion**：三个 agent 共享"知识库"，Recommender 的建议会被 Decision 修正
- **LLM-Driven Warm-Starting**：利用 LLM 的领域知识做初始配置，而非随机起点
- **Adaptive Search**：搜索策略随实验进展动态调整（早期粗搜、后期精搜）

### 与 asr-evolve 的关系

你的 skill 分工模式与 OptiMindTune 的三 Agent 模式高度对应。不同点是你的 "agent" 是同一个 Kiro 切换不同 skill，而不是独立 agent。

### 改进方向

**强化 Skill 间的知识共享协议**

1. **显式的 Skill 通信接口**：每个 Skill 完成后输出结构化摘要供其他 Skill 消费

```yaml
# skill-eval 输出 → skill-experiment 消费
eval_summary:
  cer: 0.1495
  recall: 0.7933
  error_patterns:
    - type: "substitution"
      examples: ["开→关", "灯→等"]
      frequency: 23
    - type: "deletion"
      examples: ["把...删除"]
      frequency: 8
  device_breakdown:
    DEV001: {cer: 0.12}
    DEV003: {cer: 0.28}
  recommendation: "focus on DEV003 and substitution errors"
```

2. **搜索策略自适应**：

```yaml
# configs/search_strategy.yaml
strategy:
  phase: "coarse"    # coarse / fine / exploitation
  switch_rules:
    coarse_to_fine: "cer < 0.10"     # CER 进入 10% 以内切精搜
    fine_to_exploit: "cer < 0.05"    # CER 进入 5% 以内切利用
  coarse:
    param_step_size: "large"          # 大步调参
    explore_ratio: 0.7
  fine:
    param_step_size: "small"
    explore_ratio: 0.3
  exploit:
    focus: "best_known_region"
    explore_ratio: 0.1
```

3. **Decision Gate 标准化**：在 JUDGE 阶段用统一的 Decision Protocol

```bash
asr-evolve judge --node <id>
# 输出:
# VERDICT: effective | ineffective | inconclusive
# REASONING: ...
# NEXT_ACTION: continue_branch | prune | pivot | stop
```

---

## 6. Gödel Agent

**论文**: Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement (arXiv:2410.04444, ACL 2025)
**GitHub**: [github.com/Arvid-pku/Godel_Agent](https://github.com/Arvid-pku/Godel_Agent)

### 设计思路

受哥德尔机启发，Agent 能递归改进自己的行为策略：

- **自我观察**（Self-Observation）：Agent 执行任务后审查自己的决策过程
- **策略修复**（Policy Repair）：从失败经验中抽象出"策略补丁"，更新自己的行为规则
- **持久化策略**（Persistent Policy）：改进后的规则保存并在未来任务中自动应用
- **保守检查**（Conservative Check）：新策略必须在历史用例上不退化才被采纳

后续工作 POLARIS (arXiv:2603.23129) 将此机制应用于小模型：
- 失败 → 分析错误 → 生成策略修订 → 抽象为通用规则 → 最小化代码补丁 → 保守验证

### 与 asr-evolve 的关系

你的 INSIGHTS.md 是策略改进的初级形态：记录"什么有效/无效"。但目前 insights 是被动参考，不能自动影响决策。

### 改进方向

**让 INSIGHTS 从"参考文档"升级为"可执行策略"**

1. **Insights → Rules 自动化**：当 insight 被验证两次以上，自动提取为 decision rule

```json
// configs/decision_rules.json
{
  "rules": [
    {
      "id": "rule-001",
      "source_insight": "SeaCo热词只适合专有名词",
      "condition": "hotword_candidate is common_word",
      "action": "reject",
      "confidence": 0.95,
      "validated_count": 2
    },
    {
      "id": "rule-002",
      "source_insight": "batch_type=example避免OOM",
      "condition": "training.batch_type != 'example'",
      "action": "warn: OOM risk",
      "confidence": 0.9,
      "validated_count": 3
    }
  ]
}
```

2. **DECIDE 阶段自动执行 rule check**：

```bash
asr-evolve decide validate --config configs/training.yaml
# 输出:
# ✅ rule-001: no conflict
# ⚠️ rule-002: batch_type=length detected, OOM risk per validated insight
# Suggestion: change to batch_type=example
```

3. **策略演化追踪**：

```
v1.0: 基线策略（无 rules）
v1.1: +rule-001（热词只放专有名词）
v1.2: +rule-002（batch_type=example）
v1.3: +rule-003（epoch>10 需降 lr）
```

4. **退化检测**：新 rule 上线后如果 CER 变差，自动回滚

```bash
asr-evolve rules rollback --rule rule-003 --reason "CER regression +0.5%"
```

---

## 7. EMERALD-O

**论文**: EMERALD-O: Efficient Multi-Agent Reinforcement Learning Framework for Optimised Deep Learning Hyperparameter Tuning (Springer Applied Intelligence, 2025)

### 设计思路

用 Multi-Agent RL 代替传统超参搜索：

- **Agent1**：专注数据增强策略优化（augmentation policy）
- **Agent2**：专注学习率和优化器选择
- 两个 Agent 通过共享环境反馈（模型性能）间接协作
- 使用 RL reward signal = validation performance improvement

关键洞察：将超参优化分解为独立但相关的子问题，每个 agent 优化一个子空间，避免联合搜索的指数爆炸。

### 与 asr-evolve 的关系

你的优化方向优先级（热词→增强调参→训练配方→加数据）本质上也是将优化分解为子空间。但你目前是线性优先级，不是并行优化。

### 改进方向

**将优化方向从线性优先级升级为并行子空间搜索**

1. **定义正交子空间**：

```yaml
# configs/search_spaces.yaml
subspaces:
  augmentation:
    owner: "augment.yaml"
    params: [babble_prob, noise_snr, speed_perturb, volume_perturb]
    search_budget: 2/day   # 该子空间每日最多 2 次实验
  training_recipe:
    owner: "training.yaml"
    params: [lr, max_epoch, warmup, scheduler]
    search_budget: 1/day
  hotwords:
    owner: "hotwords.txt"
    params: [word_list, bias_weight]
    search_budget: 3/day   # 免训练，成本低
```

2. **子空间独立进度追踪**：

```
📊 子空间进度
━━━━━━━━━━━━━━━━━━
增强子空间: 3/5 尝试，最优 babble=0.6 (CER -0.98%)
训练配方: 1/3 尝试，最优 epoch=10 (CER -0.98%)
热词: 2/5 尝试，无效（通用词不行）
━━━━━━━━━━━━━━━━━━
```

3. **跨子空间效果叠加验证**：当各子空间分别找到最优后，做一次联合验证

```bash
asr-evolve experiment combined --subspaces augmentation,training_recipe
```

---

## 8. LBW-Guard

**论文**: Bounded Autonomous Training Under Stress for Stability and Efficiency (arXiv:2605.19008, 2025.05)

### 设计思路

在训练过程中加入一个"治理层"（governance plane）：

- **不替代优化器**：不修改 SGD/Adam 的更新规则
- **观察训练 telemetry**：梯度范数、loss 变化率、参数更新幅度
- **识别不稳定区间**：检测 loss spike / gradient explosion / NaN
- **有界干预**：暂停更新、降低 lr、跳过 bad batch
- **保护有效计算**：目标是让 GPU 花的每一秒都产出有效的模型改善

### 与 asr-evolve 的关系

你的"预算约束"和"急停"是宏观版本的 LBW-Guard。但你没有 step 级别的稳定性保护。

### 改进方向

**增加训练稳定性监控**

1. **训练健康指标定义**：

```yaml
# configs/training.yaml 新增
health_monitors:
  loss_spike:
    threshold: 3.0         # loss 单步跳变超过 3 倍视为 spike
    action: "log_warning"
  gradient_norm:
    max: 10.0
    action: "clip_and_warn"
  nan_detection:
    action: "stop_immediately"
  convergence_stall:
    window: 50             # 50 步内 loss 无下降
    action: "reduce_lr_or_stop"
```

2. **远程训练脚本增加 telemetry 输出**：

```python
# 远程训练脚本中每 N 步输出
# logs/telemetry.jsonl
{"step": 100, "train_loss": 2.3, "val_loss": 2.5, "grad_norm": 1.2, "lr": 1e-4}
```

3. **`asr-evolve train monitor` 命令**：

```bash
asr-evolve train monitor --interval 30s --alert-on spike,stall
# 实时拉取远程 telemetry，本地判定是否需要干预
```

---

## 9. MLAgentBench / MLE-Bench

**MLAgentBench**: Evaluating Language Agents on Machine Learning Experimentation (arXiv:2310.03302, ICLR 2024)
**GitHub**: [github.com/snap-stanford/MLAgentBench](https://github.com/snap-stanford/MLAgentBench)

**MLE-Bench**: OpenAI 的 ML 工程 Agent 评估基准
**GitHub**: [github.com/openai/mle-bench](https://github.com/openai/mle-bench)

**MLR-Bench**: Evaluating AI Agents on Open-Ended ML Research (arXiv:2505.19955, 2025.05)

### 设计思路

这些基准定义了"Agent 做 ML 实验"的标准化评估维度：

- **任务完成率**：给定 ML 任务，agent 能否端到端完成
- **改善幅度**：相比 baseline，性能提升多少
- **效率**：用了多少次实验/多少 GPU 时间
- **可靠性**：多次运行的方差
- **可解释性**：agent 的决策过程是否可审计

MLAgentBench 发现 Claude v3 Opus 的任务成功率最高；MLR-Bench 发现 agent 在 ideation 和 writing 强，但 coding 产出的实验结果不可靠。

### 与 asr-evolve 的关系

你目前没有系统化的"自我评估"机制。loop-run-log 记录了过程，但没有量化 agent 自身的决策效果。

### 改进方向

**引入 Agent 效能自评估**

1. **定义 Agent 效能指标**：

```yaml
# configs/agent_metrics.yaml
metrics:
  decision_accuracy:
    definition: "有效实验 / 总实验次数"
    target: "> 0.5"    # 超过一半的实验应该有效
  convergence_speed:
    definition: "达到目标 CER 所需的实验轮数"
    baseline: null      # 待建立
  budget_efficiency:
    definition: "CER 改善量 / GPU 时间"
    unit: "% per hour"
  dead_end_rate:
    definition: "进入死胡同的频率"
    target: "< 0.3"
```

2. **定期生成效能报告**：

```bash
asr-evolve agent-report --last 10
```

输出：
```
📊 Agent 效能报告 (最近 10 轮)
━━━━━━━━━━━━━━━━━━
决策准确率: 4/10 = 40% (⚠️ 低于 50% 目标)
有效改善: CER 18% → 14.95% (-3.05%)
GPU 时间: 5.2 hours
效率: 0.59% per hour
死胡同率: 1/10 = 10% ✅
━━━━━━━━━━━━━━━━━━
改善建议: 决策准确率偏低，考虑增加 probe 机制减少无效实验
```

3. **将效能指标纳入 REPORT 阶段**：每轮结束不只报告模型指标，也报告 agent 自身表现

---

## 10. 综合改进路线图

基于以上分析，按实现难度和预期收益排序：

### Phase 1: 低成本高收益（1-2 天）

| 改进 | 来源论文 | 具体动作 |
|------|---------|---------|
| 结构化 HPO 历史 | AgentHPO | 新增 `configs/hpo_history.json`，每次实验自动追加 |
| Insights → Rules | Gödel Agent | 新增 `configs/decision_rules.json`，DECIDE 时自动 check |
| Agent 效能指标 | MLAgentBench | 在 REPORT 模板中加入效能统计 |

### Phase 2: 中等投入（3-5 天）

| 改进 | 来源论文 | 具体动作 |
|------|---------|---------|
| Probe 机制 | AI Scientist v2 | `asr-evolve train submit --mode probe`，2 epoch 快速验证 |
| 搜索策略自适应 | OptiMindTune | `configs/search_strategy.yaml` + 阶段切换 |
| 训练稳定性监控 | LBW-Guard | 远程脚本加 telemetry，`train monitor` 命令 |

### Phase 3: 架构升级（1-2 周）

| 改进 | 来源论文 | 具体动作 |
|------|---------|---------|
| 统一状态 schema | AiScientist | memory/ 目录结构化重构 |
| 训练中干预 | AI Training Manager | 实时监控 + bounded intervention |
| UCB1 树搜索 | AI Scientist v2 | `asr-evolve tree suggest-next` 基于 UCB1 |
| 并行子空间搜索 | EMERALD-O | search_spaces.yaml + 独立进度追踪 |

### Phase 4: 前沿探索（持续）

| 改进 | 来源论文 | 具体动作 |
|------|---------|---------|
| 策略自演化 | Gödel Agent | Agent 自动生成+验证+采纳新 decision rules |
| 跨项目迁移 | AgentHPO | insights/rules 在不同 ASR 数据集间复用 |
| 自我评估闭环 | MLR-Bench | Agent 定期审查自己的决策效能并调整策略 |

---

## 参考链接

| 项目/论文 | URL |
|-----------|-----|
| AI Training Manager | https://arxiv.org/abs/2606.29871 |
| AiScientist (Long-Horizon) | https://arxiv.org/abs/2604.13018 |
| AI Scientist v2 (Sakana) | https://github.com/SakanaAI/AI-Scientist-v2 |
| AgentHPO | https://arxiv.org/abs/2402.01881 |
| OptiMindTune | https://arxiv.org/abs/2505.19205 |
| Gödel Agent | https://arxiv.org/abs/2410.04444 |
| EMERALD-O | https://link.springer.com/article/10.1007/s10489-025-06878-4 |
| LBW-Guard | https://arxiv.org/abs/2605.19008 |
| MLAgentBench | https://github.com/snap-stanford/MLAgentBench |
| MLE-Bench | https://github.com/openai/mle-bench |
| MLR-Bench | https://arxiv.org/abs/2505.19955 |
| Gödel Agent (Code) | https://github.com/Arvid-pku/Godel_Agent |
| LLM-HyperOpt (Code) | https://github.com/michaelrzhang/LLM-HyperOpt |

---

*Content was rephrased for compliance with licensing restrictions. All information synthesized from publicly available papers and repositories.*
