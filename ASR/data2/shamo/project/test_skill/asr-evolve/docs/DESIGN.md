# ASR-Evolve 单 Agent 自动调优系统 — 架构设计

> 基于 Vibe-Trading 的设计范式，适配 ASR 模型优化的时间尺度与安全需求。
> 核心理念：**Agent 做决策，Pipeline 做执行，Hook 做守卫**。

---

## 1. 设计目标

让一个 LLM Agent（Kiro）自主完成 ASR 模型的渐进式调优循环：

```
分析错误 → 提出假设 → 修改配方 → 提交训练 → 等待完成 → 评估结果 → 记录结论 → 决定下一步
```

在保证安全（不烧爆 GPU、不走死胡同、不破坏数据）的前提下，尽可能减少人工介入。

---

## 2. 顶层架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Kiro Agent（决策层）                                    │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │            THINK → EXECUTE → REFLECT 循环                         │  │
│  │                                                                    │  │
│  │  读 STATE.md / experiment_tree / INSIGHTS / DEAD_ENDS             │  │
│  │  → 决定下一步方向                                                   │  │
│  │  → 调用工具（pipeline/eval/analyse/config）                         │  │
│  │  → 读取结果，判断成败                                               │  │
│  │  → 更新记忆（STATE/INSIGHTS/DEAD_ENDS/tree）                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
┌─────────────────┐  ┌───────────────────┐  ┌──────────────────┐
│   工具层         │  │    守卫层          │  │    记忆层          │
│                 │  │                   │  │                  │
│ run_pipeline    │  │ Config Guard      │  │ STATE.md         │
│ eval            │  │ Budget Guard      │  │ INSIGHTS.md      │
│ analyse_errors  │  │ Dead-End Guard    │  │ DEAD_ENDS.md     │
│ update_config   │  │ Health Check      │  │ experiment_tree  │
│ read_state      │  │ Post-Train Verify │  │ loop-run-log.md  │
│ submit_train    │  │                   │  │                  │
│ check_training  │  │                   │  │                  │
│ gpu_status      │  │                   │  │                  │
└────────┬────────┘  └───────────────────┘  └──────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Pipeline 编排层                                        │
│                                                                         │
│   PipelineOrchestrator                                                  │
│   prepare → quality → clean → split → augment → merge → jsonl → train  │
│                                                                         │
│   幂等执行 / dry-run / force 重跑 / 暂停检测 / 失败终止                    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    环境层                                                 │
│                                                                         │
│   RemoteExecutor (SSH → GPU 服务器)                                      │
│   nohup 后台训练 / 轮询 / 文件上传下载 / GPU 状态查询                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心设计模式（从 Vibe-Trading 借鉴并适配）

### 3.1 借鉴的模式

| Vibe-Trading 模式 | ASR 适配形态 | 为什么借鉴 |
|---|---|---|
| BaseTool + ToolRegistry | 8 个 ASR 工具（标准接口） | Agent 通过工具与系统交互，统一调用方式 |
| Skills 渐进式加载 | `.kiro/steering/*.md`（ASR 优化知识） | 不塞满 prompt，按需加载 |
| Grounding（注入真实数据） | 注入最新实验结果到 prompt | 防止 Agent 用训练数据的旧认知做决策 |
| Goal Continuation（目标续行） | 实验树驱动的自动续行 | 目标未达（CER>10%）则自动规划下一步 |
| Content Filter 断路器 | 连续失败断路器 | 连续 2 次无改善 → 停止当前方向 |
| RunStateStore（产物持久化） | experiment_tree.json | 每次实验的配置+结果可追溯 |
| 提示词驱动的归因分析 | 提示词驱动的错误分析 | Agent 读 eval 结果自行分析错误分布 |

### 3.2 不借鉴的模式

| Vibe-Trading 模式 | 为什么不用 |
|---|---|
| 紧循环 ReAct（秒级迭代） | ASR 训练是小时级，不能在 while 循环里等 |
| 五层上下文压缩 | ASR 对话不会积累 40K token |
| 实盘门控 Mandate | 不涉及资金 |
| Swarm 多智能体 | 单 Agent 足够（决策不复杂） |
| SSE 流式推送 | CLI 够用 |

---

## 4. 工具层详细设计

### 4.1 工具清单

```python
# 所有工具返回结构化 JSON，Agent 据此做决策

class ReadStateTool(BaseTool):
    """读取 STATE.md + experiment_tree 最新状态"""
    name = "read_state"
    is_readonly = True

class AnalyseErrorsTool(BaseTool):
    """分析最近一次 eval 的错误分布 (Sub/Ins/Del 比例、高频错误词)"""
    name = "analyse_errors"
    is_readonly = True

class RunPipelineTool(BaseTool):
    """执行流水线阶段子集 (prepare/quality/clean/split/augment/merge/jsonl)"""
    name = "run_pipeline"
    is_readonly = False
    # 注意：不含 train（训练单独工具，因为需要确认）

class UpdateConfigTool(BaseTool):
    """修改增强/训练/模型配置 (augmentation.yaml / training.yaml)"""
    name = "update_config"
    is_readonly = False

class SubmitTrainTool(BaseTool):
    """提交远程训练任务 (L2 模式下暂停等人确认)"""
    name = "submit_train"
    is_readonly = False
    # 返回 {status: "submitted"/"awaiting_confirmation", job_id, estimated_minutes}

class CheckTrainingTool(BaseTool):
    """查询正在运行/已完成的训练状态"""
    name = "check_training"
    is_readonly = True
    # 返回 {status: "running"/"completed"/"failed", progress, checkpoints}

class EvalTool(BaseTool):
    """在指定模型上跑评估，返回 CER/召回/错误分布"""
    name = "eval"
    is_readonly = True

class GpuStatusTool(BaseTool):
    """查询远程 GPU 占用状态"""
    name = "gpu_status"
    is_readonly = True

class RecordInsightTool(BaseTool):
    """记录实验结论到 INSIGHTS.md / DEAD_ENDS.md / experiment_tree"""
    name = "record_insight"
    is_readonly = False

class UpdateStateTool(BaseTool):
    """更新 STATE.md（当前最优/待办/高优先级）"""
    name = "update_state"
    is_readonly = False
```

### 4.2 工具接口规范

每个工具返回统一的 JSON envelope：

```json
{
  "status": "ok" | "error" | "awaiting_confirmation",
  "data": { ... },            // 工具特定的结构化数据
  "message": "人类可读摘要"
}
```

Agent 根据 `status` 决定下一步。`"awaiting_confirmation"` 表示需要人工介入（L2 模式）。

### 4.3 SubmitTrain 的确认机制

```python
class SubmitTrainTool(BaseTool):
    def execute(self, **kwargs):
        autonomy = config["autonomy_level"]  # L1/L2/L3
        if autonomy == "L2":
            # 不立即训练，把方案写入待确认队列
            save_pending_train(config_snapshot)
            return {"status": "awaiting_confirmation",
                    "message": "训练方案已保存，等待人工确认 (asr-evolve train --confirm)"}
        elif autonomy == "L3":
            # 直接提交
            return self._submit_and_poll(kwargs)
```

---

## 5. Agent 决策循环（松散环路，非紧循环）

### 5.1 与 Vibe-Trading AgentLoop 的关键区别

| | Vibe-Trading AgentLoop | ASR-Evolve Agent 循环 |
|---|---|---|
| 循环方式 | `while iteration < 50` 紧循环 | **事件驱动**（hook 触发） |
| 单轮耗时 | 5-30 秒 | 分钟（决策）到小时（含训练） |
| 等待方式 | 同步 stream_chat | 提交后返回，训练完成触发下一轮 |
| 上下文 | 累积在 messages 里 | STATE.md 为脊柱，每轮重新读取 |

### 5.2 三个触发点（Hook 驱动）

```
┌─ SessionStart (会话分诊) ─────────────────────────────────────────┐
│  1. 读 STATE.md / experiment_tree / INSIGHTS / DEAD_ENDS          │
│  2. 输出当前状态简报                                                │
│  3. 如果有"下一步"待办 → 建议 Agent 继续                            │
└──────────────────────────────────────────────────────────────────┘

┌─ Agent 主动执行（用户说"继续优化"或 Agent 自行决策）─────────────────┐
│  THINK: 读状态 → 分析错误 → 检查死胡同 → 提出假设                    │
│  EXECUTE: 修改配置 → 跑 pipeline(augment/merge/jsonl) → 提交训练     │
│  （此处暂停，等训练完成）                                             │
└──────────────────────────────────────────────────────────────────┘

┌─ PostTaskExec (训练后验证) ────────────────────────────────────────┐
│  1. 检查 .pt 存在性                                                │
│  2. 跑 eval → 对比基线                                             │
│  3. 判定 PASS/FAIL                                                 │
│  4. 更新 STATE / INSIGHTS / experiment_tree                         │
│  5. 如果还有预算 + 方向有效 → 建议下一轮                              │
│  6. 连续 2 次 FAIL → 标记 DEAD_END，换方向                           │
└──────────────────────────────────────────────────────────────────┘
```

### 5.3 Agent 决策的伪代码

```
# Agent 每次被触发时的思考过程（写在 steering 里指导 LLM）：

1. 读取当前状态
   state = read_state()
   current_cer = state.best_cer        # 13.60%
   error_dist = analyse_errors()       # {Sub: 35%, Ins: 22%, Del: 43%}
   dead_ends = state.dead_ends         # ["lr=5e-5", "unfreeze_seaco"]
   insights = state.insights           # ["精简增强链有效", "augment方向3/3成功"]

2. 决定方向（THINK）
   - 当前主要矛盾：Del=43%（删除错误最多）
   - 已知有效路径：augmentation 方向
   - 上次有效操作：去掉 clipping+noise（-0.92%）
   - 下一步假设：进一步调 speaker_mix SNR 范围 or 减少 volume 衰减幅度
   - 检查 DEAD_ENDS：不冲突 ✓

3. 修改配置（EXECUTE）
   update_config({
     "augmentation.speaker_mix.snr_range": [2, 12],
     "augmentation.volume_perturb.gain_range": [-5, 0]
   })

4. 执行数据流水线
   run_pipeline(stages=["augment", "merge", "jsonl"])

5. 提交训练
   result = submit_train()
   if result.status == "awaiting_confirmation":
       → 暂停，等人确认

6. 训练完成后（由 hook 触发）
   eval_result = eval(version="v0.7.0")
   if eval_result.cer < current_cer - 0.001:  # 改善 > 0.1%
       record_insight("SNR [2,12] + volume [-5,0] 有效，CER降至 {new_cer}")
       update_state(best_cer=new_cer, next_step="继续在此基础上微调")
   else:
       record_insight(type="dead_end", "SNR [2,12] + volume [-5,0] 无效")
       update_state(next_step="换方向：尝试增加 speed_perturb 范围")
```

---

## 6. 记忆层详细设计

### 6.1 四类记忆文件（语义分离）

| 文件 | 语义 | Agent 权限 | 更新时机 |
|------|------|:---:|---------|
| `STATE.md` | 当前最优 + 高优先级待办 + 观察列表 | 读写 | 每轮结束 |
| `INSIGHTS.md` | 跨实验验证的可靠结论 | 追加 | 实验成功时 |
| `DEAD_ENDS.md` | 硬禁止方向 | 追加 | 连续2次失败时 |
| `experiment_tree.json` | 完整实验 DAG（假设→配置→结果） | 追加节点 | 每次实验 |

### 6.2 experiment_tree 节点结构

```json
{
  "node_id": "node-7",
  "parent_id": "node-6",         // 基于哪个实验分支
  "action_desc": "SNR [2,12] + volume [-5,0], epoch=10, lr=1e-4",
  "hypothesis": "缩小 SNR 范围 + 减少 volume 衰减，降低 Del 同时保持 Ins 改善",
  "config_snapshot": {           // 完整配置快照（可复现）
    "augmentation": {...},
    "training": {...}
  },
  "status": "completed",         // pending/running/completed/failed
  "cer": 0.128,
  "stripped_cer": 0.14,
  "instruction_recall": 0.82,
  "metrics": {
    "sub_rate": 0.33, "ins_rate": 0.20, "del_rate": 0.47,
    "top_errors": [...]
  },
  "created_at": "...",
  "completed_at": "...",
  "tags": ["augmentation", "snr_tuning"],
  "notes": ""
}
```

### 6.3 Grounding：注入最新实验数据到 Prompt

类比 Vibe-Trading 的行情 Grounding——防止 Agent 用"训练数据里的旧认知"做决策：

```markdown
## Ground Truth — 当前实验状态

**以下是最新验证数据，你的决策必须基于这些数字，不要凭记忆。**

| Node | 方向 | CER | 召回 | 主要错误 | 结论 |
|------|------|:---:|:---:|---------|------|
| node-6 (最优) | 精简增强链 | 13.60% | 79.67% | Del=43% | 有效 |
| node-5 | lr=5e-5 | 14.90% | 78% | - | 无效 |
| node-4 | 解冻 seaco | 14.52% | 80% | - | 无效 |
| node-3 | SNR+speed+epoch | 14.52% | 80% | Del=44% | 边际 |

**有效路径**: augmentation (3/3 有效)
**无效路径**: training_recipe (0/3)
**DEAD_ENDS**: (暂无硬禁止)
**剩余预算**: 今日 0/3 训练（已用完）
```

---

## 7. 守卫层（安全护栏）

### 7.1 五道守卫

| 守卫 | 触发时机 | 做什么 | 失败处理 |
|------|---------|-------|---------|
| **Config Guard** | 配置文件保存后 | 校验 seaco_id / freeze_param / lr 合法 | 阻止保存 |
| **Dead-End Guard** | Agent 提出方案时 | 检查方案是否与 DEAD_ENDS 冲突 | 拒绝执行 |
| **Budget Guard** | 训练提交前 | 今日训练次数 < 3 | 延迟到明天 |
| **Health Check** | 训练完成后 | .pt 存在 / 无 OOM/RuntimeError | 标记失败 |
| **Progress Guard** | 评估完成后 | CER 改善 > 0.1% | 连续2次无改善 → DEAD_END |

### 7.2 渐进信任级别

```yaml
autonomy_level: L2  # 在 configs/asr_evolve.yaml 中设置

# L1: 仅报告 + 只读分析（Agent 只能 read_state / analyse_errors / eval）
# L2: 可修改配置 + 跑 pipeline，但训练需人工确认
# L3: 全自主（Agent 可自行提交训练，仅异常时升级）
```

L2 → L3 的升级条件（建议）：
- 连续 3 次实验均改善 CER
- DEAD_ENDS 无新增
- 总训练次数 > 10

### 7.3 急停机制

```markdown
# STATE.md 顶部加这行即立即暂停所有环路：
## ⚠️ LOOPS PAUSED
```

Pipeline 执行前检查此标记。Agent 也被指示在看到此标记时停止一切动作。

---

## 8. 一次完整迭代的时序图

```
人类                Kiro Agent              工具层              Pipeline         Remote GPU
 │                     │                     │                    │                 │
 │─"继续优化"─────────▶│                     │                    │                 │
 │                     │                     │                    │                 │
 │                     │─ read_state() ─────▶│                    │                 │
 │                     │◀─ {best_cer:13.6%, │                    │                 │
 │                     │    del:43%,         │                    │                 │
 │                     │    有效方向:augment} │                    │                 │
 │                     │                     │                    │                 │
 │                     │─ analyse_errors() ──▶│                    │                 │
 │                     │◀─ {top_del_words:    │                    │                 │
 │                     │    "我","的","了"}   │                    │                 │
 │                     │                     │                    │                 │
 │                     │ [THINK]             │                    │                 │
 │                     │ "Del主要在短词。     │                    │                 │
 │                     │  假设：缩小volume    │                    │                 │
 │                     │  衰减让短词不被压"   │                    │                 │
 │                     │                     │                    │                 │
 │                     │─ update_config({     │                    │                 │
 │                     │    volume:[-3,0]}) ──▶│                    │                 │
 │                     │◀─ ok ───────────────│                    │                 │
 │                     │                     │                    │                 │
 │                     │─ run_pipeline(       │                    │                 │
 │                     │    ["augment",       │─ orchestrator ────▶│                 │
 │                     │     "merge",         │                    │─ augment        │
 │                     │     "jsonl"]) ──────▶│                    │─ merge          │
 │                     │                     │                    │─ jsonl          │
 │                     │◀─ {status:ok,        │◀───────────────── │                 │
 │                     │    entries:5000}     │                    │                 │
 │                     │                     │                    │                 │
 │                     │─ submit_train() ────▶│                    │                 │
 │                     │◀─ {status:           │                    │                 │
 │                     │    awaiting_confirm} │                    │                 │
 │                     │                     │                    │                 │
 │◀─"训练方案已就绪，  │                     │                    │                 │
 │   请确认"───────────│                     │                    │                 │
 │                     │                     │                    │                 │
 │─"确认"────────────▶│                     │                    │                 │
 │                     │─ submit_train(       │                    │                 │
 │                     │    confirmed=true)──▶│────────────────────────────────────▶│
 │                     │◀─ {submitted,        │                    │                 │
 │                     │    job_id, ~40min}   │                    │   nohup train   │
 │                     │                     │                    │                 │
 │                     │    ～～ 40 分钟 ～～   │                    │      ...        │
 │                     │                     │                    │                 │
 │  (Post-Train hook)  │                     │                    │                 │
 │                     │─ check_training() ──▶│────────────────────────────────────▶│
 │                     │◀─ {completed,        │                    │                 │
 │                     │    3 checkpoints}   │                    │                 │
 │                     │                     │                    │                 │
 │                     │─ eval() ────────────▶│────────────────── ────────────────▶│ 推理
 │                     │◀─ {cer:12.8%,        │                    │                 │
 │                     │    recall:81%,       │                    │                 │
 │                     │    del:40%}         │                    │                 │
 │                     │                     │                    │                 │
 │                     │ [REFLECT]           │                    │                 │
 │                     │ "CER 13.6→12.8%     │                    │                 │
 │                     │  改善 0.8%>0.1% ✓    │                    │                 │
 │                     │  Del 43→40% ✓        │                    │                 │
 │                     │  方向有效"           │                    │                 │
 │                     │                     │                    │                 │
 │                     │─ record_insight(     │                    │                 │
 │                     │    "volume[-3,0]     │                    │                 │
 │                     │     有效") ──────────▶│                    │                 │
 │                     │─ update_state(       │                    │                 │
 │                     │    best_cer=12.8%)──▶│                    │                 │
 │                     │                     │                    │                 │
 │◀─"CER 从 13.6% 降到│                     │                    │                 │
 │   12.8%，方向正确。 │                     │                    │                 │
 │   建议继续微调      │                     │                    │                 │
 │   volume 范围"──────│                     │                    │                 │
```

---

## 9. 知识管理（Steering / Skills）

### 9.1 Steering 文件（持久知识，每次会话自动加载）

```
.kiro/steering/
├── asr-optimization.md          # ASR 优化方法论（Agent 的"教科书"）
│   - 错误类型(Sub/Ins/Del)的含义与应对
│   - 增强配方各参数的作用
│   - SeaCo-Paraformer 的冻结策略
│   - 训练超参调优经验法则
├── safety-rules.md              # 安全规则（硬约束）
│   - seaco_id 不可变
│   - freeze 必须含 hotword_output_layer
│   - 修改前必须检查 DEAD_ENDS
└── experiment-protocol.md       # 实验规范
    - 每次实验必须记录假设
    - 必须跑 eval 对比基线
    - CER 改善 < 0.1% = 无效
```

### 9.2 Session Triage（会话开始时自动注入）

Hook 触发 → 读取 STATE.md + experiment_tree 最新 3 节点 + 预算状态 → 生成简报注入到 Agent 上下文。类比 Vibe-Trading 的 ContextBuilder.build_system_prompt。

---

## 10. 与 Vibe-Trading 的对照关系

| Vibe-Trading 概念 | ASR-Evolve 对应 | 说明 |
|---|---|---|
| `AgentLoop.run()` | Hook 驱动的松散循环 | 时间尺度不同 |
| `BacktestTool` | `SubmitTrainTool` | 都是"提交计算→等结果" |
| `BaseEngine.run_backtest()` | `PipelineOrchestrator.run()` | 阶段化执行框架 |
| `metrics.csv` | eval 结果 (CER/recall) | 量化产出 |
| `提示词驱动的归因分析` | `analyse_errors` 工具 + steering 指导 | Agent 自行分析错误 |
| `Goal Continuation` | experiment_tree + "CER>10%" 循环条件 | 目标未达就继续 |
| `DEAD_ENDS` 断路器 | 连续2次无改善 → 换方向 | 防止打转 |
| `Grounding（行情注入）` | Grounding（最新实验结果注入） | 防止用旧数据做决策 |
| `ProviderCapabilities` | 不需要（只用一个 LLM） | 简化 |
| `Skills 渐进式加载` | `.kiro/steering/*.md` | 类似但更简单 |
| `RunStateStore` | `experiment_tree.json` | 实验可追溯 |
| `PersistentMemory` | INSIGHTS.md + DEAD_ENDS.md | 分语义比通用记忆更精确 |

---

## 11. 目录结构（最终设计）

```
asr-evolve/
├── .kiro/
│   ├── hooks/
│   │   ├── asr-session-triage.json      # SessionStart → 读状态出简报
│   │   ├── asr-post-train-verify.json   # PostTaskExec → 健康检查+评估
│   │   └── asr-config-guard.json        # PostFileSave(.yaml) → 配置校验
│   └── steering/
│       ├── asr-optimization.md          # ASR 优化方法论
│       ├── safety-rules.md              # 安全硬约束
│       └── experiment-protocol.md       # 实验规范
├── asr_evolve/                          # Python 包
│   ├── cli.py                           # CLI 入口
│   ├── commands/                        # 子命令（status/gpu/pipeline/eval/train/...）
│   ├── tools/                           # ★ 新增：BaseTool 实现（10 个工具）
│   │   ├── base.py                      # BaseTool ABC + ToolRegistry
│   │   ├── read_state.py
│   │   ├── analyse_errors.py
│   │   ├── run_pipeline.py
│   │   ├── update_config.py
│   │   ├── submit_train.py
│   │   ├── check_training.py
│   │   ├── eval_tool.py
│   │   ├── gpu_status.py
│   │   ├── record_insight.py
│   │   └── update_state.py
│   ├── pipeline/                        # 流水线（不变）
│   │   ├── orchestrator.py
│   │   ├── stages/
│   │   └── evaluation/
│   ├── environment/                     # 远程执行（不变）
│   ├── memory/                          # 记忆管理
│   └── guards/                          # ★ 新增：守卫实现
│       ├── config_guard.py
│       ├── budget_guard.py
│       ├── dead_end_guard.py
│       └── progress_guard.py
├── configs/                             # 配置文件
├── STATE.md                             # 环路状态脊柱
├── INSIGHTS.md                          # 持久洞察
├── DEAD_ENDS.md                         # 硬禁止
├── experiment_tree.json                 # 实验 DAG
└── loop-run-log.md                      # 运行日志
```

---

## 12. 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 单 Agent vs 多 Agent | 单 Agent | 决策不复杂，方向选择 + 配方调整一个角色够了 |
| 紧循环 vs 松散环路 | 松散环路（hook 驱动） | 训练耗时小时级，不适合在 while 里等 |
| 训练同步 vs 异步 | 异步（提交后返回） | Agent 不阻塞，训练完成由 hook 触发下一步 |
| 配置直接改 vs 工具改 | 工具改（`update_config`） | 可被 Config Guard 拦截和审计 |
| L2 确认粒度 | 每次训练 | 训练是不可逆的 GPU 消耗（~30min×1卡） |
| 记忆结构 | 分语义（STATE/INSIGHTS/DEAD_ENDS/tree） | 比通用 PersistentMemory 更精确 |
| 断路器阈值 | 连续 2 次无改善 | 经验值：3 次太多浪费 GPU，1 次可能误判 |
| 每日预算 | 3 次训练 | 单卡 4090D，每次 ~30min，日计 ~1.5h GPU 时间 |

---

*设计版本：v1.0 | 生成时间：2026-07-13*
