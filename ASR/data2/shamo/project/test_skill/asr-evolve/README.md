# ASR-Evolve v3: Kiro 驱动的 ASR 模型自动化优化平台

## 核心理念

**Kiro 做决策，代码做执行。** 借鉴 Loop Engineering 的设计思路，以分阶段、可观测、渐进信任的环路模式运行。

## 架构

```
┌─────────────────────────────────────────────────────┐
│  Kiro (LLM Agent) — 决策层                           │
│  消费结构化报告 → 决定阈值/配方/是否继续              │
│  ┌───────┐    ┌─────────┐    ┌─────────┐           │
│  │ THINK │───→│ EXECUTE │───→│ REFLECT │──┐        │
│  └───────┘    └─────────┘    └─────────┘  │        │
│       ↑                                    │        │
│       └────────────────────────────────────┘        │
└──────────────────────┬──────────────────────────────┘
                       │ CLI / Hooks
                       ▼
┌─────────────────────────────────────────────────────┐
│  PipelineOrchestrator — 编排层                       │
│  阶段顺序 / 跳过检测 / dry-run / 失败终止 / 决策点   │
├─────────────────────────────────────────────────────┤
│  8 个 Stage（纯执行，Kaldi 接口衔接）                │
│  prepare → quality → clean → split                  │
│  → augment → merge → jsonl → train                  │
├─────────────────────────────────────────────────────┤
│  环境层（复用）                                      │
│  RemoteExecutor（SSH 执行 / 上传下载 / GPU 查询）    │
├─────────────────────────────────────────────────────┤
│  记忆层                                             │
│  STATE / DEAD_ENDS / INSIGHTS / experiment_tree     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Remote GPU (单卡 NVIDIA RTX 4090 D)                │
│  FunASR SeaCo-Paraformer 训练 & 推理                │
└─────────────────────────────────────────────────────┘
```

## 环路模式（来自 Loop Engineering）

| 环路 | 触发 | 级别 | 做什么 |
|------|------|:---:|--------|
| Session Triage | 会话开始 | L1 | 读 STATE.md 简报当前状态 |
| Post-Train Verify | 任务完成 | L2 | 验证训练产物、更新状态 |
| Config Guard | 配置保存 | L1 | 检查配置变更合理性 |

## 快速开始

```bash
cd asr-evolve
pip install -e .

# 查看系统状态
asr-evolve status

# 运行全流水线（dry-run 预览）
asr-evolve pipeline run --mode aishell --dry-run

# 单独运行某阶段
asr-evolve pipeline stage --only prepare,quality

# 评估
asr-evolve eval --version v0.1.0
```

## 自治级别（渐进信任）

| 级别 | 含义 | 需要确认的操作 |
|:---:|------|--------------|
| L1 | 仅报告+数据处理 | 无（只读+非破坏性写） |
| L2 | 辅助修复 | 训练启动、破坏性清理 |
| L3 | 自主迭代 | 仅异常时升级 |

当前默认：**L2**（训练需确认）

## 目录结构

```
asr-evolve/
├── .kiro/hooks/          环路调度（Kiro agent hooks，3 个）
├── .kiro/steering/       环路技能（持久知识，asr-optimization/safety-rules/skill-*）
├── asr_evolve/           Python 包
│   ├── cli.py            CLI 入口（asr-evolve 命令）
│   ├── commands/         各子命令实现（status/gpu/pipeline/eval/analyse/tree/memory）
│   ├── pipeline/         流水线核心
│   │   ├── orchestrator.py   编排器（顺序/幂等/dry-run/暂停检测）
│   │   ├── stages/           8 个阶段实现
│   │   ├── evaluation/       DomainEvaluator + ErrorAnalyser
│   │   └── health.py         训练健康检查（静默崩溃检测）
│   ├── environment/      远程执行（RemoteExecutor / SSH）
│   └── memory/           状态/日志/死胡同/实验树
├── configs/              训练/增强/评估/模型/远程 配置
├── scripts/              运维脚本（run_train/check_log/diag_gpu/kill）
├── logs/                 运维脚本输出日志
├── tests/                属性测试（hypothesis）
├── STATE.md              环路状态脊柱
├── LOOP.md               环路运维说明
├── AGENTS.md             协作规约
├── DEAD_ENDS.md          已验证无效的方向
├── INSIGHTS.md           持久化的优化洞察
├── loop-budget.md        GPU/训练 预算
└── loop-run-log.md       运行日志
```
