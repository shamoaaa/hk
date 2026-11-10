# ASR-Evolve: 基于 LLM Agent 的自主 ML 实验闭环系统

## 项目定位

设计并实现了一套 **Agentic ML Optimization System**——以 LLM 作为有界监督控制器（Bounded Supervisory Controller），在智能家居 ASR 微调场景中自主执行 Experiment Lifecycle Management。系统替代人工 ML 工程师完成"假设生成 → 实验设计 → 执行 → 评估 → 知识沉淀"的完整循环。

## 核心架构

**Sense-Plan-Act 闭环 + Durable State Continuity**

系统采用 7 阶段 Agent Loop（THINK → DECIDE → EXECUTE → OBSERVE → JUDGE → RECORD → REPORT），每阶段通过 Skill 文档注入领域知识，通过 CLI Tool Layer 获取量化信号。Agent 的推理能力负责"判断"，工具层负责"计算"，实现了 Cognitive-Computational Separation。

## 关键技术设计

**1. Experiment Space as Tree Search**

将超参/数据/配方的优化建模为带剪枝的树搜索。引入 UCB1（Upper Confidence Bound）作为节点选择策略，在 exploitation（最优分支深入）和 exploration（未知方向探测）之间数学最优地分配有限的训练预算。支持三阶段自适应搜索（Coarse/Fine/Exploit），C 值随目标指标水平动态调整。

**2. Cognitive-Computational Separation**

将 Agent 系统严格分为两层：
- **Computational Layer**（CLI 工具）：纯函数，无副作用判断。输出 UCB1 score、CER delta、confusion matrix、per-device breakdown——全部是数字
- **Cognitive Layer**（Agent 推理）：基于工具输出 + 持久记忆 + Skill 分析框架，做出"哪个方向值得""这次实验有没有效""下一步该做什么"的决策

这种分离使系统可审计（所有决策有据可查）、可调试（更换 Agent 不影响工具层）、无硬编码偏差（分析框架是 soft guidance 不是 hard rule）。

**3. Durable State & Cross-Session Memory**

解决 LLM stateless 的本质局限：通过 File-as-Bus 模式实现跨 session 的持久记忆层。
- INSIGHTS（正面经验累积）+ DEAD_ENDS（方向级剪枝）+ Experiment Tree（拓扑结构化历史）
- 每次 session cold-start 时自动 re-ground：从文件恢复全部上下文
- 区别于 RAG：不是检索碎片知识，而是维护结构化的 decision state

**4. Graduated Autonomy with Safety Bounds**

三级自主体系（L1 Report-Only / L2 Human-Gated / L3 Unattended），参考工业控制系统的 Safety Integrity Level 思想：
- 升级需满足量化前提条件（准确率阈值 + 无退化记录）
- L3 运行时的 bounded action space：预算硬上限、Probe 预验证机制、连续失败自动降级
- 可逆操作优先：增强数据按 namespace 缓存可回退，训练产物不覆盖原始数据

**5. Remote Orchestration without Shared Filesystem**

本地 Agent 通过 SSH 编排远程 GPU 集群：
- 增强引擎（771 行，7 效果 audio augmentation chain）按需上传执行，产物按配置哈希缓存
- 训练生命周期管理：nohup 异步提交 + 本地轮询等待（解决 paramiko 长连接不稳定）
- Probe 机制：2 epoch 快速验证假设有效性，pass 后再提交全量训练

## 量化成果

- 单轮迭代：人工 2-3h → Agent 自主 ~15min（10x 加速）
- 首两轮自主决策均有效，CER 16.14% → 14.90%
- 实现完整无人工干预的 Plan-Execute-Evaluate-Learn 闭环

## 技术栈

Python / Paramiko / FunASR / SeaCo-Paraformer (247M params) / NumPy+SciPy / NVIDIA RTX 4090D / Kiro Agent Framework
