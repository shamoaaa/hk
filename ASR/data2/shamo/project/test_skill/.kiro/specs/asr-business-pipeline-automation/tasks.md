# Implementation Plan: ASR Business Pipeline Automation

## Overview

将业务侧 10 阶段 ASR 训练全流程封装为 `BusinessPipeline`，作为新的环境层组件加入现有 `asr-evolve/`（`asr_evolve/` 包）。实现策略为自底向上、增量集成：先建立数据模型与配置骨架，再实现 Stage 协议与编排器（阶段顺序/跳过/dry-run/失败终止），随后按 Kaldi 接口契约逐一实现 10 个阶段（prepare → quality → clean → pretag → cross_valid → split → augment → merge → jsonl → train），最后接入业务实战配方训练、健康检查、领域 KPI 评估，并通过 `cli.py` 与 `RemoteExecutor`/`ExperimentTree` 完成端到端串联。

实现语言为 **Python**（与现有 `asr_evolve/` 代码栈一致）。属性测试使用 `hypothesis`（`@settings(max_examples=100)`），每个属性测试以注释 `# Feature: asr-business-pipeline-automation, Property {N}: {text}` 标注，并放在对应组件附近以尽早暴露缺陷。

新增代码组织在 `asr-evolve/asr_evolve/pipeline/` 子包内，复用 `environment/remote.py`、`environment/sharded_augmentor.py`、`environment/trainer.py`、`environment/evaluator.py`、`experiment_tree.py` 与 `core/config.py`。

## Tasks

- [ ] 1. 建立流水线子包、数据模型与配置骨架
  - [ ] 1.1 创建 `asr_evolve/pipeline/` 子包与运行时数据模型
    - 创建 `asr_evolve/pipeline/__init__.py` 与 `asr_evolve/pipeline/models.py`
    - 实现 `StageContext`、`ArtifactStatus`、`StageResult`、`PipelineRunReport`、`QualityThresholds`、`GoodBadLists`、`KaldiEntry`、`Sample`、`RecallReport`、`DomainReport`、`ComparisonReport` 数据类
    - 定义 `CANONICAL_STAGES` 常量与 Kaldi 接口文件路径布局辅助函数（`data/pipeline/<dataset>/...`、`all_label/...`）
    - _Requirements: 1.1, 3.1_
  - [ ] 1.2 扩展 `core/config.py` 增加 pipeline / recipe / evaluation 配置段
    - 在配置加载中新增 `pipeline`（mode、datasets、stages_root、quality_thresholds、split）、`recipe`、`evaluation`（wakewords、test_sets、baselines）段并提供默认值
    - 同步更新 `configs/asr_evolve.yaml` 示例配置
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 19.1, 20.1, 20.2_
  - [ ]* 1.3 为配置与模型加载编写单元测试
    - 验证配置段默认值解析、aishell/business 模式映射到正确数据集
    - _Requirements: 20.1, 20.2_

- [ ] 2. 实现 Stage 协议与编排器
  - [ ] 2.1 定义 `Stage` 协议与上下文
    - 在 `asr_evolve/pipeline/stage.py` 定义 `Stage`（`name`、`destructive`、`artifact_exists`、`plan`、`execute`）协议
    - _Requirements: 2.1, 2.4_
  - [ ] 2.2 实现 `PipelineOrchestrator` 编排控制流
    - 在 `asr_evolve/pipeline/orchestrator.py` 实现 `run`、`_plan`（按 CANONICAL_STAGES 过滤并排序）、`_should_skip`（artifact 存在且未 force）、`record_decision_point`（决策点结构化持久化）
    - 实现 dry-run（输出动作不写盘）、force 重跑、非破坏性阶段失败终止并记录 `halted_at`
    - 将决策点输入经 `ExperimentTree` / journal 持久化
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.4, 19.5, 20.3_
  - [ ]* 2.3 编写阶段规划不变式属性测试
    - **Property 1: 阶段规划不变式（顺序 + 子集 + 模式无关）**
    - **Validates: Requirements 1.2, 1.3, 20.3**
  - [ ]* 2.4 编写跳过/强制决策属性测试
    - **Property 2: 跳过/强制决策**
    - **Validates: Requirements 2.1, 2.2**
  - [ ]* 2.5 编写非破坏性阶段失败即终止属性测试
    - **Property 3: 非破坏性阶段失败即终止**
    - **Validates: Requirements 1.5**
  - [ ]* 2.6 编写 dry-run 不写盘属性测试
    - **Property 4: dry-run 不写盘**
    - **Validates: Requirements 1.4**

- [ ] 3. 实现 prepare 阶段
  - [ ] 3.1 实现 `PrepareStage`
    - 在 `asr_evolve/pipeline/stages/prepare.py` 实现产物组织为 labeled/（wav.scp+text）与 unlabeled/
    - labeled/ 已存在则跳过并报告条目数；未检测到 labeled/ 时输出提示需先运行 prepare 子脚本
    - _Requirements: 4.1, 4.2, 4.3_
  - [ ]* 3.2 编写 prepare 跳过分支单元测试
    - 测试 labeled/ 已存在跳过、缺失提示分支
    - _Requirements: 4.1, 4.2, 4.3_

- [ ] 4. 实现 quality 阶段（扫描与判断解耦）
  - [ ] 4.1 实现 `QualityScanner`
    - 在 `asr_evolve/pipeline/stages/quality.py` 实现 `scan`，为每条音频采集 duration/sample_rate/channels/rms/clip_rate 并写入 metrics.jsonl
    - _Requirements: 5.1_
  - [ ]* 4.2 编写质量扫描完整性属性测试
    - **Property 6: 质量扫描完整性**
    - **Validates: Requirements 5.1**
  - [ ] 4.3 实现 `QualityJudge`
    - 在 `asr_evolve/pipeline/stages/quality_judge.py` 基于既有 metrics 与可配置阈值判定 good/bad，bad 条目记录违反的具体问题类型；不重跑扫描
    - quality 阶段产物（bad 清单）已存在则跳过该数据集
    - 编排器将 metrics 分布作为决策点交给 Agent（代码不内置阈值策略）
    - _Requirements: 5.2, 5.3, 5.4, 5.5, 19.1_
  - [ ]* 4.4 编写质量判定划分与违规明细属性测试
    - **Property 7: 质量判定的划分与违规明细**
    - **Validates: Requirements 5.2, 5.3**
  - [ ]* 4.5 编写判定纯依赖既有指标属性测试
    - **Property 8: 判定纯依赖既有指标（调阈值不重扫）**
    - **Validates: Requirements 5.4**
  - [ ]* 4.6 编写 quality 跳过分支单元测试
    - 测试 bad 清单已存在跳过分支
    - _Requirements: 5.5_

- [ ] 5. 实现 clean 阶段（破坏性，默认关闭）
  - [ ] 5.1 实现 `Cleaner`
    - 在 `asr_evolve/pipeline/stages/clean.py` 实现：未启用清理时保留全部音频；启用时按 bad 清单删除音频并从 text/wav.scp 索引移除（先移除索引再删文件）
    - dry-run 时输出将删除条目数不执行删除；缺 bad 清单则跳过并记录原因
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  - [ ]* 5.2 编写清理开关安全不变式属性测试
    - **Property 9: 清理开关安全不变式**
    - **Validates: Requirements 6.1, 6.2**
  - [ ]* 5.3 编写 clean dry-run 与缺清单单元测试
    - 测试 dry-run 计数、缺 bad 清单跳过分支
    - _Requirements: 6.3, 6.4_

- [ ] 6. 实现 pretag 阶段（断点续标）
  - [ ] 6.1 实现 `Pretagger`
    - 在 `asr_evolve/pipeline/stages/pretag.py` 实现 FireRedASR 与 Qwen3-ASR 双模型预标注产出两份文本
    - 某模型产物已存在则跳过该模型；中断后从已完成条目之后续标
    - _Requirements: 7.1, 7.2, 7.3_
  - [ ]* 6.2 编写预标注断点续标属性测试
    - **Property 29: 预标注断点续标**
    - **Validates: Requirements 7.3**
  - [ ]* 6.3 编写 pretag 跳过分支单元测试
    - 测试某模型产物已存在跳过分支
    - _Requirements: 7.2_

- [ ] 7. 实现 cross_valid 阶段
  - [ ] 7.1 实现 `CrossValidator`
    - 在 `asr_evolve/pipeline/stages/cross_valid.py` 对人工/FireRedASR/Qwen3-ASR 三方文本逐条比对，输出结构化分歧明细供 Agent 决策
    - 任一方缺失则跳过该数据集并记录缺失方；记录写入失败时继续处理其余数据集；产物已存在则跳过
    - 编排器将分歧清单作为决策点交给 Agent
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 19.2_
  - [ ]* 7.2 编写三方标签分歧检测属性测试
    - **Property 30: 三方标签分歧检测**
    - **Validates: Requirements 8.1**
  - [ ]* 7.3 编写 cross_valid 缺失与跳过单元测试
    - 测试缺失方跳过、写入失败继续、产物已存在跳过分支
    - _Requirements: 8.2, 8.3, 8.4_

- [ ] 8. 实现 split 阶段（按设备隔离）
  - [ ] 8.1 实现 `DataSplitter` 切分与设备分组
    - 在 `asr_evolve/pipeline/stages/split.py` 实现 `split`：按 Device_ID 分组整组分配，固定 seed=42，90/5/5 切分，每个 split 输出 wav.scp+text；多数据集独立切分
    - 实现 `device_id_of` 从 utt_id 提取 Device_ID
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  - [ ] 8.2 实现失效路径重映射 `remap_paths`
    - wav.scp 绝对路径失效时按实际音频目录重映射，丢弃重映射后仍不存在的条目并计数上报
    - _Requirements: 3.3_
  - [ ]* 8.3 编写切分划分与比例属性测试
    - **Property 13: 切分划分与比例**
    - **Validates: Requirements 9.1**
  - [ ]* 8.4 编写切分确定性属性测试
    - **Property 14: 切分确定性（固定种子）**
    - **Validates: Requirements 9.2**
  - [ ]* 8.5 编写按设备隔离属性测试
    - **Property 15: 按设备隔离（多数据集独立）**
    - **Validates: Requirements 9.3, 9.4**
  - [ ]* 8.6 编写失效路径重映射与丢弃属性测试
    - **Property 10: 失效路径重映射与丢弃**
    - **Validates: Requirements 3.3**

- [ ] 9. Checkpoint - 确保数据处理阶段测试通过
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. 实现 augment 阶段（仅 train 距离关联增强）
  - [ ] 10.1 实现 `Augmentor`（包装 ShardedAugmentor）
    - 在 `asr_evolve/pipeline/stages/augment.py` 仅对 train 增强（不动 dev/test），由正态分布距离因子 t 联动 SNR/RT60/低通三参数，强制背景人声能量低于主说话人
    - 复用 `ShardedAugmentor` 分片增量增强，仅为参数变化分片重新生成；输出增强后 train 的 wav.scp+text
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_
  - [ ]* 10.2 编写增强仅作用于 train 属性测试
    - **Property 16: 增强仅作用于 train**
    - **Validates: Requirements 10.1**
  - [ ]* 10.3 编写距离因子联动参数属性测试
    - **Property 17: 距离因子联动参数**
    - **Validates: Requirements 10.2**
  - [ ]* 10.4 编写背景人声能量约束属性测试
    - **Property 18: 背景人声能量低于主说话人**
    - **Validates: Requirements 10.3**
  - [ ]* 10.5 编写分片增量仅重生成变化分片属性测试
    - **Property 19: 分片增量仅重生成变化分片**
    - **Validates: Requirements 10.4**

- [ ] 11. 实现 merge 阶段（多源合并去重）
  - [ ] 11.1 实现 `Merger`
    - 在 `asr_evolve/pipeline/stages/merge.py` 按 split 合并 reflow/augmented/public → all_label；增强数据仅并入 train；重复键去重保留首现；仅纳入 wav.scp ∩ text；保持 split 间隔离
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_
  - [ ]* 11.2 编写合并 split 归属与隔离属性测试
    - **Property 20: 合并的 split 归属与隔离**
    - **Validates: Requirements 11.2, 11.5**
  - [ ]* 11.3 编写合并去重保留首现属性测试
    - **Property 21: 合并去重保留首现**
    - **Validates: Requirements 11.3**

- [ ] 12. 实现 jsonl 阶段（训练 manifest 生成）
  - [ ] 12.1 实现 `ManifestGenerator`
    - 在 `asr_evolve/pipeline/stages/jsonl.py` 将 all_label/<split> 的 wav.scp+text 转换为 <split>.jsonl，仅纳入键交集，保留 audio 路径与对应 text
    - 缺 wav.scp 则跳过该 split 并提示先 merge；单 split 可独立转换
    - _Requirements: 12.1, 12.2, 12.3_
  - [ ]* 12.2 编写 manifest 仅纳入键交集属性测试
    - **Property 11: manifest 仅纳入键交集**
    - **Validates: Requirements 3.4, 11.4, 12.1**
  - [ ]* 12.3 编写 jsonl 保真属性测试
    - **Property 12: jsonl 保真**
    - **Validates: Requirements 12.3**
  - [ ]* 12.4 编写 jsonl 缺 wav.scp 跳过单元测试
    - 测试缺 wav.scp 跳过并提示先 merge 分支
    - _Requirements: 12.2_

- [ ] 13. 验证确定性阶段幂等性
  - [ ]* 13.1 编写幂等性属性测试
    - **Property 5: 幂等性（相同输入+相同配置）**
    - 覆盖 split、merge、jsonl、quality-judge 连续两次执行产物内容等价
    - **Validates: Requirements 2.3, 2.4**

- [ ] 14. Checkpoint - 确保数据流水线全链路测试通过
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 15. 实现 train 阶段（业务实战配方 + 健康检查）
  - [ ] 15.1 实现 `RecipeTrainer` 配方覆盖与启动前校验
    - 在 `asr_evolve/pipeline/stages/train.py` 实现 `build_recipe`，以 `BUSINESS_RECIPE`（freeze_param、lr=2e-5、warmup=30000、batch=60000 token、max_epoch=50、loss 选模型、keep/avg=5、find_unused_parameters=true、seaco_id=8377）覆盖现有相悖训练配置
    - 实现 `preflight_check`：仅检测 `dataset_conf.seaco_id` 键是否被显式传入，缺失则启动前报错提示 seaco_label_pad 将为空；报错输出自身失败时以 try/except 兜底仍允许启动
    - 实现 `train` 经 `RemoteExecutor` 远程 torchrun 微调
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 13.10, 15.1, 21.1_
  - [ ]* 15.2 编写业务配方覆盖相悖配置属性测试
    - **Property 26: 业务配方覆盖相悖配置**
    - **Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 15.1**
  - [ ]* 15.3 编写 seaco_id 启动前校验属性测试
    - **Property 28: seaco_id 启动前校验（仅检测传入标志）**
    - **Validates: Requirements 13.9**
  - [ ] 15.4 实现 `TrainingHealthChecker`
    - 在 `asr_evolve/pipeline/stages/health.py` 校验 output_dir 至少一个有效 .pt；返回码 0 但无 .pt 时判失败并标记静默崩溃
    - 编排器据健康检查失败阻止进入评估并上报失败详情
    - _Requirements: 14.1, 14.2, 14.3_
  - [ ]* 15.5 编写训练健康检查判定属性测试
    - **Property 27: 训练健康检查判定**
    - **Validates: Requirements 14.1, 14.2**
  - [ ]* 15.6 编写配方常量与兜底单元测试
    - 断言构建命令包含期望键值（R13.10）、覆盖相悖项（R15）、健康失败阻断评估（R14.3）
    - _Requirements: 13.10, 14.3, 15.1_

- [ ] 16. 实现领域 KPI 评估
  - [ ] 16.1 实现 `DomainEvaluator` 核心评估
    - 在 `asr_evolve/pipeline/evaluation/domain_evaluator.py` 实现 `strip_wakeword`（按变体列表首尾剥离）、`instruction_recall`（命中=strip(ref)⊆strip(hyp)，strip(ref) 为空同时排除分子分母）、`evaluate`（同时报告剥离前 CER 与剥离后 Stripped_CER，复用 `evaluator.edit_distance_ops`，每个测试集独立报告不聚合）
    - 多测试集分别评估，业务全量数据评估，明细缺失标记 missing 并继续
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 17.1, 17.2, 17.3_
  - [ ]* 16.2 编写唤醒词首尾剥离属性测试
    - **Property 23: 唤醒词首尾剥离**
    - **Validates: Requirements 16.1**
  - [ ]* 16.3 编写指令召回率定义属性测试
    - **Property 22: 指令召回率定义**
    - **Validates: Requirements 16.2, 16.3**
  - [ ]* 16.4 编写 Stripped_CER 字符级编辑距离属性测试
    - **Property 24: Stripped_CER 为字符级编辑距离**
    - **Validates: Requirements 16.5**
  - [ ]* 16.5 编写每个测试集独立报告不聚合属性测试
    - **Property 25: 每个测试集独立报告不聚合**
    - **Validates: Requirements 17.1**
  - [ ] 16.6 实现 `evaluate_with_baselines` 基线对比与版本化评估
    - 强制纳入 seaco 与 online 基线；对每版本评估 best/epN/avgN 三类 Checkpoint_Variant；以 v0.X.0 标识；输出含各模型 Instruction_Recall 与 Stripped_CER 的对比汇总供 Agent 决策
    - _Requirements: 18.1, 18.2, 18.3, 18.4_
  - [ ]* 16.7 编写评估结构完整性单元测试
    - 断言报告含 cer 与 stripped_cer、强制基线行、检查点变体行、v0.X.0 版本号
    - _Requirements: 16.4, 18.1, 18.2, 18.3, 18.4_

- [ ] 17. 接入远程执行与版本追踪
  - [ ] 17.1 将训练/推理/评估串接 RemoteExecutor 与 ExperimentTree
    - 在编排器中将 `RecipeTrainer`/`DomainEvaluator` 经 `RemoteExecutor` 下发远程命令；每次远程训练/评估登记 `ExperimentTree` 节点
    - 远程连接失败时终止当前阶段并报告连接失败原因
    - 将配方自适应与是否继续迭代的输入作为决策点持久化供 Agent 决策
    - _Requirements: 19.3, 19.4, 21.1, 21.2, 21.3, 21.4_
  - [ ]* 17.2 编写远程执行集成/冒烟测试
    - 经 mock SSH 验证训练/推理命令下发与 experiment_tree 登记、连接失败上报
    - _Requirements: 21.1, 21.2, 21.3, 21.4_

- [ ] 18. CLI 集成与端到端骨架贯通
  - [ ] 18.1 在 `cli.py` 新增 pipeline 子命令
    - 新增 `pipeline run [--mode] [--dry-run]`、`pipeline stage --only ... [--force ...]`、`pipeline quality-judge --thresholds ...`、`pipeline eval --version --checkpoints`，复用现有参数解析风格并接入 `PipelineOrchestrator`
    - _Requirements: 1.2, 1.3, 1.4, 2.2, 20.1, 20.2, 20.3_
  - [ ]* 18.2 编写 CLI 模式解析单元测试
    - 测试 aishell/business 模式解析到正确数据集、阶段子集与 force 传递
    - _Requirements: 20.1, 20.2_
  - [ ]* 18.3 编写 AISHELL 端到端骨架冒烟测试
    - AISHELL_Mode 跑通 prepare→quality→...→jsonl，断言阶段间 Kaldi 文件契约（wav.scp/text/manifest.jsonl）
    - _Requirements: 3.1, 3.2, 20.1_

- [ ] 19. Final checkpoint - 确保全部测试通过
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- 标记 `*` 的子任务为可选（单元/属性/集成测试），可为更快 MVP 跳过，但建议实现以保证正确性。
- 每个任务引用具体需求子条款以保证可追溯。
- 属性测试使用 `hypothesis`（`@settings(max_examples=100)`），每个测试以 `# Feature: asr-business-pipeline-automation, Property {N}: {text}` 标注，对象为纯函数（编排规划、集合交集、按设备分组、合并去重、召回率、剥离唤醒词、配方合并、健康判定、preflight），mock 掉远程 SSH 与音频 I/O。
- 远程执行（R21）、外部 ASR 模型（R7.1）、配方常量与版本号格式、Kiro 决策点治理（R19）以集成/示例测试覆盖，不在属性测试范围。
- Checkpoint 任务用于增量验证；属性测试就近放置以尽早暴露缺陷。

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3", "2.1"] },
    { "id": 2, "tasks": ["2.2"] },
    { "id": 3, "tasks": ["3.1", "4.1", "5.1", "6.1", "7.1", "8.1", "10.1", "11.1", "12.1", "15.1", "15.4", "16.1"] },
    { "id": 4, "tasks": ["4.3", "8.2", "16.6", "17.1", "18.1"] },
    { "id": 5, "tasks": ["2.3", "2.4", "2.5", "2.6", "3.2", "4.2", "4.4", "4.5", "4.6", "5.2", "5.3", "6.2", "6.3", "7.2", "7.3", "8.3", "8.4", "8.5", "8.6", "10.2", "10.3", "10.4", "10.5", "11.2", "11.3", "12.2", "12.3", "12.4", "13.1", "15.2", "15.3", "15.5", "15.6", "16.2", "16.3", "16.4", "16.5", "16.7", "17.2", "18.2", "18.3"] }
  ]
}
```
