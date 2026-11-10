# Requirements Document

## Introduction

本功能将业务侧已成熟的 ASR 训练全流程（位于 `背景人声/`，10 阶段 + 训练后评估/分析/瘦身）封装为一个**自动化训练流水线**，并集成进已有的 `asr-evolve/` 自动化系统（"Kiro 决策，代码执行"哲学，远程 SSH 执行训练/推理/评估，含 ShardedAugmentor 分片增量增强、pipeline.py 原子操作编排、experiment_tree 版本追踪）。

流水线遵循阶段化、幂等、可断点的设计：每个阶段在执行前检测产物是否已存在，已完成则跳过，`--force` 可强制重跑；阶段之间通过 Kaldi 格式的 scp/text/manifest 文件衔接。所有"判断类"决策点（质量阈值取舍、交叉验证分歧处理、训练配方自适应、评估后是否继续迭代）由 Agent（Kiro）做出，代码层只负责确定性执行。

落地策略分两步：先用远程已有的 AISHELL 小数据集（含分片增强）验证整条流水线骨架的连通性与正确性，再切换到真实业务数据（s10、sp4）。本流水线还需修正现有 `asr-evolve/` 中与业务实战配方相悖的训练配置缺陷，并加固训练健康检查以捕获 FunASR/hydra 静默崩溃（返回码为 0 但无检查点）。

## Glossary

- **Pipeline_Orchestrator**: 编排 10 个阶段顺序执行、检测产物、处理跳过/强制重跑的顶层组件。
- **Stage**: 流水线中的一个阶段（prepare、quality、clean、pretag、cross_valid、split、augment、merge、jsonl、train 之一）。
- **Stage_Artifact**: 某阶段产出的文件或目录（如 reflow_label/、train.jsonl、检查点），作为该阶段完成与否的判定依据。
- **Quality_Scanner**: 采集每条音频原始质量指标（时长、采样率、声道、RMS、削波率）的组件（对应 02 慢扫描）。
- **Quality_Judge**: 依据阈值对原始质量指标做合格/问题判断、输出 good/bad 清单的组件（对应 03 快速判断）。
- **Cleaner**: 依据 bad 清单删除问题音频并清理索引的破坏性组件（对应 04/05）。
- **Pretagger**: 用 FireRedASR 与 Qwen3-ASR 对未标注数据生成预标注文本的组件。
- **Cross_Validator**: 对人工标注、FireRedASR、Qwen3-ASR 三方标签做交叉验证并输出分歧清单的组件。
- **Data_Splitter**: 将 labeled 数据按设备 ID 切分为 train/dev/test 的组件。
- **Augmentor**: 仅对 train 做多说话人距离关联增强的组件（复用 ShardedAugmentor 分片增量增强）。
- **Merger**: 将业务 reflow、增强数据、公开数据集按 split 合并去重为 all_label 的组件。
- **Manifest_Generator**: 将 scp + text 转换为 FunASR 训练所需 jsonl 的组件。
- **Trainer**: 通过 torchrun 远程微调 SeacoParaformer 的组件。
- **Training_Health_Checker**: 训练结束后校验是否产出有效检查点的组件。
- **Evaluator**: 计算领域 KPI（指令召回率、剥离唤醒词前后 CER）并与基线对比的组件。
- **Agent**: 做出策略决策的 LLM（Kiro），消费流水线产出的报告并选择下一步动作。
- **Wakeword**: 唤醒词"你好萤石"及其变体列表中的任一词。
- **Instruction_Recall**: 剥离唤醒词后参考文本字符串是否为识别文本字符串子串（ref_stripped ⊆ hyp_stripped）的命中率。
- **Stripped_CER**: 剥离唤醒词后计算的字符错误率。
- **Device_ID**: 用于标识音频录制设备的标识符，作为切分隔离的分组键。
- **Kaldi_Interface**: 阶段间衔接所用的 Kaldi 格式文件（wav.scp、text、manifest.jsonl）。
- **Recipe**: 训练超参数集合（冻结参数、学习率、warmup、batch、epoch、模型选择策略、seaco_id 等）。
- **Baseline**: 用于对比的参照模型，包括 SeaCo 原始基线（seaco）与线上基线（online）。
- **Checkpoint_Variant**: 同一版本下的不同检查点选择（best、epN、avgN）。
- **AISHELL_Mode**: 使用 AISHELL 小数据集验证流水线骨架的运行模式。
- **Business_Mode**: 使用真实业务数据（s10、sp4）运行的模式。
- **Dataset**: 一个数据来源（如 s10_reflow、sp4_reflow、AISHELL、公开数据集）。

## Requirements

### Requirement 1: 阶段化编排与执行顺序

**User Story:** 作为 ASR 工程师，我希望流水线按固定阶段顺序自动编排执行，以便从原始数据一步步推进到训练完成。

#### Acceptance Criteria

1. THE Pipeline_Orchestrator SHALL 支持以下 10 个阶段：prepare、quality、clean、pretag、cross_valid、split、augment、merge、jsonl、train。
2. WHEN 用户未指定阶段子集，THE Pipeline_Orchestrator SHALL 按 prepare → quality → clean → pretag → cross_valid → split → augment → merge → jsonl → train 的顺序执行全部阶段。
3. WHEN 用户通过参数指定阶段子集，THE Pipeline_Orchestrator SHALL 仅执行指定阶段，并保持上述固定相对顺序。
4. WHEN 用户启用预览模式（dry-run），THE Pipeline_Orchestrator SHALL 输出每个阶段将执行的动作而不产生或修改任何 Stage_Artifact。
5. IF 某个非破坏性阶段执行失败，THEN THE Pipeline_Orchestrator SHALL 终止后续阶段并输出失败阶段名称与失败原因。

### Requirement 2: 幂等与可断点续跑

**User Story:** 作为 ASR 工程师，我希望流水线可中断后续跑且不重复已完成工作，以便节省算力并安全恢复。

#### Acceptance Criteria

1. WHEN 一个阶段开始执行且其 Stage_Artifact 已存在，THE Pipeline_Orchestrator SHALL 跳过该阶段并记录跳过原因。
2. WHEN 用户对某阶段指定强制重跑（force），THE Pipeline_Orchestrator SHALL 重新执行该阶段并覆盖既有 Stage_Artifact。
3. WHERE 同一阶段以相同输入与相同配置被重复执行，THE Pipeline_Orchestrator SHALL 产出内容等价的 Stage_Artifact（幂等）。
4. WHEN 一个阶段成功完成，THE Pipeline_Orchestrator SHALL 将其 Stage_Artifact 落盘到约定路径，以供后续阶段与续跑检测使用。

### Requirement 3: 阶段间 Kaldi 格式接口

**User Story:** 作为 ASR 工程师，我希望各阶段通过标准 Kaldi 文件衔接，以便阶段解耦、可独立检视与替换。

#### Acceptance Criteria

1. THE Pipeline_Orchestrator SHALL 在阶段之间使用 Kaldi_Interface（wav.scp、text、manifest.jsonl）传递数据。
2. WHEN 一个阶段读取上游产物，THE Stage SHALL 通过约定的 Kaldi_Interface 文件路径读取输入。
3. WHERE wav.scp 中记录的音频绝对路径已失效，THE Data_Splitter SHALL 依据实际音频目录重映射路径，并丢弃重映射后仍不存在的条目。
4. WHEN 一个阶段构建 manifest.jsonl，THE Stage SHALL 仅纳入 wav.scp 与 text 键交集中的条目。

### Requirement 4: 数据准备（prepare）

**User Story:** 作为 ASR 工程师，我希望将原始音频与标签转换为 FunASR 格式的 labeled/unlabeled 数据，以便后续阶段消费。

#### Acceptance Criteria

1. WHEN prepare 阶段执行且目标数据集的 labeled/（含 wav.scp 与 text）已存在，THE Pipeline_Orchestrator SHALL 跳过该数据集并报告已存在条目数量。
2. IF 目标数据集未检测到 labeled/，THEN THE Pipeline_Orchestrator SHALL 输出提示，告知需先运行对应的 prepare 子脚本。
3. THE prepare 阶段 SHALL 将产物组织为 labeled/（已标注，FunASR 格式）与 unlabeled/（未标注）两部分。

### Requirement 5: 音频质量扫描与判断解耦

**User Story:** 作为 ASR 工程师，我希望音频质量扫描与阈值判断解耦，以便慢扫描只跑一次、调整阈值时可快速重判。

#### Acceptance Criteria

1. THE Quality_Scanner SHALL 为每条音频采集时长、采样率、声道数、RMS 能量、削波率原始指标。
2. WHILE Quality_Judge 正在执行，THE Quality_Judge SHALL 依据可配置阈值将每条音频判定为合格或问题，并输出 good 清单与 bad 清单；质量结果与清单更新约束仅在 Quality_Judge 执行期间生效。
3. THE Quality_Judge SHALL 在判定为问题时记录每条音频违反的具体问题类型（如时长过短、时长过长、采样率非预期、非单声道、静音或低能量、削波）。
4. WHEN 质量阈值发生变更，THE Quality_Judge SHALL 在不重新执行 Quality_Scanner 的前提下基于既有原始指标重新判定。
5. WHEN quality 阶段执行且 bad 清单已存在，THE Pipeline_Orchestrator SHALL 跳过该数据集。

### Requirement 6: 问题音频清理（破坏性，默认关闭）

**User Story:** 作为 ASR 工程师，我希望删除问题音频是显式可选的操作，以便避免误删数据。

#### Acceptance Criteria

1. WHILE 用户未显式启用清理选项，THE Cleaner SHALL 保留全部音频且不执行任何删除。
2. WHEN 用户显式启用清理选项，THE Cleaner SHALL 依据 bad 清单删除对应音频文件并从 labeled 索引（text、wav.scp）中移除对应条目。
3. WHEN 清理在预览模式下执行，THE Cleaner SHALL 输出将删除的音频条目数量而不执行删除。
4. IF 目标数据集缺少 bad 清单，THEN THE Cleaner SHALL 跳过该数据集并记录原因。

### Requirement 7: 预标注（pretag）

**User Story:** 作为 ASR 工程师，我希望用 FireRedASR 与 Qwen3-ASR 对未标注数据生成预标注文本，以便扩充可用训练数据。

#### Acceptance Criteria

1. THE Pretagger SHALL 分别使用 FireRedASR 与 Qwen3-ASR 对 unlabeled 数据生成两份预标注文本。
2. WHEN 某模型的预标注产物已存在，THE Pipeline_Orchestrator SHALL 跳过该模型的预标注。
3. WHEN 预标注被中断后再次执行，THE Pretagger SHALL 从已完成条目之后继续（断点续标）。

### Requirement 8: 三方交叉验证（cross_valid）

**User Story:** 作为 ASR 工程师，我希望对人工、FireRedASR、Qwen3-ASR 三方标签做交叉验证，以便识别标注分歧。

#### Acceptance Criteria

1. THE Cross_Validator SHALL 对人工标注、FireRedASR 预标注、Qwen3-ASR 预标注三方文本逐条比对并输出分歧明细。
2. IF 三方标签中任一方缺失，THEN THE Cross_Validator SHALL 跳过交叉验证并记录缺失方；IF 记录缺失方的日志写入失败，THEN THE Cross_Validator SHALL 继续处理其余 Dataset 而不终止整条流水线。
3. WHEN 交叉验证产物已存在，THE Pipeline_Orchestrator SHALL 跳过该数据集。
4. WHEN 交叉验证完成，THE Cross_Validator SHALL 将分歧明细以结构化形式输出供 Agent 决策。

### Requirement 9: 按设备隔离的数据切分（split）

**User Story:** 作为 ASR 工程师，我希望按设备 ID 切分 train/dev/test 且同设备不跨 split，以便避免数据泄漏导致评估虚高。

#### Acceptance Criteria

1. THE Data_Splitter SHALL 将 labeled 数据按 90% / 5% / 5% 比例切分为 train / dev / test。
2. THE Data_Splitter SHALL 使用固定随机种子 42 执行切分。
3. THE Data_Splitter SHALL 确保同一 Device_ID 的所有音频被分配到同一个 split，不跨 split 分布。
4. WHEN 多个 Dataset 参与切分，THE Data_Splitter SHALL 对每个 Dataset 独立执行切分隔离规则。
5. WHEN 切分完成，THE Data_Splitter SHALL 为每个 split 输出对应的 wav.scp 与 text 文件；在切分处理进行过程中，THE Data_Splitter MAY 先生成其中一种文件再生成另一种（允许处理过程中的部分文件生成）。

### Requirement 10: 仅训练集的多说话人距离关联增强（augment）

**User Story:** 作为 ASR 工程师，我希望增强只作用于 train 且模拟多说话人距离场景，以便提升背景人声抑制鲁棒性而不污染评估集。

#### Acceptance Criteria

1. THE Augmentor SHALL 仅对 train split 执行增强，且不修改 dev 与 test。
2. THE Augmentor SHALL 由单个正态分布的距离因子 t 联动驱动信噪比（SNR）、混响（RT60）与低通滤波三项参数。
3. THE Augmentor SHALL 强制背景人声的能量低于主说话人的能量。
4. THE Augmentor SHALL 复用 ShardedAugmentor 的分片增量增强，仅为参数发生变化的分片重新生成音频。
5. WHEN 增强完成，THE Augmentor SHALL 输出增强后的 wav.scp 与 text 文件供合并阶段使用。

### Requirement 11: 多源数据合并去重（merge）

**User Story:** 作为 ASR 工程师，我希望将业务 reflow、增强数据与公开数据集按 split 合并去重，以便形成统一训练集。

#### Acceptance Criteria

1. THE Merger SHALL 按 split 合并业务 reflow 数据、增强数据与公开数据集，产出 all_label。
2. THE Merger SHALL 仅将增强数据合并进 train split。
3. WHEN 合并遇到重复键，THE Merger SHALL 去重并保留首次出现的条目。
4. THE Merger SHALL 仅纳入同时存在于 wav.scp 与 text 的条目。
5. WHILE 执行合并，THE Merger SHALL 保持各 split 之间的数据隔离不被破坏。

### Requirement 12: 训练 manifest 生成（jsonl）

**User Story:** 作为 ASR 工程师，我希望将合并后的 scp+text 转换为 FunASR 训练 jsonl，以便驱动训练。

#### Acceptance Criteria

1. THE Manifest_Generator SHALL 将 all_label 的 train 与 dev 分别转换为 train.jsonl 与 dev.jsonl；WHEN 仅有一个 split 存在，THE Manifest_Generator SHALL 独立转换该 split 而不要求两个 split 同时存在。
2. IF 目标 split 缺少 wav.scp，THEN THE Manifest_Generator SHALL 跳过该 split 并提示需先执行 merge。
3. THE Manifest_Generator SHALL 在 jsonl 中保留每条记录的音频路径与对应文本。

### Requirement 13: 业务实战训练配方

**User Story:** 作为 ASR 工程师，我希望训练严格遵循业务已验证的实战配方，以便复现稳定收敛与正确的模型选择。

#### Acceptance Criteria

1. THE Trainer SHALL 冻结参数 hotword_output_layer 与 seaco_decoder（仅微调主干 encoder、decoder、predictor）。
2. THE Trainer SHALL 使用学习率 2e-5。
3. THE Trainer SHALL 使用 warmup_steps 30000。
4. THE Trainer SHALL 使用 batch_size 60000（按 token 计）。
5. THE Trainer SHALL 使用 max_epoch 50。
6. THE Trainer SHALL 设置最优模型选择类型为按 loss（avg_keep_nbest_models_type = loss），保留与平均最优 5 个检查点（keep_nbest_models = 5，avg_nbest_model = 5）。
7. THE Trainer SHALL 设置 find_unused_parameters 为 true。
8. THE Trainer SHALL 传入 dataset_conf.seaco_id = 8377（NO_BIAS）。
9. IF dataset_conf.seaco_id 未被显式传入（仅检测传入标志本身，不论其是否已具有有效值），THEN THE Trainer SHALL 在训练启动前报错，提示 seaco_label_pad 将为空导致崩溃。
10. IF seaco_id 缺失的报错输出本身失败，THEN THE Trainer SHALL 仍允许训练启动而不因报错失败而终止。

### Requirement 14: 训练健康检查

**User Story:** 作为 ASR 工程师，我希望训练后强制校验检查点产出，以便捕获 FunASR/hydra 静默崩溃（返回码 0 但无检查点）。

#### Acceptance Criteria

1. WHEN 训练进程结束，THE Training_Health_Checker SHALL 校验输出目录中是否存在至少一个有效的 .pt 检查点。
2. IF 训练进程返回码为 0 但输出目录中不存在任何有效 .pt 检查点，THEN THE Training_Health_Checker SHALL 将该次训练判定为失败并报告静默崩溃。
3. WHEN 训练被判定为失败，THE Pipeline_Orchestrator SHALL 阻止进入评估阶段并向 Agent 上报失败详情。

### Requirement 15: 修正现有 asr-evolve 训练配方缺陷

**User Story:** 作为 ASR 工程师，我希望自动化系统修正与业务相悖的旧训练配置，以便避免 LR 空转与不收敛。

#### Acceptance Criteria

1. THE Pipeline_Orchestrator SHALL 以业务实战配方（Requirement 13）覆盖现有 asr-evolve 中与之相悖的训练配置，包括冻结 hotword_output_layer 与 seaco_decoder、学习率 2e-5、warmup_steps 30000、batch_size 60000 token，以避免 LR 空转与不收敛。

### Requirement 16: 领域 KPI 评估——指令召回率与剥离唤醒词 CER

**User Story:** 作为 ASR 工程师，我希望以领域指标为先评估模型，以便衡量业务真实可用性而非泛化字错率。

#### Acceptance Criteria

1. THE Evaluator SHALL 在计算 Instruction_Recall 前，依据唤醒词变体列表从参考文本与识别文本的首尾剥离 Wakeword。
2. THE Evaluator SHALL 将 Instruction_Recall 定义为剥离唤醒词后参考文本字符串为识别文本字符串子串（ref_stripped ⊆ hyp_stripped）的命中率。
3. WHEN 某条样本剥离唤醒词后参考文本为空，THE Evaluator SHALL 将该样本同时排除出 Instruction_Recall 统计的分子与分母——召回率的总样本数（分母）仅计入剥离后参考文本非空的样本。
4. THE Evaluator SHALL 同时计算并报告剥离唤醒词前的 CER 与剥离唤醒词后的 Stripped_CER。
5. THE Evaluator SHALL 使用字符级编辑距离计算 Stripped_CER。

### Requirement 17: 多测试集与真实业务全量评估

**User Story:** 作为 ASR 工程师，我希望在多个测试集与真实业务全量数据上评估，以便反映线上真实分布而非增强分布。

#### Acceptance Criteria

1. THE Evaluator SHALL 在多个独立测试集上分别评估并分别报告结果；THE Evaluator SHALL NOT 产生任何跨测试集的合并或聚合汇总——每个测试集必须拥有完全独立、互不聚合的报告。
2. THE Evaluator SHALL 在真实业务全量数据上评估，而非在增强后的 test 上评估。
3. WHERE 某评估数据集对应的明细文件缺失，THE Evaluator SHALL 标记该数据集结果为缺失并继续评估其余数据集。

### Requirement 18: 基线对比与版本化检查点评估

**User Story:** 作为 ASR 工程师，我希望评估强制对比基线并覆盖多个检查点变体，以便判断训练是否带来真实增益。

#### Acceptance Criteria

1. THE Evaluator SHALL 在每次评估中强制纳入 SeaCo 原始基线与 online 线上基线进行对比。
2. THE Evaluator SHALL 对每个训练版本评估 best、epN、avgN 三类 Checkpoint_Variant。
3. THE Evaluator SHALL 以语义化版本号（v0.X.0）标识每个训练版本的评估结果。
4. WHEN 评估完成，THE Evaluator SHALL 输出包含各模型 Instruction_Recall 与 Stripped_CER 的对比汇总供 Agent 决策。

### Requirement 19: Agent 决策点（"Kiro 决策，代码执行"）

**User Story:** 作为使用 Kiro 的工程师，我希望策略性判断由 Agent 做出而代码只做确定性执行，以便保持自动化系统的可解释与可控。

#### Acceptance Criteria

1. WHERE 存在质量阈值取舍决策，THE Pipeline_Orchestrator SHALL 向 Agent 提供质量指标分布并由 Agent 决定阈值，代码层不内置自动阈值策略。
2. WHERE 存在交叉验证分歧处理决策，THE Pipeline_Orchestrator SHALL 向 Agent 提供分歧清单并由 Agent 决定处理方式。
3. WHERE 存在训练配方自适应决策，THE Pipeline_Orchestrator SHALL 向 Agent 提供训练与评估结果并由 Agent 决定配方调整。
4. WHEN 一轮评估完成，THE Pipeline_Orchestrator SHALL 向 Agent 提供评估汇总并由 Agent 决定是否继续迭代。
5. THE Pipeline_Orchestrator SHALL 将每个决策点的输入数据以结构化形式持久化，供 Agent 通过 experiment_tree 与记忆系统追溯。

### Requirement 20: AISHELL 骨架验证优先于业务数据

**User Story:** 作为 ASR 工程师，我希望先用 AISHELL 小数据验证整条流水线骨架，再切换业务数据，以便低成本暴露流程缺陷。

#### Acceptance Criteria

1. WHERE 运行于 AISHELL_Mode，THE Pipeline_Orchestrator SHALL 使用远程已有的 AISHELL 小数据集（含分片增强）贯通全部阶段。
2. WHERE 运行于 Business_Mode，THE Pipeline_Orchestrator SHALL 使用业务数据集 s10 与 sp4。
3. WHEN 在 AISHELL_Mode 与 Business_Mode 之间切换，THE Pipeline_Orchestrator SHALL 仅通过数据集与配置切换实现，不改变阶段编排逻辑。

### Requirement 21: 远程 SSH 执行集成

**User Story:** 作为 ASR 工程师，我希望训练、推理、评估通过已有 asr-evolve 远程执行能力运行，以便复用 GPU 服务器与现有原子操作。

#### Acceptance Criteria

1. THE Trainer SHALL 通过 asr-evolve 的远程 SSH 执行能力在远程 GPU 服务器上运行训练。
2. THE Evaluator SHALL 通过 asr-evolve 的远程执行能力运行推理与评估。
3. WHEN 训练或评估在远程执行，THE Pipeline_Orchestrator SHALL 将每次运行登记到 experiment_tree 以支持版本追踪。
4. IF 远程执行连接失败，THEN THE Pipeline_Orchestrator SHALL 终止当前阶段并报告连接失败原因。
