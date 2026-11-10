# 需求文档：ASR训练优化系统

## 简介

本系统基于FunASR框架下的 `speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` 模型，通过大语言模型（LLM）的Skill能力，构建一个智能化的ASR模型训练优化闭环系统。系统聚焦于背景人声场景下的语音识别率优化，能够自动分析训练过程、训练结果和数据集，并基于分析结果给出模型结构优化、训练参数调优、数据流优化和后处理增强的建议与执行方案。

## 术语表

- **Optimizer_System**：ASR训练优化系统，本项目的核心系统，负责协调各模块完成训练优化闭环
- **Training_Monitor**：训练监控模块，负责采集和展示训练过程中的指标数据（loss、学习率、CER/WER等）
- **Result_Analyzer**：训练结果分析模块，负责对训练完成后的模型评估结果进行深度分析
- **Dataset_Analyzer**：数据集分析模块，负责对训练数据集进行质量评估和问题数据识别
- **LLM_Skill_Engine**：大模型Skill引擎，基于LLM的能力，对分析结果进行推理并生成优化建议
- **Model_Optimizer**：模型结构优化模块，负责根据LLM建议调整模型结构参数
- **Param_Tuner**：训练参数调优模块，负责根据LLM建议调整训练超参数
- **Data_Pipeline**：数据流优化模块，负责数据增强、数据清洗和数据预处理流程的优化
- **Post_Processor**：后处理模块，负责对ASR识别结果进行语言模型纠错、热词替换等后处理操作
- **Paraformer_Model**：基础ASR模型，即 speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
- **CER**：字符错误率（Character Error Rate），中文ASR的核心评估指标
- **WER**：词错误率（Word Error Rate），ASR评估指标
- **SNR**：信噪比（Signal-to-Noise Ratio），衡量语音信号与背景噪声的比值
- **背景人声场景**：目标语音与非目标人声（如电视声、旁人交谈声）同时存在的场景
- **Skill_Evolution_Engine**：Skill自我进化引擎，负责基于历史优化效果反馈对LLM Skill的分析策略和建议生成逻辑进行持续学习与改进，包括自动更新Skill文件内容（阈值、置信度、规则）
- **Skill_Knowledge_Base**：Skill知识库，存储Skill在优化过程中积累的领域专家知识，包括错误模式、优化策略和效果关联，具有生命周期管理（active/stale/archived）
- **Skill_Version**：Skill版本，LLM_Skill_Engine在特定时间点的策略快照，包含分析规则、建议模板和知识库状态
- **Effect_Feedback**：效果反馈记录，记录每次Skill建议的内容、执行结果及其对CER/WER的实际影响（正面或负面）
- **Knowledge_Lifecycle**：知识生命周期，知识条目从创建到淘汰的状态流转：active（有效使用中）→ stale（超过60天未验证）→ archived（超过120天未验证或置信度过低）
- **Auto_Confidence**：自动置信度，基于实际反馈数据计算的知识条目可靠性指标，公式为 (success_count / total_count) * time_decay_factor
- **Skill_Quick_Reference**：Skill速查表，自动维护在每个Skill文件顶部的高置信度策略摘要表格，供Kiro推理时直接参考
- **Action_Verification_Principle**：行动验证原则，只有经过完整迭代闭环验证（有before/after CER对比）的信息才能写入知识库

## 需求

### 需求 1：训练过程监控与可视化

**用户故事：** 作为一名ASR算法工程师，我希望能够实时查看模型训练过程中的关键指标，以便及时发现训练异常并做出调整。

#### 验收标准

1. WHILE Paraformer_Model 处于训练状态, THE Training_Monitor SHALL 每隔指定步数采集一次训练指标，包括训练loss、验证loss、学习率和梯度范数
2. WHEN 一个训练epoch完成时, THE Training_Monitor SHALL 计算并记录该epoch的平均CER和平均WER
3. WHEN 用户请求查看训练过程时, THE Training_Monitor SHALL 以图表形式展示loss曲线、CER/WER变化曲线和学习率调度曲线
4. IF 训练loss在连续5个epoch内未下降超过阈值, THEN THE Training_Monitor SHALL 生成训练停滞告警并通知 LLM_Skill_Engine 进行分析
5. IF 训练loss出现NaN或Inf值, THEN THE Training_Monitor SHALL 立即暂停训练并记录异常上下文信息

### 需求 2：训练结果分析

**用户故事：** 作为一名ASR算法工程师，我希望能够对训练完成后的模型进行全面的结果分析，以便了解模型在不同场景下的表现。

#### 验收标准

1. WHEN 模型训练完成时, THE Result_Analyzer SHALL 在标准测试集上计算整体CER和WER
2. WHEN 模型训练完成时, THE Result_Analyzer SHALL 按音频场景类型（纯净语音、背景人声、背景噪声）分别计算CER和WER
3. WHEN 模型训练完成时, THE Result_Analyzer SHALL 生成badcase列表，包含识别错误率超过指定阈值的音频样本及其对应的参考文本和识别文本
4. WHEN 用户请求对比分析时, THE Result_Analyzer SHALL 对比当前模型与基线模型在各场景下的CER/WER差异，并以表格和图表形式展示
5. THE Result_Analyzer SHALL 对badcase进行错误类型分类，包括替换错误、插入错误和删除错误

### 需求 3：数据集分析与质量评估

**用户故事：** 作为一名ASR算法工程师，我希望能够全面了解训练数据集的质量和分布情况，以便识别数据层面的问题。

#### 验收标准

1. WHEN 用户指定一个数据集路径时, THE Dataset_Analyzer SHALL 统计数据集的基本信息，包括总样本数、总时长、平均时长和时长分布
2. THE Dataset_Analyzer SHALL 分析数据集中音频的SNR分布，并标记SNR低于指定阈值的样本为低质量样本
3. WHEN 用户请求数据分析时, THE Dataset_Analyzer SHALL 统计文本标注的字符分布、词频分布和句子长度分布
4. THE Dataset_Analyzer SHALL 检测数据集中的异常样本，包括空标注、标注与音频时长严重不匹配、以及音频文件损坏的样本
5. WHEN 数据集包含场景标签时, THE Dataset_Analyzer SHALL 统计各场景类型的样本数量和占比，并识别样本不足的场景类别
6. THE Dataset_Analyzer SHALL 生成数据质量报告，汇总所有发现的问题并按严重程度排序

### 需求 4：LLM Skill驱动的智能分析与优化建议

**用户故事：** 作为一名ASR算法工程师，我希望大模型能够基于训练过程、训练结果和数据集的分析结果，自动生成优化建议，以便加速模型迭代。

#### 验收标准

1. WHEN Training_Monitor 生成训练停滞告警时, THE LLM_Skill_Engine SHALL 分析训练指标趋势并生成包含可能原因和建议操作的诊断报告
2. WHEN Result_Analyzer 完成badcase分析时, THE LLM_Skill_Engine SHALL 对badcase进行模式识别，归纳常见错误模式并生成针对性优化建议
3. WHEN Dataset_Analyzer 完成数据质量报告时, THE LLM_Skill_Engine SHALL 分析数据问题并生成数据优化方案，包括数据清洗规则和数据增强策略
4. THE LLM_Skill_Engine SHALL 综合训练过程、训练结果和数据集三方面的分析结果，生成一份完整的优化方案，涵盖模型结构、训练参数、数据流和后处理四个维度
5. WHEN 用户确认采纳某项优化建议时, THE LLM_Skill_Engine SHALL 生成对应的可执行配置变更或代码变更
6. THE LLM_Skill_Engine SHALL 记录每次优化建议及其执行结果，形成优化历史日志以供后续参考

### 需求 5：模型结构优化

**用户故事：** 作为一名ASR算法工程师，我希望系统能够根据分析结果建议并执行模型结构层面的优化，以便提升模型在背景人声场景下的表现。

#### 验收标准

1. WHEN LLM_Skill_Engine 建议调整模型注意力机制参数时, THE Model_Optimizer SHALL 生成对应的模型配置变更，包括注意力头数和隐藏层维度的调整方案
2. WHEN LLM_Skill_Engine 建议添加语音增强前端模块时, THE Model_Optimizer SHALL 生成语音增强模块的配置，包括模块类型和参数设置
3. WHEN LLM_Skill_Engine 建议调整SeACo（Semantic-Aware Co-decoder）模块参数时, THE Model_Optimizer SHALL 生成SeACo模块的参数变更配置
4. THE Model_Optimizer SHALL 在应用模型结构变更前验证配置的合法性，确保变更后的模型可以正常初始化
5. IF 模型结构变更导致模型无法初始化, THEN THE Model_Optimizer SHALL 回滚到变更前的配置并报告错误原因

### 需求 6：训练参数调优

**用户故事：** 作为一名ASR算法工程师，我希望系统能够根据分析结果自动调整训练超参数，以便找到更优的训练配置。

#### 验收标准

1. WHEN LLM_Skill_Engine 生成训练参数优化建议时, THE Param_Tuner SHALL 生成新的训练配置文件，包括学习率、batch size、warmup步数和权重衰减等参数
2. THE Param_Tuner SHALL 支持学习率调度策略的切换，包括warmup-decay、cosine annealing和step decay策略
3. WHEN 用户确认新的训练参数配置时, THE Param_Tuner SHALL 将配置应用到训练流程并启动新一轮训练
4. THE Param_Tuner SHALL 保存每次训练的参数配置和对应的评估结果，以便进行参数效果对比
5. IF 新的训练参数导致训练在前100步内出现loss异常, THEN THE Param_Tuner SHALL 自动回滚到上一组有效参数并通知用户

### 需求 7：数据流优化

**用户故事：** 作为一名ASR算法工程师，我希望系统能够优化数据处理流程，包括数据增强和数据清洗，以便提升训练数据的质量和多样性。

**业务强制约束**：本项目面向背景人声场景，train / val / test 必须都经过加噪增强后才能用于训练和评估。原始 jsonl 仅作为增强器的输入。

#### 验收标准

1. WHEN LLM_Skill_Engine 建议进行数据增强时, THE Data_Pipeline SHALL 支持以下增强方式：添加背景人声噪声（babble）、背景音乐（music）、环境噪声（noise）和混响添加（reverb）
2. THE Data_Pipeline SHALL 支持基于SNR阈值的自动数据过滤，移除信噪比过低的训练样本（仅当数据含 snr 字段时）
3. WHEN LLM_Skill_Engine 识别到特定场景数据不足时, THE Data_Pipeline SHALL 通过数据增强为该场景生成额外的训练样本
4. THE Data_Pipeline SHALL 支持动态数据混合策略，根据模型在各场景下的表现自动调整各场景数据的采样比例
5. WHEN 数据增强完成时, THE Data_Pipeline SHALL 生成增强报告，包括增强前后的数据分布对比和增强样本数量统计
6. THE Data_Pipeline SHALL 确保数据增强过程中原始数据不被修改，所有增强数据存储在独立目录中
7. **WHERE `augmentation.require_augmentation` 配置为 true（默认值），THE 训练脚本（run_training.py）和推理脚本（run_inference.py） SHALL 强制校验输入数据路径位于增强目录下（含 `_aug` 后缀或位于 `augmented_dir` 内）；若校验失败，SHALL 终止执行并提示用户先运行 `run_augmentation.py --all`**
8. **THE 配置文件 SHALL 同时维护 `data.train_jsonl`（原始数据路径，仅供增强器读取）和 `data.augmented_train_jsonl`（增强后路径，供训练器读取）；val 和 test 同理**
9. **THE `run_augmentation.py --all` 命令 SHALL 同时对 train / val / test 三份数据执行增强，输出到 `{augmented_dir}/{split}_aug.jsonl`**

### 需求 8：后处理优化

**用户故事：** 作为一名ASR算法工程师，我希望系统能够在ASR识别结果上增加后处理步骤，以便进一步降低最终的字符错误率。

#### 验收标准

1. THE Post_Processor SHALL 支持基于热词表的识别结果纠正，将识别文本中的常见错误替换为热词表中的正确词汇
2. WHEN 用户提供领域热词表时, THE Post_Processor SHALL 加载热词表并在后处理阶段应用热词纠正
3. THE Post_Processor SHALL 支持基于语言模型的N-best重排序，从ASR模型输出的N-best候选中选择语言模型得分最高的结果
4. WHEN LLM_Skill_Engine 分析badcase发现规律性错误模式时, THE Post_Processor SHALL 生成对应的正则替换规则并应用到后处理流程中
5. THE Post_Processor SHALL 支持后处理规则的优先级配置，按照热词纠正、正则替换、语言模型重排序的顺序依次执行
6. WHEN 后处理完成时, THE Post_Processor SHALL 计算后处理前后的CER/WER变化，以量化后处理的效果

### 需求 9：优化闭环与迭代管理

**用户故事：** 作为一名ASR算法工程师，我希望系统能够管理完整的优化迭代流程，以便追踪每次优化的效果并持续改进。

#### 验收标准

1. THE Optimizer_System SHALL 维护一个优化迭代记录，包含每次迭代的优化内容、参数变更、数据变更和评估结果
2. WHEN 一次完整的优化迭代完成时, THE Optimizer_System SHALL 自动对比本次迭代与上一次迭代的CER/WER变化
3. WHEN 用户请求查看优化历史时, THE Optimizer_System SHALL 以时间线形式展示所有迭代的关键指标变化趋势
4. THE Optimizer_System SHALL 支持回滚到任意历史迭代的模型和配置状态
5. IF 连续3次优化迭代未带来CER改善, THEN THE LLM_Skill_Engine SHALL 生成深度诊断报告，分析优化瓶颈并建议调整优化方向

### 需求 10：LLM Skill自我进化与持续学习

**用户故事：** 作为一名ASR算法工程师，我希望LLM Skill模块能够基于历史优化效果进行自我学习和改进，以便随着使用次数的增加，Skill给出的分析和建议质量持续提升，逐步成为领域专家。

#### 验收标准

1. WHEN LLM_Skill_Engine 生成一条优化建议并被用户执行后, THE Skill_Evolution_Engine SHALL 记录该建议的完整上下文，包括输入分析数据、建议内容、执行参数变更和最终CER/WER变化
2. WHEN 一次优化迭代完成时, THE Skill_Evolution_Engine SHALL 将该迭代的Effect_Feedback标记为正面反馈（CER/WER改善）或负面反馈（CER/WER未改善或恶化）
3. WHILE Skill_Knowledge_Base 中累积的Effect_Feedback数量超过指定阈值时, THE Skill_Evolution_Engine SHALL 对历史反馈进行模式分析，提取高成功率的优化策略和低成功率的优化策略
4. WHEN Skill_Evolution_Engine 完成模式分析时, THE LLM_Skill_Engine SHALL 调整自身的建议生成策略，提高高成功率策略的推荐权重并降低低成功率策略的推荐权重
5. THE Skill_Knowledge_Base SHALL 持续积累领域专家知识，包括场景-问题-解决方案的关联映射、参数调整与效果的经验规律、以及数据特征与最优增强策略的对应关系
6. WHEN 用户请求查看Skill知识库时, THE Skill_Knowledge_Base SHALL 以结构化形式展示已积累的知识条目，包括知识类型、置信度和来源迭代编号
7. WHEN LLM_Skill_Engine 的建议生成策略发生变更时, THE Skill_Evolution_Engine SHALL 创建一个新的Skill_Version快照，记录当前策略规则、知识库状态和版本变更说明
8. WHEN 用户请求对比Skill版本时, THE Skill_Evolution_Engine SHALL 展示不同Skill_Version之间的建议质量指标对比，包括建议采纳率、正面反馈率和平均CER改善幅度
9. IF 新版本Skill的正面反馈率连续低于上一版本, THEN THE Skill_Evolution_Engine SHALL 自动回滚到上一版本的策略配置并通知用户
10. THE Skill_Evolution_Engine SHALL 支持用户手动触发Skill学习过程，基于指定时间范围内的Effect_Feedback重新优化建议生成策略

### 需求 11：Skill内容自动进化与知识库生命周期管理

**用户故事：** 作为一名ASR算法工程师，我希望Skill文件中的分析规则能够基于实际验证数据自动更新（阈值、置信度、策略描述），知识库条目具有生命周期管理（自动淘汰过时知识），以便系统的优化能力随使用持续增强而不是停留在初始配置。

#### 验收标准

1. WHEN 某条优化策略的验证次数达到3次且成功率与Skill文件中标注的置信度偏差超过0.1时, THE Skill_Evolution_Engine SHALL 自动更新该Skill文件中对应规则的置信度为基于实际数据计算的值
2. WHEN 某条优化策略的验证次数达到3次且实际最优阈值与Skill文件中标注的阈值不一致时, THE Skill_Evolution_Engine SHALL 自动更新该Skill文件中对应规则的阈值参数
3. WHEN 知识库中出现新的高置信度模式（confidence >= 0.8 且验证次数 >= 3）但Skill文件中没有对应规则时, THE Skill_Evolution_Engine SHALL 自动将该模式作为新规则追加到对应的Skill文件末尾
4. IF 某条Skill规则连续3次验证失败（success_rate < 0.3 且验证次数 >= 3）, THEN THE Skill_Evolution_Engine SHALL 在Skill文件中将该规则标记为deprecated并降低其推荐权重
5. WHEN Skill文件内容发生自动更新时, THE Skill_Evolution_Engine SHALL 在更新前创建版本快照，确保可回滚
6. THE Skill_Evolution_Engine SHALL 在每个Skill文件顶部自动维护一个"经验速查表"，包含该Skill领域内验证次数最多、成功率最高的策略摘要（不超过10条）
7. WHEN 一次优化迭代完成时, THE Optimizer_System SHALL 自动触发轻量级Skill审查，无需用户手动调用trigger_learning
8. WHEN 连续5次迭代完成后, THE Optimizer_System SHALL 自动触发深度Skill审查，包括合并相似知识条目、淘汰长期无效规则和重新计算所有策略权重
9. THE Skill_Knowledge_Base SHALL 为每个知识条目维护生命周期状态（active/stale/archived），其中active表示有效使用中，stale表示超过60天未被验证，archived表示超过120天未验证或置信度低于0.3
10. THE Skill_Knowledge_Base SHALL 基于实际反馈数据自动计算知识条目的置信度，公式为：confidence = (success_count / total_count) * time_decay_factor，其中time_decay_factor = 0.95 ^ (days_since_last_verification / 30)
11. THE Skill_Knowledge_Base SHALL 遵循"行动验证原则"：只有来源于完整迭代闭环（有before/after CER对比）且is_positive=True且在至少2次不同迭代中验证过的信息才能写入知识库
12. THE Skill_Knowledge_Base SHALL 在查询时默认只返回state为active的知识条目，stale和archived状态的条目需要显式指定才能查询
