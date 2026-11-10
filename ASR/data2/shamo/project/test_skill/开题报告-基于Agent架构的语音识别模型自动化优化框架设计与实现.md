# 基于Agent架构的语音识别模型自动化优化平台设计与实现

## 二、选题依据

### 1. 阐述该选题的研究意义，或工程设计的价值和意义，国内外概况和发展趋势，选题的先进性和实用性，技术难度及工作量。

#### 1.1 选题的研究意义

随着智能家居、车载交互与智能客服等领域的快速发展，语音识别（Automatic Speech Recognition, ASR）技术已成为人机交互的核心入口之一[1]。特别是以萤石智能摄像头为代表的物联网（IoT）设备，普遍采用"唤醒词+控制指令"的语音交互范式。这类设备部署在家庭、办公等多样化场景中，对特定业务场景下的识别准确率、唤醒词鲁棒性及远场噪声环境的适应能力提出了极高的要求[2]。

将通用预训练ASR模型适配到具体业务场景并使其达到产品级要求，本质上是一个复杂的多轮迭代优化过程。这一过程涉及数据工程（数据采集、清洗、增强）、模型训练（配置调优、分布式训练）与评估分析（多维度指标计算、错误归因）三个层面的紧密协作。在实际业务实践中，ASR模型的优化迭代长期以零散脚本与大量人工经验判断的方式运行。算法工程师需要反复执行"分析当前模型瓶颈→提出优化假设→调整数据或配置→执行训练→解读评估结果→决定下一步方向"的循环。

这种工作模式存在多方面的突出问题。优化决策高度依赖个人经验，且不可复现也不可传承。资深工程师离职后，其积累的优化经验随之丧失，新人需要重新摸索。

实验管理呈线性结构，缺乏可回溯的搜索空间管理。工程师容易重复尝试已被验证无效的优化方向，造成算力与时间的浪费。例如，"增加速度扰动增强"在某配置下已经验证无效，但三个月后另一位工程师可能因为不知情而再次尝试。

各数据处理阶段以独立脚本运行，缺乏统一的编排与状态管理。一个完整的优化迭代可能涉及数据切分、增强、合并、训练清单生成、训练、评估等多个步骤，中途某步失败后往往需要从头重跑，浪费GPU算力。

训练过程的元数据（使用的配置版本、数据来源、产物之间的依赖关系）分散存储在文件系统各处，缺乏结构化的存储和查询能力，无法方便地进行跨实验对比分析[3]。

评测流程缺乏系统化管理。测试集、评估指标与评估结果之间的关联是临时性的，评估报告以文本文件形式散落在各目录中，无法进行历史趋势分析或多实验横向对比。

近年来，基于大语言模型（Large Language Model, LLM）的智能Agent技术取得了突破性进展[4][5]。Agent通过"推理—行动—观察"的循环模式，结合工具调用（Tool Use）与记忆机制（Memory），能够自主完成复杂的多步决策任务。将Agent架构引入ASR模型优化领域，以Agent作为智能决策核心驱动优化闭环，同时配合系统化的评测模块、版本管理、决策Trace观测与Web管理平台，可以形成一套完整的智能化模型优化工程系统。这一方案既能将资深工程师的领域经验结构化地沉淀到Agent的记忆系统中实现知识的可复用，又能通过平台化的管理能力使实验数据、决策过程与产物关系均可被持久化记录、查询与审计。

因此，本课题以实际业务中ASR模型优化的工程需求为出发点，设计并实现一套基于Agent架构的语音识别模型自动化优化平台，旨在将Agent的智能决策能力与系统化的评测、版本管理、观测能力相结合，为ASR模型的持续迭代优化提供端到端的平台支撑。

#### 1.2 国内外概况和发展趋势

端到端语音识别技术

语音识别技术经历了从传统HMM-DNN混合架构向端到端（End-to-End）架构的演进过程。近年来，以Transformer和Conformer[6]为骨干网络的注意力机制模型在多个基准任务上取得了领先性能。其中，非自回归（Non-autoregressive, NAR）模型因其并行解码优势和低延迟特性而受到工业界广泛关注。

高忠孝等人提出的Paraformer模型[7]通过连续积分与解触（Continuous Integrate-and-Fire, CIF）机制和双向上下文建模，在中文语音识别任务上实现了识别精度与推理速度的良好平衡，被广泛应用于工业级部署。在此基础上，石宇等人进一步提出SeaCo-Paraformer[8]，引入语义协作（Semantic Cooperation）解码器实现热词定制能力，可以在不重新训练主干模型的前提下，通过配置热词偏置列表显著提升特定词表（如唤醒词、设备控制指令）的识别率。这一特性使其特别适合IoT设备的语音交互场景。

在开源工具方面，FunASR[9]作为达摩院开源的端到端语音识别工具包，集成了Paraformer系列模型并提供了完整的训练、推理与评估流程，支持基于Hydra的配置管理和torchrun多卡分布式训练。WeNet[10]采用统一的流式/非流式架构设计，在工业部署方面具有优势。此外，SpecAugment[28]等数据增强技术被广泛用于提升模型对噪声和变体的鲁棒性。上述工具包主要解决"如何训练一个ASR模型"的问题，而对"如何系统化地管理多轮业务驱动的优化迭代过程"则缺乏支持。

LLM Agent与自动化决策

大语言模型Agent是近两年人工智能领域最活跃的研究方向之一。Yao等人提出的ReAct范式[4]将推理（Reasoning）与行动（Acting）相结合，使Agent能够在与外部环境交互的过程中进行逐步推理与决策，而非一次性生成最终答案。Shinn等人提出Reflexion[5]，通过语言形式的反馈机制使Agent具备从历史失败经验中学习的能力，避免重复犯错。在工具使用方面，Schick等人的Toolformer研究[11]证明了LLM能够学会自主判断何时需要调用外部工具以及调用哪个工具。

在应用层面，LLM Agent已被成功应用于软件工程[12]、科学发现[13]、数据分析[14]等多个领域。其中，Schmidt等人提出的AIDE框架[15]将机器学习工程建模为解空间（Solution Space）上的树搜索问题，使用Agent自主管理实验节点的创建、评估与回溯，在多个Kaggle竞赛任务上取得了接近人类专家的表现。该工作证明了"Agent+树搜索+记忆"的技术路线在实验优化类任务上的有效性。

Xi等人[23]对LLM Agent的规划（Planning）、记忆（Memory）与工具使用（Tool Use）能力进行了系统综述，指出当前Agent系统发展的关键方向包括：工具使用的模块化设计、长期记忆的高效管理、以及决策过程的可解释性。杨等人[24]从软件工程角度分析了Agent系统的架构设计模式，提出了将通用基础能力（如文件操作、命令执行）与领域专有流程（如特定业务的处理管线）分层组织的设计原则，使Agent能够通过加载不同的领域能力模块适应不同应用场景。赵鑫等人[19]在大语言模型综述中指出，Agent系统的可靠性与可控性是从研究原型走向生产部署的关键挑战。

然而，将Agent架构系统性地应用于ASR模型优化这一特定垂直领域，目前公开报道极少。现有的Agent框架多面向通用场景设计，缺乏对ASR领域特性（如按设备隔离切分防止数据泄漏、唤醒词剥离后评估指令识别率、SeaCo偏置标签校验等）的专门支持。

机器学习工程化（MLOps）与评测

随着模型规模与数据规模的持续增长，机器学习工程化（MLOps）成为工业界与学术界的共同关注点[16]。Zaharia等人提出的MLflow平台[17]从实验跟踪（Experiment Tracking）、模型注册（Model Registry）与部署管道等角度提供了标准化的机器学习生命周期管理能力。Amershi等人[22]通过对微软内部多个ML项目的案例研究，总结了机器学习系统在数据管理、模型训练与部署运维各环节面临的工程挑战。Sculley等人[3]早在2015年便指出机器学习系统中隐藏的技术债务问题，包括数据依赖不透明、实验结果不可复现等。

在AI系统评测方面，刘宇涵等人[18]对预训练模型评测方法进行了系统综述，提出了评测流程标准化的架构思路，强调评测集管理、评估指标定义与实验运行应当形成系统化的闭环。刘群等人[25]针对智能对话系统提出了多维度质量评估方法，将规则评估（如BLEU、字符错误率）与模型评估（如LLM打分）相结合，实现评估过程的自动化。陈恩红等人[27]在智能评测技术研究中强调了评测结果的可视化分析对于指导模型改进的重要性。

上述通用MLOps平台虽然提供了基础的工程化能力，但存在两方面局限：部署与学习成本较高，不适合作为轻量级组件嵌入既有研发流程；缺乏面向特定领域（如ASR）的定制能力，工作流中的判断类决策（如数据质量阈值的取舍、训练配方的自适应调整）仍需人工介入。

发展趋势

综合以上分析，当前的技术发展趋势体现在以下几个方面：ASR模型架构趋于成熟稳定，行业竞争焦点正从模型层向数据工程与优化流程的智能化方向转移[20]；LLM Agent正从通用对话助手向垂直领域的专家型自主系统演进，日益强调工具调用的可靠性、决策的可控性与能力的模块化扩展[21][23]；机器学习工程化从"通用大平台"向"领域定制、评测系统化、全链路可观测"的方向发展[22][16]。本课题正是顺应这些趋势，将Agent智能决策能力与平台化工程管理在ASR领域深度结合的一次实践探索。

#### 1.3 选题的先进性和实用性

本选题的先进性体现在以下几个方面：

自主设计LLM Agent引擎，采用"通用工具层+领域技能层"的双层能力架构。通用工具层提供文件操作、命令执行等基础能力，领域技能层将ASR优化的各处理阶段封装为符合统一接口协议的可独立调用技能（Skill）。Agent通过技能索引动态感知当前可用的领域能力，根据优化状态自主编排调用策略，而非按固定流水线顺序执行。

将评测与分析流程系统化，支持规则型评估器（CER、指令召回率、Stripped_CER等固定算法）与LLM型评估器（对bad case做语义归因分析）两类评估机制。评测产出的结构化报告（错误类型分布、混淆对、薄弱环节定位）持久化到数据库，既供Agent自主读取以决定下一步优化方向，也供工程师在Web看板上进行跨实验对比分析。

设计完整的Trace链路观测能力，将Agent每一轮优化循环中的思考（THINK）、执行（EXECUTE）、反思（REFLECT）三阶段的全部细节结构化记录为可查询的Span链路数据，支持以链路图形式在Web控制台可视化展示，实现Agent决策过程的全面可追溯。

引入三方版本管理机制，对训练配置、数据集与模型checkpoint三者建立版本关联。每个实验节点绑定精确的版本信息，任意历史实验均可通过版本快照精确复现。

本选题的实用性体现在：平台直接面向萤石智能设备唤醒词识别的真实业务需求，部署在远程GPU训练集群上。平台提供Web管理控制台，支持团队协作与多项目管理。每位用户拥有独立的Agent会话实例，项目组共享实验树与数据产物，避免重复计算。版本管理使所有实验可精确复现，满足工业场景对可追溯性的要求。

#### 1.4 选题的技术难度和工作量

本课题的技术难度主要体现在以下几个方面：

Agent决策引擎的设计与实现。需要设计动态Prompt构建器（根据当前实验树状态、紧凑记忆、可用技能列表等动态组装上下文）、工具/技能调度器（解析Agent的工具调用指令并路由到对应的Tool或Skill执行）、上下文压缩引擎（当历史对话接近Token上限时自动摘要压缩）、迭代预算控制器（防止无限循环消耗算力）以及轨迹记录器（将每轮决策结构化持久化）等多个核心组件的协同工作。Agent的决策需具备可解释性——每一步决策的依据需要能够在Trace中被追溯。

10个ASR领域技能的接口设计与封装。每个技能需要遵循统一的Skill接口协议（包括前置条件检查、参数校验、幂等性保证、执行结果结构化返回），同时需要处理各阶段之间的数据依赖关系。特别是train_model技能需要集成GPU资源检查与用户确认机制，evaluate_model和analyze_results技能需要实现精确的ASR评估指标计算（唤醒词首尾剥离、空参考文本排除等边界情况处理）。

多用户并发场景下的资源管理。每位用户拥有独立的Agent实例，但共享实验树和GPU资源。需要设计合理的锁机制保证实验树的一致性，以及GPU占用检查与排队机制避免训练冲突。

远程异构训练环境的可靠对接。远程训练环境涉及多GPU分布式训练（torchrun），需要准确捕获FunASR框架"返回码为0但无检查点产出"的静默崩溃问题，设计SeaCo偏置标签的启动前校验机制，以及训练日志的流式采集与解析。

完整的Web平台开发。包含前端（Vue.js 3 + Element Plus，6个核心页面）、后端（FastAPI，7组RESTful接口）、数据库（MySQL 15张表 + Redis状态缓存）的全栈开发，以及用户认证、权限管理、项目容器化隔离等工程化能力。

从工作量角度，本课题涉及Agent决策引擎（5个核心组件）、10个领域技能、评测与分析模块、Trace链路观测模块、版本管理模块、Web管理控制台（含Agent交互界面、数据集管理、评测分析看板、Trace链路视图）、RESTful API服务层、MySQL数据库层（15张核心表）、Redis缓存层、权限与多项目隔离机制、Docker容器化部署、远程训练环境适配层的设计与实现，以及在AISHELL公开数据集和萤石业务数据集上的端到端验证实验。整体工作量饱满，涉及面较广，能够满足工程硕士学位论文的要求。

### 2. 主要参考文献（列出作者、论文名称、期刊名称、出版年月）

[1] 李健, 叶栋, 赵盛. 端到端自动语音识别技术进展综述[J]. APSIPA Transactions on Signal and Information Processing, 2022, 11(1): e8.
[2] 张斌, 吕鸿, 郭鹏, 等. WenetSpeech: 万小时多领域中文语音识别语料库[C]. IEEE国际声学语音信号处理会议(ICASSP), 2022: 6182-6186.
[3] Sculley D, Holt G, Golovin D, et al. Hidden Technical Debt in Machine Learning Systems[C]. Advances in Neural Information Processing Systems (NeurIPS), 2015: 2503-2511.
[4] Yao S, Zhao J, Yu D, et al. ReAct: Synergizing Reasoning and Acting in Language Models[C]. International Conference on Learning Representations (ICLR), 2023.
[5] Shinn N, Cassano F, Gopinath A, et al. Reflexion: Language Agents with Verbal Reinforcement Learning[C]. Advances in Neural Information Processing Systems (NeurIPS), 2023.
[6] Gulati A, Qin J, Chiu C C, et al. Conformer: Convolution-augmented Transformer for Speech Recognition[C]. INTERSPEECH, 2020: 5036-5040.
[7] 高忠孝, 张声岫, McLoughlin I, 等. Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition[C]. INTERSPEECH, 2022: 2063-2067.
[8] 石宇, 吴志方, 高忠孝, 等. SeaCo-Paraformer: A Non-Autoregressive ASR System with Flexible and Effective Hotword Customization Ability[C]. IEEE ICASSP, 2024: 11291-11295.
[9] 高忠孝, 李哲, 王军, 等. FunASR: A Fundamental End-to-End Speech Recognition Toolkit[C]. INTERSPEECH, 2023: 1609-1613.
[10] 姚征, 吴迪, 王旭, 等. WeNet: Production Oriented Streaming and Non-streaming End-to-End Speech Recognition Toolkit[C]. INTERSPEECH, 2021: 4054-4058.
[11] Schick T, Dwivedi-Yu J, Dessì R, et al. Toolformer: Language Models Can Teach Themselves to Use Tools[C]. NeurIPS, 2023.
[12] Jimenez C E, Yang J, Wettig A, et al. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?[C]. ICLR, 2024.
[13] Lu C, Lu C, Lange R T, et al. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery[C]. ICLR, 2025.
[14] Huang J, Chang K C, et al. Benchmarking Large Language Models as AI Research Agents[C]. NeurIPS, 2024.
[15] Schmidt D, Boularias A, et al. AIDE: AI-Driven Exploration for Machine Learning Engineering[C]. ICLR, 2025.
[16] Paleyes A, Urma R G, Lawrence N D. Challenges in Deploying Machine Learning: A Survey of Case Studies[J]. ACM Computing Surveys, 2022, 55(6): 1-29.
[17] Zaharia M, Chen A, Davidson A, et al. Accelerating the Machine Learning Lifecycle with MLflow[J]. IEEE Data Engineering Bulletin, 2018, 41(4): 39-45.
[18] 刘宇涵, 王海峰, 吴华. 面向自然语言处理的预训练模型评测方法综述[J]. 中文信息学报, 2023, 37(1): 1-16.
[19] 赵鑫, 李军辉, 周昆, 等. 大语言模型综述[J]. 中国科学: 信息科学, 2024, 54(8): 1767-1806.
[20] Ng A. Data-centric AI: A New Paradigm for AI Development[C]. AAAI Conference on Artificial Intelligence (Invited Talk), 2022.
[21] 王磊, 马超, 冯翔, 等. 基于大语言模型的自主智能体综述[J]. Frontiers of Computer Science, 2024, 18(6): 186345.
[22] Amershi S, Begel A, Bird C, et al. Software Engineering for Machine Learning: A Case Study[C]. ICSE-SEIP, 2019: 291-300.
[23] Xi Z, Chen W, Guo X, et al. The Rise and Potential of Large Language Model Based Agents: A Survey[J]. arXiv preprint arXiv:2309.07864, 2023.
[24] 杨青, 陈文亮, 张民. 面向复杂任务的大语言模型Agent系统研究进展[J]. 计算机研究与发展, 2024, 61(5): 1097-1118.
[25] 刘群, 张伟男, 赵铁军. 智能对话系统评估方法综述[J]. 计算机学报, 2023, 46(8): 1637-1660.
[26] 周志华. 机器学习[M]. 北京: 清华大学出版社, 2016.
[27] 陈恩红, 刘淇, 黄振亚, 等. 面向在线教育的智能评测技术研究[J]. 中国科学: 信息科学, 2022, 52(1): 1-18.
[28] Park D S, Chan W, Zhang Y, et al. SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition[C]. INTERSPEECH, 2019: 2613-2617.


## 三、课题内容及具体方案

### 1. 课题内容

本课题围绕"基于Agent架构的语音识别模型自动化优化平台"展开设计与实现，主要包含以下内容：

（1）Agent决策引擎的设计与实现，包含Prompt构建器、工具注册表、技能索引、上下文管理引擎、迭代预算控制、轨迹记录器等核心组件。
（2）领域技能层的设计与实现，将ASR优化的10个处理阶段封装为符合统一接口协议的独立Skill。
（3）评测与分析模块的设计与实现，包含规则型评估器与LLM型评估器，分析结果持久化存储。
（4）Trace链路观测模块的设计与实现，将Agent每轮决策结构化记录为可查询的链路数据。
（5）版本管理模块的设计与实现，对配置、数据集、热词、模型建立版本关联。
（6）Web管理控制台的设计与实现，包含Agent交互界面、项目管理、数据集管理、热词管理、评测分析看板、Trace链路视图。
（7）RESTful API服务层与MySQL数据库层的设计与实现。
（8）用户管理与权限控制的设计与实现（平台管理员/项目Owner/项目Member三级权限）。
（9）项目容器化隔离与初始化流程的设计与实现（每个项目对应一个Docker容器）。
（10）远程训练环境适配层的设计与实现，包含GPU资源检查与训练健康监测。
（11）系统集成测试与端到端验证实验。

### 2. 系统需求分析（应用软件工程专业描述工具描述）

#### 2.1 概述

本系统是一个面向ASR模型优化的智能管理平台。每位用户拥有独立的Agent会话实例，Agent作为用户的个人优化助手，驱动"评估—诊断—决策—执行—再评估"的闭环。项目组内共享实验树与数据处理产物，避免重复计算。当涉及GPU密集型任务（如模型训练）时，Agent检查资源可用性并向用户确认后方可执行。系统通过Web控制台为团队提供协作界面，通过数据库持久化保证实验可追溯与可复现。

#### 2.2 功能需求

系统包含三类参与者：平台管理员、项目Owner、项目Member。普通用户用例图如图1所示。

[此处插入图1：系统用例图]

各类用户的核心功能需求如下：

平台管理员用例：
- 用户管理：注册新用户、管理用户账号（启用/禁用/重置密码）
- 项目管理：创建项目（指定基础模型、GPU配置）、删除项目（停止容器、归档数据）、为项目分配成员及角色

项目Owner用例：
- 与Agent交互：通过对话界面下达优化指令，查看Agent思考过程，确认或否决Agent的关键决策
- 数据集管理：上传公共数据集到shared目录、删除数据集、查看数据集信息与版本
- 热词管理：上传/修改/查看热词文件
- 配置管理：修改训练配置、增强配置等全局配置文件
- 启动/停止Agent会话
- 查看实验树、Trace链路、评测分析报告

项目Member用例：
- 与Agent交互：通过对话界面发起实验请求，查看Agent过程
- 上传数据集到个人目录（datasets/{user_id}/）
- 查看实验树、Trace链路、评测分析报告
- 查看数据集列表（不能删除公共数据集）
- 不能修改配置文件与热词文件

#### 2.3 非功能性需求

（1）可复现性：相同配置版本+数据集版本下，确定性处理阶段必须产出内容等价的产物。切分使用固定种子（seed=42），增强使用确定性种子。

（2）健壮性：技能执行失败时即终止后续步骤并记录失败原因；远程SSH连接失败须自动重连；训练过程中的静默崩溃（返回码0但无检查点）须被检测并报告。

（3）可观测性：Agent每轮决策链路完整记录，各技能的执行状态（进行中/成功/失败）、耗时、输入输出摘要实时可查；训练过程中的loss曲线可通过Web看板查看。

（4）安全性：Web控制台提供JWT Token认证与基于角色的访问控制（RBAC）。普通用户不能直接操作项目工作目录的关键文件（配置、热词），只能通过Agent间接操作或由Owner管理。

（5）多租户隔离：每个项目对应一个独立的Docker容器实例，各项目的数据集、实验记录、配置和模型相互隔离。所有项目共享MySQL和Redis实例，通过project_id字段实现逻辑隔离。

（6）性能要求：Web页面响应时间不超过2秒；API接口响应时间不超过500ms；Agent对话消息的首token延迟取决于LLM提供方，平台不引入额外延迟。

### 3. 系统概要设计（应用软件工程专业描述工具描述）

#### 3.1 系统总体架构设计

系统采用B/S架构，按职责分为六层：用户层、Web控制台层、API服务层、数据库层、Agent决策引擎层（含通用工具与领域技能）、远程训练环境适配层。系统总体架构如图2所示。

[此处插入图2：系统总体架构图（01_系统总体架构图.drawio）]

各层职责说明如下：

- 用户层：用户通过浏览器访问Web管理控制台，进行项目管理、Agent交互、数据管理与结果查看等操作。
- Web控制台层（Vue.js + Element Plus）：提供6个核心页面——项目管理、数据管理、Agent交互、评测分析、Trace视图、实验看板。负责向API层发送请求并展示响应结果。
- API服务层（FastAPI）：提供RESTful接口，包括项目接口、数据接口、热词接口、Agent接口、Trace接口、版本接口、认证接口共7组。负责业务逻辑处理与数据库读写。
- 数据库层（MySQL + Redis）：MySQL存储所有结构化业务数据（实验记录、版本信息、评测结果、Trace链路、用户权限等）。Redis缓存Agent运行状态供前端实时轮询展示。所有项目共享同一个MySQL和Redis实例，通过project_id字段逻辑隔离。
- Agent决策引擎层：每位用户拥有独立的Agent会话实例。引擎内部包含Prompt Builder、LLM推理核心、Tool/Skill Dispatcher、Context Engine、Iteration Budget、Trajectory Logger等组件。通用工具层提供基础操作能力，领域技能层提供ASR专有处理能力。
- 远程训练环境适配层：封装SSH连接管理、FunASR训练引擎（含GPU检查与健康监测）、评估引擎，对接远程远程GPU训练集群 GPU集群。

部署架构：每个项目对应一个独立的Docker容器实例，容器内运行Agent引擎与技能层代码，挂载该项目专属的工作目录。Web前端、API服务、MySQL、Redis作为共享基础设施始终运行。部署架构如图3所示。

[此处插入图3：部署架构图（07_部署架构图.drawio）]

#### 3.2 Agent决策引擎设计

Agent决策引擎是平台的智能核心，负责自主驱动优化循环。引擎内部由6个组件协同工作，其架构如图4所示。

[此处插入图4：Agent决策引擎内部架构图（02_Agent决策引擎内部架构.drawio）]

各组件职责如下：

① Prompt Builder（动态提示词构建器）
负责在每轮决策前动态组装Agent的system prompt。组装内容包括：身份定义（"你是ASR模型优化专家"）、ASR领域知识（FunASR/SeaCo训练经验规则）、当前实验树状态摘要、紧凑记忆（最近N轮经验，恒定大小）、可用技能列表及其参数规格、约束规则（"不要重复尝试死胡同中已验证无效的方向"）。Prompt Builder确保Agent在每轮决策时获得充分且聚焦的上下文信息。

② LLM推理核心
调用外部LLM API（支持GPT-4、Claude等），将Prompt Builder构建的上下文与用户消息发送给LLM，获取Agent的决策输出。输出包含自然语言形式的思考过程和结构化的工具调用指令（function call）。

③ Tool/Skill Dispatcher（能力调度器）
解析LLM返回的工具调用指令，根据被调用的名称路由到通用工具层（Tool）或领域技能层（Skill）执行。执行完成后将结果以结构化形式返回给LLM继续推理。

④ Context Engine（上下文管理引擎）
监控对话历史的Token数量，当接近LLM的上下文窗口上限时自动触发历史压缩——将早期的对话轮次摘要为简短总结，保留最近2-3轮的完整细节。同时负责Agent记忆文件（紧凑记忆、死胡同、洞察）的定期compact操作，确保记忆恒定大小。

⑤ Iteration Budget（迭代预算控制）
防止Agent无限循环消耗算力。支持三种终止条件：最大轮次（max_rounds=20）、最大GPU训练时间（max_gpu_hours=24）、连续无改善停止（连续N轮CER无下降则终止）。预算耗尽时Agent自动生成优化总结报告并结束会话。

⑥ Trajectory Logger（轨迹记录器）
在Agent每一轮决策循环中，将THINK阶段（Agent的思考内容与决策依据）、EXECUTE阶段（调用了哪些工具/技能、传递的参数、返回的结果）、REFLECT阶段（对结果的分析与经验总结）完整记录为结构化的Trace Span数据，持久化到数据库的trace和trace_span表中。

Agent的优化循环工作流程如图5所示。

[此处插入图5：Agent优化循环流程图（03_Agent优化循环流程图.drawio）]

循环的核心逻辑为：THINK阶段读取紧凑记忆与实验树状态，分析当前模型瓶颈，选择搜索策略（Draft新方向/Improve当前最优/Debug上次失败）；EXECUTE阶段在实验树上分叉新节点，编辑配置并调用领域技能执行（数据增强、训练、评估等）；REFLECT阶段解析执行结果，与基线对比，将经验记入记忆系统（有效方向记入洞察，无效方向记入死胡同），更新紧凑记忆，然后判断是否继续优化。

#### 3.3 通用工具层与领域技能层设计

系统采用"通用工具层+领域技能层"的双层能力设计。通用工具层提供与领域无关的基础操作能力（read_file、write_file、run_command、edit_yaml、query_database），任何领域的Agent均可复用。领域技能层将ASR优化的各处理阶段封装为符合统一接口协议的独立技能，Agent可以单独调用某个技能，也可以组合多个技能顺序执行。

每个Skill遵循统一的接口协议，包含以下方法与属性：name（技能名称）、description（自描述文档，注入Agent上下文供其了解该技能的功能与使用方式）、parameters_schema（参数JSON Schema）、is_idempotent（是否幂等，幂等技能在产物已存在时自动跳过）、check_precondition()（前置条件检查，如train技能需要检查GPU可用性）、execute(params)（执行主逻辑，返回结构化结果）、get_output_summary()（获取输出摘要供Agent解读）。

本系统共设计10个领域技能，其功能说明如表1所示。

表1 领域技能功能说明

| 序号 | 技能名 | 关键职责 | 输入 | 输出 |
|------|--------|----------|------|------|
| 1 | prepare_data | 原始数据按FunASR格式整理 | raw目录 | wav.scp + text |
| 2 | quality_scan | 音频质量扫描与好坏分类 | wav.scp | good/bad清单 |
| 3 | split_dataset | 按设备ID隔离切分(seed=42) | labeled数据 | train/dev/test |
| 4 | augment_data | 训练集数据增强（多策略） | train数据 | 增强后数据 |
| 5 | merge_sources | 多数据源合并去重 | 多个split产物 | merged数据 |
| 6 | generate_manifest | 生成FunASR训练清单 | scp+text | manifest.jsonl |
| 7 | train_model | GPU检查+用户确认+分布式训练+健康检查 | 配置+清单 | checkpoint |
| 8 | evaluate_model | 模型推理+规则指标计算+badcase提取 | checkpoint+测试集 | metrics+badcases |
| 9 | analyze_results | 错误类型分析+混淆对提取+LLM归因 | metrics+badcases | 结构化分析报告 |
| 10 | edit_config | 修改训练/增强配置文件 | key+value | 更新后配置 |

其中，train_model技能的执行流程较为特殊：首先调用check_precondition()检查GPU占用状态，如果GPU被其他用户占用则返回占用信息（占用者、正在执行的实验、预计剩余时间）并建议用户稍后重试；如果GPU空闲，则向用户展示训练配置摘要（预计耗时、使用GPU数量、配置版本）并请求确认；用户确认后方可提交训练任务。

#### 3.4 评测与分析模块设计

评测与分析以Agent的两个核心技能（evaluate_model和analyze_results）实现，产出结构化的评估报告与归因分析，结果持久化到数据库供Web控制台展示。评测与分析流程如图6所示。

[此处插入图6：评测与分析流程图（04_评测与分析流程图.drawio）]

evaluate_model技能的执行步骤：
1. 加载指定的模型checkpoint
2. 对测试集（dev或test）执行批量推理，获取识别结果
3. 规则型评估器计算多维度指标：
   - CER（字符错误率）：整体及按设备/场景维度
   - 指令召回率：剥离唤醒词前后缀后，计算控制指令的召回率（参考文本为空的样本排除出分子和分母）
   - Stripped_CER：剥离唤醒词后计算的字符级编辑距离
4. 生成badcase清单：CER高于阈值的样本，记录其音频路径、参考文本、识别结果与逐字错误详情
5. 按设备（device_id）、按场景（scene_type）维度进行聚合统计
6. 将评测结果（eval_result表）和badcase摘要持久化到数据库

analyze_results技能的执行步骤：
1. 从数据库读取最新一次评测的badcase清单
2. 统计错误类型分布（substitution替换 / insertion插入 / deletion删除的比例）
3. 提取Top-N混淆对（如"开灯"→"开等"、"小艺"→"小一"），按出现频次排序
4. LLM型评估器：将错误分布、混淆对、设备维度指标等结构化信息组织成提示词，调用LLM对badcase进行语义归因分析，产出改进建议（如"sp4设备在噪声场景下插入错误偏高，建议增加babble增强数据"）
5. 按设备/场景维度定位薄弱环节
6. 生成结构化分析报告（analysis_report表），包含字段：overall_cer、scene_metrics、error_distribution、top_confusions、badcase_count、llm_analysis、weakness_summary
7. 持久化到数据库

Agent在REFLECT阶段读取分析报告中的weakness_summary和llm_analysis字段，作为下一轮THINK的输入依据，形成"评估→分析→决策→执行→再评估"的闭环。

#### 3.5 Trace链路观测模块设计

Trace模块将Agent每轮优化循环的完整决策链路结构化记录，支持事后追溯与可视化分析。每一轮Agent优化对应一条Trace记录，内部包含多个Span。Trace链路示例如图7所示。

[此处插入图7：Trace链路示例图（05_Trace链路示例.drawio）]

Trace数据模型设计如下：
- trace表：id, experiment_id, round（第几轮优化）, user_id（由谁发起）, status, total_duration_ms, created_at
- trace_span表：id, trace_id, type（think/execute/reflect）, skill_name（execute类型时记录调用的技能名）, duration_ms, input_summary（输入摘要，限制长度）, output_summary（输出摘要）, status

Web控制台的Trace视图页面从数据库查询Trace数据，以时间线形式展示每轮优化的各Span，支持按时间范围、技能名称、执行状态等维度筛选。点击某个Span可展开查看其详细的输入输出内容。

#### 3.6 版本管理模块设计

版本管理对配置、数据集、热词、模型四者建立关联，保证实验可精确复现。版本管理关系如图8所示。

[此处插入图8：版本管理关联关系图（06_版本管理与项目初始化.drawio）]

每个实验节点记录其使用的各项资源版本：
- config_version：训练配置YAML的版本，保存完整内容与diff
- dataset_version：数据集的版本，记录文件hash和样本数
- hotword_version：热词文件的版本，记录词条数
- model_version：训练产出的模型checkpoint版本，关联评测指标

任意历史实验均可通过其绑定的版本信息精确复现：使用同一份配置、同一版本数据集（确定性种子保证切分一致）、同一热词配置，即可得到等价的训练结果。

#### 3.7 数据库设计

数据库采用MySQL 8.0，共设计15张核心表。Redis用于缓存Agent运行状态。各表结构设计如下：

表2 user（用户表）

| 字段名 | 类型 | 是否为空 | 说明 |
|--------|------|----------|------|
| id | bigint | N | 主键，自增 |
| email | varchar(200) | N | 邮箱，唯一 |
| password_hash | varchar(200) | N | 密码哈希 |
| name | varchar(50) | N | 用户名 |
| global_role | enum('admin','user') | N | 全局角色 |
| status | enum('active','disabled') | N | 账号状态 |
| created_at | datetime | N | 注册时间 |

表3 project（项目表）

| 字段名 | 类型 | 是否为空 | 说明 |
|--------|------|----------|------|
| id | bigint | N | 主键 |
| name | varchar(100) | N | 项目名称 |
| base_model | varchar(200) | N | 基础模型标识 |
| gpu_config | json | N | GPU配置 |
| container_id | varchar(100) | Y | Docker容器ID |
| status | enum('running','stopped') | N | 运行状态 |
| work_dir | varchar(500) | N | 工作目录路径 |
| created_at | datetime | N | 创建时间 |

表4 project_member（项目成员表）

| 字段名 | 类型 | 是否为空 | 说明 |
|--------|------|----------|------|
| project_id | bigint | N | 项目ID，外键 |
| user_id | bigint | N | 用户ID，外键 |
| role | enum('owner','member') | N | 项目内角色 |

表5 experiment（实验记录表，树结构）

| 字段名 | 类型 | 是否为空 | 说明 |
|--------|------|----------|------|
| id | bigint | N | 主键 |
| project_id | bigint | N | 所属项目 |
| parent_id | bigint | Y | 父节点ID（null为根节点） |
| name | varchar(200) | N | 实验名称/描述 |
| config_ver_id | bigint | Y | 配置版本外键 |
| dataset_ver_id | bigint | Y | 数据集版本外键 |
| hotword_ver_id | bigint | Y | 热词版本外键 |
| status | enum('pending','running','success','failed') | N | 状态 |
| created_by | bigint | N | 创建者用户ID |
| created_at | datetime | N | 创建时间 |

表6 config_version（配置版本表）

| 字段名 | 类型 | 是否为空 | 说明 |
|--------|------|----------|------|
| id | bigint | N | 主键 |
| project_id | bigint | N | 所属项目 |
| version | varchar(20) | N | 版本号（如v5） |
| yaml_content | text | N | 完整YAML内容 |
| diff_json | json | Y | 与上一版本的diff |
| created_at | datetime | N | 创建时间 |

表7 dataset_version（数据集版本表）

| 字段名 | 类型 | 是否为空 | 说明 |
|--------|------|----------|------|
| id | bigint | N | 主键 |
| project_id | bigint | N | 所属项目 |
| version | varchar(20) | N | 版本号 |
| file_path | varchar(500) | N | 文件路径 |
| file_hash | varchar(64) | N | 文件MD5哈希 |
| sample_count | int | N | 样本数量 |
| created_at | datetime | N | 创建时间 |

表8 hotword_version（热词版本表）

| 字段名 | 类型 | 是否为空 | 说明 |
|--------|------|----------|------|
| id | bigint | N | 主键 |
| project_id | bigint | N | 所属项目 |
| version | varchar(20) | N | 版本号 |
| file_path | varchar(500) | N | 文件路径 |
| word_count | int | N | 词条数 |
| created_by | bigint | N | 上传者 |
| created_at | datetime | N | 创建时间 |

表9 model_version（模型版本表）

| 字段名 | 类型 | 是否为空 | 说明 |
|--------|------|----------|------|
| id | bigint | N | 主键 |
| experiment_id | bigint | N | 产出该模型的实验 |
| checkpoint_path | varchar(500) | N | checkpoint文件路径 |
| eval_metrics_json | json | Y | 评测指标快照 |
| created_at | datetime | N | 创建时间 |

表10 eval_run（评测运行记录表）

| 字段名 | 类型 | 是否为空 | 说明 |
|--------|------|----------|------|
| id | bigint | N | 主键 |
| experiment_id | bigint | N | 所属实验 |
| model_ver_id | bigint | N | 模型版本 |
| dataset_ver_id | bigint | N | 测试数据集版本 |
| status | enum('running','success','failed') | N | 状态 |
| created_at | datetime | N | 运行时间 |

表11 eval_result（评测结果表）

| 字段名 | 类型 | 是否为空 | 说明 |
|--------|------|----------|------|
| id | bigint | N | 主键 |
| eval_run_id | bigint | N | 所属评测运行 |
| metric_name | varchar(50) | N | 指标名（CER/recall等） |
| value | float | N | 指标值 |
| dimension_json | json | Y | 维度信息（设备/场景） |

表12 analysis_report（分析报告表）

| 字段名 | 类型 | 是否为空 | 说明 |
|--------|------|----------|------|
| id | bigint | N | 主键 |
| eval_run_id | bigint | N | 关联的评测运行 |
| error_distribution_json | json | N | 错误类型分布 |
| top_confusions_json | json | N | Top-N混淆对 |
| badcase_count | int | N | badcase数量 |
| llm_analysis | text | Y | LLM归因分析 |
| weakness_summary | text | N | 薄弱环节总结 |
| created_at | datetime | N | 创建时间 |

表13 trace（Agent轨迹表）

| 字段名 | 类型 | 是否为空 | 说明 |
|--------|------|----------|------|
| id | bigint | N | 主键 |
| experiment_id | bigint | N | 所属实验 |
| round | int | N | 优化轮次 |
| user_id | bigint | N | 发起者 |
| total_duration_ms | bigint | Y | 总耗时 |
| status | enum('running','success','failed') | N | 状态 |
| created_at | datetime | N | 创建时间 |

表14 trace_span（轨迹Span表）

| 字段名 | 类型 | 是否为空 | 说明 |
|--------|------|----------|------|
| id | bigint | N | 主键 |
| trace_id | bigint | N | 所属Trace |
| type | enum('think','execute','reflect') | N | Span类型 |
| skill_name | varchar(50) | Y | 调用的技能名 |
| duration_ms | bigint | Y | 耗时(ms) |
| input_summary | text | Y | 输入摘要 |
| output_summary | text | Y | 输出摘要 |
| status | enum('running','success','failed') | N | 状态 |

表15 dataset_file（数据集文件管理表）

| 字段名 | 类型 | 是否为空 | 说明 |
|--------|------|----------|------|
| id | bigint | N | 主键 |
| project_id | bigint | N | 所属项目 |
| owner_user_id | bigint | N | 上传者 |
| name | varchar(200) | N | 文件名 |
| file_path | varchar(500) | N | 存储路径 |
| file_size | bigint | N | 文件大小(bytes) |
| description | text | Y | 描述 |
| upload_time | datetime | N | 上传时间 |

表16 agent_memory_ref（Agent记忆文件引用表）

| 字段名 | 类型 | 是否为空 | 说明 |
|--------|------|----------|------|
| id | bigint | N | 主键 |
| project_id | bigint | N | 所属项目 |
| type | enum('tree','compact','dead_end','insight','journal') | N | 记忆类型 |
| file_path | varchar(500) | N | 文件路径 |
| updated_at | datetime | N | 最后更新时间 |

说明：Agent的记忆内容（实验树experiment_tree.json、紧凑记忆MEMORY_LOG.md、死胡同DEAD_ENDS.md、洞察INSIGHTS.md）以文件形式存储在项目工作目录中，数据库仅记录文件路径引用。Redis用于缓存当前Agent运行状态（正在执行的轮次、技能名、进度），供Web前端实时轮询展示。

#### 3.8 Web控制台与用户管理设计

Web控制台采用Vue.js 3 + Element Plus开发，为不同角色的用户提供相应的功能页面。

用户管理功能：
- 用户注册：通过邮箱+密码注册，注册后默认为普通用户（global_role=user），需要平台管理员将其添加到项目才能使用项目功能。
- 用户登录：基于JWT Token认证，Token有效期24小时，过期后需重新登录。
- 平台管理员可管理所有用户账号（启用/禁用/重置密码）。

项目管理功能（仅平台管理员）：
- 创建项目：指定项目名称、基础模型（如SeaCo-Paraformer）、GPU配置 → 系统自动按固定模板初始化工作目录 → 将预训练模型从共享模型仓库链接到项目目录 → 启动Docker容器 → 返回项目信息。
- 删除项目：停止容器、归档项目数据。
- 分配成员：将用户添加到项目并指定角色（Owner/Member）。

Agent交互页面：
以对话式界面展示，左侧为对话区（用户消息与Agent回复交替展示），右侧为Agent状态面板（当前轮次、正在执行的技能、进度状态）。用户可以发送自然语言指令（如"请优先优化sp4设备的识别效果"），Agent理解指令后调用相应技能执行。当train_model技能检测到GPU空闲时，Agent会在对话中展示训练配置摘要并请求用户确认。

数据集管理页面：
展示项目内所有数据集文件列表（包含文件名、上传者、大小、上传时间），Owner可上传到shared/目录，Member可上传到个人目录。支持文件的预览（显示前几行内容）和删除（仅上传者或Owner可删除）。

热词管理页面（仅Owner）：
展示当前热词列表、版本历史。支持上传新热词文件（系统自动校验格式：非空、UTF-8编码、去重）、单独添加/删除热词、查看历史版本对比。

评测分析页面：
以图表形式展示评测结果：CER趋势图（横轴为实验轮次，纵轴为CER值）、按设备维度的指标对比柱状图、错误类型分布饼图、Top-10混淆对表格。支持选择多个实验进行横向对比。

Trace链路页面：
以时间线形式展示Agent每轮优化的Span链路，每个Span显示类型（THINK/EXECUTE/REFLECT）、耗时、状态。点击Span展开查看输入输出详情。支持按时间范围、技能名、状态筛选。

实验看板页面：
以树形图展示实验搜索空间，节点颜色标识状态（绿色成功、红色失败、黄色进行中）。鼠标悬停节点显示关键指标（CER、创建者、时间）。支持折叠/展开子树。

#### 3.9 权限与项目隔离设计

系统采用三级权限体系，如表17所示。

表17 权限体系设计

| 角色 | 范围 | 核心权限 |
|------|------|----------|
| 平台管理员 | 全局 | 创建/删除项目，管理用户，分配项目成员 |
| 项目Owner | 项目内 | 管理热词，修改全局配置，管理公共数据集，启动/停止Agent |
| 项目Member | 项目内 | 与Agent交互，发起实验，上传个人数据集，查看所有结果 |

项目内资源的共享与归属规则：
- 实验树：项目级共享。所有成员看到同一棵实验树，每个节点记录created_by字段标识创建者。
- 数据增强产物：项目级共享。某成员通过Agent执行增强后，产物（存放在outputs/目录下）可被全组复用。系统自动检测已有产物避免重复执行。
- 数据集上传：按用户隔离。Member上传到datasets/{user_id}/，Owner管理datasets/shared/。
- 配置与热词：仅Owner可修改。Member如需调整配置，需通过Agent交互提出建议。

项目隔离机制：每个项目对应一个独立Docker容器，容器挂载项目专属工作目录。所有项目共享MySQL和Redis实例（通过project_id逻辑隔离）。

#### 3.10 项目初始化与容器管理

创建项目时，系统执行以下初始化流程：

1. 在数据库中创建project记录
2. 创建固定结构的项目工作目录：
```
/data/projects/{project_id}/
├── configs/
│   ├── project.yaml           # 项目元配置
│   ├── train_config.yaml      # 训练配置模板
│   ├── augment_config.yaml    # 增强配置模板
│   └── hotwords.txt           # 热词文件（初始为空）
├── datasets/
│   └── shared/                # 公共数据集目录
├── models/
│   └── pretrained/            # 预训练模型（链接）
├── workspace/                 # Agent记忆文件
│   ├── experiment_tree.json   # 实验树（初始为空树）
│   ├── MEMORY_LOG.md          # 紧凑记忆
│   ├── DEAD_ENDS.md           # 死胡同
│   └── INSIGHTS.md            # 洞察
├── outputs/                   # 各技能产物（按实验节点ID隔离）
└── traces/                    # Trace文件（备份）
```
3. 将指定的预训练基础模型从共享模型仓库软链接到项目的models/pretrained/目录
4. 从默认模板复制训练配置与增强配置文件
5. 启动项目Docker容器，挂载上述工作目录
6. 将容器ID记录到数据库

Skill层的所有操作均在该固定目录结构下进行，路径通过project.yaml配置注入。

#### 3.11 远程训练环境适配层

远程环境适配层封装了与GPU训练集群的交互能力，主要包含三个组件：

RemoteExecutor（远程执行器）： 管理SSH连接池，支持断线自动重连与会话复用。提供命令执行、文件传输、超时控制等基础能力。

TrainingEngine（训练引擎）： 负责训练任务的启动与监控。关键设计包括：
- GPU资源状态检查：训练前查询GPU占用情况，若被其他用户占用则返回占用信息
- 配置校验：检查训练配置的关键参数（学习率、warmup步数等）是否合理
- SeaCo偏置标签启动前校验：检查热词文件中的词条是否在模型词表内
- torchrun分布式训练启动与进程管理
- 训练健康检查：检测FunASR"返回码0但无检查点产出"的静默崩溃
- 训练日志流式采集：实时解析loss值，推送到Redis供Web看板展示

EvalEngine（评估引擎）： 负责模型推理与指标计算。加载checkpoint执行批量推理，计算各项评估指标。

### 4. 拟采用的设计方法、环境等

本系统采用面向对象的设计方法，前后端分离开发，通过OpenAPI 3.0规范定义API接口契约。领域技能层采用统一的Skill接口协议，便于独立开发与测试。Agent引擎各组件通过依赖注入方式组装。

开发环境及使用工具如表18所示。

表18 开发环境

| 项目 | 选型 | 说明 |
|------|------|------|
| 后端语言 | Python 3.10 | Agent引擎与技能层 |
| Web框架 | FastAPI | API服务层，自动生成Swagger文档 |
| 前端框架 | Vue.js 3 + Element Plus | Web管理控制台 |
| 数据库 | MySQL 8.0 | 结构化数据持久化 |
| 缓存 | Redis 7 | Agent状态缓存、实时指标 |
| ORM | SQLAlchemy 2.0 | Python数据库映射 |
| 可视化 | ECharts 5 | 指标图表展示 |
| ASR框架 | FunASR (SeaCo-Paraformer) | 模型训练与推理 |
| LLM调用 | OpenAI API兼容接口 | Agent推理核心 |
| 远程执行 | SSH (paramiko) + torchrun | 远程训练环境对接 |
| 训练集群 | 远程GPU训练集群 | GPU计算资源 |
| 容器化 | Docker + Docker Compose | 项目隔离部署 |
| 测试 | pytest + Hypothesis | 自动化测试 |
| 接口文档 | Swagger / OpenAPI 3.0 | API文档自动生成 |
| UML建模 | draw.io | 架构图与流程图 |

### 5. 技术难度及特色分析

本课题的技术特色主要包括：

自主设计Agent引擎，采用"通用工具+领域技能"的双层能力架构与动态Prompt构建。Agent能够根据当前实验状态和记忆内容自主决定调用哪些技能、传递什么参数，而非按固定流水线编排执行。这使得优化过程具备了自适应性——面对不同的瓶颈类型（数据不足、配置不当、模型欠拟合等），Agent可以选择不同的应对策略。

评测与分析流程作为Agent技能实现，产出的结构化分析报告（错误类型分布、混淆对、LLM归因分析）持久化到数据库，形成"评估→分析→决策→执行→再评估"的自动化闭环。工程师在Web看板上可以直观地看到每轮优化的效果变化。

完整的Trace链路观测能力使Agent的决策过程不再是黑盒。每一轮优化的思考、执行与反思全过程可被事后追溯与审计，为Agent决策的可解释性提供了工程化支撑。

三方版本管理（配置/数据集/模型）与实验树的结合，使得"回退到历史某个成功版本并从该点分支探索"成为可能，避免了线性优化中一旦方向错误需要从头重来的困境。

### 6. 本人主要工作描述

1. 系统总体架构设计与各模块接口定义。
2. Agent决策引擎的完整设计与实现（Prompt构建器、能力调度器、上下文管理、迭代预算、轨迹记录器）。
3. 10个ASR领域技能的接口设计与实现。
4. 评测与分析模块的实现及结果持久化。
5. Trace链路观测模块的设计与实现。
6. 版本管理模块的设计与实现。
7. RESTful API服务层开发（7组接口）。
8. MySQL数据库设计与ORM层开发（15张表）。
9. Web管理控制台前端开发（6个核心页面，含Agent交互界面）。
10. 用户管理、权限控制与项目容器化隔离的实现。
11. 远程训练环境适配层开发与调试。
12. 系统集成测试与业务数据端到端验证实验。

### 7. 测试方案

采用"单元测试+属性测试+接口测试+端到端测试"四层策略，如表19所示。

表19 测试方案

| 测试类型 | 方法 | 工具 | 覆盖范围 |
|----------|------|------|----------|
| 单元测试 | 示例与边界用例 | pytest | Skill执行逻辑、评估器指标计算、版本管理 |
| 属性测试 | 基于属性的随机验证(≥100次) | Hypothesis | 数据切分确定性、合并去重正确性、幂等性 |
| 接口测试 | API联调 | pytest + httpx | RESTful接口、认证鉴权、权限控制 |
| 端到端测试 | 完整业务流程 | pytest + mock | Agent闭环优化全流程（AISHELL数据集） |

### 8. 可行性分析

在技术可行性方面：FunASR框架已在生产环境中广泛使用，其SeaCo-Paraformer模型在中文识别任务上表现优异[8]；LLM Agent的工具调用能力已在多个复杂任务场景中得到验证[4][15]；Agent系统的工具层与决策层分离设计已在学术界得到理论验证[23][24]；FastAPI + Vue.js技术栈生态成熟，社区活跃；远程GPU集群（远程GPU训练集群）已配置完毕并稳定运行。

在人力与条件方面：本课题的研究与实现属于所在企业ASR业务优化工作的核心组成部分，得到实践导师和团队的支持，具备充足的计算资源与业务数据条件。

基于以上分析，该实施方案具有可行性。


## 四、工作进度的大致安排

应包括文献调研，工程设计，项目开发和调试，实验数据的分析处理，撰写论文等。

| 时间 | 工作内容 | 阶段成果 |
|------|----------|----------|
| 2025.10–2025.11 | 文献调研与技术选型；完成系统总体架构设计 | 文献综述、开题报告 |
| 2025.12–2026.01 | 完成详细需求分析与概要设计；定义数据库schema与API接口规范 | 需求分析报告、概要设计报告 |
| 2026.02–2026.03 | 实现Agent决策引擎核心组件；实现10个领域技能 | Agent引擎原型、技能代码、单元测试 |
| 2026.04–2026.05 | 实现评测与分析模块；实现Trace观测模块与版本管理 | 评测系统、Trace模块、属性测试 |
| 2026.06–2026.07 | 实现Web管理控制台、API服务层、数据库层、权限管理与容器管理 | 完整平台系统、接口测试 |
| 2026.08–2026.09 | 系统集成测试；开展真实业务数据多轮优化验证实验 | 实验数据与评估报告 |
| 2026.10–2026.11 | 实验数据分析，系统优化，撰写毕业论文 | 毕业论文初稿 |
| 2026.12 | 修改完善论文，准备答辩 | 毕业论文定稿 |


## 五、预期成果

1. 基于Agent架构的语音识别模型自动化优化平台软件系统一套，包含Agent决策引擎（6个核心组件）、10个领域技能、评测与分析模块（规则型+LLM型评估器）、Trace链路观测模块、版本管理模块、Web管理控制台（6个核心页面，含Agent交互界面）、RESTful API服务层（7组接口）、MySQL数据库层（15张表）、Redis缓存层、用户管理与三级权限控制、项目Docker容器化隔离、远程训练环境适配层。

2. 在萤石s10/sp4业务数据集上经平台优化的ASR模型，领域KPI（指令召回率、Stripped_CER）相较基线有显著提升。

3. 系统设计文档、API接口文档（Swagger）、操作手册及测试报告。

4. 工程硕士学位论文1篇。


## 六、已取得的与学位申请成果研究内容相关的成果

1. 完成了核心框架搭建，包括CLI入口、原子操作（train/infer/eval/augment）、实验树与分层记忆系统的原型实现，并已在实际业务中支撑Agent完成多轮THINK→EXECUTE→REFLECT优化循环。

2. 完成了分片增量数据增强引擎的设计与实现，支持按策略+参数管理增强版本，增量生成新数据而不重复处理已有数据。

3. 完成了远程SSH执行能力（RemoteExecutor）的实现与验证，已在远程GPU训练集群集群上稳定运行，支持多卡分布式训练启动与日志采集。

4. 完成了10个领域技能的需求分析与概要设计，定义了统一的Skill接口协议。


## 七、研究计划与进度安排

（同第四部分，此处不再重复）

---

研究生本人签名：________________

校内导师（主导师）签名：________________

实践导师签名：________________

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;年&emsp;&emsp;月&emsp;&emsp;日
