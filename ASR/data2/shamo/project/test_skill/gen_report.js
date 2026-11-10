const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, WidthType, BorderStyle, ShadingType,
  HeadingLevel, PageBreak } = require("docx");
const fs = require("fs");

// Constants
const FONT = "宋体";
const SIZE_BODY = 21;    // 10.5pt in half-points
const SIZE_H1 = 24;      // 12pt
const SIZE_H2 = 22;      // 11pt
const SIZE_SMALL = 18;   // 9pt
const INDENT = 480;      // ~2 Chinese chars in twips (DXA)
const LINE_SPACING = 360; // 1.5x line spacing

function heading(text, level = 1) {
  return new Paragraph({
    spacing: { before: 160, after: 80, line: LINE_SPACING },
    children: [new TextRun({ text, bold: true, font: FONT, size: level === 1 ? SIZE_H1 : SIZE_H2 })],
  });
}

function body(text) {
  return new Paragraph({
    indent: { firstLine: INDENT },
    spacing: { after: 40, line: LINE_SPACING },
    children: [new TextRun({ text, font: FONT, size: SIZE_BODY })],
  });
}

function emptyLine() {
  return new Paragraph({ spacing: { after: 80 } });
}

function refLine(text) {
  return new Paragraph({
    spacing: { after: 20, line: 280 },
    children: [new TextRun({ text, font: FONT, size: SIZE_SMALL })],
  });
}

function makeTable(headers, rows) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "000000" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const nCols = headers.length;
  const colW = Math.floor(9000 / nCols);
  const colWidths = Array(nCols).fill(colW);

  const headerRow = new TableRow({
    children: headers.map((h, i) => new TableCell({
      borders, width: { size: colWidths[i], type: WidthType.DXA },
      shading: { fill: "E8E8E8", type: ShadingType.CLEAR },
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, font: FONT, size: SIZE_SMALL })] })],
    })),
  });

  const dataRows = rows.map(row => new TableRow({
    children: row.map((val, i) => new TableCell({
      borders, width: { size: colWidths[i], type: WidthType.DXA },
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({ children: [new TextRun({ text: val, font: FONT, size: SIZE_SMALL })] })],
    })),
  }));

  return new Table({
    width: { size: 9000, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows],
  });
}

// ============================================================
// CONTENT
// ============================================================
const section1 = [
  heading("1.1 选题的研究意义", 2),
  body("随着智能家居、车载交互与智能客服等领域的快速发展，语音识别（ASR）技术已成为人机交互的核心入口之一[1]。以萤石智能摄像头为代表的物联网设备普遍采用"唤醒词+控制指令"的语音交互范式，对识别准确率、唤醒词鲁棒性及远场噪声环境的适应能力提出了极高要求[2]。将通用预训练ASR模型适配到具体业务场景并达到产品级要求，本质上是一个涉及数据工程、模型训练与评估分析的多轮迭代优化过程。"),
  body("在实际业务实践中，ASR模型的优化迭代长期以零散脚本与人工经验判断的方式运行。算法工程师需要反复执行"分析瓶颈→提出假设→调整配置→执行训练→解读结果→决定下一步"的循环。这种工作模式存在多方面的突出问题：优化决策高度依赖个人经验，不可复现也不可传承；实验管理呈线性结构，缺乏可回溯的搜索空间管理，容易重复尝试已验证无效的方向；各处理阶段以独立脚本运行，缺乏统一编排，中途失败往往需从头重跑；训练元数据（配置版本、数据来源、产物依赖关系）分散存储，缺乏结构化查询能力[3]；评测流程缺乏系统化管理，无法进行跨实验对比分析。"),
  body("近年来，基于大语言模型（LLM）的智能Agent技术取得了突破性进展[4][5]。Agent通过"推理—行动—观察"的循环模式，结合工具调用与记忆机制，能够自主完成复杂的多步决策任务。将Agent架构引入ASR模型优化领域，以Agent作为智能决策核心驱动优化闭环，同时配合系统化的评测、版本管理、决策Trace观测与Web管理平台，可以形成一套完整的智能化模型优化工程系统。一方面Agent可以将资深工程师的领域经验结构化地沉淀到记忆系统中实现知识复用，另一方面平台化的管理能力使得实验数据、决策过程与产物关系均可被持久化记录和审计。"),
  body("因此，本课题设计并实现一套基于Agent架构的语音识别模型自动化优化平台，旨在将Agent的智能决策能力与系统化的评测、版本管理、观测能力相结合，为ASR模型的持续优化提供端到端的平台支撑。"),
  emptyLine(),
  heading("1.2 国内外概况和发展趋势", 2),
  heading("◎ 端到端语音识别技术", 2),
  body("语音识别技术经历了从传统HMM-DNN混合架构向端到端架构的演进。以Conformer[6]为骨干的注意力机制模型在多个基准任务上取得领先性能。高忠孝等人提出的Paraformer模型[7]通过CIF机制在中文识别任务上实现了精度与速度的良好平衡。石宇等人进一步提出SeaCo-Paraformer[8]，引入语义协作解码器实现热词定制能力，可在不重训主干模型的前提下提升唤醒词和控制指令的识别率。FunASR[9]作为达摩院开源工具包集成了Paraformer系列并提供完整训练推理流程。WeNet[10]采用流式/非流式统一架构在工业部署方面具有优势。上述工具包主要解决"如何训练一个模型"的问题，对"如何系统化管理多轮优化迭代"则支持不足。"),
  heading("◎ LLM Agent与自动化决策", 2),
  body("Yao等人提出的ReAct范式[4]将推理与行动相结合使Agent能够逐步推理决策。Shinn等人提出Reflexion[5]使Agent具备从历史经验中学习的能力。Schick等人[11]证明LLM能自主调用外部工具。Schmidt等人的AIDE框架[15]将机器学习工程建模为树搜索问题，Agent自主管理实验创建、评估与回溯，在Kaggle竞赛中取得接近人类专家的表现。Xi等人[23]对Agent的规划、记忆与工具使用能力进行了综述，指出工具使用的模块化设计是能力扩展的关键方向。杨等人[24]提出将通用基础能力与领域专有流程分层组织的架构原则。然而将Agent系统性地应用于ASR模型优化领域目前公开报道极少。"),
  heading("◎ 机器学习工程化与评测", 2),
  body("MLflow[17]提供了实验跟踪与模型注册的标准化管理能力。Amershi等人[22]总结了ML系统在数据管理与部署运维方面的工程挑战。刘宇涵等人[18]对预训练模型评测方法进行综述，提出评测流程标准化的架构思路。刘群等人[25]针对对话系统提出多维度质量评估方法，将规则评估与模型评估相结合。上述平台部署成本高，缺乏对ASR领域的定制能力，判断类决策仍需人工介入。"),
  heading("◎ 发展趋势", 2),
  body("ASR竞争焦点正从模型层向数据工程与流程智能化转移[20]；LLM Agent日益强调工具可靠性与能力模块化扩展[21][23]；ML工程化从"通用大平台"向"领域定制、评测系统化、全链路可观测"方向发展[22][16]。本课题正是顺应这些趋势的实践探索。"),
  emptyLine(),
  heading("1.3 选题的先进性和实用性", 2),
  body("本选题自主设计LLM Agent引擎，采用"通用工具层+领域技能层"的双层能力架构，Agent通过技能索引动态感知可用的ASR领域能力，自主编排优化流程。评测与分析流程系统化，支持规则型评估器与LLM型评估器。设计完整的Trace链路观测能力，Agent决策过程全面可追溯。引入三方版本管理，保证实验可复现。平台面向萤石智能设备唤醒词识别的真实业务需求，每位用户拥有独立Agent实例，项目组共享实验树与产物。Web控制台支持团队协作与多项目管理。"),
  emptyLine(),
  heading("1.4 选题的技术难度和工作量", 2),
  body("技术难度体现在：Agent引擎多组件协同设计（Prompt构建、工具调度、上下文压缩、迭代预算、记忆系统）；10个ASR领域技能的接口设计与封装（幂等性、失败处理、GPU检查）；评测与分析结果的结构化持久化；Trace链路记录与可视化；远程异构训练环境的健康检查与静默崩溃捕获；完整Web平台的全栈开发（前端6页面、后端7组API、数据库15张表、容器化隔离）。工作量涉及Agent引擎、10个技能、评测模块、Trace模块、版本管理、Web控制台、API层、数据库层、权限管理、Docker容器化、远程环境适配层的设计与实现以及端到端验证实验。"),
];

const refs = [
  "[1] 李健等. 端到端自动语音识别技术进展综述[J]. APSIPA TSIP, 2022, 11(1): e8.",
  "[2] 张斌等. WenetSpeech: 万小时多领域中文语音识别语料库[C]. ICASSP, 2022.",
  "[3] Sculley D, et al. Hidden Technical Debt in ML Systems[C]. NeurIPS, 2015.",
  "[4] Yao S, et al. ReAct: Synergizing Reasoning and Acting[C]. ICLR, 2023.",
  "[5] Shinn N, et al. Reflexion: Language Agents with Verbal RL[C]. NeurIPS, 2023.",
  "[6] Gulati A, et al. Conformer[C]. INTERSPEECH, 2020.",
  "[7] 高忠孝等. Paraformer[C]. INTERSPEECH, 2022.",
  "[8] 石宇等. SeaCo-Paraformer[C]. ICASSP, 2024.",
  "[9] 高忠孝等. FunASR[C]. INTERSPEECH, 2023.",
  "[10] 姚征等. WeNet[C]. INTERSPEECH, 2021.",
  "[11] Schick T, et al. Toolformer[C]. NeurIPS, 2023.",
  "[15] Schmidt D, et al. AIDE[C]. ICLR, 2025.",
  "[16] Paleyes A, et al. Challenges in Deploying ML[J]. ACM Comp. Surveys, 2022.",
  "[17] Zaharia M, et al. MLflow[J]. IEEE Data Eng. Bulletin, 2018.",
  "[18] 刘宇涵等. 预训练模型评测方法综述[J]. 中文信息学报, 2023.",
  "[19] 赵鑫等. 大语言模型综述[J]. 中国科学:信息科学, 2024.",
  "[20] Ng A. Data-centric AI[C]. AAAI Invited Talk, 2022.",
  "[21] 王磊等. 基于LLM的自主智能体综述[J]. FCS, 2024.",
  "[22] Amershi S, et al. SE for ML[C]. ICSE-SEIP, 2019.",
  "[23] Xi Z, et al. The Rise of LLM Based Agents[J]. arXiv, 2023.",
  "[24] 杨青等. LLM Agent系统研究进展[J]. 计算机研究与发展, 2024.",
  "[25] 刘群等. 智能对话系统评估方法综述[J]. 计算机学报, 2023.",
  "[26] 周志华. 机器学习[M]. 清华大学出版社, 2016.",
  "[27] 陈恩红等. 智能评测技术研究[J]. 中国科学:信息科学, 2022.",
  "[28] Park D S, et al. SpecAugment[C]. INTERSPEECH, 2019.",
];

const section1Refs = [
  emptyLine(),
  heading("2. 主要参考文献", 1),
  ...refs.map(r => refLine(r)),
];

// Section 3: 课题内容
const section3 = [
  heading("1. 课题内容", 1),
  body("本课题围绕"基于Agent架构的语音识别模型自动化优化平台"展开设计与实现，主要包含Agent决策引擎设计与实现、领域技能层设计与实现（10个ASR处理阶段）、评测与分析模块、Trace链路观测模块、版本管理模块、Web管理控制台（含Agent交互界面、数据管理、热词管理、评测看板）、RESTful API服务层与数据库层、用户管理与三级权限控制、项目容器化隔离、远程训练环境适配层、系统集成测试与端到端验证实验。"),
  emptyLine(),
  heading("2. 系统需求分析", 1),
  heading("2.1 概述", 2),
  body("本系统是面向ASR模型优化的智能管理平台。每位用户拥有独立的Agent会话实例，Agent作为个人优化助手驱动"评估—诊断—决策—执行—再评估"的闭环。项目组内共享实验树与数据产物以避免重复计算。GPU密集型任务通过资源检查与用户确认后执行。系统通过Web控制台为团队提供协作界面。"),
  heading("2.2 功能需求", 2),
  body("系统包含三类参与者：平台管理员负责用户管理和项目管理；项目Owner可与Agent交互、管理公共数据集和热词文件、修改配置、启动Agent；项目Member可与Agent交互发起实验、上传数据集到个人目录、查看所有结果，但不能修改配置和热词。系统用例图如图1所示。"),
  body("[此处插入图1：系统用例图]"),
  heading("2.3 非功能性需求", 2),
  body("系统在非功能性方面需满足以下要求：可复现性——相同配置版本与数据集版本下确定性阶段产出等价产物；健壮性——技能失败即终止后续，训练静默崩溃须被捕获；可观测性——Agent决策链路完整记录，技能状态实时可查；安全性——JWT认证与RBAC权限控制；多租户隔离——每项目独立Docker容器。"),
  emptyLine(),
  heading("3. 系统概要设计", 1),
  heading("3.1 系统总体架构", 2),
  body("系统采用B/S架构，分为六层：用户层、Web控制台层（Vue.js）、API服务层（FastAPI）、数据库层（MySQL+Redis）、Agent决策引擎层（含通用工具与领域技能）、远程训练环境适配层。每个项目对应独立Docker容器，挂载项目工作目录，共享MySQL和Redis。系统总体架构如图2所示。"),
  body("[此处插入图2：系统总体架构图]"),
  heading("3.2 Agent决策引擎设计", 2),
  body("Agent引擎内部由6个组件协同工作。Prompt Builder负责在每轮决策前动态组装上下文，包括身份定义、ASR领域知识、实验树状态摘要、紧凑记忆、可用技能列表及约束规则。LLM推理核心调用外部API获取Agent决策输出。Tool/Skill Dispatcher将工具调用指令路由到通用工具层或领域技能层执行。Context Engine管理上下文Token数量，接近窗口上限时自动压缩历史。Iteration Budget控制迭代预算防止无限循环（支持最大轮次、最大GPU时间、连续无改善停止三种条件）。Trajectory Logger将每轮THINK/EXECUTE/REFLECT完整记录为Trace Span。Agent内部架构如图3所示。"),
  body("[此处插入图3：Agent决策引擎内部架构图]"),
  body("Agent工作流程为THINK→EXECUTE→REFLECT循环：THINK阶段读取记忆与实验树，分析当前模型瓶颈，选择搜索策略（Draft新方向、Improve当前最优、Debug上次失败）；EXECUTE阶段调用领域技能执行数据处理或模型训练；REFLECT阶段分析执行结果，与基线对比，将有效经验记入洞察、无效方向记入死胡同。流程如图4所示。"),
  body("[此处插入图4：Agent优化循环流程图]"),
];

const section3b = [
  heading("3.3 通用工具层与领域技能层", 2),
  body("系统采用双层能力设计。通用工具层提供与领域无关的基础操作能力（read_file、write_file、run_command、edit_yaml、query_database），领域技能层将ASR优化的10个处理阶段封装为符合统一接口协议的独立技能。每个Skill具有标准接口：name（名称）、description（自描述文档）、parameters_schema（参数规格）、check_precondition()（前置条件检查）、execute()（执行主逻辑）。Agent可单独调用某个技能，也可组合多个技能顺序执行。10个技能的功能说明见表1。"),
  emptyLine(),
  body("表1 领域技能功能说明"),
  makeTable(
    ["序号", "技能名", "关键职责", "输入/输出"],
    [
      ["1", "prepare_data", "原始数据按格式整理", "raw → wav.scp+text"],
      ["2", "quality_scan", "音频质量扫描分类", "wav.scp → good/bad"],
      ["3", "split_dataset", "按设备隔离切分(seed=42)", "labeled → train/dev/test"],
      ["4", "augment_data", "训练集数据增强", "train → augmented"],
      ["5", "merge_sources", "多数据源合并去重", "multi → merged"],
      ["6", "generate_manifest", "生成训练清单", "scp+text → jsonl"],
      ["7", "train_model", "GPU检查+确认+训练+健康检查", "config → checkpoint"],
      ["8", "evaluate_model", "推理+规则指标+badcase", "ckpt → metrics"],
      ["9", "analyze_results", "错误归因+LLM分析", "metrics → report"],
      ["10", "edit_config", "修改配置文件", "key+value → config"],
    ]
  ),
  emptyLine(),
  body("其中train_model技能执行前先检查GPU占用状态，若被其他用户占用则返回占用信息（占用者、实验名、预计剩余时间）建议稍后重试。GPU空闲时向用户展示配置摘要请求确认后方可执行。"),
  heading("3.4 评测与分析模块", 2),
  body("评测与分析以evaluate_model和analyze_results两个技能实现。evaluate_model对测试集执行推理并计算CER、指令召回率、Stripped_CER等规则型指标，生成badcase清单（CER高于阈值的样本）。analyze_results统计错误类型分布（替换/插入/删除比例）、提取Top-N混淆对，并使用LLM型评估器对badcase做语义归因分析产出改进建议。分析报告持久化到数据库analysis_report表，Web控制台以图表形式展示并支持跨实验对比。Agent在REFLECT阶段读取报告决定下一步方向。流程如图5所示。"),
  body("[此处插入图5：评测与分析流程图]"),
  heading("3.5 Trace链路观测模块", 2),
  body("Trace模块将Agent每轮优化循环的完整决策链路结构化记录。每轮对应一条Trace，内含多个Span（think/execute/reflect类型），记录耗时、调用技能名、输入输出摘要。Web控制台以时间线形式可视化展示，支持按时间范围、技能名、状态筛选。Trace链路示例如图6所示。"),
  body("[此处插入图6：Trace链路示例图]"),
  heading("3.6 版本管理模块", 2),
  body("对配置、数据集、热词、模型四者建立版本关联。每个实验节点绑定精确版本信息，产出的模型也注册为model_version。任意历史实验可通过版本快照精确复现。支持从历史成功节点分支探索新方向。版本关系如图7所示。"),
  body("[此处插入图7：版本管理关联关系图]"),
  heading("3.7 数据库设计", 2),
  body("数据库采用MySQL 8.0，共15张核心表，涵盖用户管理（user、project、project_member）、实验管理（experiment、config_version、dataset_version、hotword_version、model_version）、评测分析（eval_run、eval_result、analysis_report）、Trace观测（trace、trace_span）、数据管理（dataset_file）和记忆引用（agent_memory_ref）。Agent记忆内容以文件形式存储在项目工作目录中，数据库仅记录路径引用。Redis缓存Agent实时运行状态供前端轮询。"),
  heading("3.8 Web控制台与用户管理", 2),
  body("Web控制台采用Vue.js 3开发，提供6个页面：项目管理、数据管理、Agent交互（对话式界面）、评测分析（指标图表与对比）、Trace视图（链路时间线）、实验看板（树形图）。用户通过邮箱注册、JWT登录。三级权限控制分为平台管理员、项目Owner和项目Member。热词管理页面支持上传、修改和版本查看。"),
  heading("3.9 项目初始化与容器管理", 2),
  body("创建项目时按固定模板初始化工作目录（configs/、datasets/shared/与user_id/、models/pretrained/、workspace/、outputs/、traces/），将预训练模型从共享仓库链接到项目目录，复制默认配置模板，启动Docker容器挂载该目录。部署架构如图8所示。"),
  body("[此处插入图8：部署架构图]"),
  heading("3.10 远程训练环境适配层", 2),
  body("封装SSH连接管理（断线重连）、GPU资源状态检查（返回占用者信息）、FunASR训练健康检查（检测返回码0但无检查点的静默崩溃）、SeaCo偏置标签启动前校验、训练日志流式采集等能力，对接远程8×RTX 3090集群。"),
  emptyLine(),
  heading("4. 拟采用的设计方法与环境", 1),
  body("系统采用面向对象设计方法，前后端分离，API接口通过OpenAPI规范定义。"),
  body("表2 开发环境"),
  makeTable(
    ["项目", "选型"],
    [
      ["后端", "Python 3.10 + FastAPI"],
      ["前端", "Vue.js 3 + Element Plus"],
      ["数据库", "MySQL 8.0 + Redis 7"],
      ["ASR框架", "FunASR (SeaCo-Paraformer)"],
      ["LLM", "OpenAI API兼容接口"],
      ["远程训练", "SSH + torchrun (8×RTX3090)"],
      ["容器化", "Docker + Docker Compose"],
      ["测试", "pytest + Hypothesis"],
    ]
  ),
  emptyLine(),
  heading("5. 可行性分析", 1),
  body("FunASR框架在生产环境中广泛使用，SeaCo-Paraformer在中文识别表现优异[8]；LLM Agent工具调用能力已在多场景验证[4][15]；FastAPI+Vue.js技术栈生态成熟；远程GPU集群已稳定运行。本课题属于所在企业ASR业务优化核心工作，得到团队支持，具备充足资源条件。"),
];

// Section 4: 预期成果
const section4 = [
  body("1. 基于Agent架构的语音识别模型自动化优化平台软件系统一套，包含Agent决策引擎（6个核心组件）、10个领域技能、评测与分析模块、Trace链路观测模块、版本管理模块、Web管理控制台（6个页面）、RESTful API服务层、MySQL数据库层（15张表）、Redis缓存层、权限与多项目管理、Docker容器化隔离、远程训练环境适配层。"),
  body("2. 在萤石s10/sp4业务数据集上经平台优化的ASR模型，领域KPI（指令召回率、Stripped_CER）相较基线有显著提升。"),
  body("3. 系统设计文档、API接口文档（Swagger）、操作手册及测试报告。"),
  body("4. 工程硕士学位论文1篇。"),
];

// Section 5: 已取得成果
const section5 = [
  body("1. 完成了核心框架搭建，包括CLI入口、原子操作（train/infer/eval/augment）、实验树与分层记忆系统的原型实现，已在实际业务中支撑Agent完成多轮优化循环。"),
  body("2. 完成了分片增量数据增强引擎的设计与实现，支持按策略管理增强版本。"),
  body("3. 完成了远程SSH执行能力的实现与验证，已在8×RTX 3090集群上稳定运行。"),
  body("4. 完成了10个领域技能的需求分析与概要设计，定义了统一的Skill接口协议。"),
];

// Section 6: 进度安排
const section6 = [
  body("表3 工作进度安排"),
  makeTable(
    ["时间", "工作内容", "阶段成果"],
    [
      ["2025.10-2025.11", "文献调研；系统架构设计", "开题报告"],
      ["2025.12-2026.01", "需求分析与概要设计", "设计报告"],
      ["2026.02-2026.03", "Agent引擎+10个技能实现", "引擎原型"],
      ["2026.04-2026.05", "评测+Trace+版本管理", "核心模块"],
      ["2026.06-2026.07", "Web控制台+API+数据库", "完整系统"],
      ["2026.08-2026.09", "集成测试+业务验证实验", "实验报告"],
      ["2026.10-2026.11", "数据分析+撰写论文", "论文初稿"],
      ["2026.12", "修改论文+答辩准备", "论文定稿"],
    ]
  ),
];

// Innovation section (Table 2 in template)
const innovation = [
  body("1. 自主设计LLM Agent引擎，采用"通用工具层+领域技能层"的双层能力架构，Agent通过技能索引动态感知可用ASR领域能力，自主编排优化流程。"),
  body("2. 评测与分析作为Agent技能实现，支持规则型评估器与LLM型评估器，结构化分析报告持久化到数据库，形成评估-分析-决策自动化闭环。"),
  body("3. 完整的Trace链路观测能力，Agent每轮决策的THINK/EXECUTE/REFLECT全过程结构化记录为可查询、可可视化的Span链路数据。"),
  body("4. 三方版本管理（配置/数据集/模型）与实验树结合，支持从历史成功节点分支探索，任意实验可精确复现。"),
  body("5. 每位用户独立Agent会话实例，项目组共享实验树与数据产物，GPU资源通过检查与确认机制调度。"),
];

// ============================================================
// BUILD DOCUMENT
// ============================================================
async function main() {
  const doc = new Document({
    styles: {
      default: {
        document: { run: { font: FONT, size: SIZE_BODY, color: "000000" } },
      },
    },
    sections: [
      // 选题依据
      { children: [...section1, ...section1Refs] },
      // 创新之处 (separate page)
      { children: [heading("研究的创新之处", 1), ...innovation] },
      // 课题内容
      { children: [...section3, ...section3b] },
      // 预期成果
      { children: [heading("预期成果", 1), ...section4] },
      // 已取得成果
      { children: [heading("已取得的与学位申请成果研究内容相关的成果", 1), ...section5] },
      // 进度安排
      { children: [heading("研究计划与进度安排", 1), ...section6] },
    ],
  });

  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(OUTPUT.toString(), buffer);
  console.log("Done! Output:", OUTPUT.toString());
}

main().catch(console.error);
