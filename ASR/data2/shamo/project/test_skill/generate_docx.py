# -*- coding: utf-8 -*-
"""Generate opening report. Run: python generate_docx.py"""
import re
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

TEMPLATE = Path("C:/Users/shamo/Desktop/专业学位研究生学位申请成果开题报告.docx")
OUTPUT = Path("C:/Users/shamo/Desktop/开题报告_v4.docx")
CONTENT_DIR = Path("D:/project/ASR/data2/shamo/project/test_skill/report_content")

BLACK = RGBColor(0, 0, 0)
INDENT = "\u3000\u3000"  # Two full-width spaces for paragraph indent


def styled_run(para, text, bold=False, size=Pt(10.5)):
    r = para.add_run(text)
    r.font.name = "宋体"
    r._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    r.font.size = size
    r.font.color.rgb = BLACK
    r.font.bold = bold
    return r


def add_heading(cell, text, level=1, first=False):
    """Add a heading paragraph. level 1=section, 2=subsection."""
    if first:
        para = cell.paragraphs[0]
        para.text = ""
    else:
        para = cell.add_paragraph()
    pf = para.paragraph_format
    pf.space_before = Pt(8)
    pf.space_after = Pt(4)
    pf.line_spacing = Pt(18)
    pf.first_line_indent = Cm(0)
    size = Pt(12) if level == 1 else Pt(11)
    styled_run(para, text, bold=True, size=size)
    return para


def add_body(cell, text):
    """Add a body paragraph with 2-char indent."""
    para = cell.add_paragraph()
    pf = para.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(2)
    pf.line_spacing = Pt(16)
    pf.first_line_indent = Cm(0.74)  # approx 2 Chinese chars
    styled_run(para, text)
    return para


def add_item(cell, text):
    """Add a list-style item (no indent, for numbered items)."""
    para = cell.add_paragraph()
    pf = para.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(2)
    pf.line_spacing = Pt(16)
    pf.first_line_indent = Cm(0.74)
    styled_run(para, text)
    return para


def add_empty(cell):
    para = cell.add_paragraph()
    para.paragraph_format.space_after = Pt(2)
    para.paragraph_format.line_spacing = Pt(8)


def add_table(cell, headers, rows):
    """Add a real Word table after the current cell content."""
    from docx.oxml import OxmlElement
    # Create table element inside the cell
    n_rows = 1 + len(rows)
    n_cols = len(headers)
    tbl_xml = cell._element
    # Use the document to add a table and then move it
    # Workaround: add paragraphs that look like a table using tabs
    add_empty(cell)
    # Header line
    header_text = "\t".join(headers)
    p = cell.add_paragraph()
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.line_spacing = Pt(14)
    styled_run(p, header_text, bold=True, size=Pt(9))
    # Separator
    p = cell.add_paragraph()
    p.paragraph_format.space_after = Pt(1)
    styled_run(p, "─" * 60, size=Pt(8))
    # Rows
    for row in rows:
        row_text = "\t".join(row)
        p = cell.add_paragraph()
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = Pt(14)
        styled_run(p, row_text, size=Pt(9))
    add_empty(cell)


def fill_section1(cell):
    """选题依据"""
    add_heading(cell, "1.1 选题的研究意义", level=2, first=True)
    add_body(cell, "随着智能家居、车载交互与智能客服等领域的快速发展，语音识别（ASR）技术已成为人机交互的核心入口之一[1]。以萤石智能摄像头为代表的物联网设备普遍采用"唤醒词+控制指令"的语音交互范式，对识别准确率、唤醒词鲁棒性及远场噪声环境的适应能力提出了极高要求[2]。将通用预训练ASR模型适配到具体业务场景并达到产品级要求，本质上是一个涉及数据工程、模型训练与评估分析的多轮迭代优化过程。")
    add_body(cell, "在实际业务实践中，ASR模型的优化迭代长期以零散脚本与人工经验判断的方式运行。算法工程师需要反复执行"分析瓶颈→提出假设→调整配置→执行训练→解读结果→决定下一步"的循环，而这些环节之间缺乏统一的编排与管理机制。优化决策高度依赖个人经验且不可复现；实验管理呈线性结构，缺乏可回溯的搜索空间管理，容易重复尝试已验证无效的方向；各处理阶段以独立脚本运行，中途失败往往需从头重跑；训练元数据分散存储，缺乏结构化查询能力[3]；评测流程缺乏系统化管理，无法进行跨实验对比。")
    add_body(cell, "近年来，基于大语言模型（LLM）的智能Agent技术取得了突破性进展[4][5]。Agent通过"推理—行动—观察"的循环模式，结合工具调用与记忆机制，能够自主完成复杂的多步决策任务。将Agent架构引入ASR模型优化领域，以Agent作为智能决策核心驱动优化闭环，同时配合系统化的评测、版本管理、决策Trace观测与Web管理平台，可以形成一套完整的智能化模型优化工程系统。")
    add_body(cell, "因此，本课题设计并实现一套基于Agent架构的语音识别模型自动化优化平台，旨在将Agent的智能决策能力与系统化的评测、版本管理、观测能力相结合，为ASR模型的持续优化提供端到端的平台支撑。")

    add_empty(cell)
    add_heading(cell, "1.2 国内外概况和发展趋势", level=2)
    add_heading(cell, "◎ 端到端语音识别技术", level=2)
    add_body(cell, "语音识别技术经历了从传统HMM-DNN混合架构向端到端架构的演进。以Conformer[6]为骨干的注意力机制模型在多个基准任务上取得领先性能。高忠孝等人提出的Paraformer模型[7]通过CIF机制在中文识别任务上实现了精度与速度的良好平衡。石宇等人进一步提出SeaCo-Paraformer[8]，引入语义协作解码器实现热词定制能力，可在不重训主干模型的前提下提升唤醒词和控制指令的识别率。FunASR[9]作为达摩院开源工具包，集成了Paraformer系列并提供完整训练推理流程。WeNet[10]采用流式/非流式统一架构，在工业部署方面具有优势。上述工具包主要解决"如何训练一个模型"，对"如何系统化管理多轮优化迭代"则支持不足。")
    add_heading(cell, "◎ LLM Agent与自动化决策", level=2)
    add_body(cell, "Yao等人提出的ReAct范式[4]将推理与行动相结合，使Agent能够逐步推理与决策。Shinn等人提出Reflexion[5]，使Agent具备从历史经验中学习的能力。Schick等人[11]证明LLM能自主调用外部工具。Schmidt等人的AIDE框架[15]将机器学习工程建模为树搜索问题，Agent自主管理实验创建、评估与回溯，在Kaggle竞赛中取得接近人类专家的表现。Xi等人[23]对Agent的规划、记忆与工具使用能力进行了综述，指出工具使用的模块化设计是能力扩展的关键方向。杨等人[24]提出将通用基础能力与领域专有流程分层组织的架构原则。然而，将Agent系统性地应用于ASR模型优化领域，目前公开报道极少。")
    add_heading(cell, "◎ 机器学习工程化与评测", level=2)
    add_body(cell, "MLflow[17]提供了实验跟踪与模型注册的标准化管理能力。Amershi等人[22]总结了ML系统在数据管理与部署运维方面的工程挑战。刘宇涵等人[18]对预训练模型评测方法进行综述，提出评测流程标准化的架构思路。刘群等人[25]针对对话系统提出多维度质量评估方法，将规则评估与模型评估相结合。上述平台部署成本高、缺乏对ASR领域的定制能力，判断类决策仍需人工介入。")
    add_heading(cell, "◎ 发展趋势", level=2)
    add_body(cell, "ASR竞争焦点正从模型层向数据工程与流程智能化转移[20]；LLM Agent日益强调工具可靠性与能力模块化扩展[21][23]；ML工程化从"通用大平台"向"领域定制、评测系统化、全链路可观测"方向发展[22][16]。本课题正是顺应这些趋势的实践探索。")

    add_empty(cell)
    add_heading(cell, "1.3 选题的先进性和实用性", level=2)
    add_body(cell, "本选题自主设计LLM Agent引擎，采用"通用工具层+领域技能层"的双层能力架构，Agent通过技能索引动态感知可用的ASR领域能力，自主编排优化流程。评测与分析流程系统化，支持规则型评估器与LLM型评估器。设计完整的Trace链路观测能力，Agent决策过程全面可追溯。引入三方版本管理，保证实验可复现。")
    add_body(cell, "平台面向萤石智能设备唤醒词识别的真实业务需求，每位用户拥有独立Agent实例，项目组共享实验树与产物。Web控制台支持团队协作与多项目管理。")
    add_empty(cell)
    add_heading(cell, "1.4 选题的技术难度和工作量", level=2)
    add_body(cell, "技术难度体现在：Agent引擎多组件协同设计（Prompt构建、工具调度、上下文压缩、迭代预算、记忆系统）；10个ASR领域技能的接口设计与封装（幂等性、失败处理、GPU检查）；评测与分析结果的结构化持久化；Trace链路记录与可视化；远程异构训练环境的健康检查与静默崩溃捕获；完整Web平台的全栈开发（前端6页面、后端7组API、数据库15张表、容器化隔离）。")
    add_body(cell, "工作量涉及Agent引擎、10个技能、评测分析模块、Trace模块、版本管理、Web控制台、API层、数据库层、权限管理、Docker容器化、远程环境适配层的设计与实现，以及端到端验证实验。")
    add_empty(cell)
    add_heading(cell, "2. 主要参考文献", level=1)
    refs = [
        "[1] 李健等. 端到端自动语音识别技术进展综述[J]. APSIPA TSIP, 2022.",
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
        "[12] Jimenez C E, et al. SWE-bench[C]. ICLR, 2024.",
        "[13] Lu C, et al. The AI Scientist[C]. ICLR, 2025.",
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
    ]
    for ref in refs:
        p = cell.add_paragraph()
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = Pt(14)
        p.paragraph_format.first_line_indent = Cm(0)
        styled_run(p, ref, size=Pt(9))


def fill_section3(cell):
    """课题内容及具体方案 (Table 3)"""
    add_heading(cell, "1. 课题内容", level=1, first=True)
    add_body(cell, "本课题围绕"基于Agent架构的语音识别模型自动化优化平台"展开设计与实现，主要包含：Agent决策引擎的设计与实现；领域技能层的设计与实现（10个ASR处理阶段）；评测与分析模块；Trace链路观测模块；版本管理模块；Web管理控制台（含Agent交互界面、数据管理、热词管理、评测看板）；RESTful API服务层与数据库层；用户管理与三级权限控制；项目容器化隔离；远程训练环境适配层；系统集成测试与端到端验证实验。")
    add_empty(cell)
    add_heading(cell, "2. 系统需求分析", level=1)
    add_heading(cell, "2.1 概述", level=2)
    add_body(cell, "本系统是面向ASR模型优化的智能管理平台。每位用户拥有独立的Agent会话实例，Agent作为个人优化助手驱动闭环优化。项目组内共享实验树与数据产物，避免重复计算。GPU密集型任务通过资源检查与用户确认后执行。系统通过Web控制台为团队提供协作界面。")
    add_heading(cell, "2.2 功能需求", level=2)
    add_body(cell, "系统包含三类参与者。平台管理员负责用户管理（注册/启用/禁用）和项目管理（创建/删除/分配成员）。项目Owner可与Agent交互、管理公共数据集和热词文件、修改配置、启动Agent。项目Member可与Agent交互发起实验、上传数据集到个人目录、查看实验与评测结果，但不能修改配置和热词等关键文件。普通用户用例图如图1所示。")
    add_body(cell, "[此处插入图1：系统用例图]")
    add_heading(cell, "2.3 非功能性需求", level=2)
    add_body(cell, "可复现性：相同配置+数据集版本下确定性阶段产出等价产物。健壮性：技能失败即终止后续，训练静默崩溃须被捕获。可观测性：Agent决策链路完整记录，技能状态实时可查。安全性：JWT认证与RBAC权限控制。多租户隔离：每项目独立Docker容器。")
    add_empty(cell)
    add_heading(cell, "3. 系统概要设计", level=1)
    add_heading(cell, "3.1 系统总体架构", level=2)
    add_body(cell, "系统采用B/S架构，分为六层：用户层、Web控制台层（Vue.js）、API服务层（FastAPI）、数据库层（MySQL+Redis）、Agent决策引擎层（含通用工具与领域技能）、远程训练环境适配层。每个项目对应独立Docker容器，挂载项目工作目录，共享MySQL和Redis。系统总体架构如图2所示。")
    add_body(cell, "[此处插入图2：系统总体架构图]")
    add_heading(cell, "3.2 Agent决策引擎设计", level=2)
    add_body(cell, "Agent引擎由6个组件协同工作：Prompt Builder动态组装上下文（身份定义+领域知识+实验树摘要+记忆+技能索引+约束规则）；LLM推理核心调用外部API获取决策；Tool/Skill Dispatcher将调用指令路由到通用工具层或领域技能层；Context Engine管理上下文Token并自动压缩历史；Iteration Budget控制迭代预算防止无限循环；Trajectory Logger将每轮THINK/EXECUTE/REFLECT记录为Trace Span。Agent内部架构如图3所示。")
    add_body(cell, "[此处插入图3：Agent决策引擎内部架构图]")
    add_body(cell, "Agent工作流程为THINK→EXECUTE→REFLECT循环：THINK阶段分析瓶颈选择搜索策略（Draft新方向/Improve最优/Debug失败）；EXECUTE阶段调用技能执行；REFLECT阶段分析结果并更新记忆。流程如图4所示。")
    add_body(cell, "[此处插入图4：Agent优化循环流程图]")

    add_heading(cell, "3.3 通用工具层与领域技能层", level=2)
    add_body(cell, "系统采用双层能力设计。通用工具层提供基础操作（read_file、write_file、run_command、edit_yaml、query_database），领域技能层将ASR优化的10个处理阶段封装为符合统一接口协议的独立Skill。每个Skill具有name、description、parameters_schema、check_precondition()、execute()等标准接口。Agent可单独调用某个技能，也可组合多个技能顺序执行。10个技能包括：prepare_data（数据整理）、quality_scan（质量扫描）、split_dataset（按设备隔离切分）、augment_data（数据增强）、merge_sources（多源合并）、generate_manifest（训练清单生成）、train_model（GPU检查+确认+训练+健康检查）、evaluate_model（推理+指标计算）、analyze_results（错误归因分析）、edit_config（配置修改）。")
    add_body(cell, "其中train_model技能执行前先检查GPU占用状态，若被占用则返回占用信息建议稍后重试，空闲时向用户展示配置摘要请求确认后方可执行。")
    add_heading(cell, "3.4 评测与分析模块", level=2)
    add_body(cell, "评测与分析以evaluate_model和analyze_results两个技能实现。evaluate_model对测试集执行推理并计算CER、指令召回率、Stripped_CER等规则型指标，生成badcase清单。analyze_results统计错误类型分布、提取混淆对，并使用LLM型评估器对badcase做语义归因分析。分析报告持久化到数据库，在Web看板展示。Agent在REFLECT阶段读取报告决定下一步方向。流程如图5所示。")
    add_body(cell, "[此处插入图5：评测与分析流程图]")
    add_heading(cell, "3.5 Trace链路观测模块", level=2)
    add_body(cell, "Trace模块将Agent每轮优化循环结构化记录。每轮对应一条Trace，内含多个Span（think/execute/reflect类型）。Span记录耗时、调用技能名、输入输出摘要。数据存入trace和trace_span表，Web控制台以时间线形式可视化展示，支持按时间、技能、状态筛选。Trace链路示例如图6所示。")
    add_body(cell, "[此处插入图6：Trace链路示例图]")
    add_heading(cell, "3.6 版本管理模块", level=2)
    add_body(cell, "对配置、数据集、热词、模型四者建立版本关联。每个实验节点绑定精确版本信息（config_ver、dataset_ver、hotword_ver），产出的模型也注册为model_version。任意实验可通过版本快照复现。版本关系如图7所示。")
    add_body(cell, "[此处插入图7：版本管理关联关系图]")
    add_heading(cell, "3.7 数据库设计", level=2)
    add_body(cell, "数据库采用MySQL 8.0，共15张核心表，涵盖用户管理（user、project、project_member）、实验管理（experiment、config_version、dataset_version、hotword_version、model_version）、评测分析（eval_run、eval_result、analysis_report）、Trace观测（trace、trace_span）、数据管理（dataset_file）和记忆引用（agent_memory_ref）。Redis缓存Agent实时运行状态供前端轮询。所有项目共享MySQL实例，通过project_id字段逻辑隔离。")
    add_heading(cell, "3.8 Web控制台与用户管理", level=2)
    add_body(cell, "Web控制台采用Vue.js 3开发，提供6个页面：项目管理（创建/删除/成员）、数据管理（上传/查看/删除）、Agent交互（对话式界面）、评测分析（指标图表+对比）、Trace视图（链路时间线）、实验看板（树形图）。用户通过邮箱注册、JWT登录。三级权限控制（管理员/Owner/Member）。热词管理页面支持上传、修改、查看版本历史。")
    add_heading(cell, "3.9 项目初始化与容器管理", level=2)
    add_body(cell, "创建项目时按固定模板初始化工作目录（configs/、datasets/shared/+user_id/、models/pretrained/、workspace/、outputs/、traces/），将预训练模型从共享仓库链接到项目目录，从模板复制默认配置，启动Docker容器。部署架构如图8所示。")
    add_body(cell, "[此处插入图8：部署架构图]")
    add_heading(cell, "3.10 远程训练环境适配层", level=2)
    add_body(cell, "封装SSH连接管理（断线重连）、GPU资源状态检查（返回占用者信息）、FunASR训练健康检查（检测静默崩溃）、SeaCo标签启动前校验、训练日志流式采集等能力，对接8×RTX 3090集群。")
