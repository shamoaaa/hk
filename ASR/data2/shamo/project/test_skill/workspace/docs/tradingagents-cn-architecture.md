# TradingAgents-CN 项目完整架构分析

## 一、项目概述

TradingAgents-CN 是基于 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) 的中文增强版，定位为**多智能体股票分析学习平台**。它利用 LangGraph 构建多 Agent 协作工作流，模拟真实投资机构中"分析师→研究员辩论→投资经理→交易员→风控委员会"的决策链路，支持 A股/港股/美股分析。

**技术栈：**
- 核心引擎：Python + LangGraph + LangChain
- 后端：FastAPI + Uvicorn
- 前端：Vue 3 + Vite + Element Plus
- 数据库：MongoDB + Redis + ChromaDB（向量记忆）
- 部署：Docker Compose（amd64 + arm64）

---

## 二、整体架构（三层分离）

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Vue 3 + Vite)                    │
│         Element Plus UI · Pinia Store · Vue Router           │
└────────────────────────────┬────────────────────────────────┘
                             │ RESTful API / WebSocket / SSE
┌────────────────────────────▼────────────────────────────────┐
│                  Backend (FastAPI + Uvicorn)                  │
│   Routers · Services · Worker · Scheduler · Middleware       │
│   MongoDB + Redis · 用户认证 · 配置管理 · 队列系统           │
└────────────────────────────┬────────────────────────────────┘
                             │ 调用核心引擎
┌────────────────────────────▼────────────────────────────────┐
│              Core Engine (tradingagents/)                     │
│   LangGraph 状态机 · 多智能体 · 数据流 · LLM 适配层         │
│   ChromaDB 向量记忆 · 工具链 · 信号处理                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、核心引擎模块构成 (`tradingagents/`)

### 3.1 图编排层 (`graph/`)

基于 **LangGraph StateGraph** 构建有向无环图（DAG）工作流：

| 文件 | 职责 |
|------|------|
| `trading_graph.py` | 主入口类 `TradingAgentsGraph`，初始化 LLM、记忆、工具，编排整个分析流程 |
| `setup.py` | `GraphSetup` 类，构建 StateGraph 的节点和边，定义 Agent 执行顺序 |
| `conditional_logic.py` | 条件路由逻辑：控制分析师工具调用循环、辩论轮次、风险讨论轮次 |
| `propagation.py` | 状态初始化和图执行参数配置 |
| `signal_processing.py` | 将最终文本决策解析为结构化交易信号（买入/卖出/持有 + 价格 + 置信度） |
| `reflection.py` | 反思机制：根据实际收益/亏损回顾各 Agent 的决策，更新向量记忆 |

**工作流执行顺序：**

```
START
  │
  ▼
┌─────────────────────────────────────────────────┐
│ 第一阶段：数据采集与分析（串行分析师链）          │
│                                                   │
│  Market Analyst ──→ tools ──→ Msg Clear           │
│       ↓                                           │
│  Social Media Analyst ──→ tools ──→ Msg Clear     │
│       ↓                                           │
│  News Analyst ──→ tools ──→ Msg Clear             │
│       ↓                                           │
│  Fundamentals Analyst ──→ tools ──→ Msg Clear     │
└───────────────────────┬─────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│ 第二阶段：投资辩论（多轮对抗）                    │
│                                                   │
│  Bull Researcher ←──→ Bear Researcher             │
│       (看涨)              (看跌)                  │
│            ↓ (达到轮次上限)                       │
│     Research Manager（综合裁决 + 投资计划）        │
└───────────────────────┬─────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│ 第三阶段：交易决策                                │
│                                                   │
│  Trader（生成具体交易建议）                       │
└───────────────────────┬─────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│ 第四阶段：风险评估辩论（三方对抗）                │
│                                                   │
│  Risky Analyst ──→ Safe Analyst ──→ Neutral       │
│       (激进)         (保守)         (中性)        │
│            ↓ (达到轮次上限)                       │
│     Risk Judge（最终风控裁决）                     │
└───────────────────────┬─────────────────────────┘
                        ↓
                       END
```

### 3.2 智能体层 (`agents/`)

| 子模块 | Agent | 职责 |
|--------|-------|------|
| `analysts/market_analyst.py` | 市场分析师 | 调用工具获取技术指标（MA/MACD/RSI/BOLL），生成技术分析报告 |
| `analysts/fundamentals_analyst.py` | 基本面分析师 | 获取 PE/PB/ROE/营收等财务数据，生成基本面报告 |
| `analysts/news_analyst.py` | 新闻分析师 | 获取并分析相关新闻，评估新闻对股价的影响 |
| `analysts/social_media_analyst.py` | 社交媒体分析师 | 分析社交媒体情绪（Reddit/中文社区），生成情绪报告 |
| `researchers/bull_researcher.py` | 看涨研究员 | 基于所有报告构建看涨论证，与看跌方辩论 |
| `researchers/bear_researcher.py` | 看跌研究员 | 基于所有报告构建看跌论证，与看涨方辩论 |
| `managers/research_manager.py` | 研究经理 | 综合辩论结果，裁决投资方向，制定投资计划 |
| `trader/trader.py` | 交易员 | 基于投资计划生成具体交易建议（目标价/止损/置信度） |
| `risk_mgmt/aggresive_debator.py` | 激进风控 | 从高收益角度评估交易计划 |
| `risk_mgmt/conservative_debator.py` | 保守风控 | 从风险规避角度评估交易计划 |
| `risk_mgmt/neutral_debator.py` | 中性风控 | 平衡收益与风险的评估 |
| `managers/risk_manager.py` | 风控主席 | 综合三方风控辩论，做出最终交易决策 |

**状态模型 (`AgentState`)：** 基于 LangGraph 的 `MessagesState`，包含：
- 消息历史、股票代码、交易日期
- 四份分析报告（market/sentiment/news/fundamentals）
- 投资辩论状态（`InvestDebateState`）：多头/空头历史、轮次计数
- 风险辩论状态（`RiskDebateState`）：三方历史、轮次计数
- 最终交易决策

### 3.3 数据流层 (`dataflows/`)

负责所有外部数据的获取、缓存和标准化：

| 子模块 | 职责 |
|--------|------|
| `providers/china/` | A股数据提供者：AKShare、Tushare、BaoStock 三源支持 |
| `providers/hk/` | 港股数据提供者 |
| `providers/us/` | 美股数据提供者（Yahoo Finance、Finnhub、SimFin） |
| `news/` | 新闻数据：Google News、Reddit、中文财经新闻 |
| `technical/` | 技术指标计算（基于 stockstats） |
| `cache/` | 多级缓存：文件缓存 → MongoDB 缓存 → 自适应缓存 |
| `interface.py` | 统一数据接口层，屏蔽底层数据源差异 |
| `data_source_manager.py` | 数据源管理器，支持动态切换和降级 |
| `stock_data_service.py` | 股票数据服务封装 |

**数据源降级链（A股实时行情）：**

```
stock_bid_ask_em → stock_zh_a_spot → stock_zh_a_spot_em → stock_zh_a_hist
```

### 3.4 LLM 适配层 (`llm_clients/` + `llm_adapters/`)

支持 10+ 种 LLM 提供商的统一接入：

```
llm_clients/
├── factory.py          # 工厂模式，根据 provider 创建对应 Client
├── base_client.py      # 抽象基类
├── openai_client.py    # OpenAI 兼容协议（覆盖大部分国产模型）
├── google_client.py    # Google Gemini
├── anthropic_client.py # Anthropic Claude
├── provider_keys.py    # Provider 名称规范化
├── model_catalog.py    # 共享模型目录
└── validators.py       # 轻量校验

llm_adapters/
├── openai_compatible_base.py    # OpenAI 兼容基础适配器
├── dashscope_openai_adapter.py  # 阿里百炼/通义千问适配
├── deepseek_adapter.py          # DeepSeek 适配
└── google_openai_adapter.py     # Google → OpenAI 协议转换
```

**支持的 Provider：** OpenAI、Anthropic、Google、DeepSeek、阿里百炼(Qwen)、智谱(GLM)、百度千帆、SiliconFlow、OpenRouter、AiHubMix、Ollama、自定义 OpenAI 兼容端点。

**双模型架构：** 系统使用两个 LLM 实例：
- `quick_thinking_llm`：快速模型，用于分析师、研究员等高频调用
- `deep_thinking_llm`：深度模型，用于研究经理、风控主席等关键决策

### 3.5 工具层 (`tools/` + `agents/utils/agent_utils.py`)

`Toolkit` 类封装了所有可供 Agent 调用的工具函数（LangChain Tool 格式）：

- **市场数据工具：** `get_stock_market_data_unified`（统一入口，自动识别市场）
- **基本面工具：** `get_stock_fundamentals_unified`、`get_china_fundamentals`
- **新闻工具：** `get_stock_news_unified`、`get_realtime_stock_news`、`get_google_news`
- **情绪工具：** `get_stock_sentiment_unified`、`get_chinese_social_sentiment`
- **技术指标：** `get_stockstats_indicators_report`

### 3.6 记忆系统 (`agents/utils/memory.py`)

基于 **ChromaDB** 向量数据库实现经验记忆：

- 每个关键 Agent（Bull/Bear/Trader/InvestJudge/RiskManager）拥有独立记忆集合
- 记忆内容：历史市场情况 + 对应的反思/经验教训
- 检索方式：语义相似度匹配（当前市场情况 → 相似历史情况的经验）
- 反思机制：交易完成后，根据实际收益回顾决策，生成经验教训存入记忆

---

## 四、后端服务层 (`app/`)

FastAPI 企业级后端，采用分层架构：

```
app/
├── main.py              # 应用入口、生命周期管理、调度器配置
├── core/                # 基础设施
│   ├── config.py        # Settings（Pydantic BaseSettings）
│   ├── database.py      # MongoDB 连接管理
│   ├── redis_client.py  # Redis 连接
│   ├── config_bridge.py # 配置桥接（DB配置 → 环境变量 → 核心引擎）
│   └── rate_limiter.py  # 限流
├── routers/             # API 路由（35+ 个路由模块）
│   ├── analysis.py      # 分析任务提交/查询
│   ├── stocks.py        # 股票信息
│   ├── screening.py     # 智能筛选
│   ├── favorites.py     # 自选股
│   ├── reports.py       # 报告导出
│   ├── config.py        # 配置管理
│   ├── auth_db.py       # 用户认证
│   ├── sse.py           # SSE 实时推送
│   ├── queue.py         # 批量分析队列
│   └── ...（共35+模块）
├── services/            # 业务逻辑层
│   ├── analysis_service.py    # 核心：调用 TradingAgentsGraph 执行分析
│   ├── queue_service.py       # 批量任务队列管理
│   ├── config_service.py      # LLM/数据源配置管理
│   ├── screening_service.py   # 股票筛选
│   ├── favorites_service.py   # 自选股管理
│   ├── websocket_manager.py   # WebSocket 连接管理
│   ├── redis_progress_tracker.py  # Redis 进度追踪
│   └── ...
├── models/              # 数据模型（MongoDB Document）
├── schemas/             # Pydantic 请求/响应模型
├── middleware/          # 中间件（操作日志、认证）
└── worker/              # 后台任务（数据同步服务）
```

**关键服务流程（分析任务）：**

```
用户提交分析请求
    → AnalysisService.submit_single_analysis()
    → 创建 AnalysisTask，存入 MongoDB
    → 异步执行 _execute_analysis_sync_with_progress()
        → 构建配置 → 实例化 TradingAgentsGraph
        → graph.propagate(stock, date, progress_callback)
        → 通过 Redis + SSE 实时推送进度
    → 解析最终信号 → 存储分析结果
    → 返回结构化报告
```

---

## 五、前端层 (`frontend/`)

Vue 3 + Vite + Element Plus 单页应用：

| 目录 | 职责 |
|------|------|
| `views/Dashboard/` | 仪表盘首页 |
| `views/Analysis/` | 股票分析页面（核心交互） |
| `views/Stocks/` | 个股详情 |
| `views/Screening/` | 智能筛选 |
| `views/Favorites/` | 自选股管理 |
| `views/Reports/` | 分析报告查看/导出（Markdown/Word/PDF） |
| `views/Queue/` | 批量分析队列 |
| `views/PaperTrading/` | 模拟交易 |
| `views/Settings/` | 系统配置（LLM/数据源） |
| `views/Learning/` | 学习中心 |
| `views/Auth/` | 登录/注册 |
| `views/System/` | 系统管理 |
| `stores/` | Pinia 状态管理 |
| `api/` | Axios API 封装 |

---

## 六、部署架构

```
┌─────────────────────────────────────────────┐
│              Docker Compose                  │
│                                             │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Frontend│  │ Backend  │  │  MongoDB  │  │
│  │ (Nginx) │  │ (Uvicorn)│  │           │  │
│  └────┬────┘  └────┬─────┘  └───────────┘  │
│       │             │                        │
│  ┌────┴─────────────┴────┐  ┌───────────┐  │
│  │       Nginx Proxy     │  │   Redis   │  │
│  └───────────────────────┘  └───────────┘  │
└─────────────────────────────────────────────┘
```

支持 amd64 + arm64 双架构，通过 GitHub Actions 自动构建镜像。

---

## 七、核心设计思路总结

### 7.1 多智能体辩论机制

不是单一 LLM 给出答案，而是模拟投资机构的决策流程：
- 多角度分析（技术面/基本面/新闻/情绪）
- 多空辩论（Bull vs Bear，可配置轮次）
- 综合裁决（Research Manager）
- 风控审核（三方风控辩论 + Risk Judge）

通过对抗性讨论提高决策质量，避免单一视角偏见。

### 7.2 LangGraph 状态机

利用 LangGraph 的 StateGraph 实现复杂的 Agent 编排：
- 条件路由：工具调用循环（最多3次防死循环）、辩论轮次控制
- 状态传递：`AgentState` 在节点间流转，累积分析报告
- 流式执行：支持 `updates` 模式实时追踪节点进度

### 7.3 经验记忆与反思

通过 ChromaDB 向量数据库存储历史决策的反思：
- 新分析时检索相似历史情况的经验教训
- 交易完成后根据实际收益回顾决策
- 实现"从错误中学习"的闭环

### 7.4 多 LLM 提供商抽象

通过工厂模式 + 适配器模式：
- 统一接入 10+ 种 LLM
- 支持快速/深度双模型配置
- 支持混合模式（不同厂家组合）
- Provider 名称规范化和别名兼容

### 7.5 多数据源降级

- A股数据支持 Tushare/AKShare/BaoStock 三源自动切换
- 实时行情支持多级降级链
- 港股/美股采用按需获取+缓存模式
- 确保数据可用性和稳定性

### 7.6 前后端分离 + 实时通信

- FastAPI 提供 RESTful API
- SSE + WebSocket 双通道实现分析进度实时推送
- Redis 作为进度追踪中间件

### 7.7 企业级特性

- 用户认证与角色管理
- 操作日志审计
- APScheduler 定时任务调度（数据同步）
- 批量分析队列
- 多级缓存（文件/MongoDB/Redis）
- 配置热更新（数据库配置 → 环境变量桥接）

---

## 八、目录结构总览

```
TradingAgents-CN/
├── tradingagents/           # 核心引擎（开源，Apache 2.0）
│   ├── graph/               # LangGraph 工作流编排
│   ├── agents/              # 多智能体定义
│   │   ├── analysts/        # 4类分析师
│   │   ├── researchers/     # 多空研究员
│   │   ├── managers/        # 研究经理 + 风控主席
│   │   ├── trader/          # 交易员
│   │   ├── risk_mgmt/       # 3类风控分析师
│   │   └── utils/           # 状态定义、工具集、记忆
│   ├── dataflows/           # 数据流（获取/缓存/标准化）
│   │   ├── providers/       # 多市场数据提供者
│   │   ├── news/            # 新闻数据源
│   │   ├── technical/       # 技术指标计算
│   │   └── cache/           # 多级缓存
│   ├── llm_clients/         # LLM 客户端工厂
│   ├── llm_adapters/        # LLM 协议适配器
│   ├── tools/               # 分析工具
│   ├── config/              # 配置管理
│   └── utils/               # 通用工具
├── app/                     # FastAPI 后端（专有）
│   ├── core/                # 基础设施
│   ├── routers/             # API 路由
│   ├── services/            # 业务逻辑
│   ├── models/              # 数据模型
│   ├── schemas/             # 请求/响应模型
│   ├── middleware/          # 中间件
│   └── worker/              # 后台任务
├── frontend/                # Vue 3 前端（专有）
│   └── src/
│       ├── views/           # 页面组件
│       ├── stores/          # 状态管理
│       ├── api/             # API 封装
│       └── components/      # 通用组件
├── docker/                  # Docker 配置
├── docs/                    # 文档
├── config/                  # 日志配置
├── scripts/                 # 工具脚本
└── docker-compose.yml       # 编排文件
```

---

*文档生成时间: 2026-05-18*
