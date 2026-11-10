# Coze Loop 项目架构分析

## 一、项目概述

Coze Loop 是字节跳动/Coze 团队开源的一个面向 AI Agent 开发与运维的平台级解决方案。它提供从 Prompt 开发、调试、评估到监控的全生命周期管理能力，帮助开发者更高效地开发和运维 AI Agent。

**核心定位**：AI Agent 的 DevOps 平台（开发 + 评测 + 观测）

**技术栈**：
- 后端：Go (Hertz HTTP框架 + Kitex RPC框架，CloudWeGo 生态)
- 前端：TypeScript (React + Rush monorepo)
- 数据库：MySQL + ClickHouse + Redis
- 对象存储：S3 兼容（MinIO）
- 消息队列：RocketMQ
- LLM集成：Eino 框架（CloudWeGo 出品）
- 部署：Docker Compose / Kubernetes (Helm Chart)

---

## 二、总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                       用户层 (浏览器)                             │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                    Nginx 反向代理 + 静态资源                      │
├─────────────────────┬───────────────────────────────────────────┤
│   前端 SPA 应用      │        API 请求转发                       │
│   (React/TS)        │        /api/* → Backend                   │
└─────────────────────┴───────────────────────────────┬───────────┘
                                                      │
┌─────────────────────────────────────────────────────▼───────────┐
│                   后端服务 (Go / Hertz)                           │
├──────────┬──────────┬──────────┬──────────┬─────────┬───────────┤
│ Prompt   │Evaluation│Observa-  │   LLM    │  Data   │Foundation │
│ Module   │ Module   │bility    │  Module  │ Module  │  Module   │
│          │          │Module    │          │         │           │
├──────────┴──────────┴──────────┴──────────┴─────────┴───────────┤
│                      基础设施层 (Infra)                           │
│  DB│ClickHouse│Redis│MQ(RocketMQ)│S3/MinIO│ID生成│限流│i18n│指标 │
└──────────┬──────────┬──────────┬──────────┬─────────────────────┘
           │          │          │          │
     ┌─────▼──┐ ┌────▼───┐ ┌───▼──┐ ┌────▼────┐
     │ MySQL  │ │ Click- │ │Redis │ │  MinIO  │
     │        │ │ House  │ │      │ │  (S3)   │
     └────────┘ └────────┘ └──────┘ └─────────┘
```

**架构特点**：
- 单体服务设计，所有业务模块编译为一个二进制，内部通过 module 划分边界
- 前后端分离，通过 Nginx 统一入口
- DDD（领域驱动设计）分层：application → domain → infra
- IDL 优先（Thrift 定义接口），代码生成

---

## 三、目录结构说明

```
coze-loop/
├── backend/          # Go 后端服务
│   ├── api/          # HTTP 路由与 Handler（Hertz 框架）
│   ├── cmd/          # 程序入口（main.go）
│   ├── conf/         # 配置文件
│   ├── infra/        # 基础设施层（数据库、缓存、MQ、存储等）
│   ├── kitex_gen/    # Kitex RPC 代码生成
│   ├── loop_gen/     # Loop 自身代码生成（本地服务适配）
│   ├── modules/      # 业务模块（核心）
│   │   ├── data/         # 数据管理模块
│   │   ├── evaluation/   # 评测模块
│   │   ├── foundation/   # 基础模块（用户、空间、文件、认证）
│   │   ├── llm/          # LLM 模型管理与调用
│   │   ├── observability/# 观测模块（Trace）
│   │   └── prompt/       # Prompt 管理与调试
│   ├── pkg/          # 通用工具包
│   └── script/       # 代码生成脚本
├── frontend/         # TypeScript 前端（Rush monorepo）
│   ├── apps/cozeloop/    # 主应用
│   ├── packages/         # 共享包
│   │   ├── loop-base/        # 基础能力（路由、状态、API Schema）
│   │   ├── loop-components/  # 业务组件
│   │   ├── loop-config/      # 构建配置
│   │   ├── loop-modules/     # 业务逻辑模块
│   │   └── loop-pages/       # 页面组件
│   └── config/           # ESLint/TS/Tailwind 等配置
├── idl/              # Thrift IDL 接口定义
├── common/           # Rush 工程配置、Git Hooks
└── release/          # 部署相关
    ├── deployment/       # Docker Compose / Helm Chart
    └── image/            # Dockerfile
```

---

## 四、后端业务模块详解

### 4.1 Foundation 模块（基础服务）

**职责**：用户管理、空间管理、文件服务、认证鉴权

**子功能**：
- 用户注册/登录/Token 管理
- Workspace（工作空间）的创建与管理
- 文件上传/下载（对接 S3）
- 权限控制（认证 + 鉴权）

### 4.2 Prompt 模块（Prompt 开发与调试）

**职责**：Prompt 的全生命周期管理

**子功能**：
- Prompt 创建、编辑、删除
- Prompt 版本管理（多版本对比）
- Playground 调试（实时与 LLM 交互测试）
- Prompt 执行（调用 LLM 获取结果）
- Tool 管理（Prompt 中可调用的工具定义）
- OpenAPI 对外接口

**内部分层**：
```
prompt/
├── application/    # 应用服务层（debug.go, execute.go, manage.go, openapi.go）
├── domain/         # 领域层
│   ├── entity/     # 领域实体
│   ├── service/    # 领域服务
│   ├── repo/       # 仓储接口
│   └── component/  # 领域组件
├── infra/          # 基础设施实现
│   ├── repo/       # 仓储实现（MySQL）
│   ├── rpc/        # 外部 RPC 调用
│   ├── conf/       # 模块配置
│   └── collector/  # 数据采集
└── pkg/            # 模块内工具（常量、错误码、模板引擎）
```

### 4.3 Evaluation 模块（评测）

**职责**：对 Prompt 和 Agent 的输出进行系统化的多维度自动评测

**子功能**：
- **评测集管理 (Evaluation Set)**：管理标准评测数据集
- **评估器管理 (Evaluator)**：定义评估标准（准确性、简洁性、合规性等）
- **实验管理 (Experiment)**：执行评测任务、聚合结果
- **评测目标 (Eval Target)**：定义被评测的对象
- **OpenAPI**：对外评测接口

**内部分层**：
```
evaluation/
├── application/    # 应用层（evaluator_app, experiment_app, evaluation_set_app）
├── domain/
│   ├── entity/     # 实体（评测集、评估器、实验）
│   ├── service/    # 领域服务
│   ├── events/     # 领域事件
│   ├── repo/       # 仓储接口
│   └── component/  # 组件（LLM评估执行器等）
├── infra/
│   ├── repo/       # 仓储实现
│   ├── rpc/        # RPC 调用
│   ├── runtime/    # 运行时（Python FaaS 评估器执行）
│   ├── mq/         # 消息队列（异步评测任务）
│   └── storage/    # 存储
└── pkg/            # 错误码、工具
```

### 4.4 Observability 模块（观测/Trace）

**职责**：记录和展示 AI Agent 全链路执行过程

**子功能**：
- **Trace 数据接收 (Ingestion)**：接收 SDK 上报的 Span/Trace 数据
- **Trace 查询与展示**：按时间、状态、标签等维度查询和可视化
- **Metric 统计**：调用量、延迟、Token 消耗等聚合指标
- **Task 管理**：异步任务处理
- **Trajectory**：执行轨迹可视化

**数据存储**：Trace 数据存储在 ClickHouse（列式存储，适合时序/日志类数据）

**内部分层**：
```
observability/
├── application/    # ingestion.go, trace.go, metric.go, task.go, openapi.go
├── domain/
│   ├── trace/      # Trace 领域模型
│   ├── metric/     # 指标领域模型
│   ├── task/       # 任务领域模型
│   └── component/  # 组件
├── infra/
│   ├── collector/  # 数据采集器
│   ├── repo/       # 仓储（ClickHouse 实现）
│   ├── mq/         # 消息队列
│   ├── workflow/   # 工作流
│   └── span_context_extractor/ # Span 上下文提取
└── lib/otel/       # OpenTelemetry 兼容库
```

### 4.5 LLM 模块（模型管理）

**职责**：LLM 模型的配置、管理与统一调用

**子功能**：
- 模型注册与管理
- 统一的模型调用运行时（通过 Eino 框架）
- 支持 OpenAI、火山方舟等多种模型后端

### 4.6 Data 模块（数据管理）

**职责**：评测数据集和标签（Tag）的管理

**子功能**：
- Dataset 管理（评测集数据的增删改查）
- Tag 管理（数据标签分类）

---

## 五、基础设施层 (Infra)

| 组件 | 实现 | 说明 |
|------|------|------|
| db | MySQL (GORM) | 业务数据持久化 |
| ck | ClickHouse | Trace/日志类时序数据 |
| redis | go-redis | 缓存、分布式锁、限流、ID生成 |
| mq | RocketMQ | 异步任务（评测任务执行、Trace数据处理） |
| fileserver | S3 (MinIO) | 文件/对象存储 |
| idgen | Redis Snowflake | 分布式 ID 生成 |
| limiter | Redis 令牌桶 | API 限流 |
| i18n | go-i18n | 国际化 |
| metrics | Prometheus | 可观测性指标采集 |
| looptracer | CozeLoop SDK | 自身 Trace 上报（dog-fooding） |
| middleware | Hertz 中间件 | 认证、日志、错误处理 |

---

## 六、前端架构

前端采用 Rush monorepo 管理，基于 React + TypeScript 构建。

```
frontend/
├── apps/cozeloop/           # 主应用入口
├── packages/
│   ├── loop-base/           # 基础层
│   │   ├── account/         # 账号管理
│   │   ├── api-schema/      # API 类型定义
│   │   ├── env/             # 环境配置
│   │   ├── i18n/            # 国际化
│   │   ├── route/           # 路由定义
│   │   └── stores/          # 全局状态
│   ├── loop-components/     # 业务组件层
│   │   ├── prompt-components/       # Prompt 相关组件
│   │   ├── evaluate-components/     # 评测相关组件
│   │   ├── observation-components/  # 观测相关组件
│   │   └── shared-components/       # 共享组件
│   ├── loop-pages/          # 页面层
│   │   ├── prompt-pages/    # Prompt 页面
│   │   ├── evaluate-pages/  # 评测页面
│   │   ├── observation-pages/ # 观测页面
│   │   └── auth-pages/      # 认证页面
│   └── loop-modules/        # 业务逻辑模块
│       └── evaluate/        # 评测业务逻辑
```

---

## 七、设计思路总结

### 7.1 架构原则

1. **DDD 分层架构**：每个业务模块严格按 application → domain → infra 分层，领域层不依赖基础设施层，通过 repo 接口解耦。

2. **IDL 优先**：使用 Thrift IDL 定义所有接口（前后端共用），通过代码生成保证类型安全。

3. **单体+模块化**：虽然编译为单个二进制，但模块间边界清晰，每个模块有独立的 domain、infra 和 application 层，便于未来拆分微服务。

4. **依赖注入 (Wire)**：使用 Google Wire 进行依赖注入，各模块的 `wire.go` 定义了组装关系。

5. **消息驱动**：评测任务等耗时操作通过 RocketMQ 异步执行，解耦请求与执行。

### 7.2 数据架构

- **MySQL**：存储结构化业务数据（Prompt、评测集、实验配置、用户信息等）
- **ClickHouse**：存储 Trace/Span 海量时序数据（高写入吞吐 + 快速聚合查询）
- **Redis**：缓存、分布式锁、限流器、ID生成器
- **MinIO (S3)**：文件存储（评测数据文件、Prompt 附件等）

### 7.3 部署架构

支持两种部署方式：
- **Docker Compose**：适合本地开发和小规模部署
- **Kubernetes (Helm Chart)**：适合生产环境

容器组成：
- `coze-loop`（后端 Go 服务）
- `coze-loop-nginx`（前端 + 反向代理）
- `coze-loop-python-faas`（Python 评估器运行时）
- MySQL / ClickHouse / Redis / MinIO / RocketMQ（中间件）

---

## 八、核心业务流程

### 8.1 Prompt 开发调试流程

```
用户编写Prompt → Playground调试 → 选择LLM模型 → 实时查看输出 → 
→ 对比不同模型效果 → 保存版本 → 发布
```

### 8.2 评测流程

```
创建评测集 → 定义评估器(LLM打分/规则/Python自定义) → 
→ 创建实验 → 选择目标(Prompt) → 执行评测(MQ异步) → 
→ 查看实验结果(多维度聚合)
```

### 8.3 观测(Trace)流程

```
应用代码集成SDK → SDK上报Span数据 → 
→ 后端Ingestion接收 → 存入ClickHouse → 
→ 前端查询展示Trace链路 → 指标聚合分析
```

---

## 九、关键技术亮点

1. **Python FaaS 评估器**：评测模块支持用户编写 Python 代码作为自定义评估器，通过独立的 Python FaaS 容器执行，隔离运行环境。

2. **OpenTelemetry 兼容**：观测模块兼容 OTel 协议，开发者可使用标准 SDK 上报数据。

3. **多语言 SDK**：提供 Python / Go / TypeScript 三种语言的客户端 SDK。

4. **Eino 框架集成**：通过 CloudWeGo 的 Eino 框架统一管理 LLM 调用，支持多种模型后端的无缝切换。

5. **自观测 (Dog-fooding)**：Coze Loop 使用自身的 Trace 能力观测自身的运行状态。
