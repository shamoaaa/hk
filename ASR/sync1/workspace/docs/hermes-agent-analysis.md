# Hermes Agent 项目分析

## 一句话总结

Hermes Agent 是 Nous Research 开发的一个**自我改进型 AI Agent 框架**，核心特点是内置学习闭环（自动创建/改进 Skill）、40+ 工具、28+ LLM 后端、多平台网关（Telegram/Discord/Slack/WhatsApp 等），以及可插拔的记忆/上下文/模型提供者系统。代码量约 12K+ LOC（核心 `run_agent.py`），支持子 Agent 委派、定时任务、Kanban 多 Agent 协作。

---

## 核心架构

```
用户输入 (CLI / Telegram / Discord / Slack / ...)
  ↓
┌─────────────────────────────────────────────────────────┐
│  cli.py (HermesCLI) / gateway/run.py (GatewaySession)   │
│  - 前端交互层，slash 命令分发                            │
│  - Rich TUI / prompt_toolkit / React Ink TUI            │
└──────────────────────────┬──────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  run_agent.py (AIAgent ~12K LOC)                         │
│  - 核心对话循环 (run_conversation)                       │
│  - 系统提示构建 + 上下文管理                             │
│  - 工具调用分发 + 迭代预算控制                           │
│  - 流式输出 + 中断处理                                   │
└──────────────────────────┬──────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  model_tools.py (工具编排层)                              │
│  - 自动发现 tools/*.py 中注册的工具                      │
│  - handle_function_call() 分发执行                       │
│  - 异步桥接（持久化事件循环）                            │
└──────────────────────────┬──────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  tools/ (40+ 工具实现)                                    │
│  - 每个文件通过 registry.register() 自注册               │
│  - 按 toolset 分组，可按平台启用/禁用                    │
└──────────────────────────┬──────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  providers/ + plugins/model-providers/ (28+ LLM 后端)    │
│  - ProviderProfile 声明式配置                            │
│  - OpenAI / Claude / Gemini / 国产模型 / 本地模型        │
└─────────────────────────────────────────────────────────┘
```

---

## Agent Loop 核心流程

```python
# run_agent.py 中 run_conversation() 的简化逻辑
while api_call_count < max_iterations and budget.remaining > 0:
    if interrupt_requested: break
    
    # 1. 调用 LLM（带工具 schema）
    response = client.chat.completions.create(
        model=model, messages=messages, tools=tool_schemas
    )
    
    # 2. 如果有工具调用 → 执行并继续循环
    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = handle_function_call(tool_call.name, tool_call.args)
            messages.append({"role": "tool", "content": result})
        api_call_count += 1
    
    # 3. 没有工具调用 → 返回最终响应
    else:
        return response.content
```

关键设计：
- **迭代预算**：父 Agent + 所有子 Agent 共享预算（默认 90 轮），防止失控
- **Grace call**：预算耗尽后允许一次额外调用，用于优雅收尾
- **中断处理**：CLI 用 Ctrl+C，消息平台用 `/stop`
- **Fallback 模型**：主模型失败时自动切换到备用模型重试

---

## 工具系统（40+ 工具）

### 注册机制

```python
# tools/web_tools.py 示例
from tools.registry import register

@register(
    name="web_search",
    toolset="web",
    description="Search the web for information",
    parameters={...}
)
async def web_search(query: str, **kwargs) -> str:
    ...
```

所有 `tools/*.py` 在 import 时自动注册，`model_tools.py` 触发发现。

### 核心工具列表

| 类别 | 工具 | 说明 |
|------|------|------|
| Web | web_search, web_extract | 搜索和网页内容提取 |
| 终端 | terminal, process | 7 种后端（local/Docker/SSH/Modal/Daytona/Singularity/Vercel） |
| 文件 | read_file, write_file, patch, search_files | 文件读写和搜索 |
| 视觉 | vision_analyze, image_generate | 图像分析和生成 |
| 浏览器 | browser_navigate/click/type/scroll/vision/cdp... | 完整浏览器自动化（Playwright） |
| Skills | skills_list, skill_view, skill_manage | 程序化记忆管理 |
| 记忆 | memory, session_search | 持久记忆 + FTS5 历史搜索 |
| 规划 | todo, cronjob | 任务清单 + 定时调度 |
| 委派 | delegate_task, execute_code | 子 Agent 并行 + 代码执行 |
| 消息 | send_message | 跨平台消息发送 |
| 智能家居 | ha_list_entities, ha_call_service... | Home Assistant 集成 |
| Kanban | kanban_show/complete/block/create... | 多 Agent 协作看板 |
| 交互 | clarify | 向用户提问澄清 |

### Toolset 分组

工具按 toolset 分组（web、terminal、browser、skills、delegation 等），可按平台/场景启用或禁用。`_HERMES_CORE_TOOLS` 定义了所有平台共享的默认工具集。

---

## 记忆与学习系统

### 分层设计

| 层级 | 机制 | 说明 |
|------|------|------|
| **Skills** | `~/.hermes/skills/*.md` | 程序化记忆，Agent 自动创建/改进，支持 Hub 安装 |
| **Memory** | 插件化（Honcho/Mem0/Supermemory 等） | 持久化用户偏好和事实 |
| **Session Search** | SQLite FTS5 | 全文搜索历史对话，LLM 摘要辅助召回 |
| **User Model** | Honcho dialectic modeling | 跨会话构建用户画像 |

### Skill 生命周期

```
[复杂任务完成] → [Agent 自动创建 Skill] → [后续使用中自动改进]
                                              ↓
                                    [Curator 后台审查]
                                    active → stale → archived
```

- **自动创建**：完成复杂任务后，Agent 主动将流程固化为 Skill
- **自动改进**：使用 Skill 时发现问题会自动更新
- **Curator**：后台定期审查，自动归档过期 Skill（不删除）
- **Provenance 追踪**：区分用户创建 vs Agent 创建 vs Hub 安装

### Memory Nudge

系统定期提示 Agent 审查/更新记忆（可配置间隔），确保重要信息不丢失。

---

## LLM 后端支持（28+ Provider）

### Provider 架构

```python
@dataclass
class ProviderProfile:
    name: str                    # 标识符
    api_mode: str                # chat_completions | anthropic_messages | codex_responses | bedrock_converse
    base_url: str                # API 端点
    auth_type: str               # api_key | oauth_device_code | aws_sdk | copilot
    fallback_models: tuple       # 模型列表（API 不可用时的备选）
    fixed_temperature: Any       # 温度覆盖
    
    def prepare_messages(self, messages): ...    # 消息预处理钩子
    def build_extra_body(self, **ctx): ...       # 额外请求参数
```

### 支持的提供者

- **主流云**: OpenAI, Anthropic (Claude), Google Gemini, AWS Bedrock, Azure
- **聚合器**: OpenRouter (200+ 模型), Nous Portal, Hugging Face
- **国产模型**: Kimi/Moonshot, MiniMax, Qwen (阿里), Deepseek, Stepfun, 小米 MiMo, z.ai/GLM
- **本地**: Ollama, 自定义端点
- **编码专用**: GitHub Copilot, Kimi Coding, Alibaba Coding Plan
- **特殊**: xAI (Grok), NVIDIA NIM (Nemotron)

切换模型只需 `hermes model`，无需改代码。

---

## 上下文管理

### Context Engine（可插拔压缩引擎）

```python
class ContextEngine(ABC):
    def should_compress(self, prompt_tokens) -> bool: ...
    def compress(self, messages, current_tokens, focus_topic) -> list: ...
```

- **阈值触发**：prompt tokens 超过上下文限制的 75% 时自动压缩
- **会话分裂**：压缩后创建子会话（`parent_session_id` 链）
- **引导压缩**：`/compress <topic>` 指定保留重点
- **Preflight 检查**：进入主循环前检查历史是否已超限

### 系统提示（缓存优化）

- 首轮构建，后续复用（利用 LLM 的 prefix caching）
- 包含：SOUL.md（用户身份）、项目上下文、记忆上下文
- 压缩后重建（缓存失效）

---

## 会话持久化（hermes_state.py）

- **SQLite + WAL 模式**：支持并发读 + 单写（网关多平台场景）
- **FTS5 全文搜索**：跨所有会话的快速文本检索
- **Trigram 分词**：支持 CJK 子串搜索
- **Token 计费追踪**：input/output/cache_read/cache_write/reasoning tokens
- **会话链**：通过 `parent_session_id` 实现压缩后的会话延续

---

## 多平台网关

```
hermes gateway start
  ↓
┌─────────────────────────────────────────┐
│  gateway/run.py (GatewaySession)         │
│  - 统一消息处理                          │
│  - 每条消息创建新 AIAgent 实例           │
│  - 会话 DB 保持连续性                    │
└──────────────────┬──────────────────────┘
                   ↓
┌──────────┬──────────┬──────────┬─────────┐
│ Telegram │ Discord  │  Slack   │WhatsApp │ ...
└──────────┴──────────┴──────────┴─────────┘
```

支持平台：Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Mattermost, Email, SMS, DingTalk, WeCom, 微信, 飞书, QQ Bot, Home Assistant, Webhook, API Server 等。

---

## 子 Agent 委派

```python
# delegate_task 工具
delegate_task(
    task="分析这个 CSV 文件并生成报告",
    role="leaf",           # leaf=专注执行, orchestrator=可再委派
    skills=["data_analysis"],
    model="claude-sonnet"  # 可指定不同模型
)
```

- **隔离执行**：子 Agent 有独立上下文和终端
- **批量并行**：`tasks: [...]` 同时启动多个子 Agent
- **预算共享**：父子共享迭代预算，防止失控
- **深度限制**：orchestrator 可再委派，但有深度上限

---

## 定时任务（Cron）

```bash
hermes cron add "每天早上9点发送天气预报到 Telegram"
```

- 5 字段 cron 表达式 + "every" 自然语言 + ISO 时间戳
- 每个 job 可指定：skills、model/provider 覆盖、pre-run 脚本、工作目录
- 3 分钟硬中断（防止 cron 会话失控）
- 多平台投递（执行结果发送到指定平台）

---

## 与 GenericAgent 的对比

| 维度 | Hermes Agent | GenericAgent |
|------|-------------|--------------|
| 代码量 | ~12K LOC（核心） + 大量插件 | ~3K LOC |
| 工具数 | 40+ 内置 + 插件扩展 | 9 个原子工具 |
| LLM 支持 | 28+ Provider，声明式切换 | 多模型，手动配置 |
| 记忆 | 插件化（Honcho/Mem0 等）+ FTS5 搜索 | 分层文件系统（L0-L4） |
| 进化机制 | Skill 自动创建/改进 + Curator 审查 | 经验写入记忆文件 |
| 部署 | 多终端后端（Docker/SSH/Modal/Daytona） | 本地运行 |
| 多平台 | 15+ 消息平台网关 | Telegram/QQ/微信等 Bot |
| 协作 | Kanban 多 Agent + delegate_task | 单 Agent |
| 定时任务 | 内置 cron 调度器 | reflect 模式 |
| 上下文管理 | 可插拔 ContextEngine + 自动压缩 | 手动 compress_history_tags |
| 设计哲学 | 功能完备的产品级框架 | 极简种子代码，靠进化生长 |

---

## 技术栈

- Python 3.10+
- OpenAI SDK（统一 API 调用）
- SQLite + FTS5（会话持久化）
- Playwright（浏览器自动化）
- Rich + prompt_toolkit（CLI）
- React/Ink（TUI）
- asyncio + httpx（异步工具执行）
- pytest（~17K 测试）
