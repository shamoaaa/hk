# GenericAgent 项目分析

## 一句话总结

GenericAgent 是一个极简（~3K 行核心代码）、可自我进化的自主 Agent 框架，通过 9 个原子工具 + ~100 行 Agent Loop + 分层记忆系统，赋予任意 LLM 对本地计算机的系统级控制能力，并在每次任务执行后自动将经验固化为可复用的 Skill。

---

## 核心架构

```
用户输入
  ↓
┌─────────────────────────────────────────────────┐
│  agentmain.py (GenericAgent 主类)                │
│  - 管理 LLM 会话、任务队列、前端交互             │
│  - 加载 system prompt + 全局记忆                 │
└──────────────────────┬──────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  agent_loop.py (~100行 Agent Loop)               │
│  - 循环: LLM推理 → 解析工具调用 → 执行 → 反馈   │
│  - 最多 max_turns 轮（默认40，plan模式100）      │
└──────────────────────┬──────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  ga.py (GenericAgentHandler - 工具实现)           │
│  - 9个原子工具的具体实现                         │
│  - Working Memory 管理                           │
│  - 历史摘要和锚点提示                            │
└──────────────────────┬──────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  llmcore.py (LLM 通信层)                         │
│  - 支持 OpenAI / Claude / 国产模型 API           │
│  - SSE 流式解析、重试、上下文压缩                │
│  - 多模型混合 (MixinSession)                     │
└─────────────────────────────────────────────────┘
```

---

## 核心文件职责

| 文件 | 行数 | 职责 |
|------|------|------|
| `agent_loop.py` | ~100 | Agent 执行循环：LLM 调用 → 工具分发 → 结果反馈 → 下一轮 |
| `ga.py` | ~580 | 9 个原子工具的实现 + Handler 逻辑（Working Memory、历史管理） |
| `llmcore.py` | ~1020 | LLM API 通信层（OpenAI/Claude/国产模型），流式解析，上下文压缩 |
| `agentmain.py` | ~200 | 主入口，管理 GenericAgent 实例、任务队列、多模型切换 |

---

## 9 个原子工具

| 工具 | 实现方法 | 功能 |
|------|---------|------|
| `code_run` | `do_code_run` | 执行 Python/PowerShell/Bash 代码（文件模式，带超时和中断） |
| `file_read` | `do_file_read` | 读取文件（支持行号范围、关键词搜索、模糊文件名匹配） |
| `file_write` | `do_file_write` | 写入/追加/前置文件内容（从 `<file_content>` 标签提取） |
| `file_patch` | `do_file_patch` | 精确替换文件中的唯一文本块（类似 sed） |
| `web_scan` | `do_web_scan` | 获取浏览器当前页面简化 HTML + 标签页列表 |
| `web_execute_js` | `do_web_execute_js` | 在浏览器中执行 JS 脚本，完全控制浏览器行为 |
| `ask_user` | `do_ask_user` | 向用户提问，等待人工输入（Human-in-the-loop） |
| `update_working_checkpoint` | `do_update_working_checkpoint` | 更新当前任务的临时工作记忆（key_info） |
| `start_long_term_update` | `do_start_long_term_update` | 触发长期记忆结算（将经验写入 L2/L3） |

---

## 分层记忆系统

这是 GenericAgent 最核心的设计，实现了跨会话的知识积累：

| 层级 | 存储位置 | 内容 | 特点 |
|------|---------|------|------|
| **L0** | `assets/sys_prompt.txt` | Agent 基础行为规则和系统约束 | 固定不变 |
| **L1** | `memory/global_mem_insight.txt` | 极简索引（≤30行），场景关键词→记忆定位 | 每次对话加载到 system prompt |
| **L2** | `memory/global_mem.txt` | 全局环境事实（路径、凭证、配置） | 每次对话加载到 system prompt |
| **L3** | `memory/*.md` + `memory/*.py` | 任务级 SOP 和工具脚本 | 按需读取 |
| **L4** | `memory/L4_raw_sessions/` | 历史会话归档（压缩后的对话记录） | 定时归档，长程召回 |

**记忆写入原则：**
- "No Execution, No Memory" — 只有经过工具调用验证成功的信息才能写入
- 禁止存储易变状态、通用常识、未验证的猜测
- L1 只做导航索引，不写细节

---

## Agent Loop 执行流程

```python
# agent_loop.py 核心逻辑（简化）
while turn < max_turns:
    # 1. 调用 LLM
    response = client.chat(messages, tools=tools_schema)
    
    # 2. 解析工具调用
    tool_calls = parse_tool_calls(response)
    
    # 3. 逐个执行工具
    for tc in tool_calls:
        outcome = handler.dispatch(tc.name, tc.args, response)
        if outcome.should_exit: break      # ask_user 等中断
        if outcome.next_prompt is None: break  # 任务完成（no_tool）
    
    # 4. 构建下一轮消息（包含 Working Memory 锚点）
    messages = [{"role": "user", "content": next_prompt, "tool_results": results}]
```

**关键设计：**
- 每轮结束后，`turn_end_callback` 提取 `<summary>` 标签作为历史摘要
- `_get_anchor_prompt` 构建 Working Memory 锚点（包含折叠的历史 + key_info）
- 上下文窗口控制在 ~30K tokens，通过 `compress_history_tags` 压缩旧消息
- 每 10 轮重置工具描述，避免上下文膨胀

---

## 自我进化机制

```
[新任务] → [自主探索] → [固化为 Skill] → [写入 L3 记忆] → [下次直接调用]
```

具体实现：
1. Agent 完成任务后，`do_start_long_term_update` 触发记忆结算
2. Agent 读取 `memory_management_sop.md`（L0 规则）
3. 按决策树判断信息归属层级（L1/L2/L3）
4. 用 `file_patch` 或 `file_write` 更新对应记忆文件
5. 下次遇到类似任务时，L1 索引命中 → 读取 L3 SOP → 直接执行

**Skill 形式：**
- `memory/*_sop.md` — 任务流程文档（关键前置条件 + 避坑点）
- `memory/*.py` — 可复用的工具脚本（如 `ocr_utils.py`、`keychain.py`）

---

## LLM 通信层 (llmcore.py)

支持多种模型后端：
- **OpenAI 兼容 API**（GPT-4、Kimi、MiniMax、DeepSeek 等）
- **Claude 原生 API**（Anthropic Messages API）
- **MixinSession**：多模型混合调用（如用 Claude 做推理 + GPT 做工具调用）

关键特性：
- SSE 流式解析（`_parse_openai_sse`、`_parse_claude_sse`）
- 自动重试（HTTP 429/500/502/503 等可重试状态码）
- 上下文压缩（`compress_history_tags` 截断旧消息中的 thinking/tool_result 标签）
- 上下文裁剪（`trim_messages_history` 当超过 context_win*3 时删除最旧消息）
- 支持 OpenAI Responses API 和 Chat Completions API 两种模式
- Claude 缓存标记（`cache_control: ephemeral`）

---

## 运行模式

| 模式 | 启动方式 | 说明 |
|------|---------|------|
| 交互式 | `python agentmain.py` | 命令行 REPL，逐条输入任务 |
| 一次性任务 | `python agentmain.py --task <dir>` | 读取 input.txt，输出到 output.txt |
| 反射模式 | `python agentmain.py --reflect <script>` | 加载监控脚本，`check()` 返回非 None 时触发任务 |
| 定时调度 | `reflect/scheduler.py` | cron 式定时任务（daily/weekly/every_Nh） |
| 桌面 GUI | `python launch.pyw` | Streamlit + PyWebView 桌面应用 |
| 终端 TUI | `python frontends/tuiapp.py` | Textual 终端界面 |
| Bot 前端 | `frontends/tgapp.py` 等 | Telegram/QQ/微信/飞书/企微/钉钉 |

---

## 浏览器控制

通过 `TMWebDriver`（CDP 协议桥接）注入真实浏览器：
- 保留用户登录态（非无头浏览器）
- `web_scan` 获取简化 HTML（`simphtml.py` 过滤非主体内容）
- `web_execute_js` 执行任意 JS 控制页面
- 支持标签页切换、页面变化监控

---

## 与其他 Agent 框架的本质区别

1. **无预设技能**：不像 LangChain/AutoGPT 预装大量工具，GenericAgent 只有 9 个原子工具，所有高级能力通过 `code_run` 动态获得
2. **自我进化**：每次任务的经验自动沉淀为 Skill，能力随使用增长
3. **极致省 Token**：上下文 <30K（vs 其他框架 200K-1M），通过分层记忆实现信息密度最大化
4. **真实环境控制**：注入真实浏览器（保留 session），而非沙箱/无头浏览器
5. **极简代码**：核心 ~3K 行，无复杂依赖链

---

## 技术栈

- Python 3.10+
- requests（LLM API 通信）
- streamlit + pywebview（桌面 GUI）
- textual（终端 TUI）
- CDP 协议（浏览器控制）
- SQLite/JSON（记忆持久化）
