# toy-agent

一个用于学习 Agent 核心原理的最小化 Agent 实现。Python + OpenAI SDK。

[English](README.md)

## 项目结构

```
toy-agent/
├── main.py                     # 入口
├── toy_agent/
│   ├── agent.py                # Agent loop 核心
│   ├── config.py               # 多级配置加载
│   ├── hooks.py               # AgentHook 观测系统
│   ├── mcp.py                  # MCP 客户端 (stdio + SSE)
│   ├── memory.py               # 会话持久化
│   ├── planning.py             # PlanHook + ReActPlanHook (任务规划)
│   ├── guardrails.py           # GuardrailHook (人工审批)
│   ├── skills.py              # Skills 加载器
│   ├── subagent.py            # SubAgentTool（Tool-call 模式）
│   └── tools/
│       ├── __init__.py          # @tool 装饰器 + 自动注册
│       ├── file_ops.py          # read_file, write_file, edit_file
│       └── run_bash.py          # run_bash (带安全检查)
├── tests/
│   ├── mcp_stdio_server.py      # stdio MCP 测试服务器
│   ├── mcp_sse_server.py        # SSE MCP 测试服务器
│   ├── test_agent.py             # Agent 单元测试
│   ├── test_context.py           # ContextCompressor 测试
│   ├── test_hooks.py             # Hook 系统测试
│   ├── test_planning.py          # Planning 测试
│   ├── test_guardrails.py        # Guardrails 测试
│   ├── test_memory.py            # SessionMemory 测试
│   ├── test_skills.py           # Skills 加载器测试
│   └── test_subagent.py         # SubAgentTool 测试
├── .env.example
├── Makefile
└── pyproject.toml
```

## Phase 1: Agent Loop

所有 Agent 框架的核心循环：

1. 将对话历史发送给 LLM
2. LLM 返回响应，可能包含 `tool_calls`
3. 有 `tool_calls` → 执行工具，将结果追加到历史，回到步骤 1
4. 无 `tool_calls` → LLM 给出最终回答，退出循环

API 错误会被捕获并展示给用户，不会导致程序崩溃。

## Phase 2: Tools

`@tool` 装饰器根据 Python 类型标注和 docstring 自动生成 OpenAI function calling schema：

```python
from toy_agent.tools import tool

@tool(description="Read the content of a file")
def read_file(path: str) -> str:
    """path: Absolute or relative file path to read"""
    ...
```

**内置工具：**
- `read_file` / `write_file` / `edit_file` — 文件操作
- `run_bash` — 执行 Shell 命令（高危命令被禁止）

`toy_agent/tools/*.py` 中的工具会被自动导入和注册，无需手动添加。

## Phase 3: MCP 集成

通过 MCP（Model Context Protocol）连接外部服务器，扩展 Agent 工具能力。

**多级配置**（项目级覆盖用户级）：
- 用户级：`~/.toy-agent/mcp.json`
- 项目级：`.toy-agent/mcp.json`

**支持的传输方式：**

| 类型  | 配置项     | 示例 |
|-------|-----------|------|
| stdio | `command` | `{"command": "npx", "args": ["-y", "@mcp/server"]}` |
| SSE   | `url`     | `{"url": "http://localhost:9000/sse"}` |

配置示例：

```json
{
  "mcpServers": {
    "local-tool": {
      "command": "uv",
      "args": ["run", "python", "tests/mcp_stdio_server.py"]
    },
    "remote-tool": {
      "url": "http://localhost:9000/sse"
    }
  }
}
```

## Phase 4: Skills

通过 Skill 扩展 Agent 的角色和专业知识。Skill 是包含 YAML frontmatter 的 markdown 文件，放在 skills 子目录中。

**目录结构：**
```
.toy-agent/skills/
└── code-review/
    └── SKILL.md
```

**SKILL.md 格式：**
```markdown
---
description: Expert at reviewing code changes and suggesting improvements
---

You are an expert code reviewer. When asked to review code...
```

- 目录名即为 skill 的 name
- frontmatter 中的 description 告诉 LLM 何时调用此 skill
- 多级目录加载（项目级覆盖用户级）

## Phase 5: Subagent（子代理）

通过 **Tool-call 模式** 将任务委派给专职子代理。每个子代理注册为一个 `Tool`，运行独立的 Agent Loop。

```python
from toy_agent.subagent import SubAgentTool

researcher = SubAgentTool(
    name="researcher",
    description="Research a topic and return findings",
    agent=Agent(client, model, system="You are a researcher..."),
)

agent = Agent(client, tools=[..., researcher])
```

- 每个子代理拥有完全独立的上下文（独立的 messages、tools、skills）
- `max_turns` 安全限制防止无限循环（默认：10 轮）
- 错误隔离 — 子代理的异常不会影响主 Agent

## Phase 6: Streaming（流式输出）

通过可配置的流式模式实现实时 token 输出。

```python
# 构造函数启用
agent = Agent(client, stream=True)

# 或单次调用覆盖
result = await agent.run("hello", stream=True)
```

- 在 `.env` 中设置 `TOY_AGENT_STREAM=true` 启用（默认：true）
- 工具调用时暂停流式，执行完后恢复
- Subagent 内部始终使用非流式

## Phase 7: Memory / 会话恢复

跨重启的持久对话历史。每次会话保存为 JSONL 文件（每行一条消息，追加写入），下次启动时可恢复。

```
~/.toy-agent/
├── mcp.json
└── <project_hash>/sessions/
    ├── 2026-04-02_143052.jsonl
    └── 2026-04-03_091530.jsonl
```

- 每轮对话后追加保存（append-only），自动清理保留最近 10 个会话
- 按项目路径 hash 隔离，多项目互不干扰
- REPL 命令：`/resume`、`/resume <id>`、`/sessions`

```python
from toy_agent.memory import SessionMemory

memory = SessionMemory(project_path="/my/project")
memory.save(messages)  # 追加上次 save 之后的新消息
restored = memory.load_latest()
```

## Phase 8: 上下文压缩

两种压缩策略，防止长对话中的 token 溢出。

### ContextCompressor（渐进式）

三级渐进式压缩，目前仅实现 Level 1。

- **Level 1**：回合摘要 — 将工具调用链压缩为简要摘要
- **Level 2**：阶段概览 — 合并早期摘要（TODO）
- **Level 3**：滑动窗口 + 全局概览（TODO）

```python
from toy_agent.context import ContextCompressor

compressor = ContextCompressor(client=client, model="gpt-4o-mini", token_limit=80000)
agent = Agent(client=client, compressor=compressor)
```

### HermesContextCompressor（4 阶段）

灵感来自 [Hermes Agent](https://github.com/NousResearch/Hermes-Agent)。使用结构化 handoff 摘要，支持增量更新，在多次压缩间保留信息。

**4 阶段压缩（每次压缩仅需 1 次 LLM 调用）：**

| 阶段 | 操作 | 成本 |
|------|------|------|
| 1. 工具输出裁剪 | 将老的长 tool result 替换为占位符 | 零 |
| 2. 边界确定 | 按 token 预算保护头部/尾部，对齐边界避免截断 tool 配对 | 零 |
| 3. 结构化摘要 | 8 段式 handoff 摘要（Goal、Progress、Decisions 等），支持增量更新 | 1 次 LLM 调用 |
| 4. 组装 + 修复 | 角色交替检查、修复孤立的 tool_call/result 配对 | 零 |

```python
from toy_agent.context import HermesContextCompressor

compressor = HermesContextCompressor(client=client, model="gpt-4o-mini", token_limit=80000)
agent = Agent(client=client, compressor=compressor)
```

核心特性：
- **增量摘要更新**：后续压缩时更新已有摘要，而非从头重新生成
- **Token 预算动态保护**：尾部保护自动随模型上下文窗口大小缩放
- **Tool pair 完整性**：压缩后自动修复孤立的 tool_call/result 配对

在 `.env` 中设置 `TOY_AGENT_CONTEXT_TOKEN_LIMIT` 可覆盖 token 阈值（默认：80000）。

## Phase 9: 观测钩子（Hooks）

可插拔的事件回调，用于观测和监控 Agent Loop。所有事件默认是空实现 — 只需继承 `AgentHook` 即可在任意时机插入逻辑。

```python
from toy_agent.hooks import AgentHook, ConsoleHook

class MyHook(AgentHook):
    def on_tool_call(self, *, tool_name: str, arguments: dict):
        print(f"DEBUG: calling {tool_name} with {arguments}")

    def on_error(self, *, error: str):
        sentry.capture_exception(error)

agent = Agent(client=client, hooks=[MyHook()])
```

**内置的 `ConsoleHook`** 复现了之前的控制台输出行为，CLI 默认使用它。

**可用事件：**

| 事件 | 触发时机 | 关键参数 |
|------|----------|----------|
| `on_message` | 消息追加到历史 | `role`, `content` |
| `on_llm_request` | 发送 LLM 请求前 | `messages` |
| `on_llm_response` | 收到 LLM 回复后 | `message` |
| `on_tool_call` | 工具执行前 | `tool_name`, `arguments` |
| `on_tool_result` | 工具返回后 | `tool_name`, `result` |
| `on_tool_retry` | 重试前 | `tool_name`, `attempt`, `error` |
| `on_compress` | 上下文压缩后 | `before_count`, `after_count` |
| `on_error` | 发生错误时 | `error` |

### 工具重试

临时失败的工具（文件锁、网络超时）会自动重试，采用指数退避策略。

```python
agent = Agent(client=client, max_tool_retries=2)  # 最多重试 2 次
```

- `on_tool_retry` 钩子在每次重试前触发
- 默认：`max_tool_retries=0`（不重试，向后兼容）

## Phase 10: Planning（任务规划）

提供两种规划策略，展示 Agent 框架中不同的规划方法。两种都实现为可插拔的 Hook — 切换 hooks 列表即可切换策略。

### PlanHook（Plan-then-Execute）

使用独立的 LLM API 调用来分类任务并生成步骤计划。这是早期 Agent 框架（如 LangChain 的 Plan-and-Solve）中经典的"先规划后执行"模式。

```python
from toy_agent.planning import PlanHook

# 自动模式：LLM 决定何时规划
agent = Agent(
    client=client,
    hooks=[ConsoleHook(), PlanHook(client=client, model="gpt-4o-mini", auto=True)],
)

# 单次调用强制开启/关闭规划
result = await agent.run("Analyze the codebase", plan=True)   # 强制规划
result = await agent.run("Quick question", plan=False)         # 跳过规划
```

- 每次运行额外 1-2 次 LLM API 调用（分类 + 生成）
- 使用 JSON 结构化输出（`response_format`）
- 生成计划时触发 `on_plan` 事件

### ReActPlanHook（模型自规划）

注入一段 plan-aware 的提示词，让主模型在自己的上下文中完成判断、规划、执行和追踪。无额外 API 调用。这反映了生产级 Agent（Claude Code、Cursor、Devin）的实际做法 — 规划不是一个独立阶段，而是模型自然推理的一部分。

```python
from toy_agent.planning import ReActPlanHook

agent = Agent(
    client=client,
    hooks=[ConsoleHook(), ReActPlanHook()],
)
```

- 零额外 API 调用 — 规划在主 Agent Loop 内完成
- 模型通过 `[Step X complete]` 标记自报步骤完成
- `on_plan_step` / `on_plan_done` 事件在步骤完成时触发
- `ConsoleHook` 打印步骤进度和计划摘要

### 为什么同时保留两种

在生产级 Agent 中，规划通常只是 system prompt 的一部分 — 不需要专门的 Hook。这两种 Hook 作为**学习示例**保留，展示从显式规划（PlanHook）到模型自规划（ReActPlanHook）的演进，帮助理解生产级 Agent 为什么最终选择了后者。

## Phase 11: Guardrails（人工审批）

Agent 在执行高危工具前可以要求人工确认。通过 guardrail hook 在工具执行前拦截实现。

```python
from toy_agent.guardrails import GuardrailHook

# 默认：run_bash、write_file、edit_file 需要审批
agent = Agent(
    client=client,
    hooks=[ConsoleHook(), GuardrailHook()],
)

# 自定义审批列表
hook = GuardrailHook(approval_tools={"run_bash", "write_file"})

# 自动放行特定工具（不弹提示）
hook = GuardrailHook(auto_approve={"run_bash"})
```

当工具需要审批时，用户会看到提示：
```
[guardrail] Allow run_bash({"command": "rm -rf /tmp/old"})? [y/N]
```

- 输入 `n` 或直接回车 → 阻止执行，阻止信息作为工具结果返回
- 输入 `y` → 正常执行
- Hook 系统使用 `on_tool_approve`（返回 `None` 放行，返回 `str` 阻止）和 `on_guardrail_block`（观察事件）
- 同时支持流式和非流式模式
- `load_skill` 和 `read_file` 默认免审

## 快速开始

```bash
# 安装依赖
uv sync

# 配置 API
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY

# 配置 MCP 服务器（可选）
mkdir -p .toy-agent
# 创建 .toy-agent/mcp.json

# 运行
make run
```

## Make 命令

| 命令 | 说明 |
|------|------|
| `make run` | 启动 Agent |
| `make mcp` | 启动 SSE 测试 MCP 服务器 |
| `make test` | 运行单元测试 |
| `make lint` | 运行 ruff 代码检查 |
| `make fmt` | 格式化代码 |
| `make check` | 运行 lint + 格式检查 |
