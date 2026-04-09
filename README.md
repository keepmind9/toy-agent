# toy-agent

A minimal agent implementation for learning agent fundamentals. Built with Python + OpenAI SDK.

[中文文档](README_zh.md)

## Project Structure

```
toy-agent/
├── main.py                     # Entry point
├── toy_agent/
│   ├── agent.py                # Agent loop core
│   ├── config.py               # Multi-level config loader
│   ├── hooks.py               # AgentHook observability system
│   ├── mcp.py                  # MCP client (stdio + SSE)
│   ├── memory.py               # Session persistence
│   ├── planning.py             # PlanHook + ReActPlanHook (task planning)
│   ├── guardrails.py           # GuardrailHook (human-in-the-loop approval)
│   ├── skills.py              # Skills loader
│   ├── subagent.py            # SubAgentTool (tool-call pattern)
│   └── tools/
│       ├── __init__.py          # @tool decorator + auto-registry
│       ├── file_ops.py          # read_file, write_file, edit_file
│       └── run_bash.py          # run_bash (with safety checks)
├── tests/
│   ├── mcp_stdio_server.py      # stdio MCP server for testing
│   ├── mcp_sse_server.py        # SSE MCP server for testing
│   ├── test_agent.py            # Agent unit tests
│   ├── test_context.py          # ContextCompressor tests
│   ├── test_hooks.py            # Hook system tests
│   ├── test_memory.py           # SessionMemory tests
│   ├── test_planning.py         # Planning tests (both hooks)
│   ├── test_guardrails.py       # Guardrails tests
│   ├── test_skills.py           # Skills loader tests
│   └── test_subagent.py         # SubAgentTool tests
├── .env.example
├── Makefile
└── pyproject.toml
```

## Phase 1: Agent Loop

The core loop that every agent framework is built on:

1. Send conversation history to the LLM
2. LLM returns a response, possibly with `tool_calls`
3. If `tool_calls` exist → execute tools, append results, go back to step 1
4. If no `tool_calls` → LLM gives the final answer, exit loop

API errors are caught and displayed without crashing the session.

## Phase 2: Tools

A `@tool` decorator that auto-generates OpenAI function calling schema from Python type hints and docstrings:

```python
from toy_agent.tools import tool

@tool(description="Read the content of a file")
def read_file(path: str) -> str:
    """path: Absolute or relative file path to read"""
    ...
```

**Built-in tools:**
- `read_file` / `write_file` / `edit_file` — file operations
- `run_bash` — execute shell commands (dangerous commands blocked)

Tools in `toy_agent/tools/*.py` are auto-imported and registered. No manual registration needed.

## Phase 3: MCP Integration

Connect to MCP (Model Context Protocol) servers for extended tool capabilities.

**Multi-level config** (project overrides user):
- User-level: `~/.toy-agent/mcp.json`
- Project-level: `.toy-agent/mcp.json`

**Supported transports:**

| Type   | Config key | Example |
|--------|-----------|---------|
| stdio  | `command` | `{"command": "npx", "args": ["-y", "@mcp/server"]}` |
| SSE    | `url`     | `{"url": "http://localhost:9000/sse"}` |

Config example:

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

Extend the agent with specialized roles and expertise via Skills. Each skill is a markdown file with YAML frontmatter, organized in subdirectories.

**Directory structure:**
```
.toy-agent/skills/
└── code-review/
    └── SKILL.md
```

**SKILL.md format:**
```markdown
---
description: Expert at reviewing code changes and suggesting improvements
---

You are an expert code reviewer. When asked to review code...
```

- Directory name = skill name
- `description` in frontmatter tells the LLM when to invoke this skill
- Multi-level loading (project overrides user)

## Phase 5: Subagents

Delegate tasks to specialized subagents via the **Tool-call pattern**. Each subagent registers as a `Tool` and runs an independent agent loop.

```python
from toy_agent.subagent import SubAgentTool

researcher = SubAgentTool(
    name="researcher",
    description="Research a topic and return findings",
    agent=Agent(client, model, system="You are a researcher..."),
)

agent = Agent(client, tools=[..., researcher])
```

- Each subagent has fully independent context (own messages, tools, skills)
- `max_turns` safety limit prevents infinite loops (default: 10)
- Errors are isolated — subagent failures don't crash the main agent

## Phase 6: Streaming

Real-time token output via configurable streaming mode.

```python
# Enable via constructor
agent = Agent(client, stream=True)

# Or per-call override
result = await agent.run("hello", stream=True)
```

- Set `TOY_AGENT_STREAM=true` in `.env` to enable (default: true)
- Tool calls pause streaming, resume after execution
- Subagents always use non-streaming internally

## Phase 7: Memory / Session Restore

Persistent conversation history across restarts. Each session is a JSONL file (one message per line, append-only), and can be restored on next launch.

```
~/.toy-agent/
├── mcp.json
└── <project_hash>/sessions/
    ├── 2026-04-02_143052.jsonl
    └── 2026-04-03_091530.jsonl
```

- Auto-save after each turn (append-only), auto-cleanup keeps the latest 10 sessions
- Per-project isolation via path hash
- REPL commands: `/resume`, `/resume <id>`, `/sessions`

```python
from toy_agent.memory import SessionMemory

memory = SessionMemory(project_path="/my/project")
memory.save(messages)  # append new messages since last save
restored = memory.load_latest()
```

## Phase 8: Context Compression

Two compression strategies to prevent token overflow in long conversations.

### ContextCompressor (Progressive)

Three-level progressive compression. Only Level 1 is implemented.

- **Level 1**: Turn summary — compress tool call chains into brief summaries
- **Level 2**: Phase overview — merge early summaries into phase overviews (TODO)
- **Level 3**: Sliding window + global overview (TODO)

```python
from toy_agent.context import ContextCompressor

compressor = ContextCompressor(client=client, model="gpt-4o-mini", token_limit=80000)
agent = Agent(client=client, compressor=compressor)
```

### HermesContextCompressor (4-Phase)

Inspired by [Hermes Agent](https://github.com/NousResearch/Hermes-Agent). Uses a structured handoff summary with iterative updates that preserve information across multiple compressions.

**4-phase compression (1 LLM call per compression):**

| Phase | Action | Cost |
|-------|--------|------|
| 1. Tool output pruning | Replace old long tool results with placeholders | Zero |
| 2. Boundary determination | Protect head/tail by token budget, align boundaries to avoid splitting tool pairs | Zero |
| 3. Structured summarization | 8-section handoff summary (Goal, Progress, Decisions, etc.) with incremental updates | 1 LLM call |
| 4. Assembly + sanitization | Role alternation check, repair orphaned tool_call/result pairs | Zero |

```python
from toy_agent.context import HermesContextCompressor

compressor = HermesContextCompressor(client=client, model="gpt-4o-mini", token_limit=80000)
agent = Agent(client=client, compressor=compressor)
```

Key features:
- **Iterative summary updates**: subsequent compressions update the previous summary instead of re-summarizing from scratch
- **Token-budget tail protection**: scales automatically with model context window size
- **Tool pair integrity**: automatically repairs orphaned tool_call/result pairs after compression

Set `TOY_AGENT_CONTEXT_TOKEN_LIMIT` in `.env` to override the token limit (default: 80000).

## Phase 9: Observability Hooks

Pluggable event callbacks for observing and instrumenting the agent loop. All events are no-ops by default — implement a subclass to hook into any point in the execution.

```python
from toy_agent.hooks import AgentHook, ConsoleHook

class MyHook(AgentHook):
    def on_tool_call(self, *, tool_name: str, arguments: dict):
        print(f"DEBUG: calling {tool_name} with {arguments}")

    def on_error(self, *, error: str):
        sentry.capture_exception(error)

agent = Agent(client=client, hooks=[MyHook()])
```

**Built-in `ConsoleHook`** replicates the previous console output behaviour and is used by the CLI by default.

**Available events:**

| Event | When | Key Args |
|-------|------|----------|
| `on_message` | message appended to history | `role`, `content` |
| `on_llm_request` | before LLM API call | `messages` |
| `on_llm_response` | after LLM response | `message` |
| `on_tool_call` | before tool execution | `tool_name`, `arguments` |
| `on_tool_result` | after tool returns | `tool_name`, `result` |
| `on_tool_retry` | before retry attempt | `tool_name`, `attempt`, `error` |
| `on_compress` | after context compression | `before_count`, `after_count` |
| `on_error` | on any error | `error` |

### Tool Retry

Tools that fail transiently (file locks, network timeouts) are automatically retried with exponential backoff.

```python
agent = Agent(client=client, max_tool_retries=2)  # retry up to 2 times
```

- `on_tool_retry` hook fires before each retry attempt
- Default: `max_tool_retries=0` (no retry, backward compatible)

## Phase 10: Planning

Two planning strategies are provided to demonstrate different approaches used in agent frameworks. Both are implemented as pluggable hooks — swap one for the other by changing the hooks list.

### PlanHook (Plan-then-Execute)

Uses separate LLM API calls to classify tasks and generate step-by-step plans before execution. This is the classic "Plan-and-Solve" pattern found in early agent frameworks (e.g., LangChain's Plan-and-Solve executor).

```python
from toy_agent.planning import PlanHook

# Auto mode: LLM decides when to plan
agent = Agent(
    client=client,
    hooks=[ConsoleHook(), PlanHook(client=client, model="gpt-4o-mini", auto=True)],
)

# Force planning on/off per call
result = await agent.run("Analyze the codebase", plan=True)   # force plan
result = await agent.run("Quick question", plan=False)         # skip plan
```

- 1-2 extra LLM API calls per run (classify + generate)
- Plans use JSON structured output (`response_format`)
- `on_plan` hook event fires when a plan is generated

### ReActPlanHook (Model Self-Planning)

Injects a plan-aware prompt so the main model decides, plans, executes, and tracks progress all within its own context. No extra API calls. This mirrors how production-grade agents (Claude Code, Cursor, Devin) handle planning — planning is not a separate phase but part of the model's natural reasoning.

```python
from toy_agent.planning import ReActPlanHook

agent = Agent(
    client=client,
    hooks=[ConsoleHook(), ReActPlanHook()],
)
```

- Zero extra API calls — planning happens within the main agent loop
- The model self-reports step completion via `[Step X complete]` markers
- `on_plan_step` / `on_plan_done` hook events fire as steps complete
- `ConsoleHook` prints step progress and plan summary

### Why Both Exist

In production agents, planning is typically just part of the system prompt — no separate hook is needed. These two hooks are preserved as **learning examples** to illustrate the evolution from explicit planning (PlanHook) to the model-driven approach (ReActPlanHook) that production agents actually use.

## Phase 11: Guardrails (Human-in-the-Loop)

The agent can require human approval before executing dangerous tools. This is implemented as a guardrail hook that intercepts tool calls before execution.

```python
from toy_agent.guardrails import GuardrailHook

# Default: approve run_bash, write_file, edit_file
agent = Agent(
    client=client,
    hooks=[ConsoleHook(), GuardrailHook()],
)

# Custom approval list
hook = GuardrailHook(approval_tools={"run_bash", "write_file"})

# Auto-approve specific tools (no prompt)
hook = GuardrailHook(auto_approve={"run_bash"})
```

When a tool requires approval, the user is prompted:
```
[guardrail] Allow run_bash({"command": "rm -rf /tmp/old"})? [y/N]
```

- Returning `n` or pressing Enter blocks execution; the blocked message is returned as the tool result
- Returning `y` allows execution to proceed normally
- The hook system uses `on_tool_approve` (returns `None` to allow, `str` to block) and `on_guardrail_block` (observation event)
- Works for both streaming and non-streaming modes
- `load_skill` and `read_file` are allowed by default (no approval needed)

## Getting Started

```bash
# Install dependencies
uv sync

# Configure API
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Configure MCP servers (optional)
mkdir -p .toy-agent
# Create .toy-agent/mcp.json

# Run
make run
```

## Make Commands

| Command | Description |
|---------|-------------|
| `make run` | Start the agent |
| `make mcp` | Start SSE test MCP server |
| `make test` | Run unit tests |
| `make lint` | Run ruff lint |
| `make fmt` | Format code with ruff |
| `make check` | Run lint + format check |
