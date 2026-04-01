# toy-agent

A minimal agent implementation for learning agent fundamentals. Built with Python + OpenAI SDK.

[中文文档](README_zh.md)

## Project Structure

```
toy-agent/
├── main.py                     # Entry point
├── src/toy_agent/
│   ├── agent.py                # Agent loop core
│   ├── config.py               # Multi-level config loader
│   ├── mcp.py                  # MCP client (stdio + SSE)
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
│   ├── test_skills.py           # Skills loader tests
│   └── test_subagent.py         # SubAgentTool tests
├── .env.example
├── .toy-agent/mcp.json          # MCP server config (not tracked)
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
from src.toy_agent.tools import tool

@tool(description="Read the content of a file")
def read_file(path: str) -> str:
    """path: Absolute or relative file path to read"""
    ...
```

**Built-in tools:**
- `read_file` / `write_file` / `edit_file` — file operations
- `run_bash` — execute shell commands (dangerous commands blocked)

Tools in `src/toy_agent/tools/*.py` are auto-imported and registered. No manual registration needed.

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
from src.toy_agent.subagent import SubAgentTool

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
