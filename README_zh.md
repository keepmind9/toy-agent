# toy-agent

一个用于学习 Agent 核心原理的最小化 Agent 实现。Python + OpenAI SDK。

[English](README.md)

## 项目结构

```
toy-agent/
├── main.py                     # 入口
├── src/toy_agent/
│   ├── agent.py                # Agent loop 核心
│   ├── config.py               # 多级配置加载
│   ├── mcp.py                  # MCP 客户端 (stdio + SSE)
│   └── tools/
│       ├── __init__.py          # @tool 装饰器 + 自动注册
│       ├── file_ops.py          # read_file, write_file, edit_file
│       └── run_bash.py          # run_bash (带安全检查)
├── tests/
│   ├── mcp_stdio_server.py      # stdio MCP 测试服务器
│   └── mcp_sse_server.py        # SSE MCP 测试服务器
├── .env.example
├── .toy-agent/mcp.json          # MCP 服务器配置（不提交）
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
from src.toy_agent.tools import tool

@tool(description="Read the content of a file")
def read_file(path: str) -> str:
    """path: Absolute or relative file path to read"""
    ...
```

**内置工具：**
- `read_file` / `write_file` / `edit_file` — 文件操作
- `run_bash` — 执行 Shell 命令（高危命令被禁止）

`src/toy_agent/tools/*.py` 中的工具会被自动导入和注册，无需手动添加。

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
| `make lint` | 运行 ruff 代码检查 |
| `make fmt` | 格式化代码 |
| `make check` | 运行 lint + 格式检查 |
