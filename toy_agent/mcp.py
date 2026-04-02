"""MCP client: connect to MCP servers, discover tools, and execute calls.

Supports two transport types:
  - stdio: local subprocess, configured with "command" and "args"
  - sse:   remote server, configured with "url"

Tools from MCP servers are converted to our Tool format and
registered alongside built-in tools.
"""

import os
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client

from toy_agent.tools import Tool


class MCPClient:
    """Manages connections to multiple MCP servers."""

    def __init__(self):
        self._sessions: dict[str, ClientSession] = {}
        self._cleanup_handlers: list[Any] = []

    async def connect(self, servers: dict[str, dict]) -> list[Tool]:
        """Connect to all configured MCP servers and return discovered tools.

        Config format:
          stdio: {"command": "...", "args": [...], "env": {...}}
          sse:   {"url": "http://..."}
        """
        tools: list[Tool] = []

        for server_name, config in servers.items():
            try:
                server_tools = await self._connect_server(server_name, config)
                tools.extend(server_tools)
                print(f"[mcp] connected to '{server_name}': {len(server_tools)} tools")
            except Exception as e:
                print(f"[mcp] failed to connect to '{server_name}': {e}")

        return tools

    async def _connect_server(self, name: str, config: dict) -> list[Tool]:
        """Connect to a single MCP server and discover its tools."""
        if "url" in config:
            read_stream, write_stream = await self._connect_sse(config)
        else:
            read_stream, write_stream = await self._connect_stdio(config)

        session = ClientSession(read_stream, write_stream)
        await session.__aenter__()
        self._cleanup_handlers.append(session)
        await session.initialize()

        self._sessions[name] = session

        # Discover tools
        result = await session.list_tools()
        tools: list[Tool] = []

        for mcp_tool in result.tools:
            tool_name = mcp_tool.name

            schema = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": mcp_tool.description or "",
                    "parameters": mcp_tool.inputSchema
                    or {
                        "type": "object",
                        "properties": {},
                    },
                },
            }

            def make_caller(tn: str, sn: str):
                async def call_tool(**kwargs) -> str:
                    return await self._call_tool(sn, tn, kwargs)

                return call_tool

            tools.append(Tool(schema=schema, fn=make_caller(tool_name, name)))

        return tools

    async def _connect_stdio(self, config: dict):
        """Connect via stdio (local subprocess)."""
        command = config["command"]
        args = config.get("args", [])
        env = {**os.environ, **config.get("env", {})}

        server_params = StdioServerParameters(command=command, args=args, env=env)
        context = stdio_client(server_params)
        read_stream, write_stream = await context.__aenter__()
        self._cleanup_handlers.append(context)
        return read_stream, write_stream

    async def _connect_sse(self, config: dict):
        """Connect via SSE (remote server)."""
        url = config["url"]
        headers = config.get("headers", {})

        context = sse_client(url=url, headers=headers)
        read_stream, write_stream = await context.__aenter__()
        self._cleanup_handlers.append(context)
        return read_stream, write_stream

    async def _call_tool(self, server_name: str, tool_name: str, args: dict) -> str:
        """Call a tool on an MCP server."""
        session = self._sessions.get(server_name)
        if not session:
            return f"Error: MCP server '{server_name}' not connected"

        try:
            result = await session.call_tool(tool_name, args)
            parts = []
            for content in result.content:
                if hasattr(content, "text"):
                    parts.append(content.text)
            return "\n".join(parts) if parts else str(result)
        except Exception as e:
            return f"Error: MCP tool '{tool_name}' failed: {e}"

    async def cleanup(self):
        """Close all MCP server connections."""
        for handler in reversed(self._cleanup_handlers):
            try:
                await handler.__aexit__(None, None, None)
            except Exception:
                pass
        self._cleanup_handlers.clear()
        self._sessions.clear()
