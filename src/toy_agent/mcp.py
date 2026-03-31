"""MCP client: connect to MCP servers, discover tools, and execute calls.

Each MCP server is a subprocess communicating via stdio.
Tools from MCP servers are converted to our Tool format and
registered alongside built-in tools.
"""

import os
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.toy_agent.tools import Tool


class MCPClient:
    """Manages connections to multiple MCP servers."""

    def __init__(self):
        self._sessions: dict[str, ClientSession] = {}
        self._tool_to_server: dict[str, str] = {}  # tool_name -> server_name
        self._cleanup_handlers: list[Any] = []

    async def connect(self, servers: dict[str, dict]) -> list[Tool]:
        """Connect to all configured MCP servers and return discovered tools.

        Args:
            servers: {"server-name": {"command": ..., "args": [...], "env": {...}}}
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
        command = config["command"]
        args = config.get("args", [])
        env = {**os.environ, **config.get("env", {})}

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
        )

        # stdio_client returns (read_stream, write_stream)
        streams_context = stdio_client(server_params)
        read_stream, write_stream = await streams_context.__aenter__()
        self._cleanup_handlers.append(streams_context)

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

            # Convert MCP tool schema to OpenAI function calling format
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

            self._tool_to_server[tool_name] = name
            server_name = name  # capture for closure

            # Create a Tool with a closure that calls this server
            def make_caller(tn: str, sn: str):
                async def call_tool(**kwargs) -> str:
                    return await self._call_tool(sn, tn, kwargs)

                return call_tool

            tools.append(Tool(schema=schema, fn=make_caller(tool_name, server_name)))

        return tools

    async def _call_tool(self, server_name: str, tool_name: str, args: dict) -> str:
        """Call a tool on an MCP server."""
        session = self._sessions.get(server_name)
        if not session:
            return f"Error: MCP server '{server_name}' not connected"

        try:
            result = await session.call_tool(tool_name, args)
            # Concatenate text content parts
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
        self._tool_to_server.clear()
