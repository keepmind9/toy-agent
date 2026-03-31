"""Tiny Agent - A minimal Agent Loop implementation for learning agent fundamentals.

Agent Loop core flow:
  1. Send conversation history to the LLM
  2. LLM returns a response, possibly with tool_calls
  3. If tool_calls exist → execute tools, append results to history, go back to step 1
  4. If no tool_calls → LLM gives the final answer, exit loop

This is the essence of all Agent frameworks (LangChain, CrewAI, AutoGPT, etc.).
"""

import asyncio
import inspect
import json
from typing import Any

from openai import APIError, OpenAI

from src.toy_agent.tools import Tool


class Agent:
    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        system: str = "You are a helpful assistant.",
        tools: list[Tool] | None = None,
    ):
        self.client = client
        self.model = model
        self.tools = tools or []
        self.system = self._build_system_prompt(system)
        self.messages: list[dict] = [{"role": "system", "content": self.system}]

    def _build_system_prompt(self, base: str) -> str:
        """Build system prompt with tool descriptions appended dynamically."""
        if not self.tools:
            return base
        tool_list = "\n".join(
            f"- {t.name}: {t.schema['function']['description']}" for t in self.tools
        )
        return f"{base}\n\nAvailable tools:\n{tool_list}"

    async def run(self, user_input: str) -> str:
        """Run a complete agent loop and return the final answer."""
        self.messages.append({"role": "user", "content": user_input})

        # ===== Agent Loop =====
        while True:
            # 1. Call the LLM
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=[t.schema for t in self.tools] if self.tools else None,
                )
            except APIError as e:
                return f"[API Error] {e.status_code}: {e.message}"
            message = response.choices[0].message

            # 2. Check if LLM wants to call tools
            if message.tool_calls:
                # Append assistant message (with tool_calls) to history
                self.messages.append(message)

                # 3. Execute each tool call
                for tool_call in message.tool_calls:
                    result = await self._execute_tool(tool_call)
                    # Append tool result so LLM can see it in the next turn
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    })

                # Continue the loop, let LLM decide the next step based on tool results
                continue

            # 4. No tool calls → final answer, exit loop
            self.messages.append(message)
            return message.content

    async def _execute_tool(self, tool_call) -> Any:
        """Find and execute the corresponding tool function for a tool_call."""
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)

        for tool in self.tools:
            if tool.schema["function"]["name"] == fn_name:
                print(f"  [tool] {fn_name}({fn_args})")
                try:
                    if inspect.iscoroutinefunction(tool.fn):
                        return await tool.fn(**fn_args)
                    return tool.fn(**fn_args)
                except Exception as e:
                    return f"Error: tool '{fn_name}' failed: {e}"

        return f"Error: unknown tool '{fn_name}'"
