"""Tiny Agent - A minimal Agent Loop implementation for learning agent fundamentals.

Agent Loop core flow:
  1. Send conversation history to the LLM
  2. LLM returns a response, possibly with tool_calls
  3. If tool_calls exist → execute tools, append results to history, go back to step 1
  4. If no tool_calls → LLM gives the final answer, exit loop

This is the essence of all Agent frameworks (LangChain, CrewAI, AutoGPT, etc.).
"""

import json
from collections.abc import Callable
from typing import Any

from openai import APIError, OpenAI


class Agent:
    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        system: str = "You are a helpful assistant.",
    ):
        self.client = client
        self.model = model
        self.system = system
        self.messages: list[dict] = [{"role": "system", "content": system}]

    def run(self, user_input: str) -> str:
        """Run a complete agent loop and return the final answer."""
        self.messages.append({"role": "user", "content": user_input})

        # ===== Agent Loop =====
        while True:
            # 1. Call the LLM
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
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
                    result = self._execute_tool(tool_call)
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

    def _execute_tool(self, tool_call) -> Any:
        """Find and execute the corresponding tool function for a tool_call."""
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)

        for tool in self.tools:
            if tool.schema["function"]["name"] == fn_name:
                print(f"  [tool] {fn_name}({fn_args})")
                return tool.fn(**fn_args)

        return f"Error: unknown tool '{fn_name}'"
