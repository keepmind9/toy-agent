"""Tiny Agent - A minimal Agent Loop implementation for learning agent fundamentals.

Agent Loop core flow:
  1. Send conversation history to the LLM
  2. LLM returns a response, possibly with tool_calls
  3. If tool_calls exist → execute tools, append results, go back to step 1
  4. If no tool_calls → LLM gives the final answer, exit loop

This is the essence of all Agent frameworks (LangChain, CrewAI, AutoGPT, etc.).
"""

import inspect
import json
from typing import Any

from openai import APIError, OpenAI

from src.toy_agent.skills import Skill, get_skill
from src.toy_agent.tools import Tool


class Agent:
    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        system: str = "You are a helpful assistant.",
        tools: list[Tool] | None = None,
        skills: list[Skill] | None = None,
    ):
        self.client = client
        self.model = model
        self.tools = tools or []
        self.skills = skills or []
        self.system = self._build_system_prompt(system)
        self.messages: list[dict] = [{"role": "system", "content": self.system}]

    def _build_system_prompt(self, base: str) -> str:
        """Build system prompt with tool list, subagent list, and skill list."""
        from src.toy_agent.subagent import SubAgentTool

        parts = [base]

        regular_tools = [t for t in self.tools if not isinstance(t, SubAgentTool)]
        subagent_tools = [t for t in self.tools if isinstance(t, SubAgentTool)]

        if regular_tools:
            tool_list = "\n".join(f"- {t.name}: {t.schema['function']['description']}" for t in regular_tools)
            parts.append(f"Available tools:\n{tool_list}")

        if subagent_tools:
            sub_list = "\n".join(f"- {t.name}: {t.schema['function']['description']}" for t in subagent_tools)
            parts.append(f"Available subagents (call as tools to delegate tasks):\n{sub_list}")

        if self.skills:
            skill_list = "\n".join(f"- {s.name}: {s.description}" for s in self.skills)
            parts.append(f"Available skills (call load_skill to activate):\n{skill_list}")

        return "\n\n".join(parts)

    @property
    def _all_tool_schemas(self) -> list[dict]:
        """Built-in tools + user-registered tools."""
        built_in = [
            {
                "type": "function",
                "function": {
                    "name": "load_skill",
                    "description": "Activate a skill to gain its expertise. Returns the skill content.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Skill name to activate",
                            },
                        },
                        "required": ["name"],
                    },
                },
            }
        ]
        return built_in + [t.schema for t in self.tools]

    async def run(self, user_input: str, *, max_turns: int | None = None) -> str:
        """Run a complete agent loop and return the final answer.

        Args:
            user_input: The user message to send.
            max_turns: Maximum number of LLM API calls. If exceeded, returns an error.
                       None means no limit (runs until LLM stops calling tools).
        """
        self.messages.append({"role": "user", "content": user_input})

        turn = 0
        while True:
            turn += 1
            if max_turns is not None and turn > max_turns:
                return f"[Error] Exceeded max turns ({max_turns})"

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self._all_tool_schemas or None,
                )
            except APIError as e:
                return f"[API Error] {e.status_code}: {e.message}"
            message = response.choices[0].message

            if message.tool_calls:
                self.messages.append(message)

                for tool_call in message.tool_calls:
                    result = await self._execute_tool(tool_call)
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        }
                    )

                continue

            self.messages.append(message)
            return message.content

    async def _execute_tool(self, tool_call) -> Any:
        """Find and execute the corresponding tool function for a tool_call."""
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)

        # Built-in load_skill tool
        if fn_name == "load_skill":
            print(f"  [tool] {fn_name}({fn_args})")
            name = fn_args.get("name", "")
            skill = get_skill(self.skills, name)
            if not skill:
                return f"Skill '{name}' not found. Available skills: {[s.name for s in self.skills]}"
            return skill.content

        # User-registered tools
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
