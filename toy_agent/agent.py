"""Tiny Agent - A minimal Agent Loop implementation for learning agent fundamentals.

Agent Loop core flow:
  1. Send conversation history to the LLM
  2. LLM returns a response, possibly with tool_calls
  3. If tool_calls exist → execute tools, append results, go back to step 1
  4. If no tool_calls → LLM gives the final answer, exit loop

This is the essence of all Agent frameworks (LangChain, CrewAI, AutoGPT, etc.).
"""

from __future__ import annotations

import inspect
import json
import time
import warnings
from typing import TYPE_CHECKING, Any

from openai import APIError, OpenAI
from pydantic import BaseModel

from toy_agent.hooks import AgentHook
from toy_agent.skills import Skill, get_skill
from toy_agent.tools import Tool

if TYPE_CHECKING:
    from toy_agent.context import ContextCompressor
    from toy_agent.memory import SessionMemory


def _add_additional_properties_false(schema: dict) -> dict:
    """Recursively add additionalProperties: false for OpenAI strict mode."""
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
    for value in schema.values():
        if isinstance(value, dict):
            _add_additional_properties_false(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _add_additional_properties_false(item)
    return schema


def _pydantic_to_response_format(model: type[BaseModel]) -> dict:
    """Convert a Pydantic model to OpenAI's response_format dict."""
    schema = model.model_json_schema()
    _add_additional_properties_false(schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": True,
            "schema": schema,
        },
    }


def _parse_structured_output(raw: str, model: type[BaseModel]) -> BaseModel | str:
    """Parse LLM output into a Pydantic model, with fallback to raw string."""
    try:
        return model.model_validate_json(raw)
    except Exception as e:
        warnings.warn(
            f"Structured output parse failed ({type(e).__name__}: {e}), returning raw string",
            stacklevel=2,
        )
        return raw


class Agent:
    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        system: str = "You are a helpful assistant.",
        tools: list[Tool] | None = None,
        skills: list[Skill] | None = None,
        stream: bool = False,
        memory: SessionMemory | None = None,
        compressor: ContextCompressor | None = None,
        retriever: Any | None = None,
        hooks: list[AgentHook] | None = None,
        max_tool_retries: int = 0,
    ):
        self.client = client
        self.model = model
        self.tools = tools or []
        self.skills = skills or []
        self.stream = stream
        self.memory = memory
        self.compressor = compressor
        self.retriever = retriever
        self.hooks = hooks or []
        self.max_tool_retries = max_tool_retries
        self.system = self._build_system_prompt(system)
        self.messages: list[dict] = [{"role": "system", "content": self.system}]

    def _build_system_prompt(self, base: str) -> str:
        """Build system prompt with tool list, subagent list, and skill list."""
        from toy_agent.subagent import SubAgentTool

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

    def _emit(self, event: str, **kwargs) -> None:
        for h in self.hooks:
            getattr(h, event, lambda **_: None)(**kwargs)

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

    async def run(
        self,
        user_input: str,
        *,
        max_turns: int | None = None,
        stream: bool | None = None,
        plan: bool | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> str:
        """Run a complete agent loop and return the final answer.

        Args:
            user_input: The user message to send.
            max_turns: Maximum number of LLM API calls. If exceeded, returns an error.
                       None means no limit (runs until LLM stops calling tools).
            stream: Override streaming mode. If None, uses self.stream from constructor.
            plan: Override planning behavior. True forces plan generation, False skips it.
                  None leaves the decision to the PlanHook's auto setting.
            response_format: Pydantic model for structured output. When set, tools are
                             suppressed and the response is parsed into this model.
        """
        use_stream = stream if stream is not None else self.stream
        self.messages.append({"role": "user", "content": user_input})
        self._emit("on_message", role="user", content=user_input)

        # Planning phase: let hooks pre-process (e.g., generate and inject a plan)
        for h in self.hooks:
            if hasattr(h, "set_plan_override") and plan is not None:
                h.set_plan_override(plan)
            result = h.on_before_loop(agent=self)
            if inspect.isawaitable(result):
                await result

        before_count = len(self.messages)

        # RAG: inject relevant context from retriever
        if self.retriever:
            last_user_msg = next((m["content"] for m in reversed(self.messages) if m["role"] == "user"), None)
            if last_user_msg:
                docs = self.retriever.query(last_user_msg, top_k=3)
                if docs:
                    context = "\n\n".join(d.content for d in docs)
                    rag_msg = {"role": "system", "content": f"[Retrieved context]\n{context}"}
                    self.messages.insert(-1, rag_msg)

        if self.compressor:
            self.messages = self.compressor.compress(self.messages)
            if len(self.messages) < before_count:
                self._emit("on_compress", before_count=before_count, after_count=len(self.messages))

        turn = 0
        while True:
            turn += 1
            if max_turns is not None and turn > max_turns:
                return f"[Error] Exceeded max turns ({max_turns})"

            self._emit("on_llm_request", messages=self.messages)

            try:
                api_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": self.messages,
                    "stream": use_stream,
                }
                if response_format:
                    api_kwargs["response_format"] = _pydantic_to_response_format(response_format)
                    api_kwargs["tools"] = None
                else:
                    api_kwargs["tools"] = self._all_tool_schemas or None
                response = self.client.chat.completions.create(**api_kwargs)
            except APIError as e:
                self._emit("on_error", error=f"[API Error] {e.status_code}: {e.message}")
                return f"[API Error] {e.status_code}: {e.message}"

            if use_stream:
                return await self._process_stream(
                    response, max_turns=max_turns, turn=turn, response_format=response_format
                )

            message = response.choices[0].message

            if message.tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
                self._emit("on_llm_response", message=assistant_msg)
                self.messages.append(assistant_msg)
                self._emit("on_message", role="assistant", content=message.content or "")

                for tool_call in message.tool_calls:
                    fn_args = json.loads(tool_call.function.arguments)
                    self._emit("on_tool_call", tool_name=tool_call.function.name, arguments=fn_args)
                    result = await self._execute_tool(tool_call.function.name, fn_args)
                    self._emit("on_tool_result", tool_name=tool_call.function.name, result=str(result))
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        }
                    )

                continue

            self.messages.append({"role": "assistant", "content": message.content})
            self._emit("on_message", role="assistant", content=message.content)
            self._emit("on_llm_response", message={"role": "assistant", "content": message.content})
            if self.memory:
                self.memory.save(self.messages)
            if response_format:
                return _parse_structured_output(message.content, response_format)
            return message.content

    async def _check_guardrails(self, tool_name: str, arguments: dict) -> str | None:
        """Check if any guardrail hook blocks this tool call.

        Returns a block reason string if blocked, None if allowed.
        """
        for h in self.hooks:
            result = h.on_tool_approve(tool_name=tool_name, arguments=arguments)
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, str):
                self._emit("on_guardrail_block", tool_name=tool_name, arguments=arguments, reason=result)
                return result
        return None

    async def _execute_tool(self, fn_name: str, fn_args: dict) -> Any:
        """Find and execute the corresponding tool function."""
        # Guardrail check before execution
        blocked = await self._check_guardrails(fn_name, fn_args)
        if blocked is not None:
            return blocked

        # Built-in load_skill tool (no retry)
        if fn_name == "load_skill":
            name = fn_args.get("name", "")
            skill = get_skill(self.skills, name)
            if not skill:
                return f"Skill '{name}' not found. Available skills: {[s.name for s in self.skills]}"
            return skill.content

        # User-registered tools (with retry)
        for tool in self.tools:
            if tool.schema["function"]["name"] == fn_name:
                for attempt in range(self.max_tool_retries + 1):
                    try:
                        if inspect.iscoroutinefunction(tool.fn):
                            return await tool.fn(**fn_args)
                        return tool.fn(**fn_args)
                    except Exception as e:
                        error_str = str(e)
                        if attempt < self.max_tool_retries:
                            self._emit("on_tool_retry", tool_name=fn_name, attempt=attempt, error=error_str)
                            time.sleep(2**attempt)  # exponential backoff
                            continue
                        self._emit(
                            "on_error",
                            error=f"Error: tool '{fn_name}' failed after {self.max_tool_retries} retries: {e}",
                        )
                        return f"Error: tool '{fn_name}' failed: {e}"

        return f"Error: unknown tool '{fn_name}'"

    async def _process_stream(
        self, stream, *, max_turns: int | None, turn: int, response_format: type[BaseModel] | None = None
    ) -> str:
        """Process a streaming response. Print tokens, handle tool calls."""
        collected_content = ""
        tool_calls_data: dict[int, dict] = {}

        for chunk in stream:
            delta = chunk.choices[0].delta

            # Collect text content
            if delta.content:
                collected_content += delta.content
                print(delta.content, end="", flush=True)

            # Collect tool call data
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_data:
                        tool_calls_data[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls_data[idx]["id"] += tc.id
                    if tc.function.name:
                        tool_calls_data[idx]["name"] += tc.function.name
                    if tc.function.arguments:
                        tool_calls_data[idx]["arguments"] += tc.function.arguments

        print()  # newline after streamed output

        # If there are tool calls, execute them and continue the loop
        if tool_calls_data:
            message_dict = {
                "role": "assistant",
                "content": collected_content or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in tool_calls_data.values()
                ],
            }
            self.messages.append(message_dict)
            self._emit("on_llm_response", message=message_dict)
            self._emit("on_message", role="assistant", content=collected_content or "")

            for tc in tool_calls_data.values():
                fn_args = json.loads(tc["arguments"])
                self._emit("on_tool_call", tool_name=tc["name"], arguments=fn_args)
                result = await self._execute_tool(tc["name"], fn_args)
                self._emit("on_tool_result", tool_name=tc["name"], result=str(result))
                self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})

            # Continue — get next response (recursive call)
            turn += 1
            if max_turns is not None and turn > max_turns:
                return f"[Error] Exceeded max turns ({max_turns})"

            self._emit("on_llm_request", messages=self.messages)

            try:
                api_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": self.messages,
                    "stream": True,
                }
                if response_format:
                    api_kwargs["response_format"] = _pydantic_to_response_format(response_format)
                    api_kwargs["tools"] = None
                else:
                    api_kwargs["tools"] = self._all_tool_schemas or None
                response = self.client.chat.completions.create(**api_kwargs)
            except APIError as e:
                self._emit("on_error", error=f"[API Error] {e.status_code}: {e.message}")
                return f"[API Error] {e.status_code}: {e.message}"

            return await self._process_stream(response, max_turns=max_turns, turn=turn, response_format=response_format)

        # No tool calls — final answer
        final_message = {"role": "assistant", "content": collected_content}
        self._emit("on_llm_response", message=final_message)
        self.messages.append(final_message)
        self._emit("on_message", role="assistant", content=collected_content)
        if self.memory:
            self.memory.save(self.messages)
        if response_format:
            return _parse_structured_output(collected_content, response_format)
        return collected_content
