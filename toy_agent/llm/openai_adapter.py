"""OpenAI provider adapter."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from openai import APIError, OpenAI

from toy_agent.llm.base import BaseAdapter
from toy_agent.llm.types import (
    ChatRequest,
    ChatResponse,
    LLMError,
    StreamChunk,
    StreamDeltaToolCall,
    TokenUsage,
    ToolCall,
)


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI and OpenAI-compatible APIs."""

    def _create_client(self) -> OpenAI:
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if base_url := self._extra.get("base_url"):
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    def _tool_defs_to_openai(self, tools: list) -> list[dict] | None:
        """Convert ToolDef list to OpenAI tool format."""
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def _do_chat(self, request: ChatRequest) -> Any:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
        }
        if request.tools:
            kwargs["tools"] = self._tool_defs_to_openai(request.tools)
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.response_format is not None:
            kwargs["response_format"] = request.response_format

        try:
            return self._client.chat.completions.create(**kwargs)
        except APIError as e:
            raise LLMError(message=str(e.message), status_code=e.status_code) from e

    def _do_chat_stream(self, request: ChatRequest) -> Iterator[Any]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "stream": True,
        }
        if request.tools:
            kwargs["tools"] = self._tool_defs_to_openai(request.tools)
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.response_format is not None:
            kwargs["response_format"] = request.response_format

        try:
            return self._client.chat.completions.create(**kwargs)
        except APIError as e:
            raise LLMError(message=str(e.message), status_code=e.status_code) from e

    def _parse_response(self, raw: Any) -> ChatResponse:
        message = raw.choices[0].message
        u = raw.usage
        usage = (
            TokenUsage(
                prompt_tokens=getattr(u, "prompt_tokens", 0),
                completion_tokens=getattr(u, "completion_tokens", 0),
                total_tokens=getattr(u, "total_tokens", 0),
            )
            if u
            else None
        )

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments) for tc in message.tool_calls
            ]

        return ChatResponse(
            content=message.content,
            tool_calls=tool_calls,
            usage=usage,
            raw=raw,
        )

    def _parse_stream_chunk(self, raw: Any) -> StreamChunk | None:
        if not raw.choices:
            if hasattr(raw, "usage") and raw.usage:
                return StreamChunk(
                    usage=TokenUsage(
                        prompt_tokens=getattr(raw.usage, "prompt_tokens", 0),
                        completion_tokens=getattr(raw.usage, "completion_tokens", 0),
                        total_tokens=getattr(raw.usage, "total_tokens", 0),
                    ),
                )
            return None

        delta = raw.choices[0].delta
        finish_reason = getattr(raw.choices[0], "finish_reason", None)

        delta_tool_calls = None
        if getattr(delta, "tool_calls", None):
            delta_tool_calls = [
                StreamDeltaToolCall(
                    index=tc.index,
                    id=getattr(tc, "id", None),
                    name=getattr(tc.function, "name", None) if hasattr(tc, "function") else None,
                    arguments=getattr(tc.function, "arguments", None) if hasattr(tc, "function") else None,
                )
                for tc in delta.tool_calls
            ]

        return StreamChunk(
            delta_content=getattr(delta, "content", None),
            delta_tool_calls=delta_tool_calls,
            finish_reason=finish_reason,
        )
