"""Anthropic (Claude) provider adapter.

Uses lazy import via importlib to avoid hard dependency on the anthropic SDK.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator
from typing import Any

from toy_agent.llm.base import BaseAdapter
from toy_agent.llm.types import (
    ChatRequest,
    ChatResponse,
    LLMError,
    StreamChunk,
    TokenUsage,
    ToolCall,
)


def _import_anthropic():
    """Import anthropic SDK with a clear error message if missing."""
    try:
        return importlib.import_module("anthropic")
    except ImportError:
        raise ImportError("anthropic SDK is required for Anthropic provider. Install it with: pip install anthropic")


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic Claude API."""

    def _create_client(self) -> Any:
        anthropic = _import_anthropic()
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if base_url := self._extra.get("base_url"):
            kwargs["base_url"] = base_url
        return anthropic.Anthropic(**kwargs)

    def _split_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Extract system message and return (system, remaining_messages).

        Anthropic requires system to be a separate parameter, not in the messages list.
        """
        system = None
        filtered = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered.append(msg)
        return system, filtered

    def _tool_defs_to_anthropic(self, tools: list) -> list[dict] | None:
        """Convert ToolDef list to Anthropic tool format."""
        if not tools:
            return None
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    def _do_chat(self, request: ChatRequest) -> Any:
        system, messages = self._split_messages(request.messages)
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
        }
        if system:
            kwargs["system"] = system
        if request.tools:
            kwargs["tools"] = self._tool_defs_to_anthropic(request.tools)
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        try:
            return self._client.messages.create(**kwargs)
        except Exception as e:
            status = getattr(e, "status_code", None)
            msg = getattr(e, "message", str(e))
            raise LLMError(message=msg, status_code=status) from e

    def _do_chat_stream(self, request: ChatRequest) -> Iterator[Any]:
        system, messages = self._split_messages(request.messages)
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "stream": True,
        }
        if system:
            kwargs["system"] = system
        if request.tools:
            kwargs["tools"] = self._tool_defs_to_anthropic(request.tools)
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        try:
            return self._client.messages.create(**kwargs)
        except Exception as e:
            status = getattr(e, "status_code", None)
            msg = getattr(e, "message", str(e))
            raise LLMError(message=msg, status_code=status) from e

    def _parse_response(self, raw: Any) -> ChatResponse:
        content = None
        tool_calls = None

        for block in raw.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=json.dumps(block.input),
                    )
                )

        usage = TokenUsage(
            prompt_tokens=getattr(raw.usage, "input_tokens", 0),
            completion_tokens=getattr(raw.usage, "output_tokens", 0),
            total_tokens=getattr(raw.usage, "input_tokens", 0) + getattr(raw.usage, "output_tokens", 0),
        )

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            raw=raw,
        )

    def _parse_stream_chunk(self, raw: Any) -> StreamChunk | None:
        event_type = getattr(raw, "type", None)

        if event_type == "content_block_delta":
            delta = raw.delta
            if getattr(delta, "type", None) == "text_delta":
                return StreamChunk(delta_content=delta.text)
            if getattr(delta, "type", None) == "input_json_delta":
                return StreamChunk(delta_content=delta.partial_json)
            return None

        if event_type == "message_stop":
            return StreamChunk(finish_reason="stop")

        return None
