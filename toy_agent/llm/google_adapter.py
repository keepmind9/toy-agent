"""Google Gemini provider adapter.

Uses lazy import via importlib to avoid hard dependency on the google-genai SDK.
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


def _import_genai():
    """Import google.genai SDK with a clear error message if missing."""
    try:
        return importlib.import_module("google.genai")
    except ImportError:
        raise ImportError("google-genai SDK is required for Gemini provider. Install it with: pip install google-genai")


class GeminiAdapter(BaseAdapter):
    """Adapter for Google Gemini API."""

    def _create_client(self) -> Any:
        genai = _import_genai()
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if base_url := self._extra.get("base_url"):
            kwargs["http_options"] = {"base_url": base_url}
        return genai.Client(**kwargs)

    def _split_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Extract system message and return (system, remaining_messages)."""
        system = None
        filtered = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered.append(msg)
        return system, filtered

    def _messages_to_gemini_contents(self, messages: list[dict]) -> list[dict]:
        """Convert messages to Gemini contents format."""
        contents = []
        for msg in messages:
            role = msg["role"]
            if role == "assistant":
                role = "model"
            elif role == "tool":
                role = "function"

            if role == "function":
                contents.append(
                    {
                        "role": "function",
                        "parts": [
                            {
                                "function_response": {
                                    "name": msg.get("name", ""),
                                    "response": {"content": msg.get("content", "")},
                                }
                            }
                        ],
                    }
                )
            elif msg.get("tool_calls"):
                parts = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    args = json.loads(fn.get("arguments", "{}"))
                    parts.append({"function_call": {"name": fn.get("name", ""), "args": args}})
                if msg.get("content"):
                    parts.insert(0, {"text": msg["content"]})
                contents.append({"role": "model", "parts": parts})
            else:
                contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
        return contents

    def _tool_defs_to_gemini(self, tools: list) -> list[dict] | None:
        """Convert ToolDef list to Gemini FunctionDeclaration format."""
        if not tools:
            return None
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in tools
        ]

    def _do_chat(self, request: ChatRequest) -> Any:
        system, messages = self._split_messages(request.messages)
        contents = self._messages_to_gemini_contents(messages)

        kwargs: dict[str, Any] = {
            "model": request.model,
            "contents": contents,
        }
        config = {}
        if system:
            config["system_instruction"] = system
        if request.tools:
            tools = self._tool_defs_to_gemini(request.tools)
            config["tools"] = [{"function_declarations": tools}]
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.max_tokens is not None:
            config["max_output_tokens"] = request.max_tokens
        if request.response_format and request.response_format.get("type") == "json_object":
            config["response_mime_type"] = "application/json"
        if config:
            kwargs["config"] = config

        try:
            return self._client.models.generate_content(**kwargs)
        except Exception as e:
            msg = getattr(e, "message", str(e))
            raise LLMError(message=msg) from e

    def _do_chat_stream(self, request: ChatRequest) -> Iterator[Any]:
        system, messages = self._split_messages(request.messages)
        contents = self._messages_to_gemini_contents(messages)

        kwargs: dict[str, Any] = {
            "model": request.model,
            "contents": contents,
        }
        config = {}
        if system:
            config["system_instruction"] = system
        if request.tools:
            tools = self._tool_defs_to_gemini(request.tools)
            config["tools"] = [{"function_declarations": tools}]
        if config:
            kwargs["config"] = config

        try:
            return self._client.models.generate_content_stream(**kwargs)
        except Exception as e:
            msg = getattr(e, "message", str(e))
            raise LLMError(message=msg) from e

    def _parse_response(self, raw: Any) -> ChatResponse:
        content = None
        tool_calls = None

        if raw.candidates:
            candidate = raw.candidates[0]
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    content = part.text
                if hasattr(part, "function_call") and part.function_call:
                    if tool_calls is None:
                        tool_calls = []
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            id=f"gemini_{fc.name}",
                            name=fc.name,
                            arguments=json.dumps(dict(fc.args) if fc.args else {}),
                        )
                    )

        usage = TokenUsage()
        if hasattr(raw, "usage_metadata") and raw.usage_metadata:
            usage = TokenUsage(
                prompt_tokens=getattr(raw.usage_metadata, "prompt_token_count", 0),
                completion_tokens=getattr(raw.usage_metadata, "candidates_token_count", 0),
                total_tokens=getattr(raw.usage_metadata, "total_token_count", 0),
            )

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            raw=raw,
        )

    def _parse_stream_chunk(self, raw: Any) -> StreamChunk | None:
        if not hasattr(raw, "candidates") or not raw.candidates:
            return None

        candidate = raw.candidates[0]
        content = None
        tool_calls = None

        if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    content = part.text
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append(
                        ToolCall(
                            id=f"gemini_{fc.name}",
                            name=fc.name,
                            arguments=json.dumps(dict(fc.args) if fc.args else {}),
                        )
                    )

        finish_reason = None
        if hasattr(candidate, "finish_reason") and candidate.finish_reason:
            fr = candidate.finish_reason
            finish_reason = fr.name if hasattr(fr, "name") else str(fr)

        return StreamChunk(
            delta_content=content,
            delta_tool_calls=None,
            finish_reason=finish_reason,
        )
