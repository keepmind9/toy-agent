"""Unified data types for LLM abstraction layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Unified exception for all LLM provider errors."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Tool definitions and calls
# ---------------------------------------------------------------------------


@dataclass
class ToolDef:
    """Unified tool definition (provider-agnostic)."""

    name: str
    description: str
    parameters: dict  # JSON Schema


@dataclass
class ToolCall:
    """A tool call returned by the LLM."""

    id: str
    name: str
    arguments: str  # JSON string


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


@dataclass
class StreamDeltaToolCall:
    """Incremental tool call data in a streaming chunk."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str | None = None


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""

    delta_content: str | None = None
    delta_tool_calls: list[StreamDeltaToolCall] | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None


# ---------------------------------------------------------------------------
# Chat request / response
# ---------------------------------------------------------------------------


@dataclass
class ChatRequest:
    """Unified chat completion request."""

    messages: list[dict]
    model: str
    tools: list[ToolDef] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: dict | None = None
    stream: bool = False


@dataclass
class ChatResponse:
    """Unified chat completion response."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: TokenUsage | None = None
    raw: Any = None
