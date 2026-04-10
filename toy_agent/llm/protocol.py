"""LLM Protocol — the interface all providers must implement."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from toy_agent.llm.types import ChatRequest, ChatResponse, StreamChunk


class LLMProtocol(Protocol):
    """Protocol defining the LLM interface.

    All provider adapters must implement these two methods.
    """

    def chat(self, request: ChatRequest) -> ChatResponse: ...

    def chat_stream(self, request: ChatRequest) -> Iterator[StreamChunk]: ...
