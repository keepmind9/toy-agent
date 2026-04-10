"""BaseAdapter — shared logic for all LLM provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from toy_agent.llm.types import ChatRequest, ChatResponse, StreamChunk


class BaseAdapter(ABC):
    """Abstract base class for LLM provider adapters.

    Provides the template method pattern: chat() and chat_stream() handle
    the call and delegate to _do_chat/_do_chat_stream + _parse_response/_parse_stream_chunk.
    """

    def __init__(self, api_key: str, model: str, **kwargs: Any):
        self.api_key = api_key
        self.model = model
        self._extra = kwargs
        self._client = self._create_client()

    @abstractmethod
    def _create_client(self) -> Any:
        """Create and return the provider-specific SDK client."""

    @abstractmethod
    def _do_chat(self, request: ChatRequest) -> Any:
        """Call the provider SDK and return the raw response."""

    @abstractmethod
    def _do_chat_stream(self, request: ChatRequest) -> Iterator[Any]:
        """Call the provider SDK streaming and yield raw chunks."""

    @abstractmethod
    def _parse_response(self, raw: Any) -> ChatResponse:
        """Parse a raw provider response into ChatResponse."""

    @abstractmethod
    def _parse_stream_chunk(self, raw: Any) -> StreamChunk | None:
        """Parse a raw provider stream chunk into StreamChunk (or None to skip)."""

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat completion request and return a unified response."""
        raw = self._do_chat(request)
        return self._parse_response(raw)

    def chat_stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """Send a streaming chat completion request and yield unified chunks."""
        for raw_chunk in self._do_chat_stream(request):
            parsed = self._parse_stream_chunk(raw_chunk)
            if parsed is not None:
                yield parsed
