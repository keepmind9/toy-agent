"""Tests for BaseAdapter ABC."""

import pytest

from toy_agent.llm.base import BaseAdapter
from toy_agent.llm.types import ChatRequest, ChatResponse, StreamChunk, TokenUsage


class ConcreteAdapter(BaseAdapter):
    """Minimal concrete implementation for testing."""

    def _create_client(self):
        return "mock_client"

    def _do_chat(self, request):
        return {"content": "hello", "usage": {"prompt": 10, "completion": 5}}

    def _do_chat_stream(self, request):
        yield {"delta": "Hi"}

    def _parse_response(self, raw):
        return ChatResponse(
            content=raw["content"],
            usage=TokenUsage(prompt_tokens=raw["usage"]["prompt"], completion_tokens=raw["usage"]["completion"]),
        )

    def _parse_stream_chunk(self, raw):
        return StreamChunk(delta_content=raw.get("delta"))


class TestBaseAdapter:
    def test_chat_delegates_to_do_and_parse(self):
        adapter = ConcreteAdapter(api_key="test", model="test-model")
        request = ChatRequest(messages=[{"role": "user", "content": "hi"}], model="test-model")
        response = adapter.chat(request)

        assert response.content == "hello"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5

    def test_chat_stream_yields_parsed_chunks(self):
        adapter = ConcreteAdapter(api_key="test", model="test-model")
        request = ChatRequest(messages=[{"role": "user", "content": "hi"}], model="test-model", stream=True)
        chunks = list(adapter.chat_stream(request))

        assert len(chunks) == 1
        assert chunks[0].delta_content == "Hi"

    def test_chat_stream_skips_none_chunks(self):
        """_parse_stream_chunk returning None should skip that chunk."""

        class SkippingAdapter(ConcreteAdapter):
            def _parse_stream_chunk(self, raw):
                if raw.get("skip"):
                    return None
                return StreamChunk(delta_content=raw.get("delta"))

            def _do_chat_stream(self, request):
                yield {"delta": "keep", "skip": False}
                yield {"delta": "skip", "skip": True}
                yield {"delta": "keep2", "skip": False}

        adapter = SkippingAdapter(api_key="test", model="test-model")
        request = ChatRequest(messages=[], model="test-model", stream=True)
        chunks = list(adapter.chat_stream(request))

        assert len(chunks) == 2
        assert chunks[0].delta_content == "keep"
        assert chunks[1].delta_content == "keep2"

    def test_cannot_instantiate_directly(self):
        """BaseAdapter is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseAdapter(api_key="test", model="test")
