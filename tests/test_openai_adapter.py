"""Tests for OpenAI adapter."""

from unittest.mock import MagicMock, patch

import pytest

from toy_agent.llm.openai_adapter import OpenAIAdapter
from toy_agent.llm.types import ChatRequest, ToolDef


def _make_mock_openai_response(content="hello", tool_calls=None, usage=None):
    """Create a mock OpenAI chat completion response."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    resp_usage = MagicMock()
    resp_usage.prompt_tokens = (usage or {}).get("prompt_tokens", 10)
    resp_usage.completion_tokens = (usage or {}).get("completion_tokens", 5)
    resp_usage.total_tokens = (usage or {}).get("total_tokens", 15)

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = resp_usage
    return response


class TestOpenAIAdapterChat:
    def test_basic_chat(self):
        """Basic chat returns content and usage."""
        mock_response = _make_mock_openai_response(content="Hello!")

        with patch("toy_agent.llm.openai_adapter.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_cls.return_value = mock_client

            adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o-mini")
            request = ChatRequest(messages=[{"role": "user", "content": "hi"}], model="gpt-4o-mini")
            response = adapter.chat(request)

        assert response.content == "Hello!"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5

    def test_chat_with_tools(self):
        """Chat with tools returns tool_calls."""
        tc = MagicMock()
        tc.id = "tc_1"
        tc.function.name = "read_file"
        tc.function.arguments = '{"path": "/tmp/a.txt"}'

        mock_response = _make_mock_openai_response(content=None, tool_calls=[tc])

        with patch("toy_agent.llm.openai_adapter.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_cls.return_value = mock_client

            adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o-mini")
            tools = [ToolDef(name="read_file", description="Read file", parameters={})]
            request = ChatRequest(messages=[{"role": "user", "content": "read file"}], model="gpt-4o-mini", tools=tools)
            response = adapter.chat(request)

        assert response.content is None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "tc_1"
        assert response.tool_calls[0].name == "read_file"
        assert response.tool_calls[0].arguments == '{"path": "/tmp/a.txt"}'

    def test_chat_passes_openai_format_tools(self):
        """Adapter converts ToolDef to OpenAI tool format."""
        mock_response = _make_mock_openai_response(content="ok")

        with patch("toy_agent.llm.openai_adapter.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_cls.return_value = mock_client

            adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o-mini")
            tools = [ToolDef(name="my_tool", description="A tool", parameters={"type": "object", "properties": {}})]
            request = ChatRequest(
                messages=[{"role": "user", "content": "go"}],
                model="gpt-4o-mini",
                tools=tools,
                max_tokens=100,
                response_format={"type": "json_object"},
            )
            adapter.chat(request)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["response_format"] == {"type": "json_object"}
        oai_tools = call_kwargs["tools"]
        assert len(oai_tools) == 1
        assert oai_tools[0]["type"] == "function"
        assert oai_tools[0]["function"]["name"] == "my_tool"

    def test_chat_passes_base_url(self):
        """Adapter passes base_url to OpenAI client."""
        with patch("toy_agent.llm.openai_adapter.OpenAI") as mock_cls:
            OpenAIAdapter(api_key="sk-test", model="gpt-4o-mini", base_url="https://custom.api/v1")

        mock_cls.assert_called_once_with(api_key="sk-test", base_url="https://custom.api/v1")

    def test_chat_propagates_llm_error(self):
        """OpenAI APIError is converted to LLMError."""
        with patch("toy_agent.llm.openai_adapter.OpenAI") as mock_cls:
            from openai import APIError

            mock_client = MagicMock()
            fake_error = APIError(message="rate limit", request=MagicMock(), body=None)
            fake_error.status_code = 429
            mock_client.chat.completions.create.side_effect = fake_error
            mock_cls.return_value = mock_client

            adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o-mini")
            request = ChatRequest(messages=[{"role": "user", "content": "hi"}], model="gpt-4o-mini")

            from toy_agent.llm.types import LLMError

            with pytest.raises(LLMError, match="rate limit"):
                adapter.chat(request)


class TestOpenAIAdapterStream:
    def test_stream_yields_content_chunks(self):
        """Streaming yields content deltas."""
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].delta.tool_calls = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"
        chunk2.choices[0].delta.tool_calls = None

        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta.content = None
        chunk3.choices[0].delta.tool_calls = None
        chunk3.choices[0].finish_reason = "stop"

        with patch("toy_agent.llm.openai_adapter.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])
            mock_cls.return_value = mock_client

            adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o-mini")
            request = ChatRequest(messages=[{"role": "user", "content": "hi"}], model="gpt-4o-mini", stream=True)
            chunks = list(adapter.chat_stream(request))

        assert len(chunks) == 3
        assert chunks[0].delta_content == "Hello"
        assert chunks[1].delta_content == " world"
        assert chunks[2].finish_reason == "stop"

    def test_stream_yields_tool_call_chunks(self):
        """Streaming yields tool call deltas."""
        tc1 = MagicMock()
        tc1.index = 0
        tc1.id = "tc_1"
        tc1.function.name = "my_tool"
        tc1.function.arguments = None

        tc2 = MagicMock()
        tc2.index = 0
        tc2.id = None
        tc2.function.name = None
        tc2.function.arguments = '{"key":'

        tc3 = MagicMock()
        tc3.index = 0
        tc3.id = None
        tc3.function.name = None
        tc3.function.arguments = '"val"}'

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = None
        chunk1.choices[0].delta.tool_calls = [tc1, tc2]

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = None
        chunk2.choices[0].delta.tool_calls = [tc3]

        with patch("toy_agent.llm.openai_adapter.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])
            mock_cls.return_value = mock_client

            adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o-mini")
            request = ChatRequest(messages=[{"role": "user", "content": "go"}], model="gpt-4o-mini", stream=True)
            chunks = list(adapter.chat_stream(request))

        assert len(chunks) == 2
        assert chunks[0].delta_tool_calls[0].id == "tc_1"
        assert chunks[0].delta_tool_calls[0].name == "my_tool"
        assert chunks[0].delta_tool_calls[1].arguments == '{"key":'
        assert chunks[1].delta_tool_calls[0].arguments == '"val"}'
