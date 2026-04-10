"""Tests for Anthropic adapter."""

from unittest.mock import MagicMock, patch

import pytest

from toy_agent.llm.anthropic_adapter import AnthropicAdapter
from toy_agent.llm.types import ChatRequest, ToolDef


def _mock_message(content=None, tool_use=None, stop_reason="end_turn"):
    """Create a mock Anthropic message."""
    msg = MagicMock()
    msg.content = []
    if content:
        block = MagicMock()
        block.type = "text"
        block.text = content
        msg.content.append(block)
    if tool_use:
        for tu in tool_use:
            block = MagicMock()
            block.type = "tool_use"
            block.id = tu["id"]
            block.name = tu["name"]
            block.input = tu["input"]
            msg.content.append(block)
    msg.stop_reason = stop_reason
    msg.usage = MagicMock()
    msg.usage.input_tokens = 10
    msg.usage.output_tokens = 5
    return msg


class TestAnthropicAdapterChat:
    @patch("toy_agent.llm.anthropic_adapter.importlib.import_module")
    def test_basic_chat(self, mock_import):
        """Basic chat returns content and usage."""
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(content="Hello!")
        mock_anthropic.Anthropic.return_value = mock_client
        mock_import.return_value = mock_anthropic

        adapter = AnthropicAdapter(api_key="sk-ant-test", model="claude-sonnet-4-20250514")
        request = ChatRequest(messages=[{"role": "user", "content": "hi"}], model="claude-sonnet-4-20250514")
        response = adapter.chat(request)

        assert response.content == "Hello!"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5

    @patch("toy_agent.llm.anthropic_adapter.importlib.import_module")
    def test_chat_with_tools(self, mock_import):
        """Chat with tools returns tool_calls."""
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(
            content=None,
            tool_use=[{"id": "tc_1", "name": "read_file", "input": {"path": "/tmp/a.txt"}}],
            stop_reason="tool_use",
        )
        mock_anthropic.Anthropic.return_value = mock_client
        mock_import.return_value = mock_anthropic

        adapter = AnthropicAdapter(api_key="sk-ant-test", model="claude-sonnet-4-20250514")
        tools = [
            ToolDef(
                name="read_file",
                description="Read file",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            )
        ]
        request = ChatRequest(
            messages=[{"role": "user", "content": "read file"}], model="claude-sonnet-4-20250514", tools=tools
        )
        response = adapter.chat(request)

        assert response.content is None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "tc_1"
        assert response.tool_calls[0].name == "read_file"
        assert '"path"' in response.tool_calls[0].arguments

    @patch("toy_agent.llm.anthropic_adapter.importlib.import_module")
    def test_chat_propagates_error(self, mock_import):
        """Anthropic APIError is converted to LLMError."""
        mock_anthropic = MagicMock()

        class FakeAPIError(Exception):
            def __init__(self):
                self.status_code = 429
                self.message = "rate limit"
                super().__init__("rate limit")

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = FakeAPIError()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_import.return_value = mock_anthropic

        adapter = AnthropicAdapter(api_key="sk-ant-test", model="claude-sonnet-4-20250514")
        request = ChatRequest(messages=[{"role": "user", "content": "hi"}], model="claude-sonnet-4-20250514")

        from toy_agent.llm.types import LLMError

        with pytest.raises(LLMError, match="rate limit"):
            adapter.chat(request)

    @patch("toy_agent.llm.anthropic_adapter.importlib.import_module")
    def test_system_message_extracted(self, mock_import):
        """System messages are extracted from messages list and passed separately."""
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(content="ok")
        mock_anthropic.Anthropic.return_value = mock_client
        mock_import.return_value = mock_anthropic

        adapter = AnthropicAdapter(api_key="sk-ant-test", model="claude-sonnet-4-20250514")
        request = ChatRequest(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ],
            model="claude-sonnet-4-20250514",
        )
        adapter.chat(request)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "You are helpful."
        # user message should NOT have role=system
        assert call_kwargs["messages"][0]["role"] == "user"


class TestAnthropicAdapterMissingSDK:
    def test_raises_clear_error_without_sdk(self):
        """Missing anthropic SDK produces clear error message."""
        with patch("toy_agent.llm.anthropic_adapter.importlib.import_module", side_effect=ImportError("No module")):
            with pytest.raises(ImportError, match="anthropic"):
                AnthropicAdapter(api_key="sk-ant", model="claude-sonnet-4-20250514")
