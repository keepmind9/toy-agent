"""Tests for Google (Gemini) adapter."""

from unittest.mock import MagicMock, patch

import pytest

from toy_agent.llm.google_adapter import GeminiAdapter
from toy_agent.llm.types import ChatRequest, ToolDef


def _mock_gemini_response(text="Hello!", tool_calls=None):
    """Create a mock Gemini response."""
    candidate = MagicMock()
    candidate.content = MagicMock()
    candidate.content.parts = [MagicMock()]
    candidate.content.parts[0].text = text
    candidate.finish_reason = MagicMock(name="STOP")

    usage = MagicMock()
    usage.prompt_token_count = 10
    usage.candidates_token_count = 5

    response = MagicMock()
    response.candidates = [candidate]
    response.usage_metadata = usage
    return response


class TestGeminiAdapterChat:
    @patch("toy_agent.llm.google_adapter.importlib.import_module")
    def test_basic_chat(self, mock_import):
        """Basic chat returns content and usage."""
        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_gemini_response(text="Hello!")
        mock_genai.Client.return_value = mock_client
        mock_import.return_value = mock_genai

        adapter = GeminiAdapter(api_key="test-key", model="gemini-2.0-flash")
        request = ChatRequest(messages=[{"role": "user", "content": "hi"}], model="gemini-2.0-flash")
        response = adapter.chat(request)

        assert response.content == "Hello!"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5

    @patch("toy_agent.llm.google_adapter.importlib.import_module")
    def test_chat_with_tools(self, mock_import):
        """Chat with tools returns tool_calls."""
        mock_genai = MagicMock()
        mock_client = MagicMock()

        # Tool call response
        candidate = MagicMock()
        part = MagicMock()
        part.text = None
        part.function_call = MagicMock()
        part.function_call.name = "read_file"
        part.function_call.args = {"path": "/tmp/a.txt"}
        candidate.content = MagicMock()
        candidate.content.parts = [part]
        candidate.finish_reason = MagicMock(name="STOP")
        usage = MagicMock()
        usage.prompt_token_count = 10
        usage.candidates_token_count = 5
        response = MagicMock()
        response.candidates = [candidate]
        response.usage_metadata = usage
        mock_client.models.generate_content.return_value = response
        mock_genai.Client.return_value = mock_client
        mock_import.return_value = mock_genai

        adapter = GeminiAdapter(api_key="test-key", model="gemini-2.0-flash")
        tools = [
            ToolDef(
                name="read_file",
                description="Read file",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            )
        ]
        request = ChatRequest(messages=[{"role": "user", "content": "read"}], model="gemini-2.0-flash", tools=tools)
        resp = adapter.chat(request)

        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "read_file"

    @patch("toy_agent.llm.google_adapter.importlib.import_module")
    def test_system_message_extracted(self, mock_import):
        """System messages are passed as system_instruction."""
        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_gemini_response(text="ok")
        mock_genai.Client.return_value = mock_client
        mock_import.return_value = mock_genai

        adapter = GeminiAdapter(api_key="test-key", model="gemini-2.0-flash")
        request = ChatRequest(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ],
            model="gemini-2.0-flash",
        )
        adapter.chat(request)

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        # System message should be in config.system_instruction
        assert call_kwargs["config"]["system_instruction"] == "You are helpful."
        # contents should have no system messages
        assert all(m.get("role") != "system" for m in call_kwargs["contents"])


class TestGeminiAdapterMissingSDK:
    def test_raises_clear_error_without_sdk(self):
        """Missing google-genai SDK produces clear error message."""
        with patch("toy_agent.llm.google_adapter.importlib.import_module", side_effect=ImportError("No module")):
            with pytest.raises(ImportError, match="google-genai"):
                GeminiAdapter(api_key="test", model="gemini-2.0-flash")
