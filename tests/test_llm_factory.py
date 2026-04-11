"""Tests for LLM client factory."""

import os
from unittest.mock import MagicMock, patch

import pytest

from toy_agent.llm import create_llm_client


class TestFactoryOpenAI:
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "LLM_PROVIDER": "openai"})
    def test_creates_openai_adapter(self):
        adapter = create_llm_client()
        from toy_agent.llm.openai_adapter import OpenAIAdapter

        assert isinstance(adapter, OpenAIAdapter)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    def test_default_provider_is_openai(self):
        os.environ.pop("LLM_PROVIDER", None)
        adapter = create_llm_client()
        from toy_agent.llm.openai_adapter import OpenAIAdapter

        assert isinstance(adapter, OpenAIAdapter)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "OPENAI_BASE_URL": "https://custom/v1"})
    def test_passes_base_url_to_openai(self):
        adapter = create_llm_client()
        assert adapter._extra.get("base_url") == "https://custom/v1"

    def test_raises_when_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("LLM_PROVIDER", None)
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                create_llm_client()


class TestFactoryAnthropic:
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test", "LLM_PROVIDER": "anthropic"})
    @patch("importlib.import_module")
    def test_creates_anthropic_adapter(self, mock_import):
        mock_import.return_value = MagicMock()
        adapter = create_llm_client()
        from toy_agent.llm.anthropic_adapter import AnthropicAdapter

        assert isinstance(adapter, AnthropicAdapter)

    @patch.dict(
        os.environ,
        {
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "ANTHROPIC_BASE_URL": "https://custom-anthropic/v1",
            "LLM_PROVIDER": "anthropic",
        },
    )
    @patch("importlib.import_module")
    def test_passes_base_url_to_anthropic(self, mock_import):
        mock_import.return_value = MagicMock()
        adapter = create_llm_client()
        assert adapter._extra.get("base_url") == "https://custom-anthropic/v1"


class TestFactoryGemini:
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key", "LLM_PROVIDER": "gemini"})
    @patch("importlib.import_module")
    def test_creates_gemini_adapter(self, mock_import):
        mock_import.return_value = MagicMock()
        adapter = create_llm_client()
        from toy_agent.llm.google_adapter import GeminiAdapter

        assert isinstance(adapter, GeminiAdapter)

    @patch.dict(
        os.environ,
        {"GOOGLE_API_KEY": "test-key", "GEMINI_BASE_URL": "https://custom-gemini/v1", "LLM_PROVIDER": "gemini"},
    )
    @patch("importlib.import_module")
    def test_passes_base_url_to_gemini(self, mock_import):
        mock_import.return_value = MagicMock()
        adapter = create_llm_client()
        assert adapter._extra.get("base_url") == "https://custom-gemini/v1"


class TestFactoryUnknown:
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "LLM_PROVIDER": "unknown_provider"})
    def test_raises_on_unknown_provider(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_llm_client()
