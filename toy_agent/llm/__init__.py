"""Multi-provider LLM abstraction layer.

Provides a unified interface (LLMProtocol) for OpenAI, Anthropic (Claude),
and Google (Gemini) providers. Use create_llm_client() to instantiate the
correct adapter based on environment configuration.
"""

from __future__ import annotations

import os

from toy_agent.llm.protocol import LLMProtocol


def create_llm_client(provider: str | None = None) -> LLMProtocol:
    """Create an LLM client based on provider name or environment configuration.

    Environment variables:
        LLM_PROVIDER: openai | anthropic | gemini (default: openai)
        OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL: for OpenAI
        ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL / ANTHROPIC_MODEL: for Anthropic
        GOOGLE_API_KEY / GEMINI_BASE_URL / GEMINI_MODEL: for Gemini
    """
    provider = provider or os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        from toy_agent.llm.openai_adapter import OpenAIAdapter

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Set it via env var or .env file.")
        return OpenAIAdapter(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    if provider == "anthropic":
        from toy_agent.llm.anthropic_adapter import AnthropicAdapter

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set. Set it via env var or .env file.")
        return AnthropicAdapter(
            api_key=api_key,
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            base_url=os.getenv("ANTHROPIC_BASE_URL"),
        )

    if provider == "gemini":
        from toy_agent.llm.google_adapter import GeminiAdapter

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set. Set it via env var or .env file.")
        return GeminiAdapter(
            api_key=api_key,
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            base_url=os.getenv("GEMINI_BASE_URL"),
        )

    raise ValueError(f"Unsupported provider: {provider}. Supported: openai, anthropic, gemini")


__all__ = ["create_llm_client", "LLMProtocol"]
