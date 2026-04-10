"""Test fixtures for all tests."""

from __future__ import annotations

import pytest


class _MockEncoding:
    """Minimal tiktoken-like encoding for tests (avoids network download).

    This mock approximates token counts by treating each word as ~1.3 tokens
    plus a fixed per-message overhead, which is sufficient for test assertions.
    """

    def __init__(self) -> None:
        # Simple whitespace tokenizer as a rough baseline
        pass

    def encode(self, text: str) -> list[int]:
        """Approximate token count: words * 1.3, capped."""
        if not text:
            return []
        words = text.split()
        return list(range(max(1, int(len(words) * 1.3))))


_mock_encoding = _MockEncoding()


@pytest.fixture(autouse=True)
def mock_tiktoken_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch _get_encoding in context module to return a fast mock.

    This prevents tests from blocking on tiktoken network downloads
    (o200k_base / cl100k_base files from Azure blob storage).
    """
    import toy_agent.context as ctx_module

    monkeypatch.setattr(ctx_module, "_get_encoding", lambda model: _mock_encoding)
