"""Unit tests for ContextCompressor."""

from unittest.mock import MagicMock

import pytest

from toy_agent.context import ContextCompressor


class TestTokenCounting:
    def test_count_tokens_basic(self):
        compressor = ContextCompressor(client=MagicMock(), model="gpt-4o-mini")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        count = compressor.count_tokens(messages)
        assert count > 0

    def test_count_tokens_empty(self):
        compressor = ContextCompressor(client=MagicMock(), model="gpt-4o-mini")
        assert compressor.count_tokens([]) == 0

    def test_count_tokens_increases_with_content(self):
        compressor = ContextCompressor(client=MagicMock(), model="gpt-4o-mini")
        short = [{"role": "user", "content": "hi"}]
        long = [{"role": "user", "content": "This is a much longer message with many more words"}]
        assert compressor.count_tokens(long) > compressor.count_tokens(short)


class TestCompressNoop:
    def test_no_compress_when_under_limit(self):
        """Messages under token limit should be returned unchanged."""
        compressor = ContextCompressor(
            client=MagicMock(), model="gpt-4o-mini", token_limit=100000
        )
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = compressor.compress(messages)
        assert result == messages


class TestLevel1Compression:
    def test_split_turns(self):
        compressor = ContextCompressor(client=MagicMock(), model="gpt-4o-mini")
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "how are you"},
            {"role": "assistant", "content": "fine"},
        ]
        turns = compressor._split_turns(messages)
        # system + user1 are one turn, user2 is a second turn
        assert len(turns) == 3

    def test_level1_compresses_old_turns(self):
        """Level 1 should compress all turns except the most recent."""
        compressor = ContextCompressor(client=MagicMock(), model="gpt-4o-mini")
        # Mock _summarize to avoid real LLM call
        compressor._summarize = lambda msgs: "mocked summary"

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "response 1"},
            {"role": "user", "content": "turn 2"},
            {"role": "assistant", "content": "response 2"},
        ]
        result = compressor._level1(messages)

        # system + user1 + summary + user2 + assistant2
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "turn 1"
        assert "[Turn Summary]" in result[2]["content"]
        assert result[3]["content"] == "turn 2"
        assert result[4]["content"] == "response 2"

    def test_level1_keeps_tool_calls_in_summary(self):
        """Tool call details should be summarized, not kept raw."""
        compressor = ContextCompressor(client=MagicMock(), model="gpt-4o-mini")
        compressor._summarize = lambda msgs: "read file, got contents"

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "read the file"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "/tmp/a.txt"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "file contents here"},
            {"role": "user", "content": "now summarize"},
        ]
        result = compressor._level1(messages)

        # The tool_calls and tool messages should be replaced by a summary
        has_tool_calls = any(m.get("tool_calls") for m in result)
        has_tool_role = any(m.get("role") == "tool" for m in result)
        assert not has_tool_calls
        assert not has_tool_role

    def test_level1_no_compress_single_turn(self):
        """Single turn should not be compressed."""
        compressor = ContextCompressor(client=MagicMock(), model="gpt-4o-mini")
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        result = compressor._level1(messages)
        assert result == messages

    def test_cooldown_prevents_repeated_compression(self):
        """After compression, next compress() call should skip."""
        compressor = ContextCompressor(
            client=MagicMock(), model="gpt-4o-mini", token_limit=1  # very low to force trigger
        )
        compressor._summarize = lambda msgs: "summary"

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "there"},
        ]
        result1 = compressor.compress(messages)
        assert compressor._cooldown is True

        # Second call should return same messages (cooldown active)
        result2 = compressor.compress(messages)
        assert result2 == messages  # unchanged, cooldown skipped compression
