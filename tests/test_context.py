"""Unit tests for ContextCompressor and HermesContextCompressor."""

from unittest.mock import MagicMock

from toy_agent.context import ContextCompressor, HermesContextCompressor


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
        compressor = ContextCompressor(client=MagicMock(), model="gpt-4o-mini", token_limit=100000)
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
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "/tmp/a.txt"}'},
                    }
                ],
            },
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
            client=MagicMock(),
            model="gpt-4o-mini",
            token_limit=1,  # very low to force trigger
        )
        compressor._summarize = lambda msgs: "summary"

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "there"},
        ]
        compressor.compress(messages)
        assert compressor._cooldown is True

        # Second call should return same messages (cooldown active)
        result2 = compressor.compress(messages)
        assert result2 == messages  # unchanged, cooldown skipped compression


# ---------------------------------------------------------------------------
# HermesContextCompressor tests
# ---------------------------------------------------------------------------


def _make_messages_with_tools(n_turns=3, tool_result_size=500):
    """Create a message list with multiple turns and tool calls."""
    messages = [{"role": "system", "content": "You are helpful."}]
    for i in range(n_turns):
        messages.append({"role": "user", "content": f"turn {i + 1} request"})
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"tc_{i}",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": f'{{"path": "/tmp/{i}.txt"}}'},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"tc_{i}",
                "content": "x" * tool_result_size,
            }
        )
        messages.append({"role": "assistant", "content": f"turn {i + 1} done"})
    return messages


class TestHermesToolPruning:
    def test_prunes_long_tool_results(self):
        """Tool results >200 chars in compressible region are replaced."""
        comp = HermesContextCompressor(client=MagicMock(), token_limit=1)
        messages = _make_messages_with_tools(3, tool_result_size=500)

        pruned = comp._prune_tool_outputs(messages)

        # First two tool results should be pruned (in compressible region)
        # Last tool result should be kept (in tail)
        tool_msgs = [m for m in pruned if m.get("role") == "tool"]
        pruned_count = sum(1 for m in tool_msgs if m["content"] == "[Old tool output cleared]")
        assert pruned_count >= 1

    def test_keeps_short_tool_results(self):
        """Short tool results are not pruned."""
        comp = HermesContextCompressor(client=MagicMock(), token_limit=100000)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "short result"},
        ]
        pruned = comp._prune_tool_outputs(messages)
        tool_msg = next(m for m in pruned if m.get("role") == "tool")
        assert tool_msg["content"] == "short result"


class TestHermesBoundaryDetermination:
    def test_protects_head(self):
        """First protect_head messages are always in head."""
        comp = HermesContextCompressor(client=MagicMock(), protect_head=3, token_limit=100000)
        messages = _make_messages_with_tools(5)
        head, compressible, tail = comp._find_boundaries(messages)
        assert len(head) >= 3
        assert head[0]["role"] == "system"

    def test_protects_tail_by_token_budget(self):
        """Tail is protected with sufficient token budget."""
        comp = HermesContextCompressor(client=MagicMock(), token_limit=100000, tail_ratio=0.15)
        messages = _make_messages_with_tools(5)
        _, _, tail = comp._find_boundaries(messages)
        # Tail should contain at least the last turn
        assert len(tail) >= 2

    def test_aligns_boundary_forward_past_tool_results(self):
        """Boundary skips past orphaned tool results."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "orphan", "content": "orphaned"},
            {"role": "user", "content": "next"},
        ]
        comp = HermesContextCompressor(client=MagicMock(), protect_head=2)
        result = comp._align_boundary_forward(messages, 2)
        assert result == 3  # skipped past tool message

    def test_returns_empty_compressible_for_short_messages(self):
        """Short messages have no compressible region."""
        comp = HermesContextCompressor(client=MagicMock(), protect_head=3)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        _, compressible, _ = comp._find_boundaries(messages)
        assert compressible == []


class TestHermesSummaryGeneration:
    def test_generates_structured_summary(self):
        """Summary LLM is called with structured prompt."""
        client = MagicMock()
        response = MagicMock()
        response.choices[0].message.content = "## Goal\nTest goal\n## Progress\n### Done\n- stuff"
        client.chat.completions.create.return_value = response

        comp = HermesContextCompressor(client=client)
        messages = [
            {"role": "user", "content": "do things"},
            {"role": "assistant", "content": "done"},
        ]
        summary = comp._generate_summary(messages)

        assert "## Goal" in summary
        call_args = client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "TURNS TO SUMMARIZE" in prompt

    def test_iterative_update_uses_previous_summary(self):
        """Second compression uses iterative update prompt."""
        client = MagicMock()
        response = MagicMock()
        response.choices[0].message.content = "## Goal\nUpdated"
        client.chat.completions.create.return_value = response

        comp = HermesContextCompressor(client=client)
        comp._previous_summary = "## Goal\nOriginal"
        comp._generate_summary([{"role": "user", "content": "more work"}])

        call_args = client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "PREVIOUS SUMMARY" in prompt
        assert "Original" in prompt

    def test_fallback_on_llm_failure(self):
        """Falls back to plain text when LLM fails."""
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API down")

        comp = HermesContextCompressor(client=client)
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "data"},
        ]
        summary = comp._generate_summary(messages)
        assert "read_file" in summary


class TestHermesAssembly:
    def test_assembles_head_summary_tail(self):
        """Basic assembly creates head + summary + tail."""
        comp = HermesContextCompressor(client=MagicMock())
        head = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        tail = [{"role": "user", "content": "latest"}]

        result = comp._assemble(head, "summary text", tail)
        assert result[0]["role"] == "system"
        # summary is at index 2 (after head), tail at index 3
        summary_msgs = [m for m in result if "summary text" in m.get("content", "")]
        assert len(summary_msgs) == 1

    def test_role_alternation_avoids_double_collision(self):
        """When head ends with assistant and tail starts with user, summary merges into tail."""
        comp = HermesContextCompressor(client=MagicMock())
        head = [{"role": "system", "content": "sys"}, {"role": "assistant", "content": "response"}]
        tail = [{"role": "user", "content": "next question"}]

        result = comp._assemble(head, "summary text", tail)
        # Summary should be merged into first tail message, not standalone
        assert any("summary text" in m.get("content", "") for m in result)
        assert len(result) == len(head) + len(tail)  # no extra message


class TestHermesSanitization:
    def test_removes_orphaned_tool_results(self):
        """Tool results without matching tool_calls are removed."""
        comp = HermesContextCompressor(client=MagicMock())
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "tool_call_id": "orphan", "content": "no matching call"},
            {"role": "user", "content": "hi"},
        ]
        result = comp._sanitize_tool_pairs(messages)
        assert not any(m.get("tool_call_id") == "orphan" for m in result)

    def test_adds_stubs_for_missing_tool_results(self):
        """Tool calls without results get stub results."""
        comp = HermesContextCompressor(client=MagicMock())
        messages = [
            {"role": "system", "content": "sys"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
            },
            {"role": "user", "content": "next"},
        ]
        result = comp._sanitize_tool_pairs(messages)
        stubs = [m for m in result if m.get("role") == "tool" and "removed" in m.get("content", "")]
        assert len(stubs) == 1
        assert stubs[0]["tool_call_id"] == "tc1"


class TestHermesIntegration:
    def test_no_compress_under_limit(self):
        """Messages under limit are returned unchanged."""
        comp = HermesContextCompressor(client=MagicMock(), token_limit=100000)
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        result = comp.compress(messages)
        assert result == messages

    def test_full_compression_flow(self):
        """End-to-end: compression runs all 4 phases."""
        client = MagicMock()
        response = MagicMock()
        response.choices[0].message.content = "## Goal\nTest\n## Progress\n### Done\n- all"
        client.chat.completions.create.return_value = response

        comp = HermesContextCompressor(client=client, token_limit=1, protect_head=2)
        messages = _make_messages_with_tools(4, tool_result_size=500)
        result = comp.compress(messages)

        # Should have fewer messages than original
        assert len(result) < len(messages)
        # Should contain a summary
        assert any("[Context Summary]" in m.get("content", "") for m in result)
        # Head should be preserved
        assert result[0]["role"] == "system"
