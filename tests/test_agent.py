"""Unit tests for Agent max_turns safety limit."""

from unittest.mock import MagicMock

import pytest

from src.toy_agent.agent import Agent
from src.toy_agent.subagent import SubAgentTool


def _make_tool_call_response(tool_call_id="tc_1", fn_name="test_tool", fn_args="{}"):
    """Create a mock response with a single tool_call."""
    message = MagicMock()
    message.tool_calls = [MagicMock()]
    message.tool_calls[0].id = tool_call_id
    message.tool_calls[0].function.name = fn_name
    message.tool_calls[0].function.arguments = fn_args
    message.content = None
    return message


def _make_text_response(text="done"):
    """Create a mock response with plain text (no tool_calls)."""
    message = MagicMock()
    message.tool_calls = None
    message.content = text
    return message


class TestMaxTurns:
    @pytest.mark.anyio
    async def test_stops_after_max_turns(self):
        """Agent should stop and return error after exceeding max_turns."""
        client = MagicMock()

        # Always return tool_calls (infinite loop if no limit)
        tool_response = _make_tool_call_response(fn_name="some_tool")
        client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=tool_response)])

        agent = Agent(client=client)
        result = await agent.run("test", max_turns=3)

        assert "max turns" in result.lower()
        # Should have made exactly 3 API calls (3 turns)
        assert client.chat.completions.create.call_count == 3

    @pytest.mark.anyio
    async def test_default_no_limit(self):
        """Without max_turns, agent runs until LLM stops calling tools."""
        client = MagicMock()

        # First call: tool_call, second call: text response
        tool_response = _make_tool_call_response(fn_name="some_tool")
        text_response = _make_text_response("final answer")

        client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=tool_response)]),
            MagicMock(choices=[MagicMock(message=text_response)]),
        ]

        agent = Agent(client=client)
        result = await agent.run("test")

        assert result == "final answer"
        assert client.chat.completions.create.call_count == 2


class TestSubagentInSystemPrompt:
    def test_subagents_listed_in_system_prompt(self):
        client = MagicMock()

        sub_agent = Agent(client=client, system="sub prompt")
        sub_tool = SubAgentTool(
            name="researcher",
            description="Research topics thoroughly",
            agent=sub_agent,
        )

        main_agent = Agent(
            client=client,
            tools=[sub_tool],
        )

        system = main_agent.system
        assert "researcher" in system
        assert "Research topics thoroughly" in system

    def test_subagents_separate_from_tools(self):
        client = MagicMock()

        sub_agent = Agent(client=client)
        sub_tool = SubAgentTool(
            name="researcher",
            description="Research topics",
            agent=sub_agent,
        )

        regular_tool = MagicMock()
        regular_tool.name = "read_file"
        regular_tool.schema = {
            "function": {
                "name": "read_file",
                "description": "Read a file",
            }
        }

        main_agent = Agent(
            client=client,
            tools=[regular_tool, sub_tool],
        )

        system = main_agent.system
        assert "Available tools:" in system
        assert "Available subagents" in system
        assert "read_file" in system
        assert "researcher" in system
