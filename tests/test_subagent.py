"""Unit tests for SubAgentTool."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.toy_agent.subagent import SubAgentTool


class TestSubAgentToolSchema:
    def test_schema_has_correct_name_and_description(self):
        agent = MagicMock()
        tool = SubAgentTool(name="researcher", description="Research a topic", agent=agent)

        assert tool.schema["function"]["name"] == "researcher"
        assert tool.schema["function"]["description"] == "Research a topic"

    def test_schema_has_task_parameter(self):
        agent = MagicMock()
        tool = SubAgentTool(name="researcher", description="desc", agent=agent)

        params = tool.schema["function"]["parameters"]
        assert "task" in params["properties"]
        assert params["required"] == ["task"]

    def test_schema_function_name_property(self):
        agent = MagicMock()
        tool = SubAgentTool(name="coder", description="Write code", agent=agent)
        assert tool.name == "coder"


class TestSubAgentToolExecute:
    @pytest.mark.anyio
    async def test_execute_calls_agent_run(self):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value="research findings")

        tool = SubAgentTool(name="researcher", description="desc", agent=mock_agent)

        result = await tool.execute(task="look into X")

        mock_agent.run.assert_awaited_once_with("look into X", max_turns=10)
        assert result == "research findings"

    @pytest.mark.anyio
    async def test_execute_respects_max_turns(self):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value="done")

        tool = SubAgentTool(
            name="researcher",
            description="desc",
            agent=mock_agent,
            max_turns=5,
        )

        await tool.execute(task="test")

        mock_agent.run.assert_awaited_once_with("test", max_turns=5)

    @pytest.mark.anyio
    async def test_execute_catches_exception(self):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("boom"))

        tool = SubAgentTool(name="researcher", description="desc", agent=mock_agent)

        result = await tool.execute(task="test")

        assert "[subagent error]" in result
        assert "boom" in result
