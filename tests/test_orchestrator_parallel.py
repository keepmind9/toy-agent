"""Tests for ParallelOrchestrator."""

import asyncio
from unittest.mock import MagicMock

import pytest

from toy_agent.agent import Agent
from toy_agent.orchestrator.parallel import ParallelOrchestrator


def _mock_agent(name: str, delay: float = 0) -> MagicMock:
    agent = MagicMock(spec=Agent)

    async def fake_run(task: str) -> str:
        if delay:
            await asyncio.sleep(delay)
        return f"{name}: {task}"

    agent.run = fake_run
    return agent


class TestParallelOrchestrator:
    @pytest.mark.anyio
    async def test_runs_all_agents_concurrently(self):
        agent_a = _mock_agent("A", delay=0.01)
        agent_b = _mock_agent("B", delay=0.01)

        parallel = ParallelOrchestrator(agents=[agent_a, agent_b])
        result = await parallel.run("test task")

        assert "A: test task" in result
        assert "B: test task" in result

    @pytest.mark.anyio
    async def test_aggregates_with_llm(self):
        agent_a = _mock_agent("A")
        agent_b = _mock_agent("B")

        client = MagicMock()
        client.chat.return_value = MagicMock(content="merged answer")

        parallel = ParallelOrchestrator(
            agents=[agent_a, agent_b],
            client=client,
            model="gpt-4o-mini",
        )
        result = await parallel.run("test")

        assert result == "merged answer"
        client.chat.assert_called_once()

    @pytest.mark.anyio
    async def test_without_aggregation_concatenates(self):
        agent_a = _mock_agent("A")
        agent_b = _mock_agent("B")

        parallel = ParallelOrchestrator(agents=[agent_a, agent_b])
        result = await parallel.run("task")

        assert "A: task" in result
        assert "B: task" in result
        assert "---" in result  # separator

    @pytest.mark.anyio
    async def test_empty_agents_returns_empty(self):
        parallel = ParallelOrchestrator(agents=[])
        result = await parallel.run("task")

        assert result == ""
