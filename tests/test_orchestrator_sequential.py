"""Tests for SequentialOrchestrator."""

from unittest.mock import MagicMock

import pytest

from toy_agent.agent import Agent
from toy_agent.orchestrator.sequential import SequentialOrchestrator


def _mock_agent(name: str) -> MagicMock:
    agent = MagicMock(spec=Agent)

    async def fake_run(task: str) -> str:
        return f"{name} processed: {task}"

    agent.run = fake_run
    return agent


class TestSequentialOrchestrator:
    @pytest.mark.anyio
    async def test_chains_agents_in_order(self):
        agent_a = _mock_agent("A")
        agent_b = _mock_agent("B")
        agent_c = _mock_agent("C")

        pipeline = SequentialOrchestrator(agents=[agent_a, agent_b, agent_c])
        result = await pipeline.run("start")

        assert "C processed:" in result
        assert "B processed: A processed: start" in result

    @pytest.mark.anyio
    async def test_single_agent(self):
        agent = _mock_agent("only")

        pipeline = SequentialOrchestrator(agents=[agent])
        result = await pipeline.run("task")

        assert result == "only processed: task"

    @pytest.mark.anyio
    async def test_empty_agents_returns_task(self):
        pipeline = SequentialOrchestrator(agents=[])
        result = await pipeline.run("task")

        assert result == "task"
