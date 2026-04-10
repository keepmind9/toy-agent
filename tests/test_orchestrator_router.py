"""Tests for RouterOrchestrator."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from toy_agent.agent import Agent
from toy_agent.orchestrator import AgentDef
from toy_agent.orchestrator.router import RouterOrchestrator


def _mock_agent(name: str = "test_agent") -> MagicMock:
    agent = MagicMock(spec=Agent)
    agent.run = AsyncMock(return_value=f"{name} response")
    return agent


class TestRouterOrchestrator:
    @pytest.mark.anyio
    async def test_routes_to_correct_agent(self):
        coding = _mock_agent("coding")
        writing = _mock_agent("writing")

        client = MagicMock()
        router_msg = MagicMock()
        router_msg.content = "coding"
        client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=router_msg)])

        router = RouterOrchestrator(
            client=client,
            model="gpt-4o-mini",
            agents=[
                AgentDef(name="coding", description="Writes code", agent=coding),
                AgentDef(name="writing", description="Writes articles", agent=writing),
            ],
        )

        result = await router.run("write a quicksort")

        coding.run.assert_awaited_once_with("write a quicksort")
        writing.run.assert_not_awaited()
        assert result == "coding response"

    @pytest.mark.anyio
    async def test_falls_back_to_first_on_unknown(self):
        agent_a = _mock_agent("agent_a")
        agent_b = _mock_agent("agent_b")

        client = MagicMock()
        router_msg = MagicMock()
        router_msg.content = "unknown_agent"
        client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=router_msg)])

        router = RouterOrchestrator(
            client=client,
            model="gpt-4o-mini",
            agents=[
                AgentDef(name="agent_a", description="Does A", agent=agent_a),
                AgentDef(name="agent_b", description="Does B", agent=agent_b),
            ],
        )

        result = await router.run("do something")

        agent_a.run.assert_awaited_once()
        assert result == "agent_a response"
