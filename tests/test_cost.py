"""Tests for CostTracker."""

from unittest.mock import MagicMock

import pytest

from toy_agent.agent import Agent
from toy_agent.cost import CostTracker


def _make_response_with_usage(prompt=100, completion=50):
    """Create a mock API response with token usage."""
    response = MagicMock()
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    usage.total_tokens = prompt + completion
    response.usage = usage
    msg = MagicMock()
    msg.content = "test"
    msg.tool_calls = None
    response.choices = [MagicMock(message=msg)]
    return response


class TestCostTrackerBasic:
    def test_accumulates_tokens(self):
        tracker = CostTracker(model="gpt-4o-mini")

        r1 = _make_response_with_usage(prompt=100, completion=50)
        r2 = _make_response_with_usage(prompt=200, completion=80)

        tracker.on_llm_response(message={}, usage=_usage_from_response(r1))
        tracker.on_llm_response(message={}, usage=_usage_from_response(r2))

        assert tracker.prompt_tokens == 300
        assert tracker.completion_tokens == 130
        assert tracker.total_tokens == 430

    def test_calculates_cost(self):
        tracker = CostTracker(model="gpt-4o-mini")
        # gpt-4o-mini: $0.00015/1K input, $0.0006/1K output

        r = _make_response_with_usage(prompt=1000, completion=500)
        tracker.on_llm_response(message={}, usage=_usage_from_response(r))

        expected = (1000 / 1000) * 0.00015 + (500 / 1000) * 0.0006
        assert abs(tracker.total_cost - expected) < 1e-9

    def test_custom_pricing(self):
        tracker = CostTracker(model="custom", price_per_1k=(0.001, 0.002))

        r = _make_response_with_usage(prompt=1000, completion=1000)
        tracker.on_llm_response(message={}, usage=_usage_from_response(r))

        expected = 0.001 + 0.002
        assert abs(tracker.total_cost - expected) < 1e-9

    def test_none_usage_is_ignored(self):
        tracker = CostTracker()
        tracker.on_llm_response(message={}, usage=None)
        assert tracker.total_tokens == 0
        assert tracker.total_cost == 0.0

    def test_summary_format(self):
        tracker = CostTracker()
        r = _make_response_with_usage(prompt=100, completion=50)
        tracker.on_llm_response(message={}, usage=_usage_from_response(r))

        s = tracker.summary()
        assert "150" in s
        assert "prompt: 100" in s
        assert "completion: 50" in s
        assert "$" in s


class TestCostTrackerWithAgent:
    @pytest.mark.anyio
    async def test_tracks_usage_from_agent_run(self):
        """CostTracker receives usage data when wired into an Agent."""
        client = MagicMock()

        response = MagicMock()
        response.content = "test response"
        response.tool_calls = None
        response.usage = MagicMock()
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50
        response.usage.total_tokens = 150
        client.chat.return_value = response

        tracker = CostTracker(model="gpt-4o-mini")
        agent = Agent(client=client, hooks=[tracker])
        await agent.run("hello")

        assert tracker.prompt_tokens == 100
        assert tracker.completion_tokens == 50


class TestModelPrices:
    def test_known_model_prices(self):
        tracker = CostTracker(model="gpt-4o-mini")
        assert tracker.price_per_1k == (0.00015, 0.0006)

    def test_unknown_model_fallback(self):
        tracker = CostTracker(model="unknown-model-xyz")
        assert tracker.price_per_1k == (0.001, 0.003)


def _usage_from_response(response) -> dict:
    """Convert a mock response's usage to a plain dict."""
    return {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
