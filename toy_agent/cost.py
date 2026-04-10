"""CostTracker: accumulates token usage and cost from API responses.

Zero extra API calls — receives usage data from the Agent's on_llm_response
hook event and maintains cumulative totals.

Usage:
    tracker = CostTracker(model="gpt-4o-mini")
    agent = Agent(client=client, hooks=[ConsoleHook(), tracker])
    await agent.run("hello")
    print(tracker.summary())
"""

from __future__ import annotations

from toy_agent.hooks import AgentHook

# Prices per 1K tokens (input / output) in USD
MODEL_PRICES: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    # Anthropic
    "claude-sonnet-4-20250514": (0.003, 0.015),
    "claude-opus-4-20250514": (0.015, 0.075),
    "claude-haiku-4-5-20251001": (0.0008, 0.004),
    # Google
    "gemini-2.0-flash": (0.0001, 0.0004),
    "gemini-2.5-pro": (0.00125, 0.01),
}


class CostTracker(AgentHook):
    """Track token usage and cost across an agent session."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        price_per_1k: tuple[float, float] | None = None,
    ):
        self.model = model
        self.price_per_1k = price_per_1k or MODEL_PRICES.get(model, (0.001, 0.003))
        self.total_tokens: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_cost: float = 0.0

    def on_llm_response(self, *, message: dict, usage: dict | None = None) -> None:
        """Accumulate token counts from API response usage data."""
        if usage is None:
            return

        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        tt = usage.get("total_tokens", pt + ct)

        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.total_tokens += tt

        in_cost = (pt / 1000) * self.price_per_1k[0]
        out_cost = (ct / 1000) * self.price_per_1k[1]
        self.total_cost += in_cost + out_cost

    def summary(self) -> str:
        return (
            f"[cost] tokens: {self.total_tokens:,} "
            f"(prompt: {self.prompt_tokens:,}, completion: {self.completion_tokens:,}) "
            f"| cost: ${self.total_cost:.4f}"
        )
