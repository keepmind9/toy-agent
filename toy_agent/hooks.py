"""AgentHook: pluggable observability callbacks for the agent loop."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toy_agent.planning import Plan, PlanStep


class AgentHook(ABC):
    """Base class for agent observability hooks.

    Subclass and override any methods to add behaviour at specific
    points in the agent loop. All methods are no-ops by default.

    Args passed to each method are keyword-only for forward compatibility.
    """

    def on_message(self, *, role: str, content: str) -> None:
        """Called when a message is appended to the conversation history."""
        ...

    def on_llm_request(self, *, messages: list[dict]) -> None:
        """Called just before sending a request to the LLM API."""
        ...

    def on_llm_response(self, *, message: dict) -> None:
        """Called after receiving an LLM response (before tool processing)."""
        ...

    def on_tool_call(self, *, tool_name: str, arguments: dict) -> None:
        """Called before executing a tool."""
        ...

    def on_tool_result(self, *, tool_name: str, result: str) -> None:
        """Called after a tool returns a result."""
        ...

    def on_tool_retry(self, *, tool_name: str, attempt: int, error: str) -> None:
        """Called when a tool failed and will be retried."""
        ...

    def on_compress(self, *, before_count: int, after_count: int) -> None:
        """Called after context compression finishes."""
        ...

    def on_error(self, *, error: str) -> None:
        """Called when an error occurs."""
        ...

    def on_plan(self, *, plan: Plan) -> None:
        """Called when a plan is generated and injected into context."""
        ...

    def on_plan_step(self, *, step: PlanStep, plan: Plan) -> None:
        """Called when a plan step status changes."""
        ...

    def on_plan_done(self, *, plan: Plan) -> None:
        """Called when all plan steps are completed."""
        ...

    async def on_before_loop(self, *, agent) -> None:
        """Called after user message is appended, before the agent ReAct loop starts."""
        ...


class ConsoleHook(AgentHook):
    """Default hook that prints human-readable event summaries to stdout.

    Reproduces the console output behaviour that existed before hooks.
    """

    def on_message(self, *, role: str, content: str) -> None:
        if role == "user" and content:
            print(f"You: {content[:100]}")

    def on_tool_call(self, *, tool_name: str, arguments: dict) -> None:
        print(f"  [tool] {tool_name}({arguments})")

    def on_tool_retry(self, *, tool_name: str, attempt: int, error: str) -> None:
        print(f"  [tool-retry] {tool_name} attempt {attempt + 1}: {error}")

    def on_error(self, *, error: str) -> None:
        print(f"[error] {error}")

    def on_plan(self, *, plan) -> None:
        lines = [f"[plan] Goal: {plan.goal}", "[plan] Steps:"]
        for step in plan.steps:
            lines.append(f"  {step.id}. {step.description}")
        print("\n".join(lines))
