"""Planning: generate and inject multi-step plans before agent execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from openai import OpenAI

from toy_agent.hooks import AgentHook


@dataclass
class PlanStep:
    """A single step in a plan."""

    id: int
    description: str
    status: str = "pending"


@dataclass
class Plan:
    """A multi-step plan for a complex task."""

    goal: str
    steps: list[PlanStep] = field(default_factory=list)


_CLASSIFY_PROMPT = """\
You are a task classifier. Determine if this task requires multi-step planning.

Needs planning: complex analysis, multi-step workflows, tasks requiring multiple tool calls.
Does NOT need planning: simple questions, single operations, factual lookups, greetings.

Task: {task}

Respond with JSON: {{"needs_plan": true}} or {{"needs_plan": false}}"""

_GENERATE_PROMPT = """\
Create a concise step-by-step plan for this task. Each step should be a clear action.

Task: {task}

Available tools: {tool_names}

Respond with JSON:
{{"goal": "description of the goal", "steps": ["step 1", "step 2", ...]}}"""

_EXECUTION_CONTEXT = """\
I've created a plan to accomplish your task:

{plan_text}

I will execute this plan step by step, adapting as needed based on intermediate results."""


class PlanHook(AgentHook):
    """Hook that generates and injects plans before agent execution.

    Args:
        client: OpenAI client for plan generation LLM calls.
        model: Model to use for planning.
        auto: If True (default), LLM auto-detects when planning is needed.
              If False, planning only triggers when plan=True is passed to run().
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        auto: bool = True,
    ):
        self.client = client
        self.model = model
        self.auto = auto
        self._plan_override: bool | None = None

    def set_plan_override(self, value: bool | None) -> None:
        """Set explicit plan override from agent.run(plan=...)."""
        self._plan_override = value

    async def on_before_loop(self, *, agent) -> None:
        """Generate and inject a plan if needed."""
        try:
            if self._plan_override is False:
                return

            auto_classify = self._plan_override is None and self.auto

            user_msg = next(
                (m.get("content", "") for m in reversed(agent.messages) if m.get("role") == "user"),
                "",
            )
            if not user_msg:
                return

            if auto_classify and not await self._check_needs_plan(user_msg):
                return

            tool_names = [t.name for t in agent.tools]
            plan = await self._generate_plan(user_msg, tool_names)
            if not plan:
                return

            agent._emit("on_plan", plan=plan)
            plan_text = self._format_plan(plan)
            agent.messages.append({"role": "assistant", "content": _EXECUTION_CONTEXT.format(plan_text=plan_text)})
        finally:
            self._plan_override = None

    async def _check_needs_plan(self, task: str) -> bool:
        """Ask LLM if the task needs multi-step planning."""
        prompt = _CLASSIFY_PROMPT.format(task=task)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            return bool(data.get("needs_plan", False))
        except Exception:
            return False

    async def _generate_plan(self, task: str, tool_names: list[str]) -> Plan | None:
        """Ask LLM to generate a step-by-step plan."""
        prompt = _GENERATE_PROMPT.format(task=task, tool_names=", ".join(tool_names) or "none")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            steps = [PlanStep(id=i + 1, description=desc) for i, desc in enumerate(data.get("steps", []))]
            return Plan(goal=data.get("goal", task), steps=steps)
        except Exception:
            return None

    def _format_plan(self, plan: Plan) -> str:
        """Format plan as readable text for injection."""
        lines = [f"Goal: {plan.goal}", "", "Steps:"]
        for step in plan.steps:
            lines.append(f"  {step.id}. {step.description}")
        return "\n".join(lines)
