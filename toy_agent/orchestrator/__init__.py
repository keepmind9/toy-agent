"""Orchestration patterns for multi-agent collaboration.

Three patterns that demonstrate different ways multiple agents can work together:
- RouterOrchestrator: LLM picks the best agent for the task
- SequentialOrchestrator: agents run in a pipeline, each feeding the next
- ParallelOrchestrator: agents run concurrently, results are merged
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toy_agent.agent import Agent


@dataclass
class AgentDef:
    """An agent with a name and description, used for orchestration."""

    name: str
    description: str
    agent: Agent


from toy_agent.orchestrator.parallel import ParallelOrchestrator  # noqa: E402
from toy_agent.orchestrator.router import RouterOrchestrator  # noqa: E402
from toy_agent.orchestrator.sequential import SequentialOrchestrator  # noqa: E402

__all__ = [
    "AgentDef",
    "ParallelOrchestrator",
    "RouterOrchestrator",
    "SequentialOrchestrator",
]
