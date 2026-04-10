"""SequentialOrchestrator: runs agents in a pipeline where each receives the previous output.

The simplest orchestration pattern — agents execute in order, forming a processing
pipeline. The output of one agent becomes the input of the next.

Flow:
  1. First agent receives the original task
  2. Each subsequent agent receives the previous agent's output
  3. Last agent's output is the final result
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toy_agent.agent import Agent


class SequentialOrchestrator:
    """Runs agents in sequence, chaining outputs."""

    def __init__(self, agents: list[Agent]):
        self.agents = agents

    async def run(self, task: str) -> str:
        result = task
        for i, agent in enumerate(self.agents):
            print(f"  [sequential] step {i + 1}/{len(self.agents)}")
            result = await agent.run(result)
        return result
