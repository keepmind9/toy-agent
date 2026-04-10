"""ParallelOrchestrator: runs multiple agents concurrently on the same task.

Uses asyncio.gather to execute all agents in parallel, then optionally merges
results with a single LLM aggregation call.

Flow:
  1. Launch all agents simultaneously via asyncio.gather
  2. Collect all results
  3. If client is provided: one LLM call to synthesize; otherwise concatenate
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toy_agent.agent import Agent


class ParallelOrchestrator:
    """Runs agents in parallel and aggregates results."""

    def __init__(
        self,
        agents: list[Agent],
        client=None,
        model: str = "gpt-4o-mini",
    ):
        self.agents = agents
        self.client = client
        self.model = model

    async def run(self, task: str) -> str:
        if not self.agents:
            return ""

        print(f"  [parallel] running {len(self.agents)} agents concurrently")
        results = await asyncio.gather(*(a.run(task) for a in self.agents))

        if not self.client:
            return "\n\n---\n\n".join(results)

        # LLM aggregation
        responses = "\n\n".join(f"Response {i + 1}:\n{r}" for i, r in enumerate(results))
        prompt = f"Synthesize these responses into one best answer:\n\n{responses}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
