"""RouterOrchestrator: uses one LLM call to classify intent and delegate to the best agent.

This is the simplest form of multi-agent routing — a single classification call
that picks which specialized agent should handle the user's request.

Flow:
  1. Build a prompt listing all agents with their name and description
  2. LLM returns the name of the most suitable agent
  3. Delegate the task to that agent
"""

from __future__ import annotations

from toy_agent.llm.types import ChatRequest
from toy_agent.orchestrator import AgentDef


class RouterOrchestrator:
    """Routes tasks to the best agent via LLM classification."""

    def __init__(self, client, model: str, agents: list[AgentDef]):
        self.client = client
        self.model = model
        self.agents = {a.name: a for a in agents}

    async def run(self, task: str) -> str:
        agent_list = "\n".join(f"- {name}: {a.description}" for name, a in self.agents.items())

        prompt = (
            "Given the following task, which agent should handle it?\n\n"
            f"Agents:\n{agent_list}\n\n"
            f"Task: {task}\n\n"
            "Reply with ONLY the agent name, nothing else."
        )

        response = self.client.chat(
            ChatRequest(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        )
        chosen = response.content.strip().lower()

        for name, agent_def in self.agents.items():
            if name.lower() in chosen:
                print(f"  [router] → {name}")
                return await agent_def.agent.run(task)

        # Fallback: use first agent if no match
        fallback = next(iter(self.agents.values()))
        print(f"  [router] → {fallback.name} (fallback)")
        return await fallback.agent.run(task)
