"""SubAgentTool: wraps an Agent as a callable Tool.

Enables hierarchical agent architectures where one agent can delegate
tasks to a specialized sub-agent through the standard tool dispatch
mechanism.
"""

from src.toy_agent.agent import Agent
from src.toy_agent.tools import Tool


class SubAgentTool(Tool):
    """A Tool that runs a complete agent loop internally."""

    def __init__(
        self,
        name: str,
        description: str,
        agent: Agent,
        max_turns: int = 10,
    ):
        self._agent = agent
        self._max_turns = max_turns
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task to delegate to this subagent",
                        },
                    },
                    "required": ["task"],
                },
            },
        }
        super().__init__(schema=schema, fn=self._run)

    async def _run(self, task: str) -> str:
        """Execute the subagent. Errors are caught and returned as messages."""
        print(f"  [subagent] {self.name} started: {task[:80]}")
        try:
            result = await self._agent.run(task, max_turns=self._max_turns)
            print(f"  [subagent] {self.name} finished")
            return str(result)
        except Exception as e:
            return f"[subagent error] {e}"

    async def execute(self, task: str) -> str:
        """Public interface — same as _run, kept for clarity."""
        return await self._run(task)
