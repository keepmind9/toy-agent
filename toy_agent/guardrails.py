"""Guardrails: human-in-the-loop approval for dangerous tool calls."""

from __future__ import annotations

import json

from toy_agent.hooks import AgentHook


class GuardrailHook(AgentHook):
    """Hook that requires human approval before executing dangerous tools.

    Tools listed in ``approval_tools`` will prompt the user for confirmation
    before execution. Tools in ``auto_approve`` bypass the prompt.

    Args:
        approval_tools: Set of tool names requiring approval.
            Defaults to {"run_bash", "write_file", "edit_file"}.
        auto_approve: Set of tool names to auto-approve (subset of approval_tools).
    """

    def __init__(
        self,
        approval_tools: set[str] | None = None,
        auto_approve: set[str] | None = None,
    ):
        self.approval_tools = approval_tools or {"run_bash", "write_file", "edit_file"}
        self.auto_approve = auto_approve or set()

    def on_tool_approve(self, *, tool_name: str, arguments: dict) -> str | None:
        """Block execution if tool requires approval and user declines."""
        if tool_name not in self.approval_tools or tool_name in self.auto_approve:
            return None  # allow

        args_preview = json.dumps(arguments, indent=2) if arguments else "{}"
        answer = input(f"\n[guardrail] Allow {tool_name}({args_preview})? [y/N] ")
        if answer.lower() not in ("y", "yes"):
            return f"[Guardrail] Tool '{tool_name}' blocked by user"
        return None
