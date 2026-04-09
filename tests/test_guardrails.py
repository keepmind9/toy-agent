"""Unit tests for Guardrails feature."""

from unittest.mock import MagicMock, patch

import pytest

from toy_agent.agent import Agent
from toy_agent.guardrails import GuardrailHook
from toy_agent.hooks import AgentHook, ConsoleHook


def _make_tool_call_response(fn_name="test_tool", fn_args="{}"):
    """Create a mock LLM response with a tool call."""
    func_mock = MagicMock()
    func_mock.name = fn_name
    func_mock.arguments = fn_args
    tc_mock = MagicMock(id="tc_1", function=func_mock)
    msg = MagicMock()
    msg.content = None
    msg.tool_calls = [tc_mock]
    return msg


def _make_text_response(text="done"):
    """Create a mock LLM response with text only."""
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = None
    return msg


class TestAgentHookGuardrailMethods:
    def test_hook_has_guardrail_methods(self):
        """AgentHook defines on_tool_approve and on_guardrail_block."""
        hook = AgentHook()
        assert hasattr(hook, "on_tool_approve")
        assert hasattr(hook, "on_guardrail_block")

    def test_on_tool_approve_default_returns_ellipsis(self):
        """Default on_tool_approve returns Ellipsis (not a string, so allows execution)."""
        hook = AgentHook()
        result = hook.on_tool_approve(tool_name="run_bash", arguments={"command": "ls"})
        assert not isinstance(result, str)


class TestCheckGuardrails:
    @pytest.mark.anyio
    async def test_allows_when_no_hooks_block(self):
        """No guardrail hooks -> tool executes normally."""
        client = MagicMock()
        tool_msg = _make_tool_call_response()
        text_msg = _make_text_response()
        client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=tool_msg)]),
            MagicMock(choices=[MagicMock(message=text_msg)]),
        ]

        agent = Agent(client=client)
        result = await agent.run("do it")
        assert result == "done"

    @pytest.mark.anyio
    async def test_blocks_when_hook_returns_string(self):
        """Hook returning a string blocks tool execution."""
        client = MagicMock()
        tool_msg = _make_tool_call_response("run_bash", '{"command": "rm -rf /"}')
        text_msg = _make_text_response("ok")
        client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=tool_msg)]),
            MagicMock(choices=[MagicMock(message=text_msg)]),
        ]

        blocking_hook = MagicMock()
        blocking_hook.on_tool_approve.return_value = "[Guardrail] blocked"
        # on_tool_call and others return MagicMock, not str -> no extra blocking
        blocking_hook.on_tool_call.return_value = None

        agent = Agent(client=client, hooks=[blocking_hook])
        result = await agent.run("do it")

        # Tool was blocked, agent got the blocked message and continued
        assert result == "ok"
        blocking_hook.on_guardrail_block.assert_called_once()

    @pytest.mark.anyio
    async def test_async_hook_supported(self):
        """Async on_tool_approve hooks are awaited properly."""
        client = MagicMock()
        tool_msg = _make_tool_call_response("run_bash")
        text_msg = _make_text_response("ok")
        client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=tool_msg)]),
            MagicMock(choices=[MagicMock(message=text_msg)]),
        ]

        async def async_block(**kwargs):
            return "[Async Guardrail] blocked"

        blocking_hook = MagicMock()
        blocking_hook.on_tool_approve.side_effect = async_block

        agent = Agent(client=client, hooks=[blocking_hook])
        result = await agent.run("do it")

        assert result == "ok"
        blocking_hook.on_guardrail_block.assert_called_once()


class TestGuardrailHook:
    def test_allows_safe_tools(self):
        """Tools not in approval_tools are allowed without prompt."""
        hook = GuardrailHook()
        result = hook.on_tool_approve(tool_name="read_file", arguments={"path": "/tmp/a.txt"})
        assert result is None

    def test_blocks_dangerous_tool_without_approval(self):
        """Dangerous tools are blocked when user declines."""
        hook = GuardrailHook()
        with patch("builtins.input", return_value="n"):
            result = hook.on_tool_approve(tool_name="run_bash", arguments={"command": "rm -rf /"})
        assert isinstance(result, str)
        assert "blocked" in result

    def test_allows_dangerous_tool_with_approval(self):
        """Dangerous tools are allowed when user approves."""
        hook = GuardrailHook()
        with patch("builtins.input", return_value="y"):
            result = hook.on_tool_approve(tool_name="run_bash", arguments={"command": "ls"})
        assert result is None

    def test_auto_approve_bypasses_prompt(self):
        """Tools in auto_approve are allowed without prompting."""
        hook = GuardrailHook(auto_approve={"run_bash"})
        result = hook.on_tool_approve(tool_name="run_bash", arguments={"command": "ls"})
        assert result is None

    def test_custom_approval_tools(self):
        """Custom approval_tools override the defaults."""
        hook = GuardrailHook(approval_tools={"read_file"})
        with patch("builtins.input", return_value="n"):
            result = hook.on_tool_approve(tool_name="read_file", arguments={"path": "/etc/passwd"})
        assert isinstance(result, str)

        # run_bash is not in custom approval_tools, so it's allowed
        result = hook.on_tool_approve(tool_name="run_bash", arguments={"command": "ls"})
        assert result is None


class TestGuardrailIntegration:
    @pytest.mark.anyio
    async def test_blocked_tool_returns_message_to_agent(self):
        """When tool is blocked, the blocked message is returned as tool result."""
        client = MagicMock()
        tool_msg = _make_tool_call_response("run_bash", '{"command": "rm -rf /"}')
        text_msg = _make_text_response("I see the tool was blocked")
        client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=tool_msg)]),
            MagicMock(choices=[MagicMock(message=text_msg)]),
        ]

        with patch("builtins.input", return_value="n"):
            hook = GuardrailHook()
            agent = Agent(client=client, hooks=[hook])
            result = await agent.run("delete everything")

        assert result == "I see the tool was blocked"
        # Verify the blocked message was injected as tool result
        tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "blocked" in tool_msgs[0]["content"]

    @pytest.mark.anyio
    async def test_approved_tool_executes_normally(self):
        """When user approves, the tool executes normally."""
        client = MagicMock()
        tool_msg = _make_tool_call_response("run_bash", '{"command": "ls"}')
        text_msg = _make_text_response("here are the files")
        client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=tool_msg)]),
            MagicMock(choices=[MagicMock(message=text_msg)]),
        ]

        with patch("builtins.input", return_value="y"):
            hook = GuardrailHook()
            agent = Agent(client=client, hooks=[hook])
            result = await agent.run("list files")

        assert result == "here are the files"


class TestConsoleHookGuardrailOutput:
    def test_on_guardrail_block_prints(self, capsys):
        """on_guardrail_block prints blocked tool info."""
        hook = ConsoleHook()
        hook.on_guardrail_block(tool_name="run_bash", arguments={"cmd": "rm -rf /"}, reason="blocked by user")
        captured = capsys.readouterr()
        assert "[guardrail]" in captured.out
        assert "run_bash" in captured.out
        assert "blocked" in captured.out
