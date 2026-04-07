"""Unit tests for AgentHook system."""

from toy_agent.hooks import AgentHook, ConsoleHook


class TestAgentHook:
    def test_hook_has_all_event_methods(self):
        """AgentHook defines all expected event methods."""
        hook = AgentHook()
        expected = [
            "on_message",
            "on_llm_request",
            "on_llm_response",
            "on_tool_call",
            "on_tool_result",
            "on_tool_retry",
            "on_compress",
            "on_error",
        ]
        for name in expected:
            assert hasattr(hook, name), f"missing method: {name}"

    def test_console_hook_inherits_all_methods(self):
        """ConsoleHook is a valid AgentHook with all methods."""
        hook = ConsoleHook()
        assert callable(hook.on_message)
        assert callable(hook.on_llm_request)
        assert callable(hook.on_llm_response)
        assert callable(hook.on_tool_call)
        assert callable(hook.on_tool_result)
        assert callable(hook.on_tool_retry)
        assert callable(hook.on_compress)
        assert callable(hook.on_error)


class TestConsoleHookOutput:
    def test_on_message_user_prints_prefix(self, capsys):
        """on_message with role=user prints 'You:' prefix."""
        hook = ConsoleHook()
        hook.on_message(role="user", content="hello world")
        captured = capsys.readouterr()
        assert "You:" in captured.out
        assert "hello world" in captured.out

    def test_on_message_assistant_skips(self, capsys):
        """on_message with role=assistant produces no output."""
        hook = ConsoleHook()
        hook.on_message(role="assistant", content="hi there")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_on_tool_call_prints_format(self, capsys):
        """on_tool_call prints tool name and args."""
        hook = ConsoleHook()
        hook.on_tool_call(tool_name="read_file", arguments={"path": "/tmp/a.txt"})
        captured = capsys.readouterr()
        assert "read_file" in captured.out

    def test_on_error_prints_prefix(self, capsys):
        """on_error prints [error] prefix."""
        hook = ConsoleHook()
        hook.on_error(error="something went wrong")
        captured = capsys.readouterr()
        assert "[error]" in captured.out
        assert "something went wrong" in captured.out

    def test_on_tool_retry_prints(self, capsys):
        """on_tool_retry prints tool name and retry info."""
        hook = ConsoleHook()
        hook.on_tool_retry(tool_name="read_file", attempt=1, error="file locked")
        captured = capsys.readouterr()
        assert "read_file" in captured.out
        assert "retry" in captured.out.lower()
