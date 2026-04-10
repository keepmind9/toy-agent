"""Unit tests for Agent max_turns safety limit."""

from unittest.mock import MagicMock

import pytest

from toy_agent.agent import Agent
from toy_agent.context import ContextCompressor
from toy_agent.llm.types import LLMError, StreamChunk
from toy_agent.memory import SessionMemory
from toy_agent.subagent import SubAgentTool


def _make_mock_response(text="done", tool_calls=None, usage=None):
    """Create a mock ChatResponse (flat mock matching ChatResponse dataclass)."""
    mock = MagicMock()
    mock.content = text
    mock.tool_calls = tool_calls
    mock.usage = MagicMock()
    mock.usage.prompt_tokens = (usage or {}).get("prompt_tokens", 0)
    mock.usage.completion_tokens = (usage or {}).get("completion_tokens", 0)
    mock.usage.total_tokens = (usage or {}).get("total_tokens", 0)
    return mock


def _make_tool_call(name="test_tool", args="{}", tc_id="tc_1"):
    """Create a mock ToolCall."""
    tc = MagicMock()
    tc.id = tc_id
    tc.name = name
    tc.arguments = args
    return tc


def _make_text_response(text="done"):
    """Create a mock ChatResponse with plain text (no tool_calls)."""
    return _make_mock_response(text=text, tool_calls=None)


class TestAgentHooks:
    @pytest.mark.anyio
    async def test_hooks_called_on_llm_request(self):
        """on_llm_request hook is called before each API call."""
        mock_hook = MagicMock()
        client = MagicMock()
        response = _make_text_response("hello")
        client.chat.return_value = response

        agent = Agent(client=client, hooks=[mock_hook])
        await agent.run("hi")

        mock_hook.on_llm_request.assert_called_once()
        call_kwargs = mock_hook.on_llm_request.call_args.kwargs
        assert any(m["role"] == "system" for m in call_kwargs["messages"])
        assert any(m["role"] == "user" for m in call_kwargs["messages"])

    @pytest.mark.anyio
    async def test_hooks_called_on_message(self):
        """on_message hook is called when user message is appended."""
        mock_hook = MagicMock()
        client = MagicMock()
        response = _make_text_response("hello")
        client.chat.return_value = response

        agent = Agent(client=client, hooks=[mock_hook])
        await agent.run("test message")

        mock_hook.on_message.assert_called()
        roles = [call.kwargs.get("role") for call in mock_hook.on_message.call_args_list]
        assert "user" in roles

    @pytest.mark.anyio
    async def test_hooks_called_on_tool_call(self):
        """on_tool_call hook is called before each tool execution."""
        mock_hook = MagicMock()
        client = MagicMock()

        tool_response = _make_mock_response(
            text=None,
            tool_calls=[_make_tool_call(name="read_file", args='{"path": "/tmp/a.txt"}', tc_id="tc1")],
        )
        text_response = _make_text_response("done")

        client.chat.side_effect = [tool_response, text_response]

        async def dummy_tool(path: str) -> str:
            return "file contents"

        from toy_agent.tools import Tool

        tool_schema = {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "File path"}},
                    "required": ["path"],
                },
            },
        }
        t = Tool(schema=tool_schema, fn=dummy_tool)

        agent = Agent(client=client, tools=[t], hooks=[mock_hook])
        await agent.run("read the file")

        mock_hook.on_tool_call.assert_called_once_with(tool_name="read_file", arguments={"path": "/tmp/a.txt"})

    @pytest.mark.anyio
    async def test_hooks_called_on_error(self):
        """on_error hook is called when API raises an error."""
        mock_hook = MagicMock()
        client = MagicMock()

        fake_error = LLMError(message="server error", status_code=500)
        client.chat.side_effect = fake_error

        agent = Agent(client=client, hooks=[mock_hook])
        await agent.run("test")

        mock_hook.on_error.assert_called_once()
        assert "500" in mock_hook.on_error.call_args.kwargs["error"]

    @pytest.mark.anyio
    async def test_hooks_on_llm_response(self):
        """on_llm_response hook is called after each API response."""
        mock_hook = MagicMock()
        client = MagicMock()
        response = _make_text_response("the answer")
        client.chat.return_value = response

        agent = Agent(client=client, hooks=[mock_hook])
        await agent.run("question")

        mock_hook.on_llm_response.assert_called_once()
        assert mock_hook.on_llm_response.call_args.kwargs["message"]["role"] == "assistant"

    def test_hooks_empty_by_default(self):
        """Agent with no hooks should have an empty hooks list."""
        client = MagicMock()
        agent = Agent(client=client)
        assert agent.hooks == []

    def test_hooks_from_constructor(self):
        """Agent should accept hooks list in constructor."""
        client = MagicMock()
        hook = MagicMock()
        agent = Agent(client=client, hooks=[hook])
        assert agent.hooks == [hook]


class TestMaxTurns:
    @pytest.mark.anyio
    async def test_stops_after_max_turns(self):
        """Agent should stop and return error after exceeding max_turns."""
        client = MagicMock()

        # Always return tool_calls (infinite loop if no limit)
        tool_response = _make_mock_response(text=None, tool_calls=[_make_tool_call(name="some_tool", args="{}")])
        client.chat.return_value = tool_response

        agent = Agent(client=client)
        result = await agent.run("test", max_turns=3)

        assert "max turns" in result.lower()
        # Should have made exactly 3 API calls (3 turns)
        assert client.chat.call_count == 3

    @pytest.mark.anyio
    async def test_default_no_limit(self):
        """Without max_turns, agent runs until LLM stops calling tools."""
        client = MagicMock()

        # First call: tool_call, second call: text response
        tool_response = _make_mock_response(text=None, tool_calls=[_make_tool_call(name="some_tool", args="{}")])
        text_response = _make_text_response("final answer")

        client.chat.side_effect = [tool_response, text_response]

        agent = Agent(client=client)
        result = await agent.run("test")

        assert result == "final answer"
        assert client.chat.call_count == 2


class TestSubagentInSystemPrompt:
    def test_subagents_listed_in_system_prompt(self):
        client = MagicMock()

        sub_agent = Agent(client=client, system="sub prompt")
        sub_tool = SubAgentTool(
            name="researcher",
            description="Research topics thoroughly",
            agent=sub_agent,
        )

        main_agent = Agent(
            client=client,
            tools=[sub_tool],
        )

        system = main_agent.system
        assert "researcher" in system
        assert "Research topics thoroughly" in system

    def test_subagents_separate_from_tools(self):
        client = MagicMock()

        sub_agent = Agent(client=client)
        sub_tool = SubAgentTool(
            name="researcher",
            description="Research topics",
            agent=sub_agent,
        )

        regular_tool = MagicMock()
        regular_tool.name = "read_file"
        regular_tool.schema = {
            "function": {
                "name": "read_file",
                "description": "Read a file",
            }
        }

        main_agent = Agent(
            client=client,
            tools=[regular_tool, sub_tool],
        )

        system = main_agent.system
        assert "Available tools:" in system
        assert "Available subagents" in system
        assert "read_file" in system
        assert "researcher" in system


class TestStreamConfig:
    def test_default_stream_is_false(self):
        client = MagicMock()
        agent = Agent(client=client)
        assert agent.stream is False

    def test_stream_from_constructor(self):
        client = MagicMock()
        agent = Agent(client=client, stream=True)
        assert agent.stream is True


class TestStreamingRun:
    @pytest.mark.anyio
    async def test_stream_prints_tokens(self, capsys):
        """Streaming mode should print tokens to stdout."""
        client = MagicMock()

        chunk1 = StreamChunk(delta_content="Hello", delta_tool_calls=None)
        chunk2 = StreamChunk(delta_content=" world", delta_tool_calls=None)
        chunk3 = StreamChunk(delta_content=None, delta_tool_calls=None)

        client.chat_stream.return_value = iter([chunk1, chunk2, chunk3])

        agent = Agent(client=client, stream=True)
        result = await agent.run("hi")

        captured = capsys.readouterr()
        assert "Hello" in captured.out
        assert "world" in captured.out
        assert result == "Hello world"


class TestAgentMemory:
    @pytest.mark.anyio
    async def test_agent_saves_after_run(self, tmp_path):
        """Agent should call memory.save() after each run."""
        client = MagicMock()
        response = _make_text_response("saved!")
        client.chat.return_value = response

        memory = SessionMemory(project_path="/test/project", base_dir=tmp_path)
        agent = Agent(client=client, memory=memory)

        await agent.run("hello")

        sessions = memory.list_sessions()
        assert len(sessions) == 1

    @pytest.mark.anyio
    async def test_agent_no_memory_works_as_before(self):
        """Agent without memory should work exactly as before."""
        client = MagicMock()
        response = _make_text_response("no memory")
        client.chat.return_value = response

        agent = Agent(client=client)
        result = await agent.run("test")

        assert result == "no memory"
        assert agent.memory is None


class TestAgentCompressor:
    @pytest.mark.anyio
    async def test_agent_compresses_before_api_call(self, tmp_path):
        """Agent should call compressor.compress() before sending to LLM."""
        client = MagicMock()
        response = _make_text_response("compressed!")
        client.chat.return_value = response

        compressor = ContextCompressor(client=client, model="gpt-4o-mini", token_limit=100000)
        compressor.compress = MagicMock(side_effect=lambda msgs: msgs)

        agent = Agent(client=client, compressor=compressor)
        await agent.run("test")

        assert compressor.compress.called

    @pytest.mark.anyio
    async def test_agent_no_compressor_works_as_before(self):
        """Agent without compressor should work exactly as before."""
        client = MagicMock()
        response = _make_text_response("no compressor")
        client.chat.return_value = response

        agent = Agent(client=client)
        result = await agent.run("test")

        assert result == "no compressor"


class TestToolRetry:
    @pytest.mark.anyio
    async def test_tool_not_retried_by_default(self):
        """Without max_tool_retries, failed tool returns error immediately."""
        client = MagicMock()
        tool_response = _make_mock_response(text=None, tool_calls=[_make_tool_call(name="flaky_tool", args="{}")])
        text_response = _make_text_response("fallback answer")
        client.chat.side_effect = [tool_response, text_response]

        call_count = 0

        async def always_fail(**kwargs: object) -> str:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("always fail")

        from toy_agent.tools import Tool

        t = Tool(
            schema={
                "type": "function",
                "function": {
                    "name": "flaky_tool",
                    "description": "A flaky tool",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            fn=always_fail,
        )

        agent = Agent(client=client, tools=[t], max_tool_retries=0)
        await agent.run("do it")

        assert call_count == 1  # no retry, just one attempt

    @pytest.mark.anyio
    async def test_tool_retried_on_failure(self):
        """With max_tool_retries > 0, failed tool is retried until success."""
        client = MagicMock()
        tool_response = _make_mock_response(text=None, tool_calls=[_make_tool_call(name="flaky_tool", args="{}")])
        text_response = _make_text_response("success result")
        client.chat.side_effect = [tool_response, text_response]

        call_count = 0

        async def fail_twice_then_succeed(**kwargs: object) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError(f"attempt {call_count} failed")
            return "flaky succeeded"

        from toy_agent.tools import Tool

        t = Tool(
            schema={
                "type": "function",
                "function": {
                    "name": "flaky_tool",
                    "description": "A flaky tool",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            fn=fail_twice_then_succeed,
        )

        agent = Agent(client=client, tools=[t], max_tool_retries=2)
        result = await agent.run("do it")

        assert call_count == 3  # 2 failures + 1 success
        assert result == "success result"

    @pytest.mark.anyio
    async def test_tool_retry_hook_fires(self):
        """on_tool_retry hook is called for each retry attempt."""
        mock_hook = MagicMock()
        client = MagicMock()
        tool_response = _make_mock_response(text=None, tool_calls=[_make_tool_call(name="flaky_tool", args="{}")])
        text_response = _make_text_response("done")
        client.chat.side_effect = [tool_response, text_response]

        call_count = 0

        async def fail_twice_then_succeed(**kwargs: object) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError(f"fail {call_count}")
            return "ok"

        from toy_agent.tools import Tool

        t = Tool(
            schema={
                "type": "function",
                "function": {
                    "name": "flaky_tool",
                    "description": "A flaky tool",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            fn=fail_twice_then_succeed,
        )

        agent = Agent(client=client, tools=[t], hooks=[mock_hook], max_tool_retries=2)
        await agent.run("do it")

        assert mock_hook.on_tool_retry.call_count == 2
        assert mock_hook.on_tool_retry.call_args_list[0].kwargs["attempt"] == 0
        assert mock_hook.on_tool_retry.call_args_list[1].kwargs["attempt"] == 1

    def test_max_tool_retries_default_zero(self):
        """max_tool_retries defaults to 0 (no retry)."""
        client = MagicMock()
        agent = Agent(client=client)
        assert agent.max_tool_retries == 0
