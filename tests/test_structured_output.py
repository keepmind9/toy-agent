"""Tests for structured output (response_format with Pydantic models)."""

import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from toy_agent.agent import (
    Agent,
    _pydantic_to_response_format,
)
from toy_agent.llm.types import StreamChunk

# --- Test models ---


class UserInfo(BaseModel):
    name: str
    age: int


class Address(BaseModel):
    city: str
    zip_code: str


class UserProfile(BaseModel):
    name: str
    addresses: list[Address]


# --- Helper functions ---


def _make_text_response(text="done"):
    """Create a mock ChatResponse with plain text (no tool_calls)."""
    mock = MagicMock()
    mock.content = text
    mock.tool_calls = None
    mock.usage = None
    return mock


def _make_stream_chunks(text: str):
    """Create StreamChunk objects that form the given text."""
    chunks = [StreamChunk(delta_content=char, delta_tool_calls=None) for char in text]
    chunks.append(StreamChunk(delta_content=None, delta_tool_calls=None))  # final empty chunk
    return chunks


# --- Tests: _pydantic_to_response_format ---


class TestPydanticToResponseFormat:
    def test_simple_model(self):
        """Simple model produces correct OpenAI response_format structure."""
        result = _pydantic_to_response_format(UserInfo)

        assert result["type"] == "json_schema"
        js = result["json_schema"]
        assert js["name"] == "UserInfo"
        assert js["strict"] is True

        schema = js["schema"]
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
        assert schema["additionalProperties"] is False

    def test_nested_model(self):
        """Nested model with $defs produces valid schema."""
        result = _pydantic_to_response_format(UserProfile)

        schema = result["json_schema"]["schema"]
        assert "addresses" in schema["properties"]
        # $defs should exist for the nested Address model
        assert "$defs" in schema

    def test_additional_properties_false_recursive(self):
        """All object nodes get additionalProperties: false."""
        result = _pydantic_to_response_format(UserProfile)
        schema_str = json.dumps(result)

        # Parse back and verify all objects have additionalProperties: false
        schema = json.loads(schema_str)

        def check_objects(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    assert node.get("additionalProperties") is False, (
                        f"Object missing additionalProperties: false: {node}"
                    )
                for v in node.values():
                    check_objects(v)
            elif isinstance(node, list):
                for item in node:
                    check_objects(item)

        check_objects(schema)


# --- Tests: Agent.run() with response_format ---


class TestStructuredOutput:
    @pytest.mark.anyio
    async def test_returns_pydantic_model(self):
        """run() with response_format returns a Pydantic model instance."""
        client = MagicMock()
        json_str = '{"name": "Alice", "age": 30}'
        response = _make_text_response(json_str)
        client.chat.return_value = response

        agent = Agent(client=client)
        result = await agent.run("Extract info", response_format=UserInfo)

        assert isinstance(result, UserInfo)
        assert result.name == "Alice"
        assert result.age == 30

    @pytest.mark.anyio
    async def test_suppresses_tools(self):
        """When response_format is set, tools are not passed to the API."""
        client = MagicMock()
        response = _make_text_response('{"name": "Bob", "age": 25}')
        client.chat.return_value = response

        agent = Agent(client=client)
        await agent.run("Extract", response_format=UserInfo)

        call_args = client.chat.call_args
        request = call_args[0][0]  # ChatRequest passed as positional arg
        assert request.tools is None
        assert request.response_format is not None

    @pytest.mark.anyio
    async def test_without_response_format_returns_str(self):
        """Without response_format, run() still returns str (backward compat)."""
        client = MagicMock()
        response = _make_text_response("hello world")
        client.chat.return_value = response

        agent = Agent(client=client)
        result = await agent.run("hi")

        assert isinstance(result, str)
        assert result == "hello world"

    @pytest.mark.anyio
    async def test_streaming(self):
        """Streaming mode correctly parses structured output."""
        client = MagicMock()
        json_str = '{"name": "Charlie", "age": 40}'
        chunks = _make_stream_chunks(json_str)
        client.chat_stream.return_value = iter(chunks)

        agent = Agent(client=client, stream=True)
        result = await agent.run("Extract", response_format=UserInfo)

        assert isinstance(result, UserInfo)
        assert result.name == "Charlie"
        assert result.age == 40

    @pytest.mark.anyio
    async def test_invalid_json_falls_back_to_str(self):
        """Malformed JSON falls back to returning raw string with a warning."""
        client = MagicMock()
        response = _make_text_response("not valid json")
        client.chat.return_value = response

        agent = Agent(client=client)
        with pytest.warns(UserWarning, match="Structured output parse failed"):
            result = await agent.run("Extract", response_format=UserInfo)

        assert isinstance(result, str)
        assert result == "not valid json"
