"""AgentHook: pluggable observability callbacks for the agent loop."""

from abc import ABC


class AgentHook(ABC):
    """Base class for agent observability hooks.

    Subclass and override any methods to add behaviour at specific
    points in the agent loop. All methods are no-ops by default.

    Args passed to each method are keyword-only for forward compatibility.
    """

    def on_message(self, *, role: str, content: str) -> None:
        """Called when a message is appended to the conversation history."""
        ...

    def on_llm_request(self, *, messages: list[dict]) -> None:
        """Called just before sending a request to the LLM API."""
        ...

    def on_llm_response(self, *, message: dict) -> None:
        """Called after receiving an LLM response (before tool processing)."""
        ...

    def on_tool_call(self, *, tool_name: str, arguments: dict) -> None:
        """Called before executing a tool."""
        ...

    def on_tool_result(self, *, tool_name: str, result: str) -> None:
        """Called after a tool returns a result."""
        ...

    def on_tool_retry(self, *, tool_name: str, attempt: int, error: str) -> None:
        """Called when a tool failed and will be retried."""
        ...

    def on_compress(self, *, before_count: int, after_count: int) -> None:
        """Called after context compression finishes."""
        ...

    def on_error(self, *, error: str) -> None:
        """Called when an error occurs."""
        ...


class ConsoleHook(AgentHook):
    """Default hook that prints human-readable event summaries to stdout.

    Reproduces the console output behaviour that existed before hooks.
    """

    def on_message(self, *, role: str, content: str) -> None:
        if role == "user" and content:
            print(f"You: {content[:100]}")

    def on_tool_call(self, *, tool_name: str, arguments: dict) -> None:
        print(f"  [tool] {tool_name}({arguments})")

    def on_tool_retry(self, *, tool_name: str, attempt: int, error: str) -> None:
        print(f"  [tool-retry] {tool_name} attempt {attempt + 1}: {error}")

    def on_error(self, *, error: str) -> None:
        print(f"[error] {error}")
