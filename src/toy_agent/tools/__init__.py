"""Tool framework: @tool decorator, Tool class, and TOOLS registry.

Usage:
    @tool(description="Read file content")
    def read_file(path: str) -> str:
        '''path: File path'''
        ...

    # The tool is automatically registered to TOOLS.
"""

import importlib
import inspect
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any


class Tool:
    """Wraps a Python function with its OpenAI function calling schema."""

    def __init__(self, schema: dict, fn: Callable[..., Any]):
        self.schema = schema
        self.fn = fn

    @property
    def name(self) -> str:
        return self.schema["function"]["name"]


# The global registry, populated by @tool decorator
TOOLS: list[Tool] = []


def tool(description: str) -> Callable[..., Tool]:
    """Decorator that converts a function into a Tool and registers it.

    Uses the function signature to build parameter types, and the docstring
    to fill in parameter descriptions.
    """

    def decorator(fn: Callable[..., Any]) -> Tool:
        sig = inspect.signature(fn)
        params = sig.parameters

        # Parse docstring for parameter descriptions
        # Format: "param_name: description" one per line
        param_docs = {}
        if fn.__doc__:
            for line in fn.__doc__.strip().splitlines():
                line = line.strip()
                if ":" in line:
                    name, _, desc = line.partition(":")
                    name = name.strip()
                    if name in params:
                        param_docs[name] = desc.strip()

        # Build JSON Schema properties
        type_map = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}
        properties: dict[str, Any] = {}
        required: list[str] = []

        for name, param in params.items():
            type_name = type_map.get(
                getattr(param.annotation, "__name__", "string") if param.annotation is not inspect.Parameter.empty else "string",
                "string",
            )
            prop: dict[str, Any] = {"type": type_name}
            if name in param_docs:
                prop["description"] = param_docs[name]
            properties[name] = prop

            if param.default is inspect.Parameter.empty:
                required.append(name)

        schema = {
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    **({"required": required} if required else {}),
                },
            },
        }

        t = Tool(schema=schema, fn=fn)
        TOOLS.append(t)
        return t

    return decorator


# Auto-import all tool modules in this directory
_tools_dir = Path(__file__).parent
for _file in sorted(_tools_dir.glob("*.py")):
    if _file.name.startswith("_"):
        continue
    importlib.import_module(f"src.toy_agent.tools.{_file.stem}")
