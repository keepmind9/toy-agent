"""Toy Agent - an interactive agent with built-in and MCP tools."""

import asyncio
import os

from dotenv import load_dotenv
from openai import OpenAI

from src.toy_agent.agent import Agent
from src.toy_agent.config import load_mcp_config
from src.toy_agent.mcp import MCPClient
from src.toy_agent.tools import TOOLS

load_dotenv()


async def async_main():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        print("Error: OPENAI_API_KEY is not set. Set it via env var or .env file.")
        return

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Load MCP servers from config
    config = load_mcp_config()
    mcp_client = MCPClient()
    mcp_tools: list = []

    if config.get("mcpServers"):
        print("[mcp] connecting to servers...")
        mcp_tools = await mcp_client.connect(config["mcpServers"])

    # Combine built-in tools + MCP tools
    all_tools = TOOLS + mcp_tools
    print(f"[tools] {len(all_tools)} tools loaded ({len(TOOLS)} built-in, {len(mcp_tools)} MCP)\n")

    agent = Agent(
        client=client,
        model=model,
        system="You are toy-agent, a helpful assistant. Use tools when needed.",
        tools=all_tools,
    )

    print("Toy Agent - type 'quit' to exit\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit"):
                break
            if not user_input:
                continue

            response = await agent.run(user_input)
            print(f"Agent: {response}\n")
    finally:
        await mcp_client.cleanup()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
