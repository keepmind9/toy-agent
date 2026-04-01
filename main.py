"""Toy Agent - an interactive agent with built-in and MCP tools."""

import asyncio
import os

from dotenv import load_dotenv
from openai import OpenAI

from src.toy_agent.agent import Agent
from src.toy_agent.config import load_mcp_config
from src.toy_agent.mcp import MCPClient
from src.toy_agent.skills import load_skills
from src.toy_agent.subagent import SubAgentTool
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

    # Load skills
    skills = load_skills()

    # Load MCP servers from config
    config = load_mcp_config()
    mcp_client = MCPClient()
    mcp_tools: list = []

    if config.get("mcpServers"):
        print("[mcp] connecting to servers...")
        mcp_tools = await mcp_client.connect(config["mcpServers"])

    # Regular tools (built-in + MCP) — shared with subagents
    regular_tools = TOOLS + mcp_tools

    # Create subagents (inherit regular tools + skills, exclude SubAgentTool to prevent nesting)
    # TODO: Known issues with current subagent implementation:
    #   1. LLM tends to call irrelevant tools/skills instead of answering directly.
    #      Need better system prompts or stronger models.
    #   2. No per-subagent tool/skill filtering — all subagents get the same set.
    #      Ideally each subagent should only receive relevant tools/skills.
    #   3. MCP tool calls may slow down subagent execution if the MCP server is unresponsive.
    #   4. No result summarization — subagent returns raw output which can be very long.
    researcher_agent = Agent(
        client=client,
        model=model,
        system="You are a researcher. Analyze topics thoroughly and return concise findings.",
        tools=regular_tools,
        skills=skills,
    )
    researcher = SubAgentTool(
        name="researcher",
        description="Research a topic and return findings",
        agent=researcher_agent,
    )

    # All tools for main agent
    all_tools = regular_tools + [researcher]
    print(f"[tools] {len(all_tools)} tools loaded ({len(TOOLS)} built-in, {len(mcp_tools)} MCP, 1 subagent)\n")
    print(f"[skills] {len(skills)} skills loaded\n")

    agent = Agent(
        client=client,
        model=model,
        system="You are toy-agent, a helpful assistant. Use tools when needed.",
        tools=all_tools,
        skills=skills,
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
