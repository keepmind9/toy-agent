"""Toy Agent - an interactive agent with built-in and MCP tools."""

import asyncio
import os

from dotenv import load_dotenv
from openai import OpenAI

from toy_agent.agent import Agent
from toy_agent.config import load_mcp_config
from toy_agent.mcp import MCPClient
from toy_agent.memory import SessionMemory
from toy_agent.skills import load_skills
from toy_agent.subagent import SubAgentTool
from toy_agent.tools import TOOLS

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

    stream = os.getenv("TOY_AGENT_STREAM", "true").lower() in ("true", "1", "yes")

    agent = Agent(
        client=client,
        model=model,
        system="You are toy-agent, a helpful assistant. Use tools when needed.",
        tools=all_tools,
        skills=skills,
        stream=stream,
    )

    # Session memory
    memory = SessionMemory(project_path=os.getcwd())
    agent.memory = memory

    print(f"[stream] {'on' if stream else 'off'}")
    print("Toy Agent - type 'quit' to exit\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in (
                "quit",
                "exit",
                "/quit",
                "/exit",
            ):
                break
            if not user_input:
                continue

            # Session commands
            if user_input == "/resume":
                msgs = memory.load_latest()
                if msgs:
                    agent.messages = msgs
                    print(f"[memory] Restored session ({len(msgs)} messages)\n")
                else:
                    print("[memory] No sessions to restore\n")
                continue

            if user_input.startswith("/resume "):
                session_id = user_input[len("/resume ") :].strip()
                msgs = memory.load(session_id)
                if msgs:
                    agent.messages = msgs
                    print(f"[memory] Restored session {session_id} ({len(msgs)} messages)\n")
                else:
                    print(f"[memory] Session '{session_id}' not found\n")
                continue

            if user_input == "/sessions":
                sessions = memory.list_sessions()
                if not sessions:
                    print("[memory] No saved sessions\n")
                else:
                    for s in sessions:
                        print(f"  {s.session_id}  {s.created_at}")
                    print()
                continue

            if stream:
                print("Agent: ", end="", flush=True)
                response = await agent.run(user_input)
                print()
            else:
                response = await agent.run(user_input)
                print(f"Agent: {response}")
            print()
    finally:
        memory.cleanup(max_sessions=10)
        await mcp_client.cleanup()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
