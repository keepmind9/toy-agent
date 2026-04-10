"""Toy Agent - an interactive agent with built-in and MCP tools."""

import asyncio
import os
import readline  # noqa: F401  # side-effect: enables readline features in input()

from dotenv import load_dotenv

from toy_agent.agent import Agent
from toy_agent.config import load_mcp_config
from toy_agent.context import HermesContextCompressor
from toy_agent.guardrails import GuardrailHook
from toy_agent.hooks import ConsoleHook
from toy_agent.llm import create_llm_client
from toy_agent.mcp import MCPClient
from toy_agent.memory import SessionMemory
from toy_agent.planning import ReActPlanHook
from toy_agent.retriever import BM25Retriever
from toy_agent.skills import load_skills
from toy_agent.subagent import SubAgentTool
from toy_agent.tools import TOOLS

load_dotenv()

# Configure readline for command history and cursor navigation.
# This makes input() support:
#   - Up/Down arrows: navigate command history
#   - Left/Right arrows: move cursor within the line
#   - Ctrl+A/E: jump to start/end of line
#   - Ctrl+U/K: clear line before/after cursor
_readline_history: list[str] = []


def _get_input(prompt: str) -> str:
    """Wrapper around input() with readline history support."""
    try:
        line = input(prompt)
        if line.strip():
            _readline_history.append(line)
            # Keep history bounded to avoid unbounded memory growth
            if len(_readline_history) > 100:
                _readline_history.pop(0)
        return line
    except (KeyboardInterrupt, EOFError):
        raise KeyboardInterrupt


async def async_main():
    try:
        client = create_llm_client()
    except ValueError as e:
        print(f"Error: {e}")
        return

    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if provider == "anthropic":
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    elif provider == "gemini":
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

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
    print("[plan] ReAct planning enabled (model self-tracks step progress)\n")

    stream = os.getenv("TOY_AGENT_STREAM", "true").lower() in ("true", "1", "yes")
    context_token_limit = int(os.getenv("TOY_AGENT_CONTEXT_TOKEN_LIMIT", "80000"))

    # RAG: index documents from configured directory (graceful no-op if missing)
    knowledge_dir = os.getenv("TOY_AGENT_KNOWLEDGE_DIR", "knowledge")
    retriever = BM25Retriever.from_directory(knowledge_dir)
    if retriever:
        print(f"[rag] {len(retriever._chunks)} chunks indexed from {knowledge_dir}/\n")

    agent = Agent(
        client=client,
        model=model,
        system="You are toy-agent, a helpful assistant. Use tools when needed.",
        tools=all_tools,
        skills=skills,
        stream=stream,
        retriever=retriever,
        hooks=[ConsoleHook(), ReActPlanHook(), GuardrailHook()],
    )

    # Session memory
    memory = SessionMemory(project_path=os.getcwd())
    agent.memory = memory

    # Context compression
    compressor = HermesContextCompressor(client=client, model=model, token_limit=context_token_limit)
    agent.compressor = compressor

    print(f"[stream] {'on' if stream else 'off'}")
    print("Toy Agent - type 'quit' to exit\n")

    try:
        while True:
            try:
                user_input = _get_input("You: ").strip()
            except KeyboardInterrupt:
                print()  # clean up terminal line
                continue
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

            # Planning command: /plan <task> forces plan=True
            plan_mode = False
            if user_input.startswith("/plan "):
                user_input = user_input[len("/plan ") :].strip()
                plan_mode = True

            if not user_input:
                continue

            if stream:
                print("Agent: ", end="", flush=True)
                response = await agent.run(user_input, plan=plan_mode or None)
                print()
            else:
                response = await agent.run(user_input, plan=plan_mode or None)
                print(f"Agent: {response}")
            print()
    finally:
        memory.cleanup(max_sessions=10)
        await mcp_client.cleanup()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
