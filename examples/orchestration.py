"""Multi-Agent Orchestration Examples.

Demonstrates three orchestration patterns using mock agents:
1. RouterOrchestrator — LLM picks the best agent for the task
2. SequentialOrchestrator — agents run in a pipeline
3. ParallelOrchestrator — agents run concurrently

Usage:
    OPENAI_API_KEY=your-key python examples/orchestration.py
"""

import asyncio
import os

from dotenv import load_dotenv
from openai import OpenAI

from toy_agent.agent import Agent
from toy_agent.orchestrator import AgentDef, ParallelOrchestrator, RouterOrchestrator, SequentialOrchestrator

load_dotenv()


def create_agents(client: OpenAI, model: str):
    """Create specialized agents for orchestration demos."""

    coding_agent = Agent(
        client=client,
        model=model,
        system="You are a coding expert. Write clean, efficient code.",
    )

    writing_agent = Agent(
        client=client,
        model=model,
        system="You are a technical writer. Explain concepts clearly.",
    )

    reviewer_agent = Agent(
        client=client,
        model=model,
        system="You are a code reviewer. Point out issues and suggest improvements. Be concise.",
    )

    return coding_agent, writing_agent, reviewer_agent


async def demo_router(client: OpenAI, model: str):
    """Demo: RouterOrchestrator routes tasks to specialized agents."""
    print("=" * 60)
    print("1. RouterOrchestrator — LLM picks the best agent")
    print("=" * 60)

    coding, writing, reviewer = create_agents(client, model)

    router = RouterOrchestrator(
        client=client,
        model=model,
        agents=[
            AgentDef(name="coding", description="Writes and debugs code", agent=coding),
            AgentDef(name="writing", description="Writes articles and explains concepts", agent=writing),
            AgentDef(name="reviewer", description="Reviews code and suggests improvements", agent=reviewer),
        ],
    )

    # Task 1: should route to coding
    print("\n--- Task: 'Write a binary search in Python' ---")
    result = await router.run("Write a binary search in Python")
    print(f"Result: {result[:200]}...\n")

    # Task 2: should route to writing
    print("--- Task: 'Explain what is RAG in AI' ---")
    result = await router.run("Explain what is RAG in AI")
    print(f"Result: {result[:200]}...\n")


async def demo_sequential(client: OpenAI, model: str):
    """Demo: SequentialOrchestrator chains agents in a pipeline."""
    print("=" * 60)
    print("2. SequentialOrchestrator — agents run in a pipeline")
    print("=" * 60)

    coding, writing, reviewer = create_agents(client, model)

    pipeline = SequentialOrchestrator(agents=[coding, reviewer])
    # coding writes code → reviewer reviews it

    print("\n--- Task: 'Write a function to check if a string is a palindrome, then review it' ---")
    result = await pipeline.run("Write a Python function to check if a string is a palindrome")
    print(f"Final result: {result[:300]}...\n")


async def demo_parallel(client: OpenAI, model: str):
    """Demo: ParallelOrchestrator runs agents concurrently."""
    print("=" * 60)
    print("3. ParallelOrchestrator — agents run concurrently")
    print("=" * 60)

    coding, writing, reviewer = create_agents(client, model)

    # Without LLM aggregation
    print("\n--- Without aggregation (concatenated results) ---")
    parallel = ParallelOrchestrator(agents=[coding, writing])
    result = await parallel.run("What are the best practices for Python error handling?")
    print(f"Result: {result[:300]}...\n")

    # With LLM aggregation
    print("--- With LLM aggregation (synthesized best answer) ---")
    parallel = ParallelOrchestrator(agents=[coding, writing], client=client, model=model)
    result = await parallel.run("What are the best practices for Python error handling?")
    print(f"Result: {result[:300]}...\n")


async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        print("Error: OPENAI_API_KEY is not set.")
        print("Usage: OPENAI_API_KEY=your-key python examples/orchestration.py")
        return

    client = OpenAI(api_key=api_key, base_url=base_url)

    await demo_router(client, model)
    await demo_sequential(client, model)
    await demo_parallel(client, model)

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
