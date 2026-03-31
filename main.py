"""A minimal agent loop"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from src.toy_agent.agent import Agent
from src.toy_agent.tools import TOOLS

load_dotenv()


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        print("Error: OPENAI_API_KEY is not set. Set it via env var or .env file.")
        return

    client = OpenAI(api_key=api_key, base_url=base_url)

    agent = Agent(
        client=client,
        model=model,
        system="You are toy-agent, a helpful assistant. Use tools when needed.",
        tools=TOOLS,
    )

    print("Toy Agent - type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ('quit', 'exit', ):
            break
        if not user_input:
            continue

        response = agent.run(user_input)
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    main()
