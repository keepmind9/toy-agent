"""Three-level progressive context compression for long conversations.

Design doc: docs/CONTEXT_COMPRESSION_STRATEGY.md

Compression levels:
  Level 1: Turn summary — compress tool call chains into brief summaries
  Level 2: Phase overview — merge early turn summaries into phase overviews (TODO)
  Level 3: Sliding window + global overview — keep only recent turns (TODO)
"""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING

import tiktoken

if TYPE_CHECKING:
    from openai import OpenAI


class ContextCompressor:
    """Manages progressive context compression to prevent token overflow.

    See docs/CONTEXT_COMPRESSION_STRATEGY.md for the full design.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        token_limit: int = 80000,
    ):
        self.client = client
        self.model = model
        self.token_limit = token_limit
        self._cooldown = False

        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, messages: list[dict]) -> int:
        """Estimate token count for a list of messages.

        Uses tiktoken for accurate counting. Accounts for message formatting
        overhead (role labels, separators) with per-message tokens.
        """
        if not messages:
            return 0

        total = 0
        for msg in messages:
            # Per-message overhead (role, separators)
            total += 4
            for key, value in msg.items():
                if isinstance(value, str):
                    total += len(self._encoding.encode(value))
                elif isinstance(value, list):
                    # tool_calls or similar structured data
                    total += len(self._encoding.encode(json.dumps(value)))
            # Message separator
            total += 2
        # Priming tokens for the response
        total += 3
        return total

    def compress(self, messages: list[dict]) -> list[dict]:
        """Check token count and compress if needed.

        Applies compression levels progressively:
        Level 1 (turn summary) → Level 2 (phase overview) → Level 3 (sliding window)

        Returns:
            Compressed message list, or original if under limit.
        """
        if self._cooldown:
            self._cooldown = False
            return messages

        tokens = self.count_tokens(messages)
        if tokens < self.token_limit:
            return messages

        print(f"[context] Token count {tokens} exceeds limit {self.token_limit}, compressing...")
        messages = self._level1(messages)
        messages = self._level2(messages)
        messages = self._level3(messages)
        self._cooldown = True
        return messages

    def _split_turns(self, messages: list[dict]) -> list[list[dict]]:
        """Split messages into turns, each starting with a user message."""
        turns: list[list[dict]] = []
        current_turn: list[dict] = []

        for msg in messages:
            if msg.get("role") == "user" and current_turn:
                turns.append(current_turn)
                current_turn = []
            current_turn.append(msg)

        if current_turn:
            turns.append(current_turn)

        return turns

    def _level1(self, messages: list[dict]) -> list[dict]:
        """Level 1: Turn summary — compress tool call chains into brief summaries.

        For each turn (user message + subsequent assistant/tool messages),
        replace the execution details with a summary while keeping the user message.
        The most recent turn is left uncompressed.
        """
        turns = self._split_turns(messages)

        if len(turns) <= 1:
            return messages

        system_msg: list[dict] = []
        if turns and turns[0][0].get("role") == "system":
            system_msg = turns[0]
            turns = turns[1:]

        if not turns:
            return messages

        # Compress all turns except the last one
        compressed = list(system_msg)
        turns_to_compress = turns[:-1]
        last_turn = turns[-1]

        for turn in turns_to_compress:
            # Keep user message, summarize the rest
            user_msgs = [m for m in turn if m.get("role") == "user"]
            exec_msgs = [m for m in turn if m.get("role") != "user"]

            compressed.extend(user_msgs)

            if exec_msgs:
                summary = self._summarize(exec_msgs)
                compressed.append({"role": "assistant", "content": f"[Turn Summary] {summary}"})

        compressed.extend(last_turn)
        return compressed

    def _level2(self, messages: list[dict]) -> list[dict]:
        """Level 2: Phase overview — merge early turn summaries into phase overviews.

        See docs/CONTEXT_COMPRESSION_STRATEGY.md for design details.
        """
        # TODO: implement phase overview compression
        return messages

    def _level3(self, messages: list[dict]) -> list[dict]:
        """Level 3: Sliding window + global overview.

        See docs/CONTEXT_COMPRESSION_STRATEGY.md for design details.
        """
        # TODO: implement sliding window compression
        return messages

    def _summarize(self, messages: list[dict]) -> str:
        """Generate a summary of execution messages using LLM.

        Falls back to plain text summary if LLM call fails.
        """
        # Build a readable text of the execution
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "assistant":
                if msg.get("tool_calls"):
                    tool_names = [tc["function"]["name"] for tc in msg["tool_calls"]]
                    parts.append(f"Called tools: {', '.join(tool_names)}")
                if content:
                    parts.append(f"Assistant: {content[:200]}")
            elif role == "tool":
                preview = (content or "")[:200]
                parts.append(f"Tool result: {preview}")

        text = "\n".join(parts)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Summarize the following agent execution in 1-2 sentences. Focus on what was done and the outcome."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            return response.choices[0].message.content or text[:300]
        except Exception as e:
            warnings.warn(f"Summary generation failed: {e}, using plain text fallback", stacklevel=2)
            return text[:300]
