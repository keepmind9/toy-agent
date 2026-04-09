"""Context compression strategies for long conversations.

Two strategies are provided:

1. ContextCompressor — Three-level progressive compression (Level 1 only implemented).
   Simple turn summaries, good for basic usage.

2. HermesContextCompressor — Hermes-style 4-phase compression.
   Tool output pruning, token-budget tail protection, structured handoff summaries
   with iterative updates, and tool pair sanitization.
"""

from __future__ import annotations

import copy
import json
import warnings
from typing import TYPE_CHECKING

import tiktoken

if TYPE_CHECKING:
    from openai import OpenAI


def count_tokens(messages: list[dict], encoding) -> int:
    """Estimate token count for a list of messages."""
    if not messages:
        return 0
    total = 0
    for msg in messages:
        total += 4
        for value in msg.values():
            if isinstance(value, str):
                total += len(encoding.encode(value))
            elif isinstance(value, list):
                total += len(encoding.encode(json.dumps(value)))
        total += 2
    total += 3
    return total


def _get_encoding(model: str):
    """Get tiktoken encoding for a model."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Strategy 1: Original three-level progressive compression
# ---------------------------------------------------------------------------


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
        self._encoding = _get_encoding(model)

    def count_tokens(self, messages: list[dict]) -> int:
        return count_tokens(messages, self._encoding)

    def compress(self, messages: list[dict]) -> list[dict]:
        """Check token count and compress if needed."""
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
        """Level 1: Turn summary — compress tool call chains into brief summaries."""
        turns = self._split_turns(messages)

        if len(turns) <= 1:
            return messages

        system_msg: list[dict] = []
        if turns and turns[0][0].get("role") == "system":
            system_msg = turns[0]
            turns = turns[1:]

        if not turns:
            return messages

        compressed = list(system_msg)
        turns_to_compress = turns[:-1]
        last_turn = turns[-1]

        for turn in turns_to_compress:
            user_msgs = [m for m in turn if m.get("role") == "user"]
            exec_msgs = [m for m in turn if m.get("role") != "user"]

            compressed.extend(user_msgs)

            if exec_msgs:
                summary = self._summarize(exec_msgs)
                compressed.append({"role": "assistant", "content": f"[Turn Summary] {summary}"})

        compressed.extend(last_turn)
        return compressed

    def _level2(self, messages: list[dict]) -> list[dict]:
        # TODO: implement phase overview compression
        return messages

    def _level3(self, messages: list[dict]) -> list[dict]:
        # TODO: implement sliding window compression
        return messages

    def _summarize(self, messages: list[dict]) -> str:
        """Generate a summary of execution messages using LLM."""
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
                    {
                        "role": "system",
                        "content": "Summarize this agent turn in 1-2 sentences: what was done and the outcome.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            return response.choices[0].message.content or text[:300]
        except Exception as e:
            warnings.warn(f"Summary generation failed: {e}, using plain text fallback", stacklevel=2)
            return text[:300]


# ---------------------------------------------------------------------------
# Strategy 2: Hermes-style 4-phase compression
# ---------------------------------------------------------------------------

_FIRST_SUMMARY_PROMPT = """\
Create a structured handoff summary for a later assistant that will continue
this conversation after earlier turns are compacted.

TURNS TO SUMMARIZE:
{content}

Use this exact structure:
## Goal
## Constraints & Preferences
## Progress
### Done
### In Progress
### Blocked
## Key Decisions
## Relevant Files
## Next Steps
## Critical Context
## Tools & Patterns"""

_ITERATIVE_SUMMARY_PROMPT = """\
You are updating a context compaction summary. A previous compaction produced
the summary below. New conversation turns have occurred since then and need
to be incorporated.

PREVIOUS SUMMARY:
{previous_summary}

NEW TURNS TO INCORPORATE:
{content}

Update the summary. PRESERVE all existing information that is still relevant.
ADD new progress. Move items from "In Progress" to "Done" when completed.
Use this exact structure:
## Goal
## Constraints & Preferences
## Progress
### Done
### In Progress
### Blocked
## Key Decisions
## Relevant Files
## Next Steps
## Critical Context
## Tools & Patterns"""

_PRUNED_PLACEHOLDER = "[Old tool output cleared]"


class HermesContextCompressor:
    """Hermes-style 4-phase context compression.

    Inspired by Hermes Agent's context compressor. Phases:
      1. Tool output pruning — replace old long tool results with placeholders
      2. Boundary determination — protect head/tail by token budget
      3. Structured LLM summarization — 8-section handoff with iterative updates
      4. Assembly + sanitization — role alternation and tool pair repair

    Args:
        client: OpenAI client for summary generation.
        model: Model to use for summarization.
        token_limit: Token threshold that triggers compression.
        tail_ratio: Fraction of token_limit reserved for tail protection.
        protect_head: Number of leading messages to always protect.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        token_limit: int = 80000,
        tail_ratio: float = 0.15,
        protect_head: int = 3,
    ):
        self.client = client
        self.model = model
        self.token_limit = token_limit
        self.tail_ratio = tail_ratio
        self.protect_head = protect_head
        self._encoding = _get_encoding(model)
        self._previous_summary: str | None = None

    def count_tokens(self, messages: list[dict]) -> int:
        return count_tokens(messages, self._encoding)

    def compress(self, messages: list[dict]) -> list[dict]:
        """Run 4-phase compression if token count exceeds limit."""
        tokens = self.count_tokens(messages)
        if tokens < self.token_limit:
            return messages

        print(f"[context] Token count {tokens} exceeds limit {self.token_limit}, compressing...")

        # Phase 1: cheap tool output pruning
        messages = self._prune_tool_outputs(messages)

        # Phase 2: determine compression boundaries
        head, compressible, tail = self._find_boundaries(messages)

        if not compressible:
            return messages

        # Phase 3: structured LLM summarization
        summary_text = self._generate_summary(compressible)
        self._previous_summary = summary_text

        # Phase 4: assemble and sanitize
        result = self._assemble(head, summary_text, tail)
        result = self._sanitize_tool_pairs(result)
        return result

    # -- Phase 1: Tool output pruning ------------------------------------------

    def _prune_tool_outputs(self, messages: list[dict]) -> list[dict]:
        """Replace old long tool results with placeholders (in compressible region)."""
        # Find tail boundary to know what NOT to prune
        _, _, tail = self._find_boundaries(messages)
        tail_ids = {id(m) for m in tail}

        result = []
        for msg in messages:
            if id(msg) in tail_ids:
                result.append(msg)
            elif msg.get("role") == "tool" and len(msg.get("content", "")) > 200:
                result.append({**msg, "content": _PRUNED_PLACEHOLDER})
            else:
                result.append(msg)
        return result

    # -- Phase 2: Boundary determination ----------------------------------------

    def _find_boundaries(self, messages: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
        """Split messages into (head, compressible, tail) regions."""
        n = len(messages)
        if n <= self.protect_head + 1:
            return messages, [], []

        # Head: first protect_head messages, aligned forward past tool results
        head_end = self._align_boundary_forward(messages, self.protect_head)
        head = messages[:head_end]

        # Tail: walk backward by token budget
        tail_budget = int(self.token_limit * self.tail_ratio)
        tail_start = self._find_tail_cut(messages, head_end, tail_budget)
        tail_start = self._align_boundary_backward(messages, tail_start)

        if tail_start <= head_end:
            return head, [], messages[head_end:]

        compressible = messages[head_end:tail_start]
        tail = messages[tail_start:]
        return head, compressible, tail

    def _find_tail_cut(self, messages: list[dict], head_end: int, budget: int) -> int:
        """Walk backward from end accumulating tokens until budget is reached."""
        total = 0
        for i in range(len(messages) - 1, head_end, -1):
            msg_tokens = count_tokens([messages[i]], self._encoding)
            total += msg_tokens
            if total >= budget:
                return i + 1
        return head_end + 1

    def _align_boundary_forward(self, messages: list[dict], idx: int) -> int:
        """Move boundary forward past orphaned tool results."""
        while idx < len(messages) and messages[idx].get("role") == "tool":
            idx += 1
        return idx

    def _align_boundary_backward(self, messages: list[dict], idx: int) -> int:
        """Move boundary backward past orphaned assistant tool_calls."""
        while 0 < idx < len(messages) and messages[idx].get("role") == "tool":
            idx -= 1
        return idx

    # -- Phase 3: Structured LLM summarization ----------------------------------

    def _generate_summary(self, messages: list[dict]) -> str:
        """Generate a structured handoff summary via LLM."""
        content = self._serialize_for_summary(messages)

        if self._previous_summary:
            prompt = _ITERATIVE_SUMMARY_PROMPT.format(previous_summary=self._previous_summary, content=content)
        else:
            prompt = _FIRST_SUMMARY_PROMPT.format(content=content)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )
            return response.choices[0].message.content or self._fallback_summary(messages)
        except Exception as e:
            warnings.warn(f"Summary generation failed: {e}", stacklevel=2)
            return self._fallback_summary(messages)

    def _serialize_for_summary(self, messages: list[dict]) -> str:
        """Render messages as labeled text for the summary LLM."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "assistant":
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        args = fn.get("arguments", "")
                        if len(args) > 1200:
                            args = args[:1200] + "..."
                        parts.append(f"[assistant calls {fn.get('name', '?')}({args})]")
                if content:
                    parts.append(f"[assistant] {content[:1000]}")

            elif role == "tool":
                text = content or ""
                if len(text) > 5500:
                    text = text[:4000] + "\n...[truncated]...\n" + text[-1500:]
                parts.append(f"[tool result] {text}")

            elif role == "user":
                parts.append(f"[user] {content[:1000]}")

        return "\n".join(parts)

    def _fallback_summary(self, messages: list[dict]) -> str:
        """Plain text fallback when LLM summarization fails."""
        parts = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                names = [tc["function"]["name"] for tc in msg["tool_calls"]]
                parts.append(f"Called: {', '.join(names)}")
            elif msg.get("role") == "tool":
                parts.append(f"Result: {(msg.get('content') or '')[:200]}")
        return "\n".join(parts) if parts else "Context was compressed."

    # -- Phase 4: Assembly + sanitization ----------------------------------------

    def _assemble(self, head: list[dict], summary_text: str, tail: list[dict]) -> list[dict]:
        """Assemble head + summary + tail with role alternation."""
        summary_msg = {
            "role": "assistant",
            "content": f"[Context Summary]\n{summary_text}",
        }

        # Check role alternation: avoid consecutive same-role messages
        head_last_role = head[-1]["role"] if head else None
        tail_first_role = tail[0]["role"] if tail else None

        if head_last_role == "assistant" and tail_first_role == "user":
            # Double collision: merge summary into first tail message
            result = list(head)
            tail = copy.deepcopy(tail)
            tail[0]["content"] = f"{summary_msg['content']}\n\n---\n{tail[0].get('content', '')}"
            result.extend(tail)
            return result

        if head_last_role == "assistant":
            summary_msg["role"] = "user"

        return head + [summary_msg] + tail

    def _sanitize_tool_pairs(self, messages: list[dict]) -> list[dict]:
        """Fix orphaned tool_call/result pairs."""
        # Collect tool call IDs and result IDs
        call_ids = set()
        result_ids = set()
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    call_ids.add(tc["id"])
            elif msg.get("role") == "tool":
                result_ids.add(msg.get("tool_call_id", ""))

        if not call_ids and not result_ids:
            return messages

        result = []
        for msg in messages:
            if msg.get("role") == "tool" and msg.get("tool_call_id") not in call_ids:
                continue  # remove orphaned result
            result.append(msg)

        # Add stub results for calls missing results
        result_ids_in_output = {m.get("tool_call_id") for m in result if m.get("role") == "tool"}
        for i, msg in enumerate(result):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc["id"] not in result_ids_in_output:
                        stub = {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": "[result removed during compression]",
                        }
                        # Insert stub after the assistant message
                        result.insert(i + 1, stub)
                        result_ids_in_output.add(tc["id"])

        return result
