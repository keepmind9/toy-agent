# Hermes-style Context Compression Strategy

> Detailed analysis of the 4-phase context compression strategy inspired by [Hermes Agent](https://github.com/NousResearch/Hermes-Agent).

## Problem

In long agent conversations, message history grows continuously and eventually exceeds the LLM context window. Simple truncation ("keep last N messages") loses early important context, while "do nothing" causes token overflow. Hermes Agent solves this with a structured 4-phase compression that preserves far more information than simple summarization.

---

## 4-Phase Compression Overview

Each compression cycle costs exactly **1 LLM API call** (Phase 3). The other three phases are zero-cost local operations.

```
Phase 1: Tool Output Pruning          (zero cost, local)
    ↓
Phase 2: Boundary Determination       (zero cost, local)
    ↓
Phase 3: Structured LLM Summarization (1 LLM call)
    ↓
Phase 4: Assembly + Sanitization      (zero cost, local)
```

---

## Phase 1: Tool Output Pruning

### Goal

Reduce token count cheaply before expensive LLM summarization. Long tool results from earlier turns are the biggest token consumers.

### Strategy

Walk the message list and replace old tool result content (in the compressible region, not the protected tail) with a fixed placeholder.

| Condition | Action |
|-----------|--------|
| `role == "tool"` and `len(content) > 200` and message is in compressible region | Replace content with `"[Old tool output cleared]"` |
| `role == "tool"` and `len(content) <= 200` | Keep as-is |
| Message is in protected tail | Keep as-is regardless of length |

### Key Details

- This is a **pre-pass** before boundary determination and summarization
- It is idempotent — already-pruned results are short and won't be pruned again
- The threshold (200 chars) is intentionally low — tool results tend to be either very short (error messages, file paths) or very long (file contents, command output)

---

## Phase 2: Boundary Determination

### Goal

Split the message list into three regions: **head** (always preserved), **compressible** (will be summarized), and **tail** (always preserved).

### Strategy

```
[head: protected] | [compressible: will be summarized] | [tail: protected]
```

#### Head Protection

The first `protect_head` messages (default: 3 — system + first user/assistant exchange) are always preserved. The boundary is **aligned forward** past any orphaned tool results to avoid splitting a tool_call/result pair.

#### Tail Protection by Token Budget

Rather than using a fixed message count, the tail is protected by a **token budget**:

```
tail_budget = token_limit * tail_ratio  (default: 0.15)
```

The algorithm walks backward from the end, accumulating tokens message by message, until the budget is reached. This means:
- A 200K context model gets a ~30K token tail budget
- A 1M context model gets a ~150K token tail budget
- The tail automatically scales with model capability

The boundary is **aligned backward** past tool results to avoid splitting pairs.

#### Boundary Alignment

**Forward alignment** (for head boundary):
```
idx = protect_head
while messages[idx].role == "tool":
    idx += 1  # skip past orphaned tool results
```

**Backward alignment** (for tail boundary):
```
while messages[idx].role == "tool":
    idx -= 1  # skip back past orphaned tool results
```

This ensures we never split an assistant's `tool_calls` from their corresponding `tool` results.

---

## Phase 3: Structured LLM Summarization

### Goal

Generate a structured handoff summary of the compressible region that a "fresh" assistant can use to continue the conversation seamlessly.

### Serialization

Before sending to the LLM, messages are rendered as labeled text:

| Message Type | Serialization Rule |
|-------------|-------------------|
| `assistant` with `tool_calls` | `[assistant calls tool_name(args)]` — args truncated at 1200 chars |
| `assistant` with text | `[assistant] {content}` — truncated at 1000 chars |
| `tool` result | `[tool result] {content}` — if >5500 chars, keep head 4000 + tail 1500 chars |
| `user` | `[user] {content}` — truncated at 1000 chars |

### Structured Summary Format

The summary uses a fixed 8-section structure:

```
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
## Tools & Patterns
```

This structure ensures:
- **Goal** preserves the user's original intent
- **Progress** tracks what's been completed vs. what's pending
- **Key Decisions** preserves architectural choices the user made
- **Relevant Files** preserves file paths the agent was working with
- **Critical Context** catches anything else important

### Iterative Updates (Key Innovation)

On the **first compression**, the LLM generates a summary from scratch:

```
Create a structured handoff summary...

TURNS TO SUMMARIZE:
{serialized content}

Use this exact structure: ...
```

On **subsequent compressions**, the previous summary is preserved and incrementally updated:

```
You are updating a context compaction summary. A previous compaction produced
the summary below. New conversation turns have occurred since then and need
to be incorporated.

PREVIOUS SUMMARY:
{previous_summary}

NEW TURNS TO INCORPORATE:
{serialized content}

Update the summary. PRESERVE all existing information that is still relevant.
ADD new progress. Move items from "In Progress" to "Done" when completed.
```

**Why this matters**: Instead of re-summarizing everything from scratch (which loses information accumulated across multiple compressions), the iterative approach preserves all existing information and only adds new progress. This is inspired by Pi-mono and OpenCode.

### Fallback

If the LLM call fails, a plain-text fallback extracts tool names and result previews:

```python
parts = ["Called: read_file, write_file", "Result: file contents..."]
return "\n".join(parts)
```

---

## Phase 4: Assembly + Sanitization

### Goal

Combine the preserved head, the new summary, and the preserved tail into a valid message list — ensuring no broken tool_call/result pairs.

### Assembly with Role Alternation

The summary message is inserted between head and tail. Its role is chosen carefully:

```
Default: role = "assistant"

if head[-1].role == "assistant" and tail[0].role == "user":
    # Double collision! Can't place assistant before user.
    # Solution: merge summary into first tail message
    tail[0].content = summary + "\n---\n" + tail[0].content

elif head[-1].role == "assistant":
    # Would have two consecutive assistant messages
    summary.role = "user"
```

### Tool Pair Sanitization

After compression, some tool_call/result pairs may be broken — a call might be in the head while its result was in the compressed region, or vice versa.

**Step 1: Remove orphaned results**
```python
# Tool results whose tool_call was removed during compression
if msg.role == "tool" and msg.tool_call_id not in call_ids:
    remove(msg)
```

**Step 2: Add stubs for missing results**
```python
# Tool calls whose results were removed during compression
if tc.id not in result_ids:
    insert_stub(tc.id, content="[result removed during compression]")
```

This prevents API errors from broken tool_call/result pairs.

---

## Comparison: Our Two Strategies

| Dimension | ContextCompressor | HermesContextCompressor |
|-----------|-------------------|------------------------|
| **Phases** | 1 (turn summary) | 4 (prune → boundary → summary → assemble) |
| **LLM calls per compression** | N (one per turn) | 1 (single summary) |
| **Summary format** | 1-2 sentence plain text | 8-section structured handoff |
| **Iterative updates** | No (re-summarizes each time) | Yes (incrementally updates previous summary) |
| **Tail protection** | Fixed (last 1 turn) | Dynamic (token budget scales with context window) |
| **Tool pair repair** | No | Yes (removes orphans, adds stubs) |
| **Tool output pruning** | No | Yes (cheap pre-pass) |
| **Role alternation** | Not handled | Handled (merge on double collision) |
| **Best for** | Simple conversations | Complex multi-tool agent sessions |

---

## Configuration

```python
HermesContextCompressor(
    client=client,
    model="gpt-4o-mini",
    token_limit=80000,    # Token threshold that triggers compression
    tail_ratio=0.15,      # Fraction of token_limit reserved for tail protection
    protect_head=3,        # Number of leading messages to always protect
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `token_limit` | 80000 | Token count threshold that triggers compression |
| `tail_ratio` | 0.15 | Fraction of `token_limit` reserved for tail (15% = ~12K tokens at 80K limit) |
| `protect_head` | 3 | Number of leading messages to always preserve (system + first exchange) |

---

## References

- [Hermes Agent](https://github.com/NousResearch/Hermes-Agent) — Original implementation
- `toy_agent/context.py` — Our implementation of both strategies
- `docs/CONTEXT_COMPRESSION_STRATEGY.md` — Original three-level progressive compression design

---

## Date

2026-04-09
