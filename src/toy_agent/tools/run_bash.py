"""Tool: run bash commands with safety checks."""

import subprocess

from src.toy_agent.tools import tool

# Dangerous commands that must be blocked
BLOCKED_COMMANDS = [
    "rm -rf /",
    "rm -rf /*",
    "mkfs.",
    "dd if=",
    ":(){:|:&};:",
    "> /dev/sd",
    "chmod -R 777 /",
    "chown -R",
]

# Dangerous patterns within any command
BLOCKED_PATTERNS = [
    "rm -rf /",
    "shutdown",
    "reboot",
    "init 0",
    "init 6",
    "poweroff",
    "halt",
]


def _is_dangerous(command: str) -> str | None:
    """Return a reason string if command is dangerous, None otherwise."""
    stripped = command.strip().lower()

    for blocked in BLOCKED_COMMANDS:
        if blocked.lower() in stripped:
            return f"blocked destructive command: {blocked}"

    for pattern in BLOCKED_PATTERNS:
        if pattern.lower() in stripped:
            return f"blocked dangerous pattern: {pattern}"

    # Block shell operators that chain destructive actions
    if stripped.startswith("rm ") and "-rf" in stripped and "/" in stripped:
        return "blocked: rm -rf with root path"

    return None


@tool(description="Run a bash command and return its output. High-risk operations are disabled.")
def run_bash(command: str) -> str:
    """command: The bash command to execute"""
    reason = _is_dangerous(command)
    if reason:
        return f"Error: {reason}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr] {result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code] {result.returncode}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out (30s)"
    except Exception as e:
        return f"Error: {e}"
