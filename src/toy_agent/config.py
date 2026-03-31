"""Config loader: merge mcp.json from user-level and project-level.

Priority (higher overrides lower):
  1. Project-level: .toy-agent/mcp.json
  2. User-level:    ~/.toy-agent/mcp.json

mcp.json format:
{
  "mcpServers": {
    "server-name": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"],
      "env": {"KEY": "value"}
    }
  }
}
"""

import json
from pathlib import Path
from typing import Any


def load_mcp_config(project_dir: str | Path = ".") -> dict[str, Any]:
    """Load and merge mcp.json from user-level and project-level."""
    project_dir = Path(project_dir).resolve()

    user_path = Path.home() / ".toy-agent" / "mcp.json"
    project_path = project_dir / ".toy-agent" / "mcp.json"

    config: dict[str, Any] = {}

    # User-level first (lower priority)
    if user_path.exists():
        config = _load_json(user_path)

    # Project-level overrides (higher priority)
    if project_path.exists():
        project_config = _load_json(project_path)
        config = _deep_merge(config, project_config)

    return config


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"[config] failed to load {path}: {e}")
        return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base. Override wins on conflicts."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
