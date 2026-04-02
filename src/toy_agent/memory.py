"""Session persistence — save and restore conversation history across restarts."""

import hashlib
import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class SessionMeta:
    """Metadata for a saved session."""

    session_id: str  # filename without extension, e.g. "2026-04-02_143052"
    created_at: str  # ISO format timestamp
    message_count: int  # number of messages in the session


class SessionMemory:
    """Manages session persistence for an agent.

    Sessions are stored as JSON files under <base_dir>/<project_hash>/sessions/.
    Each session is a timestamped file containing the full message history.
    """

    def __init__(self, project_path: str, base_dir: Path | None = None):
        """Initialize SessionMemory.

        Args:
            project_path: Absolute path of the project directory.
            base_dir: Override base directory (for testing). Defaults to ~/.toy-agent/.
        """
        project_hash = hashlib.sha256(project_path.encode()).hexdigest()[:12]
        root = base_dir or Path.home() / ".toy-agent"
        self.sessions_dir = root / project_hash / "sessions"
        self.project_path = project_path

    def save(self, messages: list[dict]) -> str:
        """Save messages to a new timestamped session file.

        Returns:
            Path to the saved session file.
        """
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filepath = self.sessions_dir / f"{timestamp}.json"

        data = {
            "version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "project_path": self.project_path,
            "messages": messages,
        }

        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        return str(filepath)

    def load(self, session_id: str) -> list[dict] | None:
        """Load messages from a specific session.

        Args:
            session_id: Session identifier (filename without .json extension).

        Returns:
            List of message dicts, or None if session not found or corrupted.
        """
        filepath = self.sessions_dir / f"{session_id}.json"
        if not filepath.exists():
            return None

        try:
            data = json.loads(filepath.read_text())
            return data["messages"]
        except (json.JSONDecodeError, KeyError):
            warnings.warn(f"Corrupted session file: {filepath}", stacklevel=2)
            return None

    def list_sessions(self) -> list[SessionMeta]:
        """List all saved sessions, sorted by time descending (newest first).

        Returns:
            List of SessionMeta objects.
        """
        if not self.sessions_dir.exists():
            return []

        sessions = []
        for f in sorted(self.sessions_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(f.read_text())
                sessions.append(
                    SessionMeta(
                        session_id=f.stem,
                        created_at=data.get("created_at", ""),
                        message_count=len(data.get("messages", [])),
                    )
                )
            except (json.JSONDecodeError, KeyError):
                continue  # skip corrupted files

        return sessions

    def load_latest(self) -> list[dict] | None:
        """Load messages from the most recent session.

        Returns:
            List of message dicts, or None if no sessions exist.
        """
        sessions = self.list_sessions()
        if not sessions:
            return None
        return self.load(sessions[0].session_id)

    def cleanup(self, max_sessions: int = 10) -> None:
        """Delete old sessions, keeping only the N most recent.

        Args:
            max_sessions: Maximum number of sessions to keep.
        """
        sessions = self.list_sessions()  # already sorted newest-first
        for meta in sessions[max_sessions:]:
            filepath = self.sessions_dir / f"{meta.session_id}.json"
            filepath.unlink(missing_ok=True)
