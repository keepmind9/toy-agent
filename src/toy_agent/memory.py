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


class SessionMemory:
    """Manages session persistence for an agent.

    Sessions are stored as JSONL files under <base_dir>/<project_hash>/sessions/.
    Each session is a timestamped .jsonl file where:
      - Line 1: session metadata (version, created_at, project_path)
      - Lines 2+: one message per line (append-only)
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
        self._current_session_id: str | None = None
        self._saved_count: int = 0  # number of messages already written to file

    def save(self, messages: list[dict]) -> str:
        """Append new messages to the current session file.

        Creates a new session on first call, appends only new messages on
        subsequent calls.

        Returns:
            Path to the saved session file.
        """
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        if self._current_session_id is None:
            now = datetime.now()
            self._current_session_id = now.strftime("%Y-%m-%d_%H%M%S")
            header = {
                "version": 1,
                "created_at": now.isoformat(timespec="seconds"),
                "project_path": self.project_path,
            }
            filepath = self.sessions_dir / f"{self._current_session_id}.jsonl"
            with filepath.open("w") as f:
                f.write(json.dumps(header, ensure_ascii=False) + "\n")
            self._saved_count = 0

        # Append only new messages
        new_messages = messages[self._saved_count :]
        if new_messages:
            filepath = self.sessions_dir / f"{self._current_session_id}.jsonl"
            with filepath.open("a") as f:
                for msg in new_messages:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            self._saved_count = len(messages)

        return str(self.sessions_dir / f"{self._current_session_id}.jsonl")

    def load(self, session_id: str) -> list[dict] | None:
        """Load messages from a specific session.

        Args:
            session_id: Session identifier (filename without .jsonl extension).

        Returns:
            List of message dicts, or None if session not found or corrupted.
        """
        filepath = self.sessions_dir / f"{session_id}.jsonl"
        if not filepath.exists():
            return None

        try:
            lines = filepath.read_text().strip().split("\n")
            if not lines:
                return None
            # First line must be valid header
            json.loads(lines[0])
            # Remaining lines are messages
            return [json.loads(line) for line in lines[1:]]
        except (json.JSONDecodeError, IndexError):
            warnings.warn(f"Corrupted session file: {filepath}", stacklevel=2)
            return None

    def list_sessions(self, max_num: int = 0) -> list[SessionMeta]:
        """List saved sessions, sorted by time descending (newest first).

        Args:
            max_num: Maximum number of sessions to return. 0 means all.

        Returns:
            List of SessionMeta objects.
        """
        if not self.sessions_dir.exists():
            return []

        sessions = []
        for f in sorted(self.sessions_dir.glob("*.jsonl"), reverse=True):
            try:
                with f.open() as fh:
                    header_line = fh.readline()
                    if not header_line:
                        continue
                    header = json.loads(header_line)
                sessions.append(
                    SessionMeta(
                        session_id=f.stem,
                        created_at=header.get("created_at", ""),
                    )
                )
                if max_num and len(sessions) >= max_num:
                    break
            except (json.JSONDecodeError, IndexError):
                continue  # skip corrupted files

        return sessions

    def load_latest(self) -> list[dict] | None:
        """Load messages from the most recent session.

        Returns:
            List of message dicts, or None if no sessions exist.
        """
        sessions = self.list_sessions(max_num=1)
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
            filepath = self.sessions_dir / f"{meta.session_id}.jsonl"
            filepath.unlink(missing_ok=True)
