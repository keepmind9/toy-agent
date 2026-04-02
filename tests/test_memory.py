"""Unit tests for SessionMemory."""

import json
from pathlib import Path

import pytest

from src.toy_agent.memory import SessionMemory, SessionMeta


class TestSessionMemorySaveLoad:
    def test_save_creates_session_file(self, tmp_path):
        memory = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        filepath = memory.save(messages)

        assert Path(filepath).exists()
        data = json.loads(Path(filepath).read_text())
        assert data["version"] == 1
        assert data["project_path"] == "/fake/project"
        assert len(data["messages"]) == 3

    def test_save_returns_filepath_with_timestamp(self, tmp_path):
        memory = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
        filepath = memory.save([{"role": "system", "content": "test"}])

        filename = Path(filepath).stem
        # Should match pattern: YYYY-MM-DD_HHMMSS
        assert len(filename) == 17
        assert filename[4] == "-"
        assert filename[10] == "_"

    def test_load_returns_messages(self, tmp_path):
        memory = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
        messages = [{"role": "user", "content": "hello"}]
        filepath = memory.save(messages)

        session_id = Path(filepath).stem
        loaded = memory.load(session_id)
        assert loaded == messages

    def test_load_nonexistent_returns_none(self, tmp_path):
        memory = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
        result = memory.load("nonexistent-session")
        assert result is None

    def test_load_corrupted_json_returns_none(self, tmp_path):
        memory = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        (sessions_dir / "bad.json").write_text("not valid json{{{")

        result = memory.load("bad")
        assert result is None

    def test_project_hash_isolation(self, tmp_path):
        mem_a = SessionMemory(project_path="/project/a", base_dir=tmp_path)
        mem_b = SessionMemory(project_path="/project/b", base_dir=tmp_path)

        mem_a.save([{"role": "user", "content": "from A"}])
        mem_b.save([{"role": "user", "content": "from B"}])

        # Different project paths -> different directories
        assert mem_a.sessions_dir != mem_b.sessions_dir
