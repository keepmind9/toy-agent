"""Unit tests for SessionMemory."""

import json
from pathlib import Path

from toy_agent.memory import SessionMemory


class TestSessionMemorySaveLoad:
    def test_save_creates_jsonl_file(self, tmp_path):
        memory = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        filepath = memory.save(messages)

        assert Path(filepath).exists()
        assert filepath.endswith(".jsonl")

        lines = Path(filepath).read_text().strip().split("\n")
        assert len(lines) == 4  # 1 header + 3 messages

        header = json.loads(lines[0])
        assert header["version"] == 1
        assert header["project_path"] == "/fake/project"

    def test_save_returns_filepath_with_timestamp(self, tmp_path):
        memory = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
        filepath = memory.save([{"role": "system", "content": "test"}])

        filename = Path(filepath).stem
        # Should match pattern: YYYY-MM-DD_HHMMSS
        assert len(filename) == 17
        assert filename[4] == "-"
        assert filename[10] == "_"

    def test_save_appends_new_messages(self, tmp_path):
        """Multiple save() calls should append new messages, not rewrite."""
        memory = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
        filepath1 = memory.save([{"role": "user", "content": "turn 1"}])
        filepath2 = memory.save([{"role": "user", "content": "turn 1"}, {"role": "assistant", "content": "reply 1"}])

        # Same file
        assert filepath1 == filepath2

        # Two messages appended (header + 2 lines)
        lines = Path(filepath2).read_text().strip().split("\n")
        assert len(lines) == 3  # 1 header + 2 messages

        msg1 = json.loads(lines[1])
        assert msg1["content"] == "turn 1"
        msg2 = json.loads(lines[2])
        assert msg2["content"] == "reply 1"

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

    def test_load_corrupted_jsonl_returns_none(self, tmp_path):
        memory = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
        memory.sessions_dir.mkdir(parents=True, exist_ok=True)
        (memory.sessions_dir / "bad.jsonl").write_text("not valid json{{{")

        result = memory.load("bad")
        assert result is None

    def test_project_hash_isolation(self, tmp_path):
        mem_a = SessionMemory(project_path="/project/a", base_dir=tmp_path)
        mem_b = SessionMemory(project_path="/project/b", base_dir=tmp_path)

        mem_a.save([{"role": "user", "content": "from A"}])
        mem_b.save([{"role": "user", "content": "from B"}])

        # Different project paths -> different directories
        assert mem_a.sessions_dir != mem_b.sessions_dir


class TestSessionMemoryList:
    def test_list_sessions_sorted_newest_first(self, tmp_path):
        import time

        # Each SessionMemory instance = one session (one agent run)
        mem1 = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
        mem1.save([{"role": "user", "content": "first"}])
        time.sleep(1.1)
        mem2 = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
        mem2.save([{"role": "user", "content": "second"}])

        # Use a fresh instance to list (no _current_session_id bias)
        listing = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
        sessions = listing.list_sessions()
        assert len(sessions) == 2
        assert sessions[0].session_id > sessions[1].session_id

    def test_list_sessions_empty_dir(self, tmp_path):
        memory = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
        assert memory.list_sessions() == []

    def test_list_sessions_skips_corrupted(self, tmp_path):
        memory = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
        memory.save([{"role": "user", "content": "good"}])
        # Add a corrupted file directly into sessions_dir
        memory.sessions_dir.mkdir(parents=True, exist_ok=True)
        (memory.sessions_dir / "2020-01-01_000000.jsonl").write_text("bad json")

        sessions = memory.list_sessions()
        assert len(sessions) == 1


class TestSessionMemoryLoadLatest:
    def test_load_latest_returns_newest(self, tmp_path):
        import time

        mem1 = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
        mem1.save([{"role": "user", "content": "old"}])
        time.sleep(1.1)
        mem2 = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
        mem2.save([{"role": "user", "content": "new"}])

        listing = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
        messages = listing.load_latest()
        assert messages == [{"role": "user", "content": "new"}]

    def test_load_latest_returns_none_when_empty(self, tmp_path):
        memory = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
        assert memory.load_latest() is None


class TestSessionMemoryCleanup:
    def test_cleanup_keeps_newest_n(self, tmp_path):
        import time

        # Create 5 separate sessions
        for i in range(5):
            mem = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
            mem.save([{"role": "user", "content": f"msg-{i}"}])
            if i < 4:
                time.sleep(1.1)

        listing = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
        listing.cleanup(max_sessions=2)
        sessions = listing.list_sessions()
        assert len(sessions) == 2

    def test_cleanup_noop_when_under_limit(self, tmp_path):
        memory = SessionMemory(project_path="/fake/project", base_dir=tmp_path)
        memory.save([{"role": "user", "content": "only one"}])

        memory.cleanup(max_sessions=10)
        sessions = memory.list_sessions()
        assert len(sessions) == 1
